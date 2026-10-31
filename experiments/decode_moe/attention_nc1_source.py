"""Package only the two screened nc1 entry points; no CUDA runtime calls."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/decode-moe-plan-20261030/attention-nc1'
PARENT_SHA = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'

LAUNCH = r'''
void launch_nc1(const Params& p, dim3 grid, cudaStream_t stream) {
  const int sm_f = (int)sizeof(Smem<1, true>) + 1024;
  const int sm_n = (int)sizeof(Smem<1, false>) + 1024;
  // No cooperative barrier or cluster/PDL launch is reachable in this module.
  const int sm = p.flat ? sm_f : sm_n;
  typedef void (*KernFn)(const Params);
  KernFn kern = p.flat ? (KernFn)attn_kernel_nc1<1, true, 0>
                      : (KernFn)attn_kernel_nc1<1, false, 13>;
  kern<<<grid, p.flat ? Cfg<1, true>::NTX : Cfg<1, false>::NTX, sm, stream>>>(p);
}
'''

GUARD = r'''
  TORCH_CHECK(L == 1 && nc == 1 && tma == 0 && cs <= 1 && B >= 1 &&
              window >= 0 && n_splits >= 1 && p.max_pages >= 1,
              "nc1 module used outside its guarded metadata family");
  const long long kmax = window ? std::min<long long>(window + L - 1,
                              (long long)p.max_pages * P) : (long long)p.max_pages * P;
  TORCH_CHECK(flat || (B == 1 && P == 1 && n_splits >= (kmax + 63) / 64),
              "nc1 module requires flat-L1 or noncooperative SPEC-ONE-L1");
  launch_nc1(p, grid, stream);
'''

RESOURCES = r'''
std::vector<int64_t> nc1_resources() {
  typedef void (*KernFn)(const Params);
  const KernFn functions[2] = {attn_kernel_nc1<1, true, 0>,
                               attn_kernel_nc1<1, false, 13>};
  const int sm[2] = {(int)sizeof(Smem<1, true>) + 1024,
                    (int)sizeof(Smem<1, false>) + 1024};
  const int threads[2] = {Cfg<1, true>::NTX, Cfg<1, false>::NTX};
  std::vector<int64_t> out;
  for (int i = 0; i < 2; ++i) {
    auto err = cudaFuncSetAttribute(functions[i], cudaFuncAttributeMaxDynamicSharedMemorySize, sm[i]);
    TORCH_CHECK(err == cudaSuccess, "nc1 shared-memory attribute: ", cudaGetErrorString(err));
    cudaFuncAttributes attr{};
    err = cudaFuncGetAttributes(&attr, functions[i]);
    TORCH_CHECK(err == cudaSuccess, "nc1 function attributes: ", cudaGetErrorString(err));
    int blocks = 0;
    err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, functions[i], threads[i], sm[i]);
    TORCH_CHECK(err == cudaSuccess, "nc1 occupancy query: ", cudaGetErrorString(err));
    out.insert(out.end(), {attr.numRegs, (int64_t)attr.localSizeBytes,
                           (int64_t)attr.sharedSizeBytes, sm[i], threads[i], blocks});
  }
  return out;
}
'''


def derive():
    raw = (ROOT / 'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    vals = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(raw).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('CUDA_SRC', 'CPP_SRC')}
    original = vals['CUDA_SRC']
    screened = (OUT / 'candidate.cu').read_text()
    assert hashlib.sha256(screened.encode()).hexdigest() == 'e29efa5e07587dfa5f61d1c0a9b51946bfc6eca6c4462370a86750359a0fd6bc'
    split = 'template <int L>\nvoid launch('
    device, rest = screened.split(split, 1)
    offset = rest.index('typedef CUresult (*EncodeTiledFn)')
    host = rest[offset:]
    old = '  if (L == 1) launch<1>(p, grid, stream, (int)cs, pdl);\n  else launch<8>(p, grid, stream, (int)cs, pdl);'
    assert host.count(old) == 1 and host.count('void run(') == 1
    guarded = host.replace(old, GUARD, 1).replace('void run(', 'void run_nc1(', 1)
    assert guarded.replace('void run_nc1(', 'void run(', 1).replace(GUARD, old, 1) == host
    src = device + LAUNCH + guarded + '\n' + RESOURCES
    src = src.replace('attn_kernel_fp8pv', 'attn_kernel_nc1')
    assert src.count(LAUNCH) == 1
    assert src.split(LAUNCH, 1)[0].replace('attn_kernel_nc1', 'attn_kernel_fp8pv') == device
    cpp = vals['CPP_SRC'].replace('void run(', 'void run_nc1(', 1) + '\nstd::vector<int64_t> nc1_resources();\n'
    return raw, original, src, cpp


def main():
    _, original, src, cpp = derive()
    (OUT / 'packaged.cu').write_bytes(src.encode())
    (OUT / 'candidate.cpp').write_bytes(cpp.encode())
    binding = dict(parent_sha256=PARENT_SHA, parent_CUDA_sha256=hashlib.sha256(original.encode()).hexdigest(),
                   packaged_CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),
                   device_prefix_inverse_exact=True, run_parameter_setup_inverse_exact=True,
                   instantiated_entries=['attn_kernel_nc1<1,true,0>', 'attn_kernel_nc1<1,false,13>'],
                   kernel_math='Screened prefix unchanged except identifier rename;nc==1 integer specialization only',
                   host_scope='Preserve original Params setup and grid;guard requested coop/PDL off;instantiate two entries only',
                   resource_query='Six values per family;called before launch to set shared-memory attributes and validate resources',
                   GPU_calls=0, admitted_for_GPU=False)
    (OUT / 'packaging-binding.json').write_bytes((json.dumps(binding, indent=2) + '\n').encode())
    print(json.dumps(binding, indent=2))


if __name__ == '__main__':
    main()
