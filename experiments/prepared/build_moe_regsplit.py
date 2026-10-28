"""One full producer warpgroup; fixed, audited dynamic register ownership."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
BASE_SHA = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
NTL_OLD = 'constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);'
NTL_NEW = 'constexpr int NTL = NT + (OKC_WGS ? 128 : OKC_WS ? 32 : 0);'
ROLE_OLD = '''#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif'''
ROLE_NEW = '''#if OKC_WGS
  // Local change: a complete producer warpgroup releases registers before
  // any wait on the consumers. All four compute warpgroups claim the same
  // fixed budget. The last two producer warps only join barrier 2 and exit.
  static_assert(NT == 512 && NTL == 640, "register split requires four compute warpgroups");
  if (tid >= NT) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;\\n" ::: "memory");
    if (tid >= NT + 64) { WS_JOIN(); return; }
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
  asm volatile("setmaxnreg.inc.sync.aligned.u32 112;\\n" ::: "memory");
#endif'''
RESOURCES = '''
// Local change: ordinary compiler/device resource preflight, prepare only.
std::vector<int64_t> okc_regsplit_resources() {
  cudaFuncAttributes a{}, b{};
  TORCH_CHECK(cudaFuncGetAttributes(&a, okc::moe_fused<false>) == cudaSuccess, "small resource query failed");
  TORCH_CHECK(cudaFuncGetAttributes(&b, okc::moe_fused<true>) == cudaSuccess, "large resource query failed");
  const size_t sm = okc_smem_bytes();
  TORCH_CHECK(cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) == cudaSuccess,
              "small shared-memory opt-in failed");
  TORCH_CHECK(cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm) == cudaSuccess,
              "large shared-memory opt-in failed");
  int oa = 0, ob = 0, device = 0;
  cudaDeviceProp prop{};
  TORCH_CHECK(cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess,
              "device resource query failed");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oa, okc::moe_fused<false>, okc::NTL, sm) == cudaSuccess,
              "small occupancy query failed");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ob, okc::moe_fused<true>, okc::NTL, sm) == cudaSuccess,
              "large occupancy query failed");
  return {(int64_t)okc::NTL, (int64_t)a.numRegs, (int64_t)b.numRegs,
          (int64_t)a.localSizeBytes, (int64_t)b.localSizeBytes, (int64_t)sm,
          (int64_t)oa, (int64_t)ob, (int64_t)prop.regsPerMultiprocessor,
          (int64_t)prop.multiProcessorCount};
}
'''
PREPARE_PREFLIGHT = '''
regsplit_enabled = bool(state['sfx'] and '-DOKC_WGS=1' in tuple(_main_defines(state['sfx'])))
if regsplit_enabled:
    resources = tuple(_ext(tuple(_main_defines(state['sfx'])) + state['sfx']).okc_regsplit_resources())
    print('MOE_REGSPLIT_RESOURCES', resources, flush=True)
    assert len(resources) == 10 and resources[0] == 640
    assert resources[1] == resources[2] == 96, 'Require the audited 96-register initial budget'
    assert min(resources[6:8]) >= 1 and resources[8] >= 61440
    assert resources[9] * min(resources[6:8]) >= state['grid'], 'Persistent grid must fit'
    assert torch.cuda.get_device_capability(dev) == (9, 0)
print('MOE_REGSPLIT_ENABLED', regsplit_enabled, flush=True)
'''


def build():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    text = raw.decode('utf8')
    tree = ast.parse(text)
    literals = {n.targets[0].id: n.value.value for n in tree.body
                if isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
    cuda = literals['CUDA_SRC']
    for old, new in ((NTL_OLD, NTL_NEW), (ROLE_OLD, ROLE_NEW)):
        assert cuda.count(old) == 1
        cuda = cuda.replace(old, new, 1)
    cuda += RESOURCES
    cpp = literals['CPP_SRC'] + '\nstd::vector<int64_t> okc_regsplit_resources();\n'
    edits = []
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            if n.targets[0].id in ('CUDA_SRC', 'CPP_SRC'):
                v = cuda if n.targets[0].id == 'CUDA_SRC' else cpp
                edits.append((n, n.targets[0].id + ' = ' + repr(v)))
        if not isinstance(n, ast.FunctionDef):
            continue
        if n.name == '_ext':
            for child in ast.walk(n):
                if isinstance(child, ast.Constant) and isinstance(child.value, str) and 'v216x_pair1dq8_' in child.value:
                    child.value = child.value.replace('v216x_pair1dq8_', 'v216x_pair1dq8regsplit_')
                if isinstance(child, ast.keyword) and child.arg == 'functions':
                    assert ast.literal_eval(child.value) == ['okc_run', 'okc_set_pdl']
                    child.value.elts.append(ast.Constant('okc_regsplit_resources'))
            edits.append((n, ast.unparse(n)))
        elif n.name == 'prepare':
            assert isinstance(n.body[-1], ast.Return)
            n.body[-1:-1] = ast.parse(PREPARE_PREFLIGHT).body
            edits.append((n, ast.unparse(n)))
    lines = text.splitlines(keepends=True)
    for n, v in sorted(edits, key=lambda x: x[0].lineno, reverse=True):
        lines[n.lineno-1:n.end_lineno] = [v + '\n']
    result = ''.join(lines).encode('utf8')
    compile(result, '<moe-register-split>', 'exec')
    (ROOT / '.tools/moe-regsplit-variant.py').write_bytes(result)
    (ROOT / '.tools/moe-regsplit-inspect.cu').write_bytes(cuda.encode('utf8'))
    record = {'base_sha256': BASE_SHA, 'source_sha256': hashlib.sha256(result).hexdigest(),
              'packaged_bytes': len(result), 'consumer_threads': 512, 'producer_group_threads': 128,
              'useful_producer_threads': 64, 'padding_threads': 64, 'initial_registers': 96,
              'compute_registers': 112, 'producer_registers': 32, 'total_registers': 61440,
              'compile_variants': 4, 'extra_scratch_or_weight_copies': 0,
              'preflight': 'actual attributes must show initial96, occupancy>=1 and resident persistent grid',
              'gpu_status': 'unmeasured'}
    (ROOT / '.tools/moe-regsplit-source-audit.json').write_text(json.dumps(record, indent=2)+'\n',encoding='utf8')
    print(json.dumps(record))


if __name__ == '__main__':
    build()
