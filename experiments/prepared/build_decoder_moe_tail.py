"""Source-bound tiny Decode-tail prototype; CPU preparation only, no launch."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
CUDA = '1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'


def main():
    raw = (ROOT / 'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '_native_moe_CUDA_SRC' for t in n.targets))
    old = ast.literal_eval(node.value)
    assert hashlib.sha256(old.encode()).hexdigest() == CUDA
    helper = (ROOT / 'experiments/prepared/decoder_moe_tail.cuh').read_text()
    marker = '// Sixteen original BF16 output rows per CTA.'
    struct, functions = helper.split(marker, 1)
    functions = marker + functions
    edits = []
    new = old

    def replace(a, b):
        nonlocal new
        assert new.count(a) == 1, a[:90]
        new = new.replace(a, b)
        edits.append(dict(old=a, new=b))

    replace('struct Params {', struct + '\nstruct Params {')
    replace('  unsigned long long* dbg;\n};', '  unsigned long long* dbg;\n  DecoderTail tail;\n};')
    replace('template <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {',
            functions + '\ntemplate <bool BIG>\n__global__ void __launch_bounds__(NTL, 1) moe_fused(Params p) {')
    replace('    if (blockIdx.x == 0 && tid == 0) p.ctr[4 + ((S.gen + 1) & 1)] = 0;',
            '    if (blockIdx.x == 0 && tid == 0) {\n'
            '      p.ctr[4 + ((S.gen + 1) & 1)] = 0;\n'
            '      p.ctr[6 + ((S.gen + 1) & 1)] = 0;\n    }')
    replace('    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;\n  }',
            '    *reinterpret_cast<uint2*>(p.out + (size_t)tkp * HD + col) = o;\n'
            '    __threadfence();  // every writer publishes its own four BF16 values\n'
            '    CTA_SYNC();\n'
            '    if (tid == 0) {\n'
            '      int prior;\n'
            '      asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\\n"\n'
            '                   : "=r"(prior) : "l"(p.tail.ready + tkp) : "memory");\n'
            '    }\n  }')
    replace('  DBG(5);  // nothing to reset at exit: the flags were zeroed after the barrier, ss is double buffered',
            '  decoder_tail_project(p, S);\n'
            '  __threadfence();\n  CTA_SYNC();\n'
            '  if (tid == 0) {\n'
            '    int* bc = p.ctr + 6 + (S.gen & 1);\n'
            '    int prior;\n'
            '    asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;\\n"\n'
            '                 : "=r"(prior) : "l"(bc) : "memory");\n'
            '    while (ld_acquire(bc) < (int)gridDim.x) __nanosleep(32);\n'
            '  }\n  CTA_SYNC();\n'
            '  decoder_tail_boundary(p, S);\n'
            '  DBG(5);  // tail readiness cleared only after every consumer finished')
    replace('void okc_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n'
            '             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid) {',
            'void okc_run_tail(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch, torch::Tensor x, torch::Tensor ids,\n'
            '             torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t nloc, int64_t grid, std::vector<torch::Tensor> tail) {')
    replace('  p.out = (__nv_bfloat16*)out.data_ptr();',
            '  TORCH_CHECK(p.M >= 1 && p.M <= 8 && grid >= 64 && grid >= p.M * 4, "tiny tail bounds");\n'
            '  TORCH_CHECK(tail.size() == 9, "tiny tail tensors");\n'
            '  p.tail.weight = (const uint4*)tail[0].data_ptr();\n'
            '  p.tail.branch = (__nv_bfloat16*)tail[1].data_ptr();\n'
            '  p.tail.stream = (const float*)tail[2].data_ptr();\n'
            '  p.tail.post_weight = (const __nv_bfloat16*)tail[3].data_ptr();\n'
            '  p.tail.pre_weight = (const __nv_bfloat16*)tail[4].data_ptr();\n'
            '  p.tail.alpha = (const float*)tail[5].data_ptr();\n'
            '  p.tail.beta = (const float*)tail[6].data_ptr();\n'
            '  p.tail.output = (__nv_bfloat16*)tail[7].data_ptr();\n'
            '  p.tail.updated = (float*)tail[8].data_ptr();\n'
            '  p.tail.ready = ib + 8 + 768;\n'
            '  p.out = (__nv_bfloat16*)out.data_ptr();')
    launch_begin = new.index('  if (g_okc_pdl) {')
    launch_end = new.index('\n}', launch_begin)
    old_launch = new[launch_begin:launch_end]
    replace(old_launch,
            '  void* args[] = {&p};\n'
            '  cudaError_t launched = p.M >= OKC_BIGM\n'
            '    ? cudaLaunchCooperativeKernel((const void*)okc::moe_fused<true>, dim3((unsigned)grid), dim3(okc::NTL), args, sm, st)\n'
            '    : cudaLaunchCooperativeKernel((const void*)okc::moe_fused<false>, dim3((unsigned)grid), dim3(okc::NTL), args, sm, st);\n'
            '  TORCH_CHECK(launched == cudaSuccess, "tiny tail cooperative launch: ", cudaGetErrorString(launched));')
    inverse = new
    for edit in reversed(edits):
        assert inverse.count(edit['new']) == 1
        inverse = inverse.replace(edit['new'], edit['old'])
    assert inverse == old
    # Dedicated namespace is visible in official profiles. No original module is
    # modified at runtime, and no other retained path is redirected.
    assert new.count('namespace okc {') == 1
    new = new.replace('namespace okc {', 'namespace okc_tail {').replace('okc::', 'okc_tail::')
    host = (ROOT / 'experiments/prepared/decoder_moe_tail_host.cuh').read_text()
    new += '\n' + host
    out = ROOT / '.tools/decoder-moe-tail-20261029'
    out.mkdir(exist_ok=True)
    (out / 'prototype.cu').write_bytes(new.encode())
    report = dict(parent_decode_sha256=PARENT, parent_CUDA_sha256=CUDA,
                  prototype_CUDA_sha256=hashlib.sha256(new.encode()).hexdigest(),
                  helper_sha256=hashlib.sha256(helper.encode()).hexdigest(),
                  host_sha256=hashlib.sha256(host.encode()).hexdigest(),
                  exact_inverse_restores_parent=True, namespace_change='okc -> okc_tail',
                  edits=edits, GPU_jobs=0, Python_candidate=False,
                  admission='Not admitted: native resources, ownership, numerical, replay, host ABI and net-cost gates pending')
    (out / 'binding.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k != 'edits'}, indent=2))


if __name__ == '__main__':
    main()
