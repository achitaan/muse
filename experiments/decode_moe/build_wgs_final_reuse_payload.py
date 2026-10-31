"""Freeze the one WGS read-reuse diagnostic; no GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from screen_wgs_final_reuse_cpu import ROOT, OUT, BASE, derive
from harness import append_harness

HERE = Path(__file__).resolve().parent


def sources():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    parent, candidate, vals = derive()
    assert hashlib.sha256(candidate.encode()).hexdigest() == 'b6199d97ff4907e55b674e55b1f8f04f38df7f7c90c2f0ed2ac5c5df429d9520'
    tree = ast.parse(raw)
    cpp = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], 'id', '') == 'CPP_SRC')
    changed = candidate.replace('namespace okc {', 'namespace wr {').replace('okc::', 'wr::')
    for old, new in [('moe_fused', 'moe_readreuse'), ('okc_run', 'wr_run'),
                     ('okc_set_pdl', 'wr_set_pdl'), ('okc_smem_bytes', 'wr_smem_bytes'), ('g_okc_pdl', 'g_wr_pdl')]:
        assert old in changed
        changed = changed.replace(old, new)
        cpp = cpp.replace(old, new)
    changed += '''
std::vector<int64_t> wr_resources() {
  std::vector<int64_t> result;
  const size_t sm = wr_smem_bytes();
  for (int big = 0; big < 2; ++big) {
    const auto kernel = big ? wr::moe_readreuse<true> : wr::moe_readreuse<false>;
    auto error = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    TORCH_CHECK(error == cudaSuccess, "wr shared attribute: ", cudaGetErrorString(error));
    cudaFuncAttributes attrs{};
    error = cudaFuncGetAttributes(&attrs, kernel);
    TORCH_CHECK(error == cudaSuccess, "wr attributes: ", cudaGetErrorString(error));
    int blocks = 0;
    error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, wr::NTL, sm);
    TORCH_CHECK(error == cudaSuccess, "wr occupancy: ", cudaGetErrorString(error));
    result.insert(result.end(), {(int64_t)attrs.numRegs, (int64_t)attrs.localSizeBytes,
      (int64_t)attrs.sharedSizeBytes, (int64_t)blocks, (int64_t)sm, (int64_t)wr::NTL});
  }
  return result;
}
'''
    cpp += '\nstd::vector<int64_t> wr_resources();\n'
    return raw, parent, changed, cpp, vals


def main():
    raw, parent, candidate, cpp, vals = sources()
    (OUT / 'packaged.cu').write_bytes(candidate.encode())
    (OUT / 'candidate.cpp').write_bytes(cpp.encode())
    live = (HERE / 'wgs_live_snapshot.py').read_bytes()
    runtime = (HERE / 'wgs_final_reuse_runtime.py').read_bytes()
    payload = (append_harness(raw.decode()) + '\nWR_CUDA = ' + repr(candidate)
               + '\nWR_CPP = ' + repr(cpp) + '\n' + live.decode() + '\n' + runtime.decode()).encode()
    ast.parse(payload)
    assert payload.startswith(raw.rstrip(b'\n')) and len(payload) < 1_000_000
    (HERE / 'moe_wgs_final_reuse.py').write_bytes(payload)
    if '--repack' in sys.argv:
        report = json.loads((OUT / 'payload-binding.json').read_bytes())
        assert report['packaged_CUDA_sha256'] == hashlib.sha256(candidate.encode()).hexdigest()
        report.update(source_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                      runtime_sha256=hashlib.sha256(runtime).hexdigest(),
                      live_snapshot_sha256=hashlib.sha256(live).hexdigest())
        (OUT / 'payload-binding.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
        print('Repacked runtime only; native source and successful compiler evidence unchanged.', report['source_sha256'])
        return
    sys.path.insert(0, str(ROOT.parent / 'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv

    class Capture(nv.NVRTC):
        def output(self, program, kind):
            value = super().output(program, kind)
            if kind == 'CUBIN':
                self.cubin = value
            return value

    flags = list(vals['MAIN_DEFINES']) + ['-DOKC_PRMTSC=1', '-DOKC_NS=3', '-DOKC_NSF=3', '-DOKC_DQ7=1']
    compiles = {}
    exe = ROOT.parents[2] / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    for label, src, namespace, kernel in [('parent-both', parent, 'okc', 'moe_fused'),
                                         ('packaged-both', candidate, 'wr', 'moe_readreuse')]:
        code = src.split('#include <torch/extension.h>', 1)[0]
        anchor = 'template <bool PB>\n__device__ __forceinline__ void wgs_phase'
        declaration = '__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'
        assert code.count(anchor) == 1
        code = code.replace(anchor, declaration + anchor, 1)
        compiler = Capture()
        result = compiler.compile(code.encode(), ['&' + namespace + '::' + kernel + '<false>',
                                                  '&' + namespace + '::' + kernel + '<true>'], flags,
                                  tags=True, lineinfo=True)
        (OUT / (label + '.log')).write_bytes(result['log'].encode())
        assert result['rc'] == 0, result['log'][-2500:]
        binary = OUT / (label + '.cubin')
        binary.write_bytes(compiler.cubin)
        sass = subprocess.check_output([str(exe), '-sass', str(binary)], text=True)
        (OUT / (label + '.sass')).write_bytes(sass.encode())
        compiles[label] = dict(kernels=nv.parse_ptxas(result['log']),
            cubin_sha256=result['cubin_sha256'], local_instructions=len(re.findall(r'\b(?:LDL|STL)\b', sass)))
        print(label, json.dumps(compiles[label]), flush=True)
    a, b = compiles['parent-both'], compiles['packaged-both']
    assert len(a['kernels']) == len(b['kernels']) == 2
    for x, y in zip(a['kernels'], b['kernels']):
        assert all(y[key] <= x[key] for key in ('registers', 'stack_frame', 'spill_stores', 'spill_loads'))
        assert y['registers'] <= 96 and y['stack_frame'] <= 16
    assert b['local_instructions'] <= a['local_instructions']
    report = dict(source_sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
        parent_sha256=BASE, packaged_CUDA_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        runtime_sha256=hashlib.sha256(runtime).hexdigest(), live_snapshot_sha256=hashlib.sha256(live).hexdigest(),
        flags=flags, compiles=compiles, actual_marker='wr::moe_readreuse',
        probes='M17,31,32,33,128,256,257,384; five changing routes; live xB/ybuf/out bytes and parent repeat.',
        timing='M32,128,256; normal/skew; both orders;25 samples/arm;all12 below.97;then6 unchanged controls within2%.',
        numerical='Strict bytes even though bounded MoE norm variance is authorized. No legacy WGS inter/SS gates.',
        stopping='One focused check after admission; no noise retries or sweep; public fallback if any gate fails.',
        GPU_jobs=0, admitted_for_GPU=False)
    (OUT / 'payload-binding.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print('payload', report['source_sha256'], report['bytes'], flush=True)


if __name__ == '__main__':
    main()
