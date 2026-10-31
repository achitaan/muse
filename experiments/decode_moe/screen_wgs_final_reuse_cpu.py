"""One WGS finalizer read-reuse resource screen. NVRTC only; no GPU calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/decode-moe-plan-20261030/wgs-final-reuse'
BASE = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'


def derive():
    raw = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE
    vals = {n.targets[0].id: ast.literal_eval(n.value)
            for n in ast.parse(raw).body if isinstance(n, ast.Assign)
            and getattr(n.targets[0], 'id', '') in ('CUDA_SRC', 'MAIN_DEFINES', 'DQ7', 'MAIN_NS')}
    parent = vals['CUDA_SRC']
    first = parent.index('#if OKC_WGS && OKC_WGSFINSS\n')
    last = parent.index('#elif OKC_FINCOL\n', first)
    original = parent[first:last]
    changed = original
    edits = [
        ('      float ps[8];', '      float ps[8];\n      float4 keep_a[8], keep_b[8];'),
        ('          ps[sl] = a.x * a.x',
         '          keep_a[sl] = a; keep_b[sl] = b;\n          ps[sl] = a.x * a.x'),
        ('          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 2;\n'
         '          const float4 a = __ldcg(src), b = __ldcg(src + 1);\n          acc[0]',
         '          const float4 a = keep_a[sl], b = keep_b[sl];\n          acc[0]'),
    ]
    for old, new in edits:
        assert changed.count(old) == 1, old
        changed = changed.replace(old, new, 1)
    reverse = changed
    for old, new in reversed(edits):
        assert reverse.count(new) == 1
        reverse = reverse.replace(new, old, 1)
    assert reverse == original
    candidate = parent[:first] + changed + parent[last:]
    assert parent.count('float4 keep_a') == 0
    assert changed.count('__ldcg(src)') == 1 and original.count('__ldcg(src)') == 2
    assert changed.count('if (qv[sl] >= 0)') == original.count('if (qv[sl] >= 0)')
    return parent, candidate, vals


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / '.gitignore').write_bytes(b'*.cubin\n')
    parent, candidate, vals = derive()
    (OUT / 'candidate.cu').write_bytes(candidate.encode())
    sys.path.insert(0, str(ROOT.parent / 'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv

    class Capture(nv.NVRTC):
        def output(self, program, kind):
            result = super().output(program, kind)
            if kind == 'CUBIN':
                self.cubin = result
            return result

    # Retain the exact selected WGS geometry and PRMT eligibility. No sweep.
    flags = list(vals['MAIN_DEFINES']) + ['-DOKC_PRMTSC=1', '-DOKC_NS=' + str(vals['MAIN_NS']), '-DOKC_NSF=3']
    if vals['DQ7']:
        flags += ['-DOKC_DQ7=1']
    report = dict(baseline_sha256=BASE,
                  baseline_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                  candidate_CUDA_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
                  exact_inverse=True, flags=flags, resources={},
                  retained_math='Same FP32 sum expressions, shuffle/reduction order, epsilon, route order and BF16 output conversion.',
                  ownership='Retained float4 values are thread-private. Both passes use the same qv>=0 guard; invalid slots are never read.',
                  unchanged='All GEMM, routing, scratch capacities, counter resets, barriers, fallbacks and dispatch.',
                  added_live_values_per_thread=64,
                  logical_read_saving_bytes_per_local_route=4096 * 4,
                  CPU_portability_shim='Forward-declare wgs_cta_of in both NVRTC inputs. Declaration only; not part of candidate production source.',
                  stopping_rule='One variant and two compiles. Reject any register, stack, spill or local-instruction regression. A clean screen alone does not admit GPU work.',
                  GPU_jobs=0, admitted_for_GPU=False)
    exe = ROOT.parents[2] / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    for label, src in [('parent', parent), ('candidate', candidate)]:
        compiler = Capture()
        code = src.split('#include <torch/extension.h>', 1)[0]
        anchor = 'template <bool PB>\n__device__ __forceinline__ void wgs_phase'
        declaration = '__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'
        assert code.count(anchor) == 1
        code = code.replace(anchor, declaration + anchor, 1)
        result = compiler.compile(code.encode(),
                                  ['&okc::moe_fused<false>'], flags, tags=True, lineinfo=True)
        (OUT / (label + '.log')).write_bytes(result['log'].encode())
        assert result['rc'] == 0, result['log'][-3000:]
        binary = OUT / (label + '.cubin')
        binary.write_bytes(compiler.cubin)
        sass = subprocess.check_output([str(exe), '-sass', str(binary)], text=True)
        (OUT / (label + '.sass')).write_bytes(sass.encode())
        report['resources'][label] = dict(kernels=nv.parse_ptxas(result['log']),
            cubin_sha256=result['cubin_sha256'],
            instructions=len(re.findall(r'/\*[0-9a-fA-F]+\*/\s+.*?;', sass)),
            local_instructions=len(re.findall(r'\b(?:LDL|STL)\b', sass)),
            global_load_instructions=len(re.findall(r'\bLDG(?:\.|\s)', sass)))
        print(label, json.dumps(report['resources'][label]), flush=True)
    a, b = report['resources']['parent'], report['resources']['candidate']
    clean = b['local_instructions'] <= a['local_instructions']
    assert len(a['kernels']) == len(b['kernels'])
    for x, y in zip(a['kernels'], b['kernels']):
        assert x['symbol'] == y['symbol']
        clean &= all(y[k] <= x[k] for k in ('registers', 'stack_frame', 'spill_stores', 'spill_loads'))
    report['no_resource_regression'] = clean
    report['decision'] = ('CPU resource gate passed; exposed-time evidence missing, no GPU admission'
                          if clean else 'CPU resource regression; reject, no GPU')
    (OUT / 'cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(report['decision'])


if __name__ == '__main__':
    main()
