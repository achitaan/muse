"""Fixed nc==1 compiler specialization; no launch-plan tuning or GPU calls."""
import hashlib
import json
import re
import subprocess

from build_attention_finoutline import ROOT, derive
from audit_attention_finoutline_cpu import Capture, blocks, nv

OUT = ROOT / 'results/decode-moe-plan-20261030/attention-nc1'
FAMILIES = {'flat_L1': '&attn_kernel_fp8pv<1,true,0>',
            'spec_one_L1': '&attn_kernel_fp8pv<1,false,13>'}


def main():
    OUT.mkdir(exist_ok=True)
    raw, _, _, _, parent, _ = derive()
    split = 'template <int L>\nvoid launch('
    device, host = parent.split(split, 1)
    token = 'OKC_FIXED_NC'
    assert token not in parent
    changed, count = re.subn(r'\bp\.nc\b', token, device)
    assert count > 0 and changed.replace(token, 'p.nc') == device
    candidate = '#define ' + token + ' 1\n' + changed + split + host
    (OUT / 'parent.cu').write_bytes(parent.encode())
    (OUT / 'candidate.cu').write_bytes(candidate.encode())
    exe = ROOT.parents[2] / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    resources = {}
    for family, symbol in FAMILIES.items():
        resources[family] = {}
        for label, src in [('parent', parent), ('nc1', candidate)]:
            code = src.split(split, 1)[0]
            code = re.sub(r'#include <(torch/extension.h|ATen/cuda/CUDAContext.h|type_traits|cuda.h)>\n', '', code)
            code = ('#include <math_constants.h>\n#define INFINITY CUDART_INF_F\n'
                    'namespace std { template<class T,T V> struct integral_constant { static constexpr T value=V; }; }\n' + code)
            compiler = Capture()
            r = compiler.compile(code.encode(), [symbol], [], tags=True, lineinfo=True)
            stem = family + '-' + label
            (OUT / (stem + '.log')).write_bytes(r['log'].encode())
            assert r['rc'] == 0, r['log'][-3000:]
            binary = OUT / (stem + '.cubin')
            binary.write_bytes(compiler.cubin)
            sass = subprocess.check_output([str(exe), '-sass', str(binary)], text=True)
            (OUT / (stem + '.sass')).write_bytes(sass.encode())
            entry = next(v for k, v in blocks(sass).items() if 'attn_kernel_fp8pv' in k)
            resources[family][label] = dict(kernels=nv.parse_ptxas(r['log']), entry=entry,
                                            cubin_sha256=r['cubin_sha256'])
            print(family, label, json.dumps(resources[family][label])[:1100], flush=True)
    gates = {}
    for family, pair in resources.items():
        a, b = pair['parent'], pair['nc1']
        ratio = b['entry']['instructions'] / a['entry']['instructions']
        clean = not b['entry']['local_instructions']
        assert len(a['kernels']) == len(b['kernels'])
        for x, y in zip(a['kernels'], b['kernels']):
            clean = clean and all(y[k] <= x[k] for k in ('registers', 'stack_frame', 'spill_stores', 'spill_loads'))
        gates[family] = dict(instruction_ratio=ratio, no_resource_regression=clean,
                             passes_fixed_CPU_gate=clean and ratio <= .95)
    passed = all(g['passes_fixed_CPU_gate'] for g in gates.values())
    report = dict(parent_sha256=hashlib.sha256(raw).hexdigest(),
                  parent_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
                  CUDA_sha256=hashlib.sha256(candidate.encode()).hexdigest(), exact_inverse=True,
                  substitutions=count, fixed_metadata={'nc': 1}, resources=resources, gates=gates,
                  decision='CPU gate passed;not GPU admitted' if passed else 'CPU reject;no GPU',
                  scope='Same nc1 behavior;no copy-count,split,coop,geometry,mask,precision or arithmetic changes.',
                  required_fallback='Any nc!=1 or other kernel family retains original dispatch;host eligibility/resource proof not built.',
                  limitation='Static instruction footprint is not dynamic work,cache-miss latency or a speedup prediction.',
                  GPU_jobs=0, admitted_for_GPU=False)
    (OUT / 'cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k: v for k, v in report.items() if k != 'resources'}, indent=2))


if __name__ == '__main__':
    main()
