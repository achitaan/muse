"""CPU-only source, resources and instruction-order screens for U16/U32."""

import hashlib
import json
import re
import subprocess

from pdlu_sources import derive, OUT, U16_INST, U32_INST
from check_pdl_diag_cpu import Capture, MAIN
from check_r2_cpu import device_part
from check_pdl16_cpu import instructions
import nvrtc_screen_alternatives as nv


def proxy(name, src):
    if name == 'u16':
        return device_part(src)
    src = src.split('using EncodeFn=', 1)[0]
    src = src.replace('#include <cuda.h>', 'struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };')
    return src.replace('#include <cuda_runtime.h>', '')


def order(name, sass):
    ins = instructions(sass)
    acq = [a for a, t in ins if 'ACQBULK' in t]
    calls = [(a, int(re.search(r'0x([0-9a-f]+)', t)[1], 16)) for a, t in ins if 'CALL.REL.NOINC' in t]
    matches = [(a, target) for a, target in calls if any(target <= x < target + 64 for x in acq)]
    assert len(matches) == 1
    call, target = matches[0]
    assert any(a < call and 'PREEXIT' in t for a, t in ins)
    if name == 'u16':
        independent = [(a, t) for a, t in ins if a < call and ('LDGSTS' in t or 'LDG.' in t)]
        dependent = [(a, t) for a, t in ins if 'LDG.' in t and 'STRONG.GPU' in t]
        assert sum('LDGSTS' in t for a, t in independent) == 6
        assert len([t for a, t in independent if 'LDG.' in t]) == 1
    else:
        independent = [(a, t) for a, t in ins if a < call and 'LDG.' in t]
        dependent = [(a, t) for a, t in ins if 'UTMALDG.2D' in t]
        assert len(independent) == 8
    assert dependent and all(a > call for a, t in dependent)
    return dict(wait_call=call, helper_target=target, independent=independent,
                first_dependent=dependent[0], CPU_order_only=True)


def main():
    sources, _ = derive()
    nvr = Capture()
    exe = MAIN / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    out = {}
    for name, (parent, candidate, _) in sources.items():
        for arm, src in [('parent', parent), ('candidate', candidate)]:
            symbol = ('dd_cap2_b_cpssm_gemv' if name == 'u16' else 'm32w_gemm') if arm == 'parent' else name.replace('u', 'pdlu') + ('_gemv' if name == 'u16' else '_gemm')
            instance = U16_INST if name == 'u16' else U32_INST
            expr = '&' + symbol + '<' + ','.join(str(x).lower() for x in instance) + '>'
            code = proxy(name, src)
            r = nvr.compile(code.encode(), [expr], [], tags=True, lineinfo=True)
            key = name + '-' + arm
            r.update(source_sha256=hashlib.sha256(src.encode()).hexdigest(), proxy_sha256=hashlib.sha256(code.encode()).hexdigest(), CPU_only=True)
            (OUT / (key + '-proxy.cu')).write_bytes(code.encode())
            (OUT / (key + '.log')).write_text(r['log'])
            r['kernels'] = nv.parse_ptxas(r['log']) if not r['rc'] else []
            print(key, r['rc'], r['kernels'], flush=True)
            if not r['rc']:
                p = OUT / (key + '.cubin'); p.write_bytes(nvr.cubin)
                sass = subprocess.check_output([str(exe), '-sass', str(p)], text=True)
                (OUT / (key + '.sass')).write_bytes(sass.encode())
                if arm == 'candidate':
                    r['ordering'] = order(name, sass)
                    print(r['ordering'], flush=True)
            else:
                print(r['log'][-2500:], flush=True)
            (OUT / (key + '.json')).write_text(json.dumps(r, indent=2) + '\n')
            out[key] = r
    (OUT / 'compile-summary.json').write_text(json.dumps(out, indent=2) + '\n')
    assert all(r['rc'] == 0 for r in out.values())
    assert all(not any(k[f] for f in ['stack_frame', 'spill_stores', 'spill_loads'])
               for r in out.values() for k in r['kernels'])


if __name__ == '__main__':
    main()
