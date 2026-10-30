"""Reproduce the CPU ordering/resource screen of the existing FPR PDL mode."""

import ast
import hashlib
import json
import re
import subprocess

from pdl16_sources import HERE, BASE_SHA, literal
from pdlu_sources import OUT
from check_pdl_diag_cpu import Capture, MAIN
from check_pdl16_cpu import instructions
import nvrtc_screen_alternatives as nv


def main():
    raw = (HERE / 'decoder_pdli.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == BASE_SHA
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') == '_fpr_CUDA')
    source = literal(node.value)
    assert source.encode() == (OUT.parent / 'pdl16/fpr-parent.cu').read_bytes()
    end = source.index('\nstatic void launch_sk2(')
    device = source[:source.rfind('\ntemplate <', 0, end)]
    start = device.index('static PFN_cuTensorMapEncodeTiled_v12000 get_encode()')
    end = device.index('#define SK2_WAIT', start)
    device = device[:start] + device[end:]
    device = re.sub(r'^#include <(?:ATen/[^>]*|c10/[^>]*|cudaTypedefs.h)>\s*$', '', device, flags=re.M)
    device = device.replace('#include <cstring>', '// host-only header excluded')
    device = device.replace('static int g_sk2_pdl = 0;', '// host-only flag excluded')
    device = device.replace('void sk2_set_pdl(int64_t n) { g_sk2_pdl = (int)n; }', '// host-only setter excluded')
    device = 'struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };\n' + device
    expressions = [f'&sk2_kernel<{mt},{m},4,1,8,1,{st},true,false,2,1,__nv_bfloat16,false,true>'
                   for mt, m, st in [(1, 16, 5), (2, 32, 3)]]
    compiler = Capture()
    result = compiler.compile(device.encode(), expressions, [], tags=True, lineinfo=True)
    (OUT / 'fpr-device-proxy.cu').write_bytes(device.encode())
    (OUT / 'fpr-compile.log').write_text(result['log'])
    assert result['rc'] == 0, result['log'][-2000:]
    result['kernels'] = nv.parse_ptxas(result['log'])
    assert sorted(k['registers'] for k in result['kernels']) == [85, 126]
    assert all(not any(k[f] for f in ['stack_frame', 'spill_stores', 'spill_loads']) for k in result['kernels'])
    cubin = OUT / 'fpr.cubin'
    cubin.write_bytes(compiler.cubin)
    dump = subprocess.run([str(MAIN / '.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),
                           '-sass', str(cubin)], check=True, capture_output=True, text=True).stdout
    (OUT / 'fpr.sass').write_text(dump)
    order = {}
    for section in dump.split('Function : ')[1:]:
        name = section.splitlines()[0].strip()
        ins = instructions(section)
        weights = [a for a, t in ins if 'UBLKCP.S.G' in t]
        waits = [a for a, t in ins if 'ACQBULK' in t]
        activations = [a for a, t in ins if 'UTMALDG.3D' in t]
        assert weights and len(waits) >= 2 and activations
        assert weights[0] < waits[0] < activations[0] < waits[1]
        order[name] = dict(first_weight=weights[0], producer_wait=waits[0],
                           first_activation=activations[0], all_thread_wait=waits[1])
    assert len(order) == 2
    result.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  device_proxy_sha256=hashlib.sha256(device.encode()).hexdigest(),
                  instruction_order=order, fixed_depths=[5, 3], GPU_jobs=0,
                  limitation='Instruction positions plus source control-flow review; not a dynamic branch trace or GPU proof.')
    (OUT / 'fpr-compile.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'log'}, indent=2))


if __name__ == '__main__':
    main()
