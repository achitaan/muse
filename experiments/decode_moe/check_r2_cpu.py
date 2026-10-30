"""CPU-only NVRTC sm_90a compile of the R2 fused pooler+router kernels vs the parent instances (resources only)."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'opus-fresh' / 'experiments' / 'opus_fresh'))
import nvrtc_screen_alternatives as nv  # noqa: E402

PRE = """namespace std {
template <bool B, class T, class F> struct conditional { using type = T; };
template <class T, class F> struct conditional<false, T, F> { using type = F; };
}
#include <cuda_bf16.h>
typedef unsigned long long size_t_;
"""


def device_part(src):
    s = src[:src.index('static int ', src.index('__global__'))]
    s = re.sub(r'#include <(cuda_runtime\.h|algorithm|type_traits)>\n', '', s)
    return PRE + s


def main():
    nvr = nv.NVRTC()
    res = {}
    for p in ('v18s', 'smallm2'):
        src = device_part((HERE / f'r2_{p}.cu').read_text(encoding='utf-8'))
        names = [f'&{p}_gemv<1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, {o}>' for o in (6, 70)]
        r = nvr.compile(src.encode(), names, [], tags=True, lineinfo=False)
        res[p] = {'rc': r['rc'], 'kernels': nv.parse_ptxas(r['log']) if r['rc'] == 0 else r['log'][-2500:]}
    (HERE / 'check_r2_cpu.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    for p, v in res.items():
        if v['rc'] == 0:
            for k in v['kernels']:
                print(p, k['symbol'][-40:], k['registers'], k['stack_frame'], k['spill_stores'], k['spill_loads'], k['warnings'])
        else:
            print(p, 'FAILED', v['kernels'])


if __name__ == '__main__':
    main()
