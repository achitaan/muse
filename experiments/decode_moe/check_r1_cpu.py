"""CPU-only checks for R1: NVRTC sm_90a compile of route_fp32<TM,BPC> and block/counter coverage model."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / '..' / 'opus-fresh' / 'experiments' / 'opus_fresh'))
import nvrtc_screen_alternatives as ns  # noqa: E402

src = (HERE / 'rf_r1.cu').read_text(encoding='utf-8')
dev = src[src.index('template<int TM, int BPC>'):src.index('void run(')]
prelude = '#include <math.h>\n#include <limits.h>\n' if False else (
    '#define INFINITY __int_as_float(0x7f800000)\n#define INT_MAX 2147483647\n')
code = (prelude + dev).encode()
names = [f'route_fp32<{t}, {b}>' for t in (1, 8) for b in (1, 2, 4, 8)]
nv = ns.NVRTC()
res = nv.compile(code, names, tags=False, lineinfo=False)
print(type(res), (res[:2] if isinstance(res, tuple) else res) if not isinstance(res, dict) else {k: v for k, v in res.items() if k != 'cubin'})

# Coverage: each of the 128 (nb, ks) blocks exactly once per BPC; counter low bits return to 0 per call
for bpc in (1, 2, 4, 8):
    seen = sorted(cta * bpc + bi for cta in range(128 // bpc) for bi in range(bpc))
    assert seen == list(range(128))
    for start in (0, 128 * 7, 2**32 - 128):
        cnt, lasts = start % 2**32, 0
        for _ in range(128 // bpc):
            old = cnt
            cnt = (cnt + bpc) % 2**32
            lasts += ((old + bpc) & 127) == 0
        assert lasts == 1 and cnt % 128 == 0, (bpc, start)
print('coverage/counter model ok for BPC 1,2,4,8')
