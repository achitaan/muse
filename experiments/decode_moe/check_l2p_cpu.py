"""CPU-only NVRTC sm_90a compile of the L2P device kernels (prefetch + spin)."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'opus-fresh' / 'experiments' / 'opus_fresh'))
import nvrtc_screen_alternatives as nv  # noqa: E402

src = (HERE / 'l2p.cu').read_text(encoding='utf-8')
dev = src[:src.index('int l2p_launch(')]
dev = re.sub(r'#include <(cuda_runtime\.h|stdint\.h)>\n', '', dev)
r = nv.NVRTC().compile(dev.encode(), ['&l2p_prefetch', '&l2p_spin'], [], tags=True, lineinfo=False)
print('rc', r['rc'])
if r['rc'] == 0:
    for k in nv.parse_ptxas(r['log']):
        print(k['symbol'], k['registers'], k['stack_frame'], k['spill_stores'], k['spill_loads'], k['warnings'])
else:
    print(r['log'][-3000:])
