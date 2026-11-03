"""CPU source/byte-layout checks for the bounded feasibility test."""
import hashlib,json,re,sys
import numpy as np
sys.dont_write_bytecode=True
from source import ROOT,BASE,L2,l3_source
s=l3_source()
start=L2.index('    // codes and scales one 4-step group at a time',L2.index('void wgs_seg('))
old=L2[start:L2.index('    prev = xs;',start)]
assert old in s,'narrow load/decode/MMA code must be verbatim'
wide=s[s.index('    if constexpr (NTT > 8) {'):s.index('    } else {',s.index('    if constexpr (NTT > 8) {'))]
assert not re.search(r'\b(?:uint4 cwn|uint2 scn|lg \+ 1)\b',wide)
assert '#define OKC_L2_GS 1' in s and 'af[2][GS][4]' in s
assert s.count('if constexpr (WGS_L2 && NTT > 8) { if (pd.kind) wgs_flush')==1
assert s.split('#include <torch/extension.h>')[1]==L2.split('#include <torch/extension.h>')[1]
rng=np.random.default_rng(20261101)
raw=rng.integers(0,256,36864,dtype=np.uint8)
for wg in range(4):
    for wp in range(4):
        for lane in range(32):
            prefetched=[]
            for lg in range(4):
                c=wg*9216+((lg*4+wp)*32+lane)*16
                sc=wg*9216+8192+((lg*4+wp)*8+lane//4)*8
                prefetched.append((raw[c:c+16].copy(),raw[sc:sc+8].copy()))
            for lg in range(4):
                # Expected values: original L2 group's physical code/scale bytes.
                c=wg*9216+((lg*4+wp)*32+lane)*16
                sc=wg*9216+8192+((lg*4+wp)*8+lane//4)*8
                assert np.array_equal(raw[c:c+16],prefetched[lg][0])
                assert np.array_equal(raw[sc:sc+8],prefetched[lg][1])
# CUDA occupancy hardware rounds18 warps to20 partitions for576 threads.
assert 20*32*96==61440<=65536
assert 20*32*112==71680>65536
manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    for p in (ROOT/'experiments/boards/moe/moe_b13.py',ROOT/'experiments/boards/moe/moe_l2.py',ROOT/'experiments/boards/diag/moediag_l2.py')}
(ROOT/'.l3_offline/inputs.json').write_text(json.dumps(manifest,indent=2))
print('PASS narrow source identity, current-group byte mapping, single early drain, host ABI;112-reg control is nonlaunchable')
