"""Audit a proposed WGS weight permutation and logical coverage, not GPU code.

The original _wgs_copy documents [256-row unit][16stage][4group][9216byte].
Each whole group piece preserves its 8192 code and 1024 scale bytes unchanged.
No candidate kernel, resource measurement or barrier protocol is supplied here.
"""
from pathlib import Path
import json
import numpy as np

root=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
rng=np.random.default_rng(20261028)
piece=9216
audited_bytes=0
for units in (32,16):
    for expert in range(32):
        old=rng.integers(0,256,size=(units,16,4,piece),dtype=np.uint8)
        new=old.reshape(units,16,2,2,piece).transpose(0,2,1,3,4).copy().reshape(2*units,16,2,piece)
        inverse=new.reshape(units,2,16,2,piece).transpose(0,2,1,3,4).copy().reshape(old.shape)
        assert np.array_equal(old,inverse)
        # Check stage and group ownership directly, independently of inverse.
        for unit in range(2*units):
            for group in range(2):
                assert np.array_equal(new[unit,:,group],old[unit//2,:,2*(unit%2)+group])
        audited_bytes+=old.nbytes

coverage={}
for phase,old_units in [('A',32),('B',16)]:
    seen=[]
    for unit in range(old_units*2):
        for wg in range(2):
            old_unit,old_wg=unit//2,2*(unit%2)+wg
            for wp in range(4):
                old_r16=old_unit*16+old_wg*4+wp
                new_r16=unit*8+wg*4+wp
                assert new_r16==old_r16
                for g in range(8):
                    if phase=='A':seen.append(new_r16*8+g)
                    else:seen.extend([new_r16*16+g,new_r16*16+g+8])
    assert sorted(seen)==list(range(4096))
    coverage[phase]=len(seen)

# Original A unit writes128inter columns, new A unit writes64. Both produce
# exactly eight group completion arrivals for each256-column input stage of B.
done_old=np.zeros(16,np.int32);done_new=np.zeros(16,np.int32)
for unit in range(32):done_old[unit>>1]+=4
for unit in range(64):done_new[unit>>2]+=2
assert np.array_equal(done_old,done_new) and np.all(done_new==8)

# Virtual gather/final stores: preserve the complete512logical-thread domain
# with256physical consumers, not just its first half. This proves addresses,
# not the future floating reduction tree or CTA synchronization protocol.
logical=[physical+turn*256 for turn in range(2) for physical in range(256)]
columns=[8*tid+lane for tid in logical for lane in range(8)]
assert sorted(columns)==list(range(4096))

for count in range(384*8+1):
    visited=[route for first in range(0,count,32) for route in range(first,min(first+32,count))]
    assert visited==list(range(count))

report=dict(status='geometry prototype only; no implemented GPU candidate',seed=20261028,
    full_packed_weight_bytes_roundtripped=audited_bytes,experts_per_phase=32,
    weight_piece_code_bytes=8192,weight_piece_scale_bytes=1024,
    stage_order_preserved=True,whole_pieces_preserved=True,epilogue_columns=coverage,
    phase_A_completions_per_B_stage=done_new.tolist(),virtual_gather_columns=4096,
    route_counts_audited=[0,384*8],route_multiplicity_preserved=True,
    estimated_shared_bytes=3*2*piece+2*4*4096+22272,
    shared_estimate_assumption='other storage fixed at22272; JMAX/NW sizeof and alignment changes unverified',
    estimated_two_CTA_registers=2*320*96,
    unverified=['actual packing implementation','GPU epilogue implementation','resource attributes and two-CTA occupancy','all cross-CTA/barrier protocols','fixup/counter/scratch bounds','full numerical CUDA correctness','GPU performance'])
p=root/'experiments/prepared/moe_two_warpgroup_geometry_cpu_audit.json'
p.write_bytes((json.dumps(report,indent=2)+'\n').encode('utf8'))
print(json.dumps(report,indent=2))
