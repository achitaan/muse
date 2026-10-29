"""Screen direct phase-A activation gathering against the exact WGS layout.

This is a CPU index/arrival-count model, not CUDA or timing validation.
"""
import ast
import hashlib
import json
from pathlib import Path
import random

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PARENT='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
NATIVE='1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'


def barrier(seed,early_release=False):
    rng=random.Random(seed);pending=1;registered=set();completed=set();released=False
    while len(completed)<32 or not released:
        choices=[('register',i) for i in range(32) if i not in registered]
        choices += [('complete',i) for i in registered-completed]
        if not released and (early_release or len(registered)==32):choices.append(('release',0))
        action,lane=rng.choice(choices)
        if action=='register':registered.add(lane);pending+=1
        elif action=='complete':completed.add(lane);pending-=1
        else:released=True;pending-=1
        assert pending>=0
        if pending==0:
            assert len(registered)==len(completed)==32 and released,'phase completed before every lane copy'
    assert pending==0


def main():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PARENT
    tree=ast.parse(raw)
    # Locate by exact native hash rather than assuming a historical literal name.
    literals=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)]
    native=next(s for s in literals if hashlib.sha256(s.encode()).hexdigest()==NATIVE)
    assert 'void wgs_gather(' in native and 'void wgs_xproducer(' in native
    assert 'const int ubl = pb ? 4 : 5;' in native
    rng=random.Random(20261029);cases=0;vectors=0;traffic=[]
    for count in range(1,65):
        padded=(count+7)//8*8;nt=padded//8
        # Duplicated routes are deliberate: different routed rows may read the
        # same input token. Their destination rows must still remain distinct.
        tokens=[rng.randrange(0,257) for _ in range(count)]
        for stage in range(16):
            old={}
            for row in range(count):
                for k8 in range(32):
                    # Independently expanded from retained gather's uint4
                    # address (tid=stage*32+k8), then take this stage's slice.
                    dst=(((stage*nt+(row>>3))*32+k8)*8+(row&7))-stage*padded*32
                    old[dst]=(tokens[row]*4096+stage*256+k8*8)
            seen=set()
            for lane in range(32):
                for index in range(lane,padded*32,32):
                    row=(index//256)*8+(index&7);k8=(index>>3)&31
                    assert 0<=row<padded and 0<=k8<32 and index not in seen
                    seen.add(index)
                    if row<count:
                        source=tokens[row]*4096+stage*256+k8*8
                        assert old[index]==source
                        assert source%8==0 and 0<=source<257*4096
                    else:
                        assert index not in old # zero-fill; no entry/token read
            assert len(seen)==padded*32
            vectors+=len(seen);cases+=1
        before=(2*count+32*padded)*4096*2
        after=32*count*4096*2
        traffic.append(dict(job_rows=count,padded_rows=padded,
            old_logical_phaseA_activation_bytes=before,new_logical_phaseA_activation_bytes=after,
            direct_copy_rounds_per_lane_per_stage=padded,
            original_bulk_instructions_per_stage=1))
    for seed in range(256):barrier(seed)
    rejected=False
    for seed in range(256):
        try:barrier(seed,early_release=True)
        except AssertionError:rejected=True;break
    assert rejected
    report=dict(parent_source_sha256=PARENT,native_CUDA_sha256=NATIVE,CPU_only=True,GPU_jobs=0,
        CUDA_prototype_written=False,index_cases=cases,uint4_destinations_checked=vectors,
        barrier_random_orders=256,early_release_mutant_rejected=True,seed=20261029,
        traffic=traffic,
        preserved='Packed weights and scales, all matrix stages, routing, phase-B xB/slice readiness, BF16 epilogues and FP32 finalizer',
        proposed='Use all32 lanes of existing x producer for phase-A cp.async16 gather; skip original global xA gather; keep phase-B bulk copy on lane0',
        pending_arrival_design='Default cp.async.mbarrier.arrive adds a pending arrival per lane; syncwarp after all registrations, then lane0 releases original count1. No noinc shortcut.',
        extra_memory_bytes=0,initial_scratch_allocation_retained=True,
        source='https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive',
        unresolved=['Logical bytes are not measured HBM traffic; original xA requests may hit L2',
            'Direct copies add8..64 producer rounds per stage instead of one bulk instruction',
            'Need exact native prototype, async-proxy lifetime proof and generated-code/resource admission',
            'Need representative routed job counts and exposed gather time before a GPU check',
            'Changed routes, duplicates, all-nonlocal and chunk boundaries require fresh GPU graph tests'])
    out=ROOT/'results/moe-direct-stage-gather-model-20261029';out.mkdir(exist_ok=True)
    data=(json.dumps(report,indent=2)+'\n').encode();path=out/'model.json'
    assert not path.exists() or path.read_bytes()==data;path.write_bytes(data)
    print(json.dumps({k:v for k,v in report.items() if k!='traffic'},indent=2))


if __name__=='__main__':main()
