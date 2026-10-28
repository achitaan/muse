"""Fair interleavings of the proposed ring/deferred-fixup protocol.

This models readiness and counts, not CUDA memory ordering or MMA arithmetic.
Source release/acquire instructions are unchanged; real GPU evidence is still
required. The tested geometry includes264simultaneously resident CTAs.
"""
from pathlib import Path
import json
import random
from bisect import bisect_right
from collections import defaultdict

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
CA=[112,135,140,158];CB=[100,104,115,123]


def plan(counts,grid):
    all_segments=[[] for _ in range(grid)]
    partial_owner={};total_stages=0
    for pb,table in enumerate((CA,CB)):
        per_job=(64>>pb)*16;units=64>>pb
        costs=[table[(n+7)//8-1] for n in counts]
        prefix=[0]
        for c in costs:prefix.append(prefix[-1]+c*per_job)
        total=prefix[-1]
        def stage_at(x):
            j=bisect_right(prefix,x)-1
            return j*per_job+(x-prefix[j])//costs[j]
        boundaries=[stage_at(total*c//grid) if c<grid and counts else len(counts)*per_job for c in range(grid+1)]
        assert boundaries[0]==0 and boundaries[-1]==len(counts)*per_job
        assert all(a<=b for a,b in zip(boundaries,boundaries[1:]))
        seen=set();slot_units={}
        for c,(a,b) in enumerate(zip(boundaries,boundaries[1:])):
            tail=((b-1)//16)*16
            sp=tail if pb==0 and b%16 and tail>a else a
            ranges=[(sp,b),(a,sp)] if sp>a else [(a,b)]
            for lo,hi in ranges:
                while lo<hi:
                    unit=lo//16;j,ub=divmod(unit,units);s0=lo%16;s1=min(16,s0+hi-lo)
                    all_segments[c].append((pb,j,ub,s0,s1,unit))
                    for stage in range(lo,lo+s1-s0):
                        assert stage not in seen;seen.add(stage)
                    if s1-s0!=16:
                        # Exact independent owner formula at unit stage0.
                        xe=prefix[j]+((unit%units)*16+1)*costs[j]
                        owner=min((xe*grid-1)//total,grid-1)
                        assert boundaries[owner]<=unit*16<boundaries[owner+1]
                        partial_owner[pb,unit]=owner
                        assert (pb,owner) not in slot_units or slot_units[pb,owner]==unit
                        slot_units[pb,owner]=unit
                    lo+=s1-s0
        assert len(seen)==len(counts)*per_job
        total_stages+=len(seen)
        for c in range(grid):all_segments[c].append(None) # mandatory phase-end pending drain
    return all_segments,partial_owner,total_stages


def simulate(counts,grid,seed):
    segments,owners,stages=plan(counts,grid)
    ctas=[];consumers=[]
    for c,items in enumerate(segments):
        seq=[];entries=[]
        for segment in items:
            if segment is None:entries.append(None);continue
            pb,j,ub,s0,s1,u=segment
            entries.append((*segment,len(seq)))
            seq.extend((pb,j,ub,s) for s in range(s0,s1))
        ctas.append(dict(items=entries,seq=seq,pw=0,px=0,wr=[0]*len(seq),xr=[0]*len(seq)))
        for wg in range(2):consumers.append(dict(cta=c,wg=wg,index=0,offset=0,sub=0,pending=None))
    done=defaultdict(int);fix=defaultdict(int)
    actors=[('w',c) for c in range(grid)]+[('x',c) for c in range(grid)]+[('c',i) for i in range(len(consumers))]
    rng=random.Random(seed);actions=0;rounds=0
    def flush(w):
        pd=w['pending']
        if pd is None:return True
        pb,j,ub,s0,s1,unit=pd;n=s1-s0;wg=w['wg']
        if n==16:
            assert pb==0
            done[j,ub//4]+=1
        else:
            key=(pb,owners[pb,unit],wg)
            if pb==1 and s0==0:
                if fix[key]<16-n:return False
                assert fix[key]==16-n;fix[key]=0
            else:
                fix[key]+=n
                assert fix[key]<=16
                if pb==0 and fix[key]==16:
                    fix[key]=0;done[j,ub//4]+=1
        w['pending']=None;return True

    while True:
        progress=0;rounds+=1;rng.shuffle(actors)
        for kind,i in actors:
            if kind in ('w','x'):
                block=ctas[i];key='pw' if kind=='w' else 'px';at=block[key]
                if at>=len(block['seq']):continue
                slots=3 if kind=='w' else 2;mask=block['wr'] if kind=='w' else block['xr']
                if at>=slots and mask[at-slots]!=3:continue
                pb,j,ub,s=block['seq'][at]
                if kind=='x' and pb==1 and done[j,s]<8:continue
                block[key]+=1;progress+=1;continue
            w=consumers[i];block=ctas[w['cta']];items=block['items'];at=w['index']
            if at==len(items):continue
            item=items[at]
            if item is None:
                if flush(w):w['index']+=1;progress+=1
                continue
            pb,j,ub,s0,s1,unit,first=item;seq=first+w['offset'];bit=1<<w['wg']
            if w['sub']==0:
                if seq>=block['pw'] or seq>=block['px']:continue
                if w['offset']>0:block['xr'][seq-1]|=bit
                w['sub']=1;progress+=1
            elif w['sub']==1:
                # The deferred flush executes at first-stage CI3, after that
                # stage starts but before its weights are released at LG3.
                if w['offset']==0 and not flush(w):continue
                w['sub']=2;progress+=1
            else:
                block['wr'][seq]|=bit
                w['offset']+=1;w['sub']=0;progress+=1
                if w['offset']==s1-s0:
                    block['xr'][seq]|=bit
                    w['offset']=0;w['index']+=1
                    w['pending']=(pb,j,ub,s0,s1,unit) if pb==0 or s1-s0!=16 else None
        actions+=progress
        terminal=all(w['index']==len(ctas[w['cta']]['items']) for w in consumers)
        if terminal:
            assert all(b['pw']==b['px']==len(b['seq']) for b in ctas)
            assert all(value==0 for value in fix.values())
            assert all(done[j,st]==8 for j in range(len(counts)) for st in range(16))
            break
        assert progress>0,dict(counts=counts,grid=grid,round=rounds,blocked=consumers[:4])
        assert rounds<1000000
    return dict(jobs=len(counts),grid=grid,seed=seed,stages=stages,actions=actions,fair_rounds=rounds)


cases=[]
for counts in ([],[9],[32],[9,31,32,1,8],[32,1,32,7,19]):
    for grid in (1,2,8,31,132,264):
        for seed in (20261028,17):cases.append(simulate(counts,grid,seed))
for counts in ([32]*32, [n for c in [97]*31+[65] for n in ([32]*(c//32)+([c%32] if c%32 else []))]):
    cases.append(simulate(counts,264,20261028))
report=dict(scope='count/readiness simulation only, no CUDA memory model or numerical verification',
            cases=cases,all_complete=True,total_actions=sum(c['actions'] for c in cases),
            partial_slots_unique_per_CTA_phase=True,deferred_finalizers_drained=True,
            phase_A_completions_per_B_stage=8,all_fixup_counters_zero_after_calls=True,
            requires_every_CTA_resident=True,actual_CUDA_occupancy_unverified=True)
(ROOT/'experiments/prepared/moe_two_warpgroup_protocol_cpu_audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
