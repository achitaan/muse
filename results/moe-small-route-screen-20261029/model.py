"""Independent stable-sort oracle for a <=64-entry one-warp route model."""
import hashlib
import itertools
import json
from pathlib import Path
import random


def parent(ids,lo,nloc):
    eid=[v-lo if 0<=v-lo<nloc else -1 for v in ids]
    E=len(ids);CH=((E+255)//256)*32
    wc=[[0]*32 for _ in range(8)]
    for w in range(8):
        for i in range(w*CH,min(E,(w+1)*CH)):
            if eid[i]>=0:wc[w][eid[i]]+=1
    cnt=[sum(wc[w][e] for w in range(8)) if e<nloc else 0 for e in range(32)]
    off=[];total=0
    for c in cnt:off.append(total);total+=c
    ent=[None]*total;pos=[-1]*E
    for e in range(32):
        offset=off[e]
        for w in range(8):
            begin=w*CH;entries=[i for i in range(begin,min(E,begin+CH)) if eid[i]==e]
            for k,i in enumerate(entries):ent[offset+k]=i;pos[i]=offset+k
            offset+=wc[w][e]
    return cnt,off,eid,ent,pos


def candidate(ids,lo,nloc,mutant=None):
    assert len(ids)<=64
    eid=[v-lo if 0<=v-lo<nloc else -1 for v in ids]
    if mutant=='invalid_as_local':eid=[0 if e<0 else e for e in eid]
    e0=eid[:32];e1=eid[32:]
    first=[e0.count(e) for e in range(32)]
    cnt=[first[e]+e1.count(e) for e in range(32)]
    if mutant=='late_first_snapshot':first=cnt.copy()
    off=[];total=0
    for c in cnt:off.append(total);total+=c
    ent=[None]*total;pos=[-1]*len(ids)
    for base,group in ((0,e0),(32,e1)):
        for lane,e in enumerate(group):
            if e<0:continue
            count_before=first[e] if base and mutant!='missing_first_slice_offset' else 0
            q=off[e]+count_before+group[:lane].count(e)
            assert 0<=q<total and ent[q] is None,'Duplicate/out-of-bounds writer'
            ent[q]=base+lane;pos[base+lane]=q
    assert all(i is not None for i in ent),'Missing route'
    return cnt,off,eid,ent,pos


def oracle(ids,lo,nloc):
    valid=sorted((v-lo,i) for i,v in enumerate(ids) if 0<=v-lo<nloc)
    ent=[i for e,i in valid];pos=[-1]*len(ids)
    for q,i in enumerate(ent):pos[i]=q
    cnt=[sum(e==expert for e,i in valid) for expert in range(32)]
    off=[sum(cnt[:e]) for e in range(32)]
    eid=[v-lo if 0<=v-lo<nloc else -1 for v in ids]
    return cnt,off,eid,ent,pos


def main():
    cases=0;digest=hashlib.sha256()
    def check(ids,lo,nloc):
        nonlocal cases
        old,new,expected=parent(ids,lo,nloc),candidate(ids,lo,nloc),oracle(ids,lo,nloc)
        assert old==new==expected,(ids,lo,nloc)
        digest.update(json.dumps(expected,separators=(',',':')).encode());cases+=1
    for ids in itertools.product((0,1,2),repeat=8):check(ids,0,2)
    for m in range(9):
        E=m*8
        for lo,nloc in ((0,0),(0,1),(0,32),(32,16),(32,32),(224,32)):
            for seed in range(64):
                rng=random.Random(20261029+m*10000+lo*100+nloc+seed)
                check([rng.randrange(256) for _ in range(E)],lo,nloc)
        for count in range(E+1):
            ids=[0]*count+[32]*(E-count)
            check(ids,0,32)
            ids=ids[1::2]+ids[::2];check(ids,0,32)
    rejected=[]
    for mutant in ('invalid_as_local','late_first_snapshot','missing_first_slice_offset'):
        ids=[0,1,2,32]*16
        expected=oracle(ids,0,32)
        try:assert candidate(ids,0,32,mutant)==expected
        except AssertionError:rejected.append(mutant)
        else:raise AssertionError(('Mutant survived',mutant))
    source=Path(__file__).with_name('full.cu').read_text(encoding='utf-8')
    start=source.index('const int first_count=S.cnt[lane];')
    stop=source.index('S.cnt[e1]+=__popc(peers1);',start)
    assert '__syncwarp();' in source[start:stop]
    report=dict(cases_checked=cases,oracle='Independent stable expert/index sort',
                metadata_identical_to_parent=True,entry_coverage_and_unique_writers=True,
                duplicates_nonlocal_empty_and_two_slice_boundaries=True,rejected_mutants=rejected,
                fixture_sha256=digest.hexdigest(),source_snapshot_barrier_present=True,
                kernel_execution=False,GPU_jobs=0,
                limitations=['CPU algorithm model, not generated CUDA correctness',
                             'Not a PTX memory-order or complete-kernel proof',
                             'No routing critical-path fraction or performance measurement'])
    out=Path(__file__).with_name('model-audit.json')
    data=(json.dumps(report,indent=2)+'\n').encode()
    assert not out.exists() or out.read_bytes()==data,'Frozen model collision'
    out.write_bytes(data);print(json.dumps(report))


if __name__=='__main__':main()
