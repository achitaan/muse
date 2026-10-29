"""Source-bound ownership screen for disjoint stream-K partials; no GPU work."""
import ast
import bisect
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
BASE='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
TABLES=([112,135,140,158,173,185,200,207],[100,104,115,123,132,144,156,174])


def model(jobs,pb,grid):
    assert 1<=grid<=160 and all(1<=n<=64 for n in jobs)
    units=32 if pb==0 else 16
    costs=[TABLES[pb][(n+7)//8-1] for n in jobs]
    prefix=[0]
    for cost in costs:prefix.append(prefix[-1]+units*16*cost)
    total=prefix[-1]
    def at_cost(x):
        j=bisect.bisect_right(prefix,x)-1
        return j*units*16+(x-prefix[j])//costs[j]
    cuts=[at_cost(total*c//grid) if total and c<grid else len(jobs)*units*16
          for c in range(grid+1)]
    shares=defaultdict(list);written={};full=0
    for c,(lo,hi) in enumerate(zip(cuts,cuts[1:])):
        for u in range(lo//16,(hi+15)//16):
            a,b=max(lo,u*16)-u*16,min(hi,(u+1)*16)-u*16
            if a>=b:continue
            if b-a==16:
                full+=1;continue
            # Endpoint0 owns a leading partial; endpoint1 a trailing partial.
            # A one-unit range strictly inside a unit uses endpoint0 only.
            side=int(a==0)
            key=(pb,c,side)
            assert key not in written, ('Private slot reused in one phase',jobs,pb,grid,key)
            written[key]=(u,a,b)
            shares[u].append((c,a,b,side))
    value_bytes=0;atomic_payload=0;direct_payload=0;private_reads=0;contributors=0
    for u,parts in shares.items():
        assert parts[0][1]==0 and parts[-1][2]==16
        assert all(a[2]==b[1] for a,b in zip(parts,parts[1:])), 'Stage gap or overlap'
        assert len(parts)>=2
        first,last=parts[0][0],parts[-1][0]
        assert [p[0] for p in parts]==list(range(first,last+1)), 'Noncontiguous contributor CTAs'
        assert parts[0][3]==1 and all(p[3]==0 for p in parts[1:])
        # This matches the exact retained inverse-owner formula, including
        # variable stage cost and cut boundaries at integer quotient edges.
        def stage_owner(stage):
            job=stage//(units*16)
            xe=prefix[job]+((stage%(units*16))+1)*costs[job]
            c=min(grid-1,(xe-1)*grid//total)
            while c+1<grid and total*(c+1)//grid<xe:c+=1
            while c>0 and total*c//grid>=xe:c-=1
            return c
        assert first==stage_owner(u*16) and last==stage_owner(u*16+15)
        job=u//units
        end_cost=prefix[job]+((u%units+1)*16)*costs[job]
        assert total*grid<2**32
        limit=end_cost*grid
        from_first=first
        while from_first+1<grid and total*(from_first+1)<limit:from_first+=1
        assert from_first==last and last-first<16
        for c,a,b,side in parts:
            lo,hi=cuts[c],cuts[c+1]
            precomputed_unit=(hi-1)//16 if side else lo//16
            assert precomputed_unit==u
            flush_side=int((u<<4)>=lo)
            assert flush_side==side
            packed=stage_owner(precomputed_unit*16) | (last<<8)
            assert (packed&255)==first and (packed>>8)==last
        assert sum(b-a for c,a,b,side in parts)==16
        ntt=(jobs[u//units]+7)//8
        payload=4*128*4*ntt*4  # four WGs,128lanes,4FP32 values per n-tile
        value_bytes+=payload
        contributors+=len(parts)
        atomic_payload+=len(parts)*payload
        direct_payload+=len(parts)*payload
        private_reads+=len(parts)*payload
        # Every read in two fresh generations has a matching current write;
        # deliberately leave all other storage poisoned rather than zeroed.
        storage={}
        for generation in (1,2):
            for c,a,b,side in reversed(parts):storage[pb,c,side]=(generation,u,a,b)
            readback=[storage[pb,c,int(c==first)] for c in range(first,last+1)]
            assert all(x[0]==generation and x[1]==u for x in readback)
    return dict(jobs=len(jobs),full_units=full,split_units=len(shares),
                partial_contributors=contributors,private_endpoint_slots=len(written),
                old_atomic_FP32_payload_bytes=atomic_payload,
                old_finisher_read_bytes=value_bytes,old_finisher_zero_store_bytes=value_bytes,
                proposed_disjoint_store_bytes=direct_payload,proposed_finisher_read_bytes=private_reads)


def main():
    raw=(ROOT/'best/moe_verified.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==BASE
    cuda=next(n.value.value for n in ast.parse(raw).body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    assert hashlib.sha256(cuda.encode()).hexdigest()=='1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'
    assert 'red.global.add.v4.f32' in cuda
    assert 'wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);' in cuda
    assert 'while (ld_acquire(p.doneAs + j * 16 + st) < 8)' in cuda
    source=ROOT/'results/moe-unit-ownership-screen-20261029/screen.json'
    draw=json.loads(source.read_bytes());rows=[]
    for row in draw['rows']:
        for pb in (0,1):rows.append(dict(m=row['m'],dist=row['dist'],trial=row['trial'],phase=pb,
                                       **model(row['jobs'],pb,132)))
    cases=len(rows)
    for count in range(1,65):
        for grid in (1,2,16,32,66,128,132,159,160):
            for pb in (0,1):model([count],pb,grid);cases+=1
    for m in (1,4,8,16,17,31,32,33,63,64,65,128,256,257,384,385):
        for distribution in ('remote','onehot','alllocal'):
            for begin in range(0,m,384):
                chunk=min(384,m-begin)
                counts=[0]*32 if distribution=='remote' else [chunk]+[0]*31 if distribution=='onehot' else [chunk//4+int(e<8*(chunk%4)) for e in range(32)]
                assert sum(counts)==(0 if distribution=='remote' else chunk if distribution=='onehot' else chunk*8)
                jobs=[min(64,n-i) for n in counts if n>8 for i in range(0,n,64)]+[n for n in counts if 0<n<=8]
                for pb in (0,1):model(jobs,pb,132);cases+=1
    report=dict(baseline_sha256=BASE,routing_draw_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                ownership_cases=cases,rows=rows,
                private_extra_float_capacity=2*160*4*(8*512),private_extra_bytes=2*160*4*(8*512)*4,
                original_count_and_doneAs_offsets_unchanged=True,
                reduction_order='FP32, increasing contributor CTA ID, before retained BF16/global-scale epilogue',
                proposal='Two endpoint slots per CTA and phase; first contributor uses trailing bank, later contributors leading bank.',
                scope='EP8 MAIN only; all original numerical, routing and geometry fallbacks retained',
                GPU_jobs=0,kernel_implemented=False,
                limitations=['Synthetic routing draws are not official inputs or device-side route observations',
                             'Payload counts are logical operations, not HBM traffic, atomic cache transactions or timing predictions',
                             'Sequential ownership and fresh-write model is not a CUDA memory-order or forward-progress proof',
                             'Uniform random packed nibbles in the official generator do not justify entropy compression assumptions'])
    report['model_revision']=4
    report['division_free_last_owner_equal_original_inverse']=True
    report['precomputed_endpoint_and_flush_side_equal_original_shares']=True
    report['prior_revision_limit']='Root model v1 mislabeled an incomplete balanced all-local remainder and did not split385 into384+1; preserved, superseded by this revision.'
    out=ROOT/'results/moe-private-fixup-model-20261029/v4';out.mkdir(exist_ok=True,parents=True)
    data=(json.dumps(report,indent=2)+'\n').encode();p=out/'model.json'
    assert not p.exists() or p.read_bytes()==data,'Frozen model collision'
    p.write_bytes(data)
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows']}))


if __name__=='__main__':main()
