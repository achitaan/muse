"""Prove the capacity/index repair retains parent math/configuration and covers all keys."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_attention_three_cta import ROOT,BASE,parent_source,literal
sys.path.insert(0,str(ROOT/'scripts'))
from verify_attention_precision import verify

raw=(ROOT/'.tools/attention-segment-repair-variant.py').read_bytes();verify(raw)
old=literal(parent_source())[1];new=literal(raw)[1]
reverse=new.replace('int seg[F ? ((L == 1) ? MAXB : MAXSEG) : 1][3];','int seg[F ? MAXSEG : 1][3];')
reverse=reverse.replace('S.cum[b] < ceg && n < MAXB','S.cum[b] < ceg && n < MAXSEG')
assert reverse==old,'Only segment capacity/control may change in CUDA'
trees=[ast.parse(s) for s in (parent_source(),raw)]
plans=[next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='_plan') for t in trees]
assert ast.dump(plans[0])==ast.dump(plans[1]),'Original two-block grid remains'
assert 'static constexpr int MINB = 2;' in new and '(65536 / (2 * NTX))' in new

# Reuse only the independently written layout and interval model functions.
audittree=ast.parse((ROOT/'experiments/prepared/audit_attention_three_cta_cpu.py').read_bytes())
fns=[n for n in audittree.body if isinstance(n,ast.FunctionDef) and n.name in ('partition','smem_layout')]
import re
scope={'np':np,'re':re};exec(compile(ast.Module(body=fns,type_ignores=[]),'<partition-and-layout>','exec'),scope)
layouts={}
for L in (1,8):
    for F in (False,True):
        a,b=scope['smem_layout'](old,L,F),scope['smem_layout'](new,L,F)
        if (L,F)!=(1,True):assert a==b
        layouts[f'L{L}_flat{int(F)}']=b['size']
assert layouts['L1_flat1']==66560 and 2*(66560+1024)<=233472

rng=np.random.default_rng(20261028);partitions=0;segments_count=0;valid_counterexamples=[];fallbacks=0
for B in (2,3,8,16,32,33,64,128,256,512):
    for pairs in (1,2,4,8):
        for kind in ('tiny','mixed','long_skew','longer_skew'):
            lengths=(rng.integers(1,100001,size=B,dtype=np.int64) if kind=='mixed' else np.ones(B,np.int64))
            if kind in ('long_skew','longer_skew'):lengths[0]=(1048576 if kind=='long_skew' else 4194304)+1
            maxpages=int(lengths.max())
            # Input shapes that cannot fit the protected prefix space now
            # use the existing fallback; do not call them native coverage.
            if B*(maxpages+1+600)>=2**31:
                fallbacks+=1;continue
            G=264//pairs
            parts,lost,maximum=scope['partition'](lengths,G,512)
            assert lost==0 and maximum<=B
            byrequest=[[] for _ in range(B)]
            for b,s,e in parts:byrequest[b].append((s,e))
            for b,items in enumerate(byrequest):
                items.sort();assert items[0][0]==0 and items[-1][1]==lengths[b]
                assert all(a[1]==c[0] for a,c in zip(items,items[1:]))
                assert sum(e-s for s,e in items)==lengths[b]
                assert sum(s<=lengths[b]-1<e for s,e in items)==1
            if kind in ('long_skew','longer_skew'):
                _,oldlost,oldmax=scope['partition'](lengths,G,64)
                if oldlost:
                    valid_counterexamples.append(dict(batch=B,pairs=pairs,max_pages=maxpages,grid=G,old_lost_keys=oldlost,old_max_segments=oldmax,new_lost_keys=0))
            partitions+=1;segments_count+=len(parts)
assert valid_counterexamples
assert any(c['batch']==512 and c['pairs']==8 and c['max_pages']==1048577 and c['old_lost_keys']==154 for c in valid_counterexamples)
report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),parent_sha256=BASE,
            original_math_registers_grids_preserved=True,layout_sizes=layouts,
            partitions=partitions,segments=segments_count,native_domain_counterexamples=valid_counterexamples,
            oversized_shape_fallback_cases=fallbacks,full_key_intervals=True,one_cache_write_owner=True,
            gpu_verified=False,scope='CPU/source evidence only; official exact-source check required')
(ROOT/'.tools/attention-segment-repair-cpu-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps({k:v for k,v in report.items() if k!='native_domain_counterexamples'},indent=2))
