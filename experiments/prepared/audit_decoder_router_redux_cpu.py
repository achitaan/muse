"""Independent exact top-eight selection proof; no GPU or timing claims."""
import json
from pathlib import Path
import numpy as np
import ast
import hashlib

from build_decoder_router_redux import ROOT, BASE_SHA, REDUCE_OLD, REDUCE_NEW

base=(ROOT/'best/decoder_verified.py').read_bytes()
source=(ROOT/'.tools/decoder-router-redux-variant.py').read_bytes()
assert hashlib.sha256(base).hexdigest()==BASE_SHA
bt,ct=ast.parse(base),ast.parse(source)
def payload(tree):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='_RF_CUDA' for t in n.targets))
old_cuda,new_cuda=payload(bt),payload(ct)
assert new_cuda.count(REDUCE_NEW)==1
assert new_cuda.replace('route_fp32_redux','route_fp32').replace(REDUCE_NEW,REDUCE_OLD,1)==old_cuda
assert 'xshared' not in new_cuda
for n in ct.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_RF_CUDA' for t in n.targets):
        n.value=ast.Constant(old_cuda)
assert ast.dump(bt,include_attributes=False)==ast.dump(ct,include_attributes=False)
assert new_cuda.count('__fmaf_rn')==old_cuda.count('__fmaf_rn')==2
assert new_cuda.count('__shfl_down_sync')==old_cuda.count('__shfl_down_sync')==2
assert '__shfl_xor_sync' not in new_cuda
assert '__reduce_max_sync(0xffffffffu, key)' in new_cuda
assert '__reduce_min_sync(0xffffffffu, key == global_key ? id : INT_MAX)' in new_cuda
assert '__shfl_sync(0xffffffffu, raw, id & 31)' in new_cuda
assert 'if (best == 0.f) bits = 0u;' in new_cuda

INTMAX = np.int32(2147483647)
LANE = np.arange(32)


def local(sel,raw):
    best=np.full(32,-np.inf,dtype=np.float32)
    ids=np.full(32,INTMAX,dtype=np.int32)
    value=np.zeros(32,dtype=np.float32)
    for j in range(8):
        e=LANE+32*j
        take=(sel[e]>best)|((sel[e]==best)&(e<ids))
        best=np.where(take,sel[e],best)
        value=np.where(take,raw[e],value)
        ids=np.where(take,e,ids).astype(np.int32)
    return best,value,ids


def one_old(sel,raw):
    best,value,ids=local(sel,raw)
    for d in (16,8,4,2,1):
        ob,ov,oi=best[LANE^d],value[LANE^d],ids[LANE^d]
        take=(ob>best)|((ob==best)&(oi<ids))
        best=np.where(take,ob,best)
        value=np.where(take,ov,value)
        ids=np.where(take,oi,ids)
    assert np.all(ids==ids[0])
    return int(ids[0]),value[0]


def one_redux(sel,raw):
    best,value,ids=local(sel,raw)
    bits=best.view(np.uint32).copy()
    bits[best==0]=0 # float comparisons equate signed zeros
    key=np.where((bits&0x80000000)!=0,~bits,bits^np.uint32(0x80000000))
    winner=int(np.min(np.where(key==key.max(),ids,INTMAX)))
    return winner,value[winner&31]


def one_oracle(sel,raw):
    # Scalar global max with ascending-ID ties, no warp layout or bit keys.
    best=np.float32(-np.inf);winner=int(INTMAX);value=np.float32(0)
    for e in range(256):
        if sel[e]>best or (sel[e]==best and e<winner):
            best=sel[e];winner=e;value=raw[e]
    return winner,value


rng=np.random.default_rng(20261028)
specials=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,
  0x00800000,0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,
  0x7f800000,0xff800000,0x7fc00001,0xffc00001],dtype=np.uint32).view(np.float32)
cases=[np.full(256,x,dtype=np.float32) for x in specials]
cases.extend([np.resize(specials,256),np.resize(specials[::-1],256),
              np.zeros(256,dtype=np.float32),np.tile(np.array([0.,-0.],dtype=np.float32),128)])
for _ in range(128):
    cases.append(rng.integers(0,2**32,size=256,dtype=np.uint32).view(np.float32))
for _ in range(128):
    score=rng.random(256).astype(np.float32)
    bias=(rng.standard_normal(256)*.1).astype(np.float32)
    cases.append(score+bias)
    cases.append(rng.integers(-3,4,size=256).astype(np.float32)) # many ties
for index,sel in enumerate(cases):
    raw=rng.integers(0,2**32,size=256,dtype=np.uint32).view(np.float32)
    a,b,c=sel.copy(),sel.copy(),sel.copy()
    for pick in range(8):
        with np.errstate(invalid='ignore'):
            old=one_old(a,raw);new=one_redux(b,raw);oracle=one_oracle(c,raw)
        assert old[0]==new[0]==oracle[0],(index,pick,old,new,oracle)
        assert old[1].view(np.uint32)==new[1].view(np.uint32)==oracle[1].view(np.uint32)
        if old[0]<256:
            a[old[0]]=b[new[0]]=c[oracle[0]]=np.float32(-np.inf)
# This mutation must disagree: ordered keys without zero canonicalization
# incorrectly prefer +zero over -zero even when the latter has the lower ID.
zeros=np.full(256,-1,dtype=np.float32)
zeros[0]=np.float32(-0.)
zeros[1]=np.float32(0.)
best,_,ids=local(zeros,np.zeros(256,dtype=np.float32))
bits=best.view(np.uint32)
badkey=np.where((bits&0x80000000)!=0,~bits,bits^np.uint32(0x80000000))
wrong=int(np.min(np.where(badkey==badkey.max(),ids,INTMAX)))
assert wrong==1 and one_redux(zeros,np.zeros(256,dtype=np.float32))[0]==0

record={'source_sha256':hashlib.sha256(source).hexdigest(),'base_sha256':BASE_SHA,
        'original_module_ast_recovered':True,'complete_cuda_edit_reversible':True,
        'dot_and_split_reduction_bytes_unchanged':True,
        'seed':20261028,'numpy_version':np.__version__,'score_vectors':len(cases),
        'eight_round_selections':8*len(cases),'old_shuffle_and_redux_match_scalar_oracle':True,
        'raw_score_bits_unchanged':True,'signed_zero_ties_canonicalized':True,
        'missing_zero_canonicalization_mutant_detected':True,
        'covers':'ties, adjacent/extreme/subnormal values, +/-infinity, NaNs ignored by original comparisons, changing inputs',
        'scope_limit':'CPU selection algebra only; CUDA implementation, compilation and speed unverified'}
(ROOT/'.tools/decoder-router-redux-cpu-audit.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf8')
print(json.dumps(record))
