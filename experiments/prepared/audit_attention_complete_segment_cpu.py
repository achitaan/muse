"""Public-shape guard and nearby large-L8 fallback versus independent FP64 math."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_attention_three_cta import ROOT,literal
sys.path.insert(0,str(ROOT/'scripts'))
from verify_attention_precision import verify
torch.set_num_threads(2)
raw=(ROOT/'.tools/attention-complete-segment-variant.py').read_bytes();verify(raw)
tree=ast.parse(raw)
fits=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fits')
scope={'INDEX_LIMIT':2**31}
exec(compile(ast.Module(body=[fits],type_ignores=[]),'<shape-guard>','exec'),scope)
class Shape:
    def __init__(self,n,shape):self.n,self.shape=n,shape
    def numel(self):return self.n
count=0
for B in (1,2,8,16,32,33,64,65,128,256,512):
    for L in (1,8):
        for P in (1,8,64):
            cfg=dict(batch=B,q_len=L,num_q_heads=16,head_dim=64,page_size=P)
            pages=(80000+P-1)//P
            data=dict(config=cfg,qkv=Shape(B*L*2304,(B,L,2304)),page_table=Shape(B*pages,(B,pages)))
            got=scope['_fits'](data)
            assert got==(L==1 or B<=64)
            if got:assert B<=512 and (L==1 or B<=64)
            count+=1

# Execute exactly the unchanged fallback functions, adapting only CPU float8
# advanced indexing to byte indexing (Torch2.3 lacks that CPU operation).
code=ast.unparse(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_rms','_rope','_fallback')],type_ignores=[]))
for name,value in [('k_cache','k / k_scale'),('v_cache','v_in.reshape(batch, length, hkv, d).float() / v_scale')]:
    before=f'{name}[slots] = ({value}).to(torch.float8_e4m3fn)'
    after=f'{name}.view(torch.uint8)[slots] = ({value}).to(torch.float8_e4m3fn).view(torch.uint8)'
    assert code.count(before)==1
    code=code.replace(before,after).replace(f'{name}[rows].float()',f'{name}.view(torch.uint8)[rows].view(torch.float8_e4m3fn).float()')
scope={'torch':torch,'_inv_freq':lambda theta,device,n:1/(theta**(torch.arange(n,device=device).float()/n))}
exec(compile(code,'<unchanged-fallback-byte-indexing>','exec'),scope)
gen=torch.Generator().manual_seed(20261028)
errors=[]
for window in (0,4):
    B,L,HQ,HK,D,S,P=65,8,16,2,64,16,8
    cfg=dict(batch=B,q_len=L,num_q_heads=HQ,num_kv_heads=HK,head_dim=D,page_size=P,
             window=window,rope_theta=500000. if window else 0.,sm_scale=1/8,qk_scale_factor=5.473)
    positions=torch.arange(3,3+L).repeat(B,1)
    pages=torch.arange(B*S//P).reshape(B,-1)
    slots=pages.gather(1,positions//P)*P+positions%P
    qkv=torch.randn(B,L,(2*HQ+2*HK)*D,generator=gen).bfloat16()
    kc=torch.randn(B*S,HK,D,generator=gen).to(torch.float8_e4m3fn)
    vc=torch.randn(B*S,HK,D,generator=gen).to(torch.float8_e4m3fn)
    data=dict(config=cfg,qkv=qkv,positions=positions,page_table=pages,cache_loc=slots,
              k_cache=kc,v_cache=vc,k_scale=torch.tensor(1.),v_scale=torch.tensor(1.))
    got=scope['_fallback'](data)
    q,_,_,gate=qkv.split([HQ*D,HK*D,HK*D,HQ*D],-1)
    q=q.reshape(B,L,HQ,D).double()
    q=(q/torch.sqrt((q*q).mean(-1,keepdim=True)+1e-5)).bfloat16().double()
    if window:
        inv=1/(cfg['rope_theta']**(torch.arange(D//2).double()/(D//2)))
        angle=positions.double()[...,None,None]*inv
        a,b=q[...,::2].clone(),q[...,1::2].clone()
        q[...,::2]=a*angle.cos()-b*angle.sin();q[...,1::2]=a*angle.sin()+b*angle.cos()
        q=q.bfloat16().double()
    expected=torch.empty_like(got)
    gates=gate.reshape(B,L,HQ,D).double()
    for b in range(B):
        for j in range(L):
            end=4+j;start=max(0,end-window) if window else 0
            rows=torch.arange(b*S+start,b*S+end)
            key=kc.view(torch.uint8)[rows].view(torch.float8_e4m3fn).double().repeat_interleave(8,dim=1)
            val=vc.view(torch.uint8)[rows].view(torch.float8_e4m3fn).double().repeat_interleave(8,dim=1)
            scores=torch.einsum('hd,khd->hk',q[b,j],key)*(5.473/8)
            attn=torch.einsum('hk,khd->hd',scores.softmax(-1),val).bfloat16().double()
            y=attn/torch.sqrt((attn*attn).mean(-1,keepdim=True)+1e-5)*gates[b,j].sigmoid()
            expected[b,j]=y.reshape(-1).bfloat16()
    rel=float(torch.linalg.vector_norm(got.float()-expected.float())/torch.linalg.vector_norm(expected.float()))
    assert rel<.01 and torch.isfinite(got.float()).all(),(window,rel)
    errors.append(rel)
report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),guard_cases=count,
            large_l8_fallback_cases=2,batch=65,max_relative_l2_vs_fp64=max(errors),
            native_cuda_matches_checked_c050=True,bf16_boundaries_preserved=True,
            official_full_scored_evaluation_pending=True)
(ROOT/'.tools/attention-complete-segment-cpu-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report,indent=2))
