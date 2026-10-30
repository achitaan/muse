"""Independent forward-layout oracle and wrong-buffer/padding mutants."""

import ast
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from wgs_live_snapshot import jobs_from_routes,canonical_xb,snapshot_wgs_inter

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/decode-moe-plan-20261030/wgs-live-snapshot'


def main():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
    tree=ast.parse(raw)
    cuda=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')=='CUDA_SRC')
    assert 'WGS_XROWS = EMAX + 8 * JMAX'in cuda
    assert 'p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;'in cuda
    assert 'st_keep_bf16(xb +'in cuda
    assert 'constexpr int MMAX = 384, EMAX = MMAX * 8, JMAX = 32 + EMAX / FT_T + 8;'in cuda
    rng=random.Random(20261030);cases=[];mutants=0
    for m in (1,8,17,32,65,128,257,384):
        for mode in ('none','one','hot','all'):
            rows=[]
            for token in range(m):
                if mode=='none':row=rng.sample(range(32,256),8)
                elif mode=='one':row=[rng.randrange(32)]+rng.sample(range(32,256),7)
                elif mode=='hot':row=rng.sample(range(8),8)
                else:row=rng.sample(range(32),8)
                rows.append(row)
            jobs,n,padded=jobs_from_routes(rows)
            # Independent forward pack via tensor-axis permutation, not the
            # CUDA inverse-index formula. Reduced hidden512 exercises2stages.
            hidden=512;row=np.arange(n,dtype=np.uint32)[:,None];col=np.arange(hidden,dtype=np.uint32)[None,:]
            bits=((row*37+col*13)%20000+0x3000).astype(np.uint16)
            bits[(bits&0x7f80)==0x7f80]=0x3f80
            expected=torch.from_numpy(bits.view(np.int16)).view(torch.bfloat16)
            packed=torch.full((padded*hidden,),float('nan'),dtype=torch.bfloat16)
            for job in jobs:
                count,start=job['count'],job['start'];pad=(count+7)//8*8
                values=torch.full((pad,hidden),float('nan'),dtype=torch.bfloat16)
                values[:count]=expected[start:start+count]
                forward=values.reshape(pad//8,8,hidden//256,32,8).permute(2,0,3,1,4).contiguous().flatten()
                first=job['padded_start']*hidden;packed[first:first+pad*hidden]=forward
            observed=canonical_xb(packed,jobs,n,hidden)
            assert observed.shape==expected.shape
            if n:assert torch.equal(observed.view(torch.uint8),expected.view(torch.uint8))
            assert bool(torch.isfinite(observed).all())
            if n:
                changed=packed.clone();changed[0]=3.25
                assert not torch.equal(canonical_xb(changed,jobs,n,hidden).view(torch.uint8),expected.view(torch.uint8));mutants+=1
            cases.append(dict(M=m,mode=mode,routes=n,jobs=len(jobs),padded_rows=padded))
    # Actual4096-column allocation offsets: poison unused inter, put one
    # known live xB row at the documented offset, then change that live row.
    hidden=4096;ids=[[0]+list(range(32,39))];jobs,n,padded=jobs_from_routes(ids)
    begin=(3072+3776)*hidden
    scratch=torch.full((begin+padded*hidden,),float('nan'),dtype=torch.bfloat16)
    scratch[begin:]=1.25;state={'scratch':[None,None,scratch]}
    assert bool((snapshot_wgs_inter(state,ids)==1.25).all())
    scratch[:hidden]=7.;assert bool((snapshot_wgs_inter(state,ids)==1.25).all())
    scratch[begin]=2.;assert snapshot_wgs_inter(state,ids)[0,0].item()==2.;mutants+=2
    out=dict(parent_sha256=hashlib.sha256(raw).hexdigest(),parent_CUDA_sha256=hashlib.sha256(cuda.encode()).hexdigest(),
        layout_cases=cases,wrong_value_and_wrong_buffer_mutants_detected=mutants,
        actual_4096_column_offset_case_pass=True,EMAX=3072,JMAX=88,WGS_XROWS=3776,
        historical_correction='M32 WGS diagnostic sampled unused inter and ss buffers. Its ybuf/output byte observations remain valid;live xB equality and WGS norm statistics were not established.',
        GPU_jobs=0,Decode_checks_used=0,limitation='CPU layout and buffer ownership only; no GPU snapshot or numerical-performance claim.')
    OUT.mkdir(exist_ok=True);(OUT/'cpu-audit.json').write_bytes((json.dumps(out,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in out.items()if k!='layout_cases'},indent=2))


if __name__=='__main__':main()
