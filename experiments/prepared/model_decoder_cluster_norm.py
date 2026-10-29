"""CPU ownership, DSM lifetime, precision and weighted-payoff screens."""
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np

from preflight_decoder_cluster_norm import ROOT,BASE,freeze


def bf16(x):
    x=np.asarray(x,dtype=np.float32)
    u=x.view(np.uint32)
    return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)


def cluster_sum(x):
    # Explicit model of rank/warp/lane ownership, local FMA chains, warp tree,
    # warp0's eight values, then the same rank0..3 addition order on every CTA.
    t=x.reshape(-1,4,8,256)
    sums=np.zeros(t.shape[:2]+(256,),dtype=np.float32)
    for i in range(8):sums=(t[:,:,i,:].astype(np.float64)**2+sums.astype(np.float64)).astype(np.float32)
    w=sums.reshape(-1,4,8,32)
    for delta in (16,8,4,2,1):
        before=w.copy();w[:,:,:,:32-delta]=(before[:,:,:,:32-delta]+before[:,:,:,delta:]).astype(np.float32)
    warps=np.zeros(w.shape[:2]+(32,),dtype=np.float32);warps[:,:,:8]=w[:,:,:,0]
    for delta in (16,8,4,2,1):
        before=warps.copy();warps[:,:,:32-delta]=(before[:,:,:32-delta]+before[:,:,delta:]).astype(np.float32)
    total=np.zeros(x.shape[0],dtype=np.float32)
    for rank in range(4):total=(total+warps[:,rank,0]).astype(np.float32)
    return total


def compute(branch,stream,post_weight,pre_weight,alpha,beta,scaled,sliced,oracle=False,mutant=None):
    b=np.zeros_like(stream)
    b[:,:branch.shape[1]]=branch
    if scaled:b=bf16((b*post_weight).astype(np.float32))
    elif mutant=='extra_round':b=bf16(b)
    denominator=1024 if mutant=='wrong_norm_domain' and sliced else 8192
    sum0=np.sum(b.astype(np.float64)**2,axis=1) if oracle else cluster_sum(b)
    post=(1/np.sqrt(sum0/denominator+1e-8)).astype(np.float32)
    normed=bf16((b*post[:,None]).astype(np.float32))
    # Products are rounded separately before addition, matching native rn ops.
    updated=((alpha*stream).astype(np.float32)+(beta*normed).astype(np.float32)).astype(np.float32)
    sum1=np.sum(updated.astype(np.float64)**2,axis=1) if oracle else cluster_sum(updated)
    pre=(1/np.sqrt(sum1/8192+1e-5)).astype(np.float32)
    router=((updated*pre[:,None]).astype(np.float32)*pre_weight).astype(np.float32)
    return bf16(router),updated,router


def schedule(seed,reuse_bank=False,omit_final=False):
    rng=random.Random(seed);phase=['write0']*4;ready=[set(),set()];last=set()
    banks=[[None,None] for _ in range(4)];alive=[True]*4;steps=0
    while any(p!='done' for p in phase):
        actors=[]
        for r,p in enumerate(phase):
            if p=='done':continue
            if p=='join0' and len(ready[0])<4:continue
            if p=='join1' and len(ready[1])<4:continue
            if p=='end' and not omit_final and len(last)<4:continue
            actors.append(r)
        assert actors,'DSM schedule deadlock'
        r=rng.choice(actors);p=phase[r];steps+=1;assert steps<200
        if p.startswith('write'):
            n=int(p[-1]);banks[r][0 if reuse_bank else n]=n
            ready[n].add(r);phase[r]='join'+str(n)
        elif p.startswith('join'):phase[r]='read'+p[-1]
        elif p.startswith('read'):
            n=int(p[-1])
            assert all(alive),'CTA exited while a peer still accessed its DSM'
            assert all(banks[peer][0 if reuse_bank else n]==n for peer in range(4)),'statistic bank overwritten before peer read'
            if n==0:phase[r]='write1'
            else:last.add(r);phase[r]='end'
        else:alive[r]=False;phase[r]='done'


def main():
    assert hashlib.sha256((ROOT/'best/decoder_verified.py').read_bytes()).hexdigest()==BASE
    owners=[rank*2048+tid+i*256 for rank in range(4) for tid in range(256) for i in range(8)]
    assert sorted(owners)==list(range(8192))
    for seed in range(256):schedule(seed)
    detected={}
    for name,kw in (('reuse_stat_bank',dict(reuse_bank=True)),('omit_DSM_lifetime_join',dict(omit_final=True))):
        detected[name]=False
        for seed in range(256):
            try:schedule(seed,**kw)
            except AssertionError:detected[name]=True;break
        assert detected[name]
    rng=np.random.default_rng(20261029);records=[];bad_domain=False
    for rows in (1,3,8,16,31,32):
        for kind in range(4):
            if kind==0 and rows>8 or kind==1 and rows<=8 or kind==2 and rows>8 or kind==3 and rows<=8:continue
            sliced=kind==2;scaled=kind>=2
            for fixture in ('random','zero_branch','zero_residual','outlier','small'):
                branch=bf16(rng.normal(size=(rows,1024 if sliced else 8192)).astype(np.float32))
                stream=rng.normal(size=(rows,8192)).astype(np.float32)
                post=bf16(rng.normal(1,.25,size=(1,8192)).astype(np.float32))
                pre=bf16(rng.normal(1,.25,size=(1,8192)).astype(np.float32))
                alpha=rng.uniform(.25,1,size=(1,8192)).astype(np.float32)
                beta=rng.uniform(.25,1,size=(1,8192)).astype(np.float32)
                if fixture=='zero_branch':branch[:]=0
                if fixture=='zero_residual':stream[:]=0
                if fixture=='outlier':branch[:,17]=32;stream[:,34]=-32
                if fixture=='small':branch=bf16(branch*np.float32(1e-4));stream*=np.float32(1e-4)
                args=(branch,stream,post,pre,alpha,beta,scaled,sliced)
                expected=compute(*args,oracle=True);value=compute(*args)
                errors=[]
                for a,b in zip(expected,value):
                    assert np.isfinite(b).all()
                    error=float(np.linalg.norm((b-a).astype(np.float64))/max(np.linalg.norm(a.astype(np.float64)),1e-12))
                    errors.append(error)
                assert max(errors)<1e-3,(rows,kind,fixture,errors)
                if sliced and fixture=='random':
                    wrong=compute(*args,mutant='wrong_norm_domain')
                    bad_domain |= float(np.linalg.norm(wrong[1]-expected[1])/max(np.linalg.norm(expected[1]),1e-12))>.01
                records.append(dict(rows=rows,kind=kind,fixture=fixture,relative_L2=errors))
    assert bad_domain
    loads=json.loads((ROOT/'results/retained-load-opportunities-20261029/loads.json').read_bytes())['studies']['decode']
    assert loads['source_sha256']==BASE
    scenarios=[]
    for saving in (1.,2.,3.,4.):
        gains=[]
        for name,row in loads['loads'].items():
            cases=row['cases'];tokens=cases[0]['tokens'];old=row['retained_load_ns']
            charged=saving*1000*sum(c['weight'] for c in cases) if tokens<=32 else 0
            gains.append(old/(old-charged))
        scenarios.append(dict(hypothetical_net_saving_per_changed_layer_us=saving,
                              weighted_track_ratio=math.prod(gains)**(1/6)))
    report=dict(parent_source_sha256=BASE,CUDA_sha256=hashlib.sha256((ROOT/'experiments/prepared/decoder_cluster_norm.cu').read_bytes()).hexdigest(),
        CPU_only=True,GPU_jobs=0,full_row_ownership_columns=8192,DSM_random_schedules=256,
        hazards_detected=detected,numerical_cases=records,wrong_1024_domain_detected=bad_domain,
        conditional_payoff_scenarios=scenarios,
        limits='CPU sum tree and FP64 row oracle are finite screens, not CUDA graph correctness or a performance forecast. Profile sums are separate-run durations, not measured critical-path fractions.')
    out=ROOT/'results/decoder-cluster-norm-preflight-20261029'
    freeze(out/'CPU-model.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(dict(numerical_cases=len(records),maximum_relative_L2=max(max(r['relative_L2']) for r in records),
        DSM_schedules=256,hazards_detected=detected,conditional_payoff_scenarios=scenarios,GPU_jobs=0),indent=2))


if __name__=='__main__':main()
