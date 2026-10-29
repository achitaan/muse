"""CPU contract/ownership tests for the compile-only cooperative O boundary.

Finite arithmetic checks are not a GPU execution, race proof, or launch test.
"""
import hashlib
import json
import ast
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
PACKET = ROOT/'results/decoder-cooperative-boundary-screen-20261029'
BASE = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
POST_EPS = 1e-8
PRE_EPS = 1e-5


def bf(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.bfloat16).float().numpy()


def reduction(x, nt):
    # Model the prototype's scalar FMA accumulation and two shuffle trees.
    a = np.zeros(nt, dtype=np.float32)
    for row in x.reshape(-1, nt):
        a = (row.astype(np.float64)**2+a.astype(np.float64)).astype(np.float32)
    a = a.reshape(-1,32)
    for d in (16,8,4,2,1):
        old=a.copy(); a[:,:32-d]=(old[:,:32-d]+old[:,d:]).astype(np.float32)
    b=np.zeros(32,dtype=np.float32); b[:len(a)]=a[:,0]
    for d in (16,8,4,2,1):
        old=b.copy(); b[:32-d]=(old[:32-d]+old[d:]).astype(np.float32)
    return b[0]


def evaluate(branch, stream, alpha, beta, gain, nt, mutant=None):
    h=8192
    denom=1024 if mutant=='cropped_norm' else h
    post=np.float32(1/np.sqrt(np.float32(reduction(branch,nt)/denom+POST_EPS)))
    normalized=(branch*post).astype(np.float32)
    if mutant!='drop_post_bf16': normalized=bf(normalized)
    residual=((alpha*stream).astype(np.float32)+(beta*normalized).astype(np.float32)).astype(np.float32)
    pre=np.float32(1/np.sqrt(np.float32(reduction(residual,nt)/denom+PRE_EPS)))
    router=((residual*pre).astype(np.float32)*gain).astype(np.float32)
    return bf(router),residual,bf(router) if mutant=='round_router' else router


def main():
    assert hashlib.sha256((ROOT/'best/decoder_verified.py').read_bytes()).hexdigest()==BASE
    constants={}
    for node in ast.parse((ROOT/'best/decoder_verified.py').read_bytes()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('RMS_EPS','POST_NORM_EPS'):
                    constants[target.id]=ast.literal_eval(node.value)
    assert constants==dict(RMS_EPS=PRE_EPS,POST_NORM_EPS=POST_EPS)
    cpp=(PACKET/'pinned/norm_residual.cuh').read_text(encoding='utf-8')
    wrapper=(PACKET/'pinned/norm_residual.py').read_text(encoding='utf-8')
    for anchor in ('normed_b = round_bf16(normed_b);','const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;',
                   'router_v[q / kFPairs][q % kFPairs] = v;'):
        assert anchor in cpp
    assert 'torch.cuda.get_device_capability(branch.device)[0] >= 10' in wrapper
    rng=np.random.default_rng(20261029)
    maximum=dict(output_l2=0.,residual_l2=0.,router_l2=0.)
    detected={k:0 for k in ('drop_post_bf16','round_router','cropped_norm')}
    cases=0; ownership=0
    for m in range(1,9):
        nt=1024 if m==1 else 256
        grid=128 if m==1 else 512
        # All CTAs take one grid barrier; only first M CTAs own final rows.
        owners=[block for block in range(grid) if block<m]
        assert owners==list(range(m))
        for row in owners:
            positions=[h for tid in range(nt) for h in range(tid,8192,nt)]
            assert len(positions)==8192 and len(set(positions))==8192
            assert min(positions)==0 and max(positions)==8191
            ownership+=1
        for fixture in ('zero','constant','random','small','outlier'):
            branch=bf(rng.normal(size=8192).astype(np.float32))
            stream=rng.normal(size=8192).astype(np.float32)
            if fixture=='zero': branch[:]=0; stream[:]=0
            if fixture=='constant': branch[:]=1; stream[:]=.25
            if fixture=='small': branch*=np.float32(1e-5); stream*=np.float32(1e-5)
            if fixture=='outlier': branch[13]=256; stream[111]=-1024
            alpha=rng.uniform(.1,1,size=8192).astype(np.float32)
            beta=rng.uniform(.1,1,size=8192).astype(np.float32)
            gain=bf(1+bf(rng.normal(0,.2,size=8192).astype(np.float32)))
            actual=evaluate(branch,stream,alpha,beta,gain,nt)
            # Independent high-precision row sum; retains the required BF16
            # materialization and FP32 two-product residual operations.
            post=np.float32(1/np.sqrt(np.mean(branch.astype(np.float64)**2)+POST_EPS))
            nb=bf((branch*post).astype(np.float32))
            residual=((alpha*stream).astype(np.float32)+(beta*nb).astype(np.float32)).astype(np.float32)
            pre=np.float32(1/np.sqrt(np.mean(residual.astype(np.float64)**2)+PRE_EPS))
            router=((residual*pre).astype(np.float32)*gain).astype(np.float32)
            expected=(bf(router),residual,router)
            for name,a,b in zip(maximum,actual,expected):
                err=float(np.linalg.norm((a-b).astype(np.float64))/max(np.linalg.norm(b.astype(np.float64)),1e-12))
                maximum[name]=max(maximum[name],err)
                assert np.isfinite(a).all() and err<1e-3,(fixture,name,err)
            for mutant in detected:
                changed=evaluate(branch,stream,alpha,beta,gain,nt,mutant)
                if any(not np.array_equal(a,b) for a,b in zip(actual,changed)): detected[mutant]+=1
            cases+=1
    assert all(detected.values())
    loads=json.loads((ROOT/'results/retained-load-opportunities-20261029/loads.json').read_text())
    decode=loads['studies']['decode']
    # Removal-only optimistic screen. These are historical separate-profile
    # durations, not critical-path measurements or forecasts.
    bounds=[]
    for saving in (1.,2.,4.5):
        gains=[]
        for load in decode['loads'].values():
            old=load['retained_load_ns']/1000
            delta=sum(c['weight']*saving for c in load['cases'] if c['tokens']<=8)
            gains.append(old/(old-delta))
        bounds.append(dict(assumed_saved_us_per_affected_case=saving,
                           optimistic_track_gain=float(np.prod(gains)**(1/len(gains)))))
    report=dict(revision=2,post_eps=POST_EPS,pre_eps=PRE_EPS,
                supersedes='model.json used1e-8 as a generic pre-epsilon proxy; retained only as historical evidence',
                baseline_sha256=BASE,CPU_only=True,GPU_jobs=0,numerical_cases=cases,
                finalizer_row_ownership_cases=ownership,mutants_detected=detected,
                max_relative_L2_vs_high_precision_row_sum=maximum,optimistic_screens=bounds,
                pinned_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PACKET/'pinned').iterdir()},
                limitations=['No GPU race/graph/occupancy/precision/timing validation',
                             'FP32 reduction grouping differs; GPU checker remains required',
                             'Selected CUDA router output is FP32 unrounded with pre_norm_bf16=false',
                             'No overall aggregation formula established'])
    path=PACKET/'model-v2.json'
    data=(json.dumps(report,indent=2)+'\n').encode()
    assert not path.exists() or path.read_bytes()==data
    path.write_bytes(data)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
