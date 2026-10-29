"""Source-bound finite ownership, rounding and DSM screens; no GPU access."""
import hashlib
import json
import numpy as np
from preflight_decoder_cluster_norm import ROOT, BASE, freeze
from model_decoder_cluster_norm import bf16, schedule
from diagnose_decoder_cluster_norm_reduction import native_sum, distributed_matching_sum

OUT=ROOT/'results/decoder-cluster-norm-ordered-preflight-20261029'


def compute(branch,stream,post_weight,pre_weight,alpha,beta,scaled,sum_fn):
    b=np.zeros_like(stream);b[:,:branch.shape[1]]=branch
    if scaled:b=bf16((b*post_weight).astype(np.float32))
    post=(np.float32(1)/np.sqrt((sum_fn(b)*np.float32(1/8192)+np.float32(1e-8)).astype(np.float32))).astype(np.float32)
    normed=bf16((b*post[:,None]).astype(np.float32))
    updated=((alpha*stream).astype(np.float32)+(beta*normed).astype(np.float32)).astype(np.float32)
    pre=(np.float32(1)/np.sqrt((sum_fn(updated)*np.float32(1/8192)+np.float32(1e-5)).astype(np.float32))).astype(np.float32)
    router=((updated*pre[:,None]).astype(np.float32)*pre_weight).astype(np.float32)
    return bf16(router),updated,router


def main():
    source=(ROOT/'experiments/prepared/decoder_cluster_norm_ordered.cu').read_bytes()
    assert hashlib.sha256(source).hexdigest()=='49bf73e68a8713a431f11d03d6f2f20e361562d4a105140962f8b6a11f269634'
    text=source.decode()
    for required in ('NT=64, LOGICAL_THREADS=256, PACKS=4, VEC=8','cluster.block_rank()*NT+threadIdx.x',
        'logical_thread+pack*LOGICAL_THREADS','vector_index*VEC+v','peer=lane/2,peer_warp=lane%2',
        'float partial[2][NT/32]','ss=fmaf(b,b,ss)','pre_ss=fmaf(value,value,pre_ss)'):
        assert required in text,required
    assert text.count('cluster.sync();')==2
    owners=[(rank*64+tid+pack*256)*8+v for rank in range(4) for tid in range(64) for pack in range(4) for v in range(8)]
    assert sorted(owners)==list(range(8192))
    for seed in range(256):schedule(seed)
    hazards={}
    for name,kw in (('reuse_stat_bank',dict(reuse_bank=True)),('omit_DSM_lifetime_join',dict(omit_final=True))):
        hazards[name]=False
        for seed in range(256):
            try:schedule(seed,**kw)
            except AssertionError:hazards[name]=True;break
        assert hazards[name]
    rng=np.random.default_rng(20261029);records=[]
    for rows in (1,3,8,16,31,32):
        for kind in ((0,2) if rows<=8 else (1,3)):
            for fixture in ('random','zero_branch','zero_residual','outlier','small'):
                branch=bf16(rng.normal(size=(rows,1024 if kind==2 else 8192)).astype(np.float32))
                stream=rng.normal(size=(rows,8192)).astype(np.float32)
                post=bf16(rng.normal(1,.25,size=(1,8192)).astype(np.float32))
                pre=bf16(rng.normal(1,.25,size=(1,8192)).astype(np.float32))
                alpha=rng.uniform(.25,1,size=(1,8192)).astype(np.float32)
                beta=rng.uniform(.25,1,size=(1,8192)).astype(np.float32)
                if fixture=='zero_branch':branch[:]=0
                if fixture=='zero_residual':stream[:]=0
                if fixture=='outlier':branch[:,17]=32;stream[:,34]=-32
                if fixture=='small':branch=bf16(branch*np.float32(1e-4));stream*=np.float32(1e-4)
                args=(branch,stream,post,pre,alpha,beta,kind>=2)
                expected=compute(*args,native_sum);actual=compute(*args,distributed_matching_sum)
                assert all(np.array_equal(a.view(np.uint32),b.view(np.uint32)) for a,b in zip(expected,actual))
                assert all(np.isfinite(a).all() for a in actual)
                records.append(dict(rows=rows,kind=kind,fixture=fixture,all_outputs_bitwise_equal_under_CPU_arithmetic=True))
    report=dict(parent_source_sha256=BASE,native_CUDA_sha256=hashlib.sha256(source).hexdigest(),
        full_row_ownership_columns=8192,DSM_random_schedules=256,hazards_detected=hazards,
        numerical_cases=records,CPU_only=True,GPU_jobs=0,
        limits='Finite source-order simulation. Does not reproduce hardware rsqrt, compiler FMA contraction, graph replay, native occupancy, performance, or the unavailable failing GPU input.')
    freeze(OUT/'CPU-model.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(dict(cases=len(records),DSM_schedules=256,hazards=hazards,CPU_only=True)))


if __name__=='__main__':main()
