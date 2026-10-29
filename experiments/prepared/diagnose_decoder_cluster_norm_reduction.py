"""Pinned SGLang reduction-order diagnosis and a finite CPU repair screen."""
import hashlib
import json
from pathlib import Path

import numpy as np

from model_decoder_cluster_norm import bf16,cluster_sum
from preflight_decoder_cluster_norm import ROOT,BASE,freeze


def xor_sum(x):
    x=x.copy()
    for delta in (16,8,4,2,1):x=(x+x[...,np.arange(32)^delta]).astype(np.float32)
    return x


def lane_sums(x):
    # Pinned scalar CTA:256threads, four packs/thread, eight contiguous BF16
    # values per pack; v.x then v.y FMA sequence for each of the four pairs.
    vec=x.reshape(-1,4,256,8)
    lanes=np.zeros((x.shape[0],256),dtype=np.float32)
    for pack in range(4):
        for v in range(8):
            value=vec[:,pack,:,v].astype(np.float64)
            lanes=(value*value+lanes.astype(np.float64)).astype(np.float32)
    return lanes


def native_sum(x):
    lanes=lane_sums(x).reshape(-1,8,32)
    partial=xor_sum(lanes)[:,:,0]
    collectors=np.zeros((x.shape[0],32),dtype=np.float32);collectors[:,:8]=partial
    return xor_sum(collectors)[:,0]


def distributed_matching_sum(x):
    # Four64thread CTAs partition the original256 logical threads. Per-thread
    # element/FMA order stays identical; every global warp partial is gathered
    # before applying the same eight-warp final tree. No per-CTA collapsed sum.
    reference=x.reshape(-1,4,256,8)
    storage=np.empty((x.shape[0],4,2),dtype=np.float32)
    seen=[]
    for rank in range(4):
        lo=rank*64;hi=lo+64
        local=np.zeros((x.shape[0],64),dtype=np.float32)
        for pack in range(4):
            for v in range(8):
                values=reference[:,pack,lo:hi,v].astype(np.float64)
                local=(values*values+local.astype(np.float64)).astype(np.float32)
                seen.extend(((np.arange(lo,hi)+pack*256)*8+v).tolist())
        storage[:,rank,:]=xor_sum(local.reshape(-1,2,32))[:,:,0]
    assert sorted(seen)==list(range(8192))
    collectors=np.zeros((x.shape[0],32),dtype=np.float32)
    for global_warp in range(8):collectors[:,global_warp]=storage[:,global_warp//2,global_warp%2]
    return xor_sum(collectors)[:,0]


def main():
    files={
        'fused_norm_residual.py':ROOT/'.tools/cluster-norm-pinned-sglang.py',
        'norm_residual.py':ROOT/'.tools/cluster-norm-pinned-dispatch.py',
        'norm_residual.cuh':ROOT/'.tools/cluster-norm-pinned-sglang.cuh',
        'cta.cuh':ROOT/'.tools/cluster-norm-pinned-cta.cuh',
        'warp.cuh':ROOT/'.tools/cluster-norm-pinned-warp.cuh',
        'math.cuh':ROOT/'.tools/cluster-norm-pinned-math.cuh'}
    text=files['norm_residual.cuh'].read_text();cta=files['cta.cuh'].read_text();warp=files['warp.cuh'].read_text()
    assert 'const int idx = threadIdx.x + p * kNumThreads;' in text
    assert 'sum_sq = fmaf(v.x, v.x, sum_sq);' in text and 'sum_sq = fmaf(v.y, v.y, sum_sq);' in text
    assert 'return warp::reduce_sum(lane < kNumWarps ? smem[lane] : T{0});' in cta
    assert '__shfl_xor_sync(active_mask, value, mask, kStart)' in warp
    rng=np.random.default_rng(20261029);records=[]
    for fixture in ('random','weighted','zero_tail','outlier','small','zero'):
        x=bf16(rng.normal(size=(128,8192)).astype(np.float32))
        if fixture=='weighted':x=bf16(x*bf16(rng.normal(1,.5,size=x.shape).astype(np.float32)))
        if fixture=='zero_tail':x[:,1024:]=0
        if fixture=='outlier':x[:,17]=32
        if fixture=='small':x=bf16(x*np.float32(1e-4))
        if fixture=='zero':x[:]=0
        original=native_sum(x);old_prototype=cluster_sum(x);repaired=distributed_matching_sum(x)
        assert np.array_equal(original.view(np.uint32),repaired.view(np.uint32))
        old_inv=(np.float32(1)/np.sqrt((old_prototype*np.float32(1/8192)+np.float32(1e-8)).astype(np.float32))).astype(np.float32)
        reference_inv=(np.float32(1)/np.sqrt((original*np.float32(1/8192)+np.float32(1e-8)).astype(np.float32))).astype(np.float32)
        changed=bf16(x*old_inv[:,None]).view(np.uint32)!=bf16(x*reference_inv[:,None]).view(np.uint32)
        records.append(dict(fixture=fixture,rows=128,
            original_vs_first_prototype_different_row_sums=int(np.count_nonzero(original.view(np.uint32)!=old_prototype.view(np.uint32))),
            differing_BF16_postnorm_values=int(np.count_nonzero(changed)),
            proposed_distributed_tree_bitwise_matches_reference_row_sums=True))
    out=ROOT/'results/decoder-cluster-norm-reduction-diagnosis-20261029'
    for name,p in files.items():freeze(out/'pinned-upstream'/name,p.read_bytes())
    report=dict(parent_source_sha256=BASE,upstream_commit='a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f',
        CPU_only=True,GPU_jobs=0,GPU_failure_source='f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf',
        exact_failed_GPU_input_unavailable=True,failed_probe_m_and_kind_unrecorded=True,
        root_cause_status='Source-backed reduction-order hypothesis with finite reproductions; not proven against the actual failing GPU row',
        proposed_repair='Preserve256logical threads, original vec8/pack order and eight-warp XOR tree; distribute64logical threads to each of four CTAs and gather all eight warp partials',
        fixed_numerical_gate_unchanged=True,no_native_repair_written_yet=True,cases=records,
        pinned_files={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()},
        upstream_scalar_url='https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh',
        next='One bounded source-order repair, offline resources/precision/DSM proof, then decide whether one paced retry is justified. Do not relax gates or import the upstream moment shortcut.')
    freeze(out/'diagnosis.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='pinned_files'},indent=2))


if __name__=='__main__':main()
