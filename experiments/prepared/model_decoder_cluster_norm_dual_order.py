"""Bind tiny reduction ownership/tree to generated PTX, then finite math/DSM.

No evaluator binary-identity or GPU-performance claim is made.
"""
import hashlib
import json
import re
import numpy as np
from preflight_decoder_cluster_norm import ROOT,BASE,freeze
from model_decoder_cluster_norm import bf16,schedule
from diagnose_decoder_cluster_norm_reduction import native_sum,distributed_matching_sum,xor_sum
from model_decoder_cluster_norm_ordered import compute

OUT=ROOT/'results/decoder-cluster-norm-dual-order-preflight-20261029'
PTX=ROOT/'results/decoder-sfn-offline-reduction-20261029/retained_sfn.ptx'


def canon(tree):
    if isinstance(tree,int):return str(tree)
    return '('+','.join(sorted(map(canon,tree)))+')'


def balanced(begin=0,count=32):
    return begin if count==1 else (balanced(begin,count//2),balanced(begin+count//2,count//2))


def ptx_proof(tid):
    text=PTX.read_text();addresses={};provenance={};stores={};post_tree=None;pre_tree=None
    base=lambda p:(p+1)*(1<<40)
    def number(s):return addresses[s] if s.startswith('%') else int(s,0)
    for line in text.splitlines():
        line=line.strip()
        m=re.fullmatch(r'ld.param.b64\s+(%rd\d+), \[_sfn_kernel_param_(\d+)\];',line)
        if m:addresses[m[1]]=base(int(m[2]));continue
        m=re.fullmatch(r'mov.u32\s+(%r\d+), %(tid|ctaid).x;',line)
        if m:addresses[m[1]]=tid if m[2]=='tid' else 0;continue
        m=re.fullmatch(r'(and.b32|or.b32|shl.b32|shr.u32|mul.wide.u32|mul.wide.s32|add.s64|add.s32)\s+(%r(?:d)?\d+), ([%\w]+), ([%\w]+);',line)
        if m:
            try:a,b=number(m[3]),number(m[4])
            except (KeyError,ValueError):continue
            op=m[1];addresses[m[2]]=a&b if op=='and.b32' else a|b if op=='or.b32' else a<<b if op=='shl.b32' else a>>b if op=='shr.u32' else a*b if op.startswith('mul.wide.') else a+b
            continue
        m=re.fullmatch(r'mad.wide.s32\s+(%rd\d+), ([%\w]+), ([%\w]+), ([%\w]+);',line)
        if m:
            try:addresses[m[1]]=number(m[2])*number(m[3])+number(m[4])
            except (KeyError,ValueError):pass
            continue
        m=re.search(r'ld.global.b16 \{ (%rs\d+) \}, \[ (%rd\d+) \+ 0 \];',line)
        if m:
            address=addresses[m[2]];param=address//(1<<40)-1;offset=address-base(param)
            assert offset%2==0
            provenance[m[1]]=('load',param,offset//2)
            continue
        m=re.fullmatch(r'mov.b32\s+(%r\d+), \{(%rs\d+), (%rs\d+)\};',line)
        if m and m[2] in provenance and m[3] in provenance:
            provenance[m[1]]=('pack',provenance[m[2]],provenance[m[3]]);continue
        m=re.fullmatch(r'mul.rn.bf16x2\s+(%r\d+), (%r\d+), (%r\d+);',line)
        if m:
            a,b=provenance[m[2]],provenance[m[3]];assert a[0]==b[0]=='pack'
            lanes=[]
            for x,y in zip(a[1:],b[1:]):
                assert {x[1],y[1]}=={0,2} and x[2]==y[2]
                assert (x[2]-tid)%256==0
                lanes.append(('leaf',(x[2]-tid)//256))
            provenance[m[1]]=('pack',*lanes);continue
        m=re.fullmatch(r'mov.b32\s+\{(%rs\d+), (%rs\d+)\}, (%r\d+);',line)
        if m and m[3] in provenance:
            packed=provenance[m[3]]
            if packed[0]=='pack':provenance[m[1]],provenance[m[2]]=packed[1:]
            continue
        m=re.fullmatch(r'cvt.f32.bf16\s+(%r\d+), (%rs\d+);',line)
        if m and m[2] in provenance:provenance[m[1]]=provenance[m[2]];continue
        m=re.search(r'st.global.b32 \[ (%rd\d+) \+ 0 \], \{ (%r\d+) \};',line)
        if m:
            offset=addresses[m[1]]-base(7)
            assert offset%4==0 and (offset//4-tid)%256==0
            stores[m[2]]=('leaf',(offset//4-tid)//256)
            continue
        m=re.fullmatch(r'mul.rn.f32\s+(%r\d+), (%r\d+), (%r\d+);',line)
        if m and m[2]==m[3]:
            if m[2] in stores:provenance[m[1]]=stores[m[2]]
            elif m[2] in provenance and provenance[m[2]][0]=='leaf':provenance[m[1]]=provenance[m[2]]
            continue
        m=re.fullmatch(r'add.rn.f32\s+(%r\d+), (%r\d+), (%r\d+);',line)
        if m and m[2] in provenance and m[3] in provenance:
            a,b=provenance[m[2]],provenance[m[3]]
            if a[0] in ('leaf','tree') and b[0] in ('leaf','tree'):
                provenance[m[1]]=('tree',(a[1],b[1]))
            continue
        m=re.fullmatch(r'shfl.sync.bfly.b32\s+%r\d+, (%r\d+), 16, 31, -1;',line)
        if m:
            tree=provenance[m[1]];assert tree[0]=='tree'
            if post_tree is None:post_tree=tree[1]
            else:pre_tree=tree[1]
    assert set(v[1] for v in stores.values())==set(range(32))
    assert canon(post_tree)==canon(balanced()) and canon(pre_tree)==canon(balanced())
    return post_tree,pre_tree


def tiny_sum(x,tree=None):
    square=(x*x).astype(np.float32).reshape(-1,32,256)
    def evaluate(node):
        if isinstance(node,int):return square[:,node,:]
        return (evaluate(node[0])+evaluate(node[1])).astype(np.float32)
    local=evaluate(balanced() if tree is None else tree)
    partial=xor_sum(local.reshape(-1,8,32))[:,:,0]
    # Model Triton's eight-way warp exchange, replicated in each warp subgroup.
    gathered=np.tile(partial,(1,4))
    for delta in (4,2,1):gathered=(gathered+gathered[:,np.arange(32)^delta]).astype(np.float32)
    return gathered[:,0]


def distributed_tiny_sum(x):
    square=(x*x).astype(np.float32).reshape(-1,32,256)
    partial=np.empty((x.shape[0],8),np.float32)
    for rank in range(4):
        local=square[:,:,rank*64:(rank+1)*64].copy()
        for count in (32,16,8,4,2):local=(local[:,0:count:2,:]+local[:,1:count:2,:]).astype(np.float32)
        partial[:,rank*2:rank*2+2]=xor_sum(local[:,0,:].reshape(-1,2,32))[:,:,0]
    # New CUDA gathers eight partials followed by24 zeros. Its extra zero
    # exchanges must preserve the same positive-sum result as Triton's tree.
    gather=np.zeros((x.shape[0],32),np.float32);gather[:,:8]=partial
    return xor_sum(gather)[:,0]


def main():
    trees=[ptx_proof(tid) for tid in (0,7,31,65,127,255)]
    assert all(canon(a)==canon(trees[0][0]) and canon(b)==canon(trees[0][1]) for a,b in trees)
    source=(ROOT/'experiments/prepared/decoder_cluster_norm_dual_order.cu').read_bytes()
    assert hashlib.sha256(source).hexdigest()=='5d189e9a2f7c5b1396c82cf60dd893886299dff5f857c23c0d93da316e1a5347'
    text=source.decode()
    for s in ('SLICE ? logical_thread+(pack*VEC+v)*LOGICAL_THREADS','tiny_tree<0,PACKS*VEC>(values)',
              '__fmul_rn(v,v)','__fadd_rn(tiny_tree<BEGIN,COUNT/2>',
              'if constexpr (TRITON) return rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/H),eps));'):
        assert s in text,s
    owners=[rank*64+tid+i*256 for rank in range(4) for tid in range(64) for i in range(32)]
    assert sorted(owners)==list(range(8192))
    for seed in range(256):schedule(seed)
    hazards={}
    for name,kw in (('reuse_stat_bank',dict(reuse_bank=True)),('omit_DSM_lifetime_join',dict(omit_final=True))):
        hazards[name]=False
        for seed in range(256):
            try:schedule(seed,**kw)
            except AssertionError:hazards[name]=True;break
        assert hazards[name]
    rng=np.random.default_rng(20261029);records=[];scalar_vs_tiny_different_rows=0
    for rows in (1,3,8,16,31,32):
        for kind in ((0,2) if rows<=8 else (1,3)):
            for fixture in ('random','zero_branch','zero_residual','outlier','small'):
                branch=bf16(rng.normal(size=(rows,1024 if kind==2 else 8192)).astype(np.float32))
                stream=rng.normal(size=(rows,8192)).astype(np.float32)
                post=bf16(rng.normal(1,.25,size=(1,8192)).astype(np.float32));pre=bf16(rng.normal(1,.25,size=(1,8192)).astype(np.float32))
                alpha=rng.uniform(.25,1,size=(1,8192)).astype(np.float32);beta=rng.uniform(.25,1,size=(1,8192)).astype(np.float32)
                if fixture=='zero_branch':branch[:]=0
                if fixture=='zero_residual':stream[:]=0
                if fixture=='outlier':branch[:,17]=32;stream[:,34]=-32
                if fixture=='small':branch=bf16(branch*np.float32(1e-4));stream*=np.float32(1e-4)
                args=(branch,stream,post,pre,alpha,beta,kind>=2)
                if kind==2:
                    expected=compute(*args,lambda x:tiny_sum(x,trees[0][0]));actual=compute(*args,distributed_tiny_sum)
                    scalar_vs_tiny_different_rows+=int(np.count_nonzero(native_sum(stream).view(np.uint32)!=tiny_sum(stream).view(np.uint32)))
                else:expected=compute(*args,native_sum);actual=compute(*args,distributed_matching_sum)
                assert all(np.array_equal(a.view(np.uint32),b.view(np.uint32)) for a,b in zip(expected,actual)),(rows,kind,fixture)
                assert all(np.isfinite(a).all() for a in actual)
                records.append(dict(rows=rows,kind=kind,fixture=fixture,all_outputs_bitwise_equal_under_CPU_arithmetic=True))
    report=dict(parent_source_sha256=BASE,native_CUDA_sha256=hashlib.sha256(source).hexdigest(),
        generated_PTX_sha256=hashlib.sha256(PTX.read_bytes()).hexdigest(),generated_Triton_tree_proved_for_logical_threads=[0,7,31,65,127,255],
        post_and_pre_trees_balanced_adjacent_32=True,scalar_vs_tiny_different_rows=scalar_vs_tiny_different_rows,
        full_row_ownership_columns=8192,DSM_random_schedules=256,hazards_detected=hazards,
        numerical_cases=records,CPU_only=True,GPU_jobs=0,
        limits='Triton3.9.0 compiler evidence, evaluator version unknown. Finite CPU arithmetic does not reproduce hardware rsqrt/div or scalar FMA contraction. Native resources, replay and performance require GPU validation.')
    freeze(OUT/'CPU-model.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k!='numerical_cases'},indent=2))


if __name__=='__main__':main()
