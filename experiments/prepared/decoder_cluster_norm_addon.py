# Cluster4 boundary-only experiment. Projections and all other layer paths stay
# in the retained source; complete inherited notices remain in that source.
import json as _cn_json
import random as _cn_random
import statistics as _cn_stats
from torch.utils.cpp_extension import load_inline as _cn_load

_cn_old_prepare=prepare
_cn_old_sfn=_sfn_boundary


def _cn_cache(boundary,scaled):
    if (not boundary.jit_selected() or boundary._post_norm_is_input_scale!=scaled
            or boundary._post_norm.has_weight!=scaled or not boundary._use_residual_gate
            or not boundary._pre_norm.has_weight or boundary._residual_add.gate.numel()!=HIDDEN):
        return None
    boundary._ensure_cached_weights(torch.bfloat16)
    cache={name:getattr(boundary,name) for name in ('_cc_pre_weight','_cc_alpha','_cc_beta')}
    if scaled:cache['_cc_post_weight']=boundary._cc_post_weight
    for key,value in cache.items():
        dtype=torch.float32 if key in ('_cc_alpha','_cc_beta') else torch.bfloat16
        if value.dtype!=dtype or value.numel()!=HIDDEN or not value.is_contiguous():return None
    cache.update(post_eps=boundary._post_norm.eps,pre_eps=boundary._pre_norm.eps)
    if cache['post_eps']!=1e-8 or cache['pre_eps']!=1e-5:return None
    return cache


def _cn_call(ext,kind,stream,branch,cache):
    m=branch.shape[0]
    output=torch.empty((m,HIDDEN),dtype=torch.bfloat16,device=branch.device)
    updated=torch.empty((m,HIDDEN),dtype=torch.float32,device=branch.device)
    router=torch.empty_like(updated) if kind==0 else None
    post=cache.get('_cc_post_weight')
    ext.cn_launch(kind,m,branch.stride(0),stream.stride(0),branch.data_ptr(),
        0 if post is None else post.data_ptr(),cache['_cc_pre_weight'].data_ptr(),stream.data_ptr(),
        cache['_cc_alpha'].data_ptr(),cache['_cc_beta'].data_ptr(),output.data_ptr(),updated.data_ptr(),
        0 if router is None else router.data_ptr(),cache['post_eps'],cache['pre_eps'],
        torch.cuda.current_stream(branch.device).cuda_stream)
    return output,updated,router


def _cn_error(actual,expected):
    values=[]
    for i,(a,b) in enumerate(zip(actual,expected)):
        if a is None:
            continue # unused router result for semantic kinds1..3
        if a.shape!=b.shape or a.dtype!=b.dtype or not bool(torch.isfinite(a).all()):
            raise RuntimeError('cluster norm shape/dtype/finite probe failed')
        af,bf=a.float(),b.float()
        diff=torch.linalg.vector_norm(af-bf,dim=-1);denom=torch.linalg.vector_norm(bf,dim=-1)
        floor=max(float(denom.mean())*.1,1e-12)
        row=float((diff/denom.clamp_min(floor)).max())
        global_error=float(torch.linalg.vector_norm(af-bf)/torch.linalg.vector_norm(bf).clamp_min(1e-12))
        limit=1e-3 if i==0 else 2e-5
        if max(global_error,row)>limit:raise RuntimeError(f'cluster norm numerical probe failed:{i},{global_error},{row}')
        values.append([global_error,row])
    return values


def _cn_paired(old,new,dev,seed):
    # Same cold-L2 flush and alternating graph-arm order for both algorithms.
    flush=torch.ones((128,1024,1024),dtype=torch.float32,device=dev)
    sink=torch.empty((),dtype=torch.float32,device=dev)
    stream=torch.cuda.Stream(device=dev);stream.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(stream):
        old();new();stream.synchronize()
        graphs=[torch.cuda.CUDAGraph(),torch.cuda.CUDAGraph()]
        for arm,call in enumerate((old,new)):
            with torch.cuda.graph(graphs[arm],stream=stream):call()
        orders=[(0,1) if (i+seed)%2==0 else (1,0) for i in range(10)]
        events=[]
        for i,order in enumerate(orders):
            for arm in order:
                torch.sum(flush,dim=(0,1,2),out=sink)
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record(stream);graphs[arm].replay();b.record(stream);events.append((i,arm,a,b))
        stream.synchronize()
    samples=[dict(pair=i,order=list(order)) for i,order in enumerate(orders)]
    for i,arm,a,b in events:samples[i][arm]=a.elapsed_time(b)*1_000_000
    old_ns=_cn_stats.median(r[0] for r in samples);new_ns=_cn_stats.median(r[1] for r in samples)
    return dict(old_ns=old_ns,new_ns=new_ns,new_over_old=new_ns/old_ns,
        samples=[dict(pair=r['pair'],order=r['order'],old_ns=r[0],new_ns=r[1],new_over_old=r[1]/r[0]) for r in samples])


class _CNBoundary:
    def __init__(self,old,ext,cache,scaled):
        self.old,self.ext,self.cache,self.scaled=old,ext,cache,scaled
    def __getattr__(self,name):return getattr(self.old,name)
    def __call__(self,stream,branch,branch_dtype=torch.bfloat16,router_dtype=None,**kwargs):
        m=branch.shape[0]
        if (1<=m<=32 and branch.dtype==torch.bfloat16 and stream.dtype==torch.float32
                and branch.is_contiguous() and stream.is_contiguous() and branch.shape[1]==HIDDEN
                and stream.shape==branch.shape and branch_dtype==torch.bfloat16
                and router_dtype in (None,torch.float32) and not kwargs):
            kind=3 if self.scaled else (0 if router_dtype==torch.float32 else 1)
            return _cn_call(self.ext,kind,stream,branch,self.cache)
        return self.old(stream,branch,branch_dtype=branch_dtype,router_dtype=router_dtype,**kwargs)


def _sfn_boundary(stream,branch,state):
    cn=state.get('cluster_norm')
    if (cn is not None and 1<=branch.shape[0]<=8 and branch.shape[1]==UNPOOL_SLICE
            and branch.is_contiguous() and stream.is_contiguous()
            and branch.dtype==torch.bfloat16 and stream.dtype==torch.float32):
        a,b,_=_cn_call(cn['ext'],2,stream,branch,cn['ffn']);return a,b
    return _cn_old_sfn(stream,branch,state)


def prepare(weights):
    state=_cn_old_prepare(weights);state['cluster_norm']=None
    dev=weights['qkv_gate'].device
    if torch.cuda.get_device_capability(dev)!=(9,0):return state
    attn,ffn=state['attn_boundary'],state['ffn_boundary']
    ac,fc=_cn_cache(attn,False),_cn_cache(ffn,True)
    if ac is None or fc is None or state['sliced_ffn_boundary'] is None:
        print('CN_DISABLED unsupported boundary',flush=True);return state
    ext=_cn_load(name='decode_cluster_norm_'+_CN_SHA[:16],cpp_sources=_CN_CPP,cuda_sources=_CN_CUDA,
        functions=['cn_info','cn_launch'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo'],verbose=False)
    resources=[list(ext.cn_info(kind)) for kind in range(4)]
    for kind,r in enumerate(resources):
        if not (len(r)==9 and r[0]==kind and 0<r[1]<=64 and r[2]==0 and r[3]==72
                and r[4]>=32 and r[5]>=132 and r[6:]==[1,4,256]):
            raise RuntimeError('cluster norm native resource admission failed:'+str(r))
    print('CN_RESOURCES '+_cn_json.dumps(resources),flush=True)
    generator=torch.Generator(device=dev).manual_seed(20261029)
    paired=[];probes=0;replays=0
    for m in (1,3,7,8,9,15,16,17,31,32):
        residual=torch.randn((m,HIDDEN),device=dev,generator=generator,dtype=torch.float32)
        branch=torch.randn((m,HIDDEN),device=dev,generator=generator).bfloat16()
        sliced=torch.randn((m,UNPOOL_SLICE),device=dev,generator=generator).bfloat16() if m<=8 else branch
        definitions=[(0 if m<=8 else 1,branch,ac,
            lambda:attn.forward_jit(residual,branch,branch_dtype=torch.bfloat16,router_dtype=torch.float32 if m<=8 else None)),
            (2 if m<=8 else 3,sliced,fc,
             lambda:(_cn_old_sfn(residual,sliced,state)+(None,)) if m<=8 else
                    ffn.forward_jit(residual,branch,branch_dtype=torch.bfloat16))]
        for kind,b,cache,old in definitions:
            new=lambda:_cn_call(ext,kind,residual,b,cache)
            errors=_cn_error(new(),old());probes+=1
            if m in (1,8,16,32):
                timing=_cn_paired(old,new,dev,20261029+m*10+kind)
                timing.update(m=m,kind=kind,errors=errors);paired.append(timing)
            if m in (3,17):
                stream=torch.cuda.Stream(device=dev);stream.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(stream):
                    new();stream.synchronize();graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph,stream=stream):captured=new()
                    for repeat in range(2):
                        b.copy_(torch.randn(b.shape,device=dev,generator=generator).bfloat16())
                        residual.copy_(torch.randn(residual.shape,device=dev,generator=generator))
                        graph.replay();stream.synchronize();_cn_error(captured,old());replays+=1
    # One predeclared decision for the whole M1..32 family. Do not retain
    # individual cases from a failed family or treat baseline-only timing as gain.
    selected=True;combined=[]
    for m in (1,8,16,32):
        rows=[r for r in paired if r['m']==m];assert len(rows)==2
        old_ns=sum(r['old_ns'] for r in rows);new_ns=sum(r['new_ns'] for r in rows)
        combined.append(dict(m=m,old_ns=old_ns,new_ns=new_ns,net_saving_ns=old_ns-new_ns,new_over_old=new_ns/old_ns))
        selected &= new_ns<=old_ns/1.03 and old_ns-new_ns>=1000 and all(r['new_over_old']<=1.02 for r in rows)
    print('CN_PROBES '+_cn_json.dumps(dict(cases=probes,changed_input_graph_replays=replays)),flush=True)
    print('CN_PAIRS '+_cn_json.dumps(paired),flush=True)
    print('CN_DECISION '+_cn_json.dumps(dict(selected=bool(selected),combined=combined,statistical_significance_claim=False)),flush=True)
    if selected:
        state['cluster_norm']={'ext':ext,'ffn':fc}
        state['attn_boundary']=_CNBoundary(attn,ext,ac,False)
        state['ffn_boundary']=_CNBoundary(ffn,ext,fc,True)
    return state
