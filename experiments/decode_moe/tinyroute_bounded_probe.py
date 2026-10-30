"""Human-approved MoE boundary limits; all expert arithmetic stays retained."""


def _trb_relative(x,y,floor=1e-30):
    if x.numel()==0:return 0.
    xf,yf=x.double().flatten(),y.double().flatten()
    return float(torch.linalg.vector_norm(xf-yf)/torch.linalg.vector_norm(xf).clamp_min(floor))


def _trb_gate(reference,candidate):
    report={};ok=True
    for key in ('inter','ybuf','ss','out'):
        a,b=reference[key],candidate[key]
        same_shape=a.shape==b.shape and a.dtype==b.dtype
        finite=same_shape and bool(torch.isfinite(a).all())and bool(torch.isfinite(b).all())
        report[key]=dict(shape_dtype=bool(same_shape),finite=bool(finite))
        if not finite:ok=False;continue
        if key in ('inter','ybuf'):
            exact=bool(torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))
            report[key]['byte_equal']=exact;ok=ok and exact
        else:
            relative=_trb_relative(a,b,1e-30 if key=='ss'else 1e-12)
            report[key]['relative_L2']=relative
            ok=ok and relative<=(1e-5 if key=='ss'else .001)
            if key=='out':
                if a.numel():
                    ar,br=a.double().reshape(a.shape[0],-1),b.double().reshape(b.shape[0],-1)
                    norms=torch.linalg.vector_norm(ar,dim=1)
                    denominator=torch.maximum(norms,.1*norms.mean()).clamp_min(1e-12)
                    worst=float((torch.linalg.vector_norm(ar-br,dim=1)/denominator).max())
                else:worst=0.
                report[key]['worst_row']=worst;ok=ok and worst<=.005
    return bool(ok),report


def _tr_probe(data,parent_state,candidate_state):
    _tr_mutate(data,parent_state,'one',0)
    graphs=[]
    for state in (parent_state,candidate_state):
        _tr_launch(data,state);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=_tr_launch(data,state)
        graphs.append((graph,out,state))
    maximum=dict(ss=0.,out=0.,worst_row=0.)
    for pattern in ('none','one','four','all','skew'):
        for seed in range(3):
            _tr_mutate(data,parent_state,pattern,seed)
            # Untimed diagnostic only. No routing read is added to kernel_fn.
            ids=data['topk_ids'];lo=parent_state['lo'];cnt=parent_state['cnt']
            n=int(((ids>=lo)&(ids<lo+cnt)).sum())
            snapshots=[]
            for graph,out,state in graphs:
                before=int(state['scratch'][0][3].item())
                graph.replay();torch.cuda.synchronize()
                snap=_trb_snapshot(state,out,n)
                if snap['generation']!=before+1:
                    print('TR8B_GENERATION_FAIL',data['hidden_states'].shape[0],pattern,seed,flush=True)
                    return False
                snapshots.append(snap)
            passed,report=_trb_gate(*snapshots)
            if pattern=='none':
                passed=passed and all(bool((s['out']==0).all())for s in snapshots)
            if not passed:
                print('TR8B_NUMERICAL_FAIL',data['hidden_states'].shape[0],pattern,seed,report,flush=True)
                return False
            maximum['ss']=max(maximum['ss'],report['ss']['relative_L2'])
            maximum['out']=max(maximum['out'],report['out']['relative_L2'])
            maximum['worst_row']=max(maximum['worst_row'],report['out']['worst_row'])
    print('TR8B_NUMERICAL_PASS',data['hidden_states'].shape[0],maximum,
          'inter/ybuf=bytes,ss<=1e-5,out<=.001,worst_row<=.005',flush=True)
    return True
