"""Only a prepared M32 sliding bundle can reach the private projection paths."""

_D32_EXT = None
_D32_ERROR = None


def _d32_build():
    global _D32_EXT, _D32_ERROR
    try:
        assert hashlib.sha256(_D32_CUDA.encode()).hexdigest() == _D32_CUDA_SHA
        _D32_EXT = load_inline('okc_d32_' + _D32_CUDA_SHA[:12], _D32_CPP,
            cuda_sources=_D32_CUDA, functions=_D32_FUNCTIONS,
            extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
    except Exception as e:
        _D32_ERROR = repr(e)[-1800:]
        print('D32_BUILD_ERROR', _D32_ERROR, flush=True)


_d32_thread = _okh_bg(_d32_build)
_d32_parent_prepare = prepare
_d32_parent_kernel_fn = kernel_fn
_d32_parent_project = _achita_dense_project


def _d32_find(d,key,depth=0):
    if not isinstance(d,dict) or depth>6:
        return None
    if key in d:
        return d[key]
    for value in d.values():
        got = _d32_find(value,key,depth+1)
        if got is not None:
            return got
    return None


def _d32_leaf(x,layer,state):
    plan = state['d32_plan'][layer]
    out = torch.empty((32,8192),dtype=torch.bfloat16,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if layer=='o':
        rc = _D32_EXT.d32o_launch(0,x.data_ptr(),plan['packed'].data_ptr(),out.data_ptr(),32,stream,0)
    else:
        rc = _D32_EXT.d32u_launch(0,x.data_ptr(),plan['w'].data_ptr(),plan['esc'].data_ptr(),
            out.data_ptr(),32,8192,4096,2,plan['table'],stream)
    if rc:
        raise RuntimeError('D32 private launch: '+str((layer,rc)))
    return out


def _achita_dense_project(x,layer,state,rows=None):
    if (rows is None and 'd32_plan' in state and layer in ('o','unpooler')
        and x.shape==(32,1024 if layer=='o' else 4096) and x.dtype==torch.bfloat16
        and x.is_contiguous() and x.data_ptr()%16==0 and _D32_EXT is not None):
        return _d32_leaf(x,layer,state)
    return _d32_parent_project(x,layer,state,rows)


def _d32_plan(state):
    if _D32_EXT is None:
        return None
    bundle = _d32_find(state['achita_dense'],'bundle2') or {}
    o = bundle.get('o',{})
    up = bundle.get('diet',{}).get('plans',{}).get((32,'unpooler'))
    if not (o.get('enabled') and (32,128,8,1,1,1) in o.get('configs',())
            and o.get('packed') is not None and up is not None):
        return None
    _,u = up
    if not (u.get('compressed') and u.get('pack_exact') and
        tuple(u.get(k) for k in ('n','k','r','wk','split'))==(8192,4096,64,2,2)):
        return None
    # A different parent override would invalidate the source-bound comparison.
    choices = state['achita_dense'].get('achita_wg_choices',{})
    if (32,'o',8192,1024) in choices or (32,'unpooler',8192,4096) in choices:
        return None
    rc = int(_D32_EXT.d32o_launch(0,0,0,0,32,0,1))
    oa = [int(_D32_EXT.d32o_attr(i)) for i in range(6)]
    ur = int(_D32_EXT.d32u_info(0)); uc = int(_D32_EXT.d32u_occupancy())
    print('D32_RESOURCES',rc,oa,ur,uc,flush=True)
    if not (rc>0 and rc%10000==0 and oa[0]>0 and oa[1]==0 and oa[2]==0
        and oa[3]==196736 and oa[4]>=128 and oa[5]>0 and ur>0 and ur%10000==0 and uc>0):
        return None
    return {'o':o,'unpooler':u}


def prepare(weights):
    start = _okh_clock.perf_counter()
    state = _d32_parent_prepare(weights)
    _d32_thread.join()
    state['d32_selected'] = False
    try:
        if _okh_budget(start,274.):
            plan = _d32_plan(state)
            if plan is not None:
                # Separate dictionaries; all underlying scratch remains sequential.
                candidate = dict(state,d32_plan=plan)
                dev = state['dense']['o'].device
                gen = torch.Generator(device=dev).manual_seed(2026103032)
                units = True
                for layer,k in (('o',1024),('unpooler',4096)):
                    if not _okh_budget(start,274.):
                        units = False
                        break
                    x = torch.randn((32,k),device=dev,generator=gen).to(torch.bfloat16)
                    def mutate(seed):
                        x.normal_(generator=gen)
                        if seed==1:x.zero_()
                        elif seed==2:x.mul_(16.)
                    arms = {'parent':lambda:(_d32_parent_project(x,layer,state),),
                            'pdl':lambda:(_d32_leaf(x,layer,candidate),)}
                    same = _pdli_probe(arms,mutate,label='D32_unit_'+layer)
                    units = units and same
                    print('D32_UNIT_BYTES',layer,same,flush=True)
                if units and _okh_budget(start,274.):
                    data = _d32_data(32,True,dev,gen)
                    def mutate_layer(seed):
                        data['attn_input'].normal_(generator=gen)
                        data['stream'].normal_(generator=gen)
                    arms = {'parent':lambda:_d32_parent_kernel_fn(data,state),
                            'pdl':lambda:_d32_parent_kernel_fn(data,candidate)}
                    if _pdli_ab('D32_M32_sliding',arms,mutate_layer,dev,start,
                                (data['k_cache'],data['v_cache'])):
                        state['d32_candidate'] = candidate
                        state['d32_selected'] = True
    except Exception as e:
        state['d32_selected'] = False
        print('D32_ERROR',repr(e)[-2200:],flush=True)
    torch.cuda.synchronize()
    print('D32_DONE',_okh_clock.perf_counter()-start,state['d32_selected'],flush=True)
    return state


def kernel_fn(data,state):
    cfg = data['config']
    if (state.get('d32_selected',False) and cfg['tokens']==32 and cfg['batch']==32
            and cfg['q_len']==1 and bool(cfg['window'])):
        return _d32_parent_kernel_fn(data,state['d32_candidate'])
    return _d32_parent_kernel_fn(data,state)
