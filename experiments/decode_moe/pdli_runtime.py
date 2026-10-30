"""Gated handoffs; every unselected regime calls the exact R2f parent."""

_PDLI_SOURCES, _PDLI_SPECS, _PDLI_PROOFS = pdli_make_sources()
assert {k:hashlib.sha256(v.encode()).hexdigest() for k,v in _PDLI_SOURCES.items()} == _PDLI_EXPECTED
_PDLI_EXT = {}
_PDLI_ERRORS = {}
_PDLI_CUDA, _PDLI_CPP, _PDLI_FUNCTIONS = pdli_pack(_PDLI_SOURCES,_PDLI_SPECS)
assert hashlib.sha256(_PDLI_CUDA.encode()).hexdigest() == _PDLI_COMBINED_HASH


class _PdliFns:
    def __init__(self,ext,name):
        self.pdli_project = self.r2_launch = getattr(ext,'pdli_'+name+'_project')
        self.pdli_info = getattr(ext,'pdli_'+name+'_info')


def _pdli_build():
    try:
        ext = load_inline('okc_pdli_pack_' + _PDLI_COMBINED_HASH[:12],_PDLI_CPP,
            cuda_sources=_PDLI_CUDA,functions=_PDLI_FUNCTIONS,
            extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True,with_pytorch_error_handling=False,verbose=False)
        for name in _PDLI_SOURCES:
            _PDLI_EXT[name] = _PdliFns(ext,name)
            print('PDLI_BUILD',name,_PDLI_EXPECTED[name],flush=True)
    except Exception as e:
        _PDLI_ERRORS['packed'] = repr(e)[-1500:]
        print('PDLI_BUILD_ERROR packed',_PDLI_ERRORS['packed'],flush=True)


def _pdli_build_attention():
    try:
        _native_attn__module()
    except Exception as e:
        _PDLI_ERRORS['attention'] = repr(e)[-1500:]
        print('PDLI_BUILD_ERROR attention',_PDLI_ERRORS['attention'],flush=True)


_pdli_build_thread = _okh_bg(_pdli_build)
_pdli_attention_thread = _okh_bg(_pdli_build_attention)
_PDLI_NORM_THREADS = []
_pdli_core_prepare = _achita_layer_prepare


def _pdli_warm_core(weights):
    state = _pdli_core_prepare(weights)
    def warm():
        try:
            dev = state['dense']['qkv_gate'].device
            with torch.cuda.device(dev),torch.cuda.stream(torch.cuda.Stream(device=dev)):
                for m in (1,8):
                    s = torch.zeros((m,8192),dtype=torch.float32,device=dev)
                    a = torch.zeros((m,8192),dtype=torch.bfloat16,device=dev)
                    state['attn_boundary'](s,a,branch_dtype=torch.bfloat16,router_dtype=torch.float32)
                torch.cuda.synchronize(dev)
            print('PDLI_NORM_WARM_DONE',flush=True)
        except Exception as e:
            _PDLI_ERRORS['norm_warm'] = repr(e)[-1000:]
            print('PDLI_NORM_WARM_ERROR',_PDLI_ERRORS['norm_warm'],flush=True)
    _PDLI_NORM_THREADS.append(_okh_bg(warm))
    return state


_achita_layer_prepare = _pdli_warm_core


def _pdli_rc(rc, what):
    if rc:
        raise RuntimeError('PDLI ' + what + ' rc=' + str(rc))


def _pdli_project(x, layer, state):
    m = x.shape[0]
    kind,n,k = {'qkv_gate':('q',2304,8192),'o':('o',8192,1024),'unpooler':('u',1024,4096)}[layer]
    name = kind + str(m)
    pp = state['pdli']['plans'][name]
    if x.dtype != torch.bfloat16 or not x.is_contiguous() or x.data_ptr()%16 or x.shape[1]!=k:
        return _achita_dense_project(x,layer,state,n)
    out = torch.empty((m,n),dtype=torch.bfloat16,device=x.device)
    wr,wk,ku,s = pp['cfg']
    _pdli_rc(_PDLI_EXT[name].pdli_project(pp['wp'].data_ptr(),x.data_ptr(),out.data_ptr(),pp['esc'].data_ptr(),
        m,n,k,wr,wk,ku,s,pp['t01'],pp['u01'],torch.cuda.current_stream().cuda_stream),name)
    return out


def _pdli_route(xb,xr,state):
    m = xb.shape[0]
    if (xb.dtype!=torch.bfloat16 or not xb.is_contiguous() or xb.data_ptr()%16 or xb.shape[1]!=8192
        or xr is None or xr.dtype!=torch.float32 or tuple(xr.shape)!=(m,8192) or not xr.is_contiguous()):
        return _r2_pool_route(xb,xr,state)
    pp = state['pdli']['plans']['r'+str(m)]
    rf, scratch = state['rf'],state['pdli']['routing_scratch']
    out = torch.empty((m,4096),dtype=torch.bfloat16,device=xb.device)
    ow = torch.empty((m,8),dtype=torch.float32,device=xb.device)
    oi = torch.empty((m,8),dtype=torch.int32,device=xb.device)
    wr,wk,ku,s = pp['cfg']
    _pdli_rc(_PDLI_EXT['r'+str(m)].r2_launch(pp['wp'].data_ptr(),xb.data_ptr(),out.data_ptr(),pp['esc'].data_ptr(),
        m,4096,8192,wr,wk,ku,s,pp['t01'],pp['u01'],torch.cuda.current_stream().cuda_stream,
        xr.data_ptr(),rf['weight'].data_ptr(),rf['bias'].data_ptr(),scratch['parts'].data_ptr(),
        scratch['counter'].data_ptr(),ow.data_ptr(),oi.data_ptr(),0),'R2f')
    return out,ow,oi


def _pdli_attention(data, on):
    if on and _native_attn__supported(data['config']) and _native_attn__fits(data):
        return _native_attn__launch(data,pdl=True)
    return _native_attn_kernel_fn(data)


def _pdli_equal(a,b):
    if a is None or b is None or not _okh_same(a,b):
        return False
    if not all(bool(torch.isfinite(x).all()) for x in tuple(a)+tuple(b)):
        return False
    return _okh_same(tuple(x.contiguous().view(torch.uint8) for x in a),
                     tuple(x.contiguous().view(torch.uint8) for x in b))


def _pdli_probe(fns, mutate, cache=None, label='probe'):
    captures = {}
    for lab,fn in fns.items():
        fn()
        torch.cuda.synchronize()
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = fn()
        captures[lab] = (gr,out)
    ok = True
    for seed in range(3):
        mutate(seed)
        ref = None
        for lab,(gr,out) in captures.items():
            gr.replay()
            torch.cuda.synchronize()
            snap = tuple(x.clone() for x in out)
            if cache is not None:
                # Byte views avoid unsupported FP8 equality and include untouched cache rows.
                snap += tuple(x.view(torch.uint8).clone() for x in cache)
            if lab=='parent':
                ref = snap
            else:
                same = _pdli_equal(ref,snap)
                if not same:
                    diffs = []
                    for index,(a,b) in enumerate(zip(ref,snap)):
                        if a.dtype!=b.dtype or a.shape!=b.shape:
                            diffs.append((index,'shape/dtype'))
                        else:
                            count = int((a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8)).sum())
                            if count:diffs.append((index,str(a.dtype),tuple(a.shape),count))
                    print('PDLI_DIFF',label,'seed=',seed,'byte_mismatches=',diffs,flush=True)
                ok = ok and same
    del captures
    return ok


def _pdli_ab(label,fns,mutate,dev,t0,cache=None):
    if not _okh_budget(t0,278.0):
        print('PDLI_BUDGET_SKIP',label,flush=True)
        return False
    ok = _pdli_probe(fns,mutate,cache,label)
    print('PDLI_BITWISE',label,ok,'changing_graph_inputs=3',flush=True)
    if not ok:
        return False
    passes = []
    for order in (('parent','pdl'),('pdl','parent')):
        if not _okh_budget(t0,278.0):
            return False
        med = _okh_time({lab:fns[lab] for lab in order},dev,reps=25)
        pick = _okh_pick(med,'parent',.97)
        passes.append(pick=='pdl')
        print('PDLI_TIME',label,'order='+','.join(order),
              'parent_us=%.6f pdl_us=%.6f saving_us=%.6f ratio=%.6f pick=%s' %
              (med['parent'],med['pdl'],med['parent']-med['pdl'],med['pdl']/med['parent'],pick),flush=True)
    return all(passes)


def _pdli_data(m,sliding,dev,g,ctx=80000):
    # Fixed-capacity representative probes, not judge inputs or data-derived dispatch.
    pages = ctx+m
    k = torch.randn((pages,1,2,64),device=dev,generator=g).to(torch.float8_e4m3fn)
    v = torch.randn((pages,1,2,64),device=dev,generator=g).to(torch.float8_e4m3fn)
    seq = torch.full((1,),ctx,dtype=torch.int32,device=dev)
    return {'attn_input':torch.randn((m,8192),device=dev,generator=g).to(torch.bfloat16),
            'stream':torch.randn((m,8192),device=dev,generator=g),
            'seq_lens':seq,'cache_loc':torch.arange(ctx,ctx+m,dtype=torch.int64,device=dev),
            'page_table':torch.arange(pages,dtype=torch.int32,device=dev).reshape(1,-1),
            'k_cache':k,'v_cache':v,'k_scale':torch.ones((),device=dev),'v_scale':torch.ones((),device=dev),
            'local_start':torch.zeros((1,),dtype=torch.int32,device=dev),'scheduler_metadata':None,
            'config':{'batch':1,'q_len':m,'tokens':m,'num_q_heads':16,'num_kv_heads':2,'head_dim':64,
                      'page_size':1,'window':2048 if sliding else 0,'rope_theta':ROPE_THETA if sliding else 0.,
                      'qk_scale_factor':5.4730064863875,'sm_scale':0.125,'kv_scale':1.,'n_experts':256}}


def _pdli_prepare_plans(state):
    dense = state['achita_dense']
    v18 = _r2_find(dense,'v18') or {}
    bundle = _r2_find(dense,'bundle') or {}
    sp1 = _r2_find(dense,'v12_smtrace1') or {}
    sp8 = _r2_find(dense,'v13_smtrace8') or {}
    p = {
        'o1':(_r2_find(dense,'v12_o1') or {}).get('plans',{}).get(('o',1)),
        'o8':v18.get('plans',{}).get(('o',8)),
        'q1':(v18.get('qkv') or {}).get('plans',{}).get(('qkv_gate','mt',1)),
        'q8':bundle.get('qkv'),
        'u1':sp1.get('plans',{}).get(('unpooler','m',1)),
        'u8':sp8.get('plans',{}).get(('unpooler','m',8)),
    }
    for m in (1,8):
        p['r'+str(m)] = state['r2'][m][1] if m in state.get('r2sel',()) else None
    out = {}
    for name,pp in p.items():
        if pp is None or name not in _PDLI_EXT:
            print('PDLI_PLAN_SKIP',name,'absent',flush=True)
            continue
        inst = _PDLI_SPECS[name]['instance']
        # Bundle QKV is a packed stream with the same fixed geometry, without a cfg entry.
        cfg = tuple(pp.get('cfg',(1,4,8,4)))
        if pp.get('fmt')!=inst[0] or cfg!=tuple(inst[10:14]):
            print('PDLI_PLAN_SKIP',name,'format/config mismatch',flush=True)
            continue
        regs = _PDLI_EXT[name].pdli_info(int(name[1:]),2)
        occ = _PDLI_EXT[name].pdli_info(int(name[1:]),1)
        if regs<0 or occ<1:
            print('PDLI_PLAN_SKIP',name,'resource query',regs,occ,flush=True)
            continue
        out[name] = dict(pp,cfg=cfg)
        print('PDLI_RESOURCE',name,'regs_local=',regs,'occupancy=',occ,'source=',_PDLI_EXPECTED[name],flush=True)
    state['pdli']['plans'] = out
    state['pdli']['routing_scratch'] = {'parts':torch.empty_like(state['rf']['parts']),
                                       'counter':torch.zeros((1,),dtype=torch.int32,device=state['rf']['weight'].device)}


def _pdli_select(state,t0):
    _pdli_prepare_plans(state)
    dev = state['rf']['weight'].device
    plans = state['pdli']['plans']
    for m in (1,8):
        if not _okh_budget(t0,274.0):
            print('PDLI_BUDGET_SKIP M=',m,flush=True)
            break
        gen = torch.Generator(device=dev).manual_seed(72100+m)
        # Real norm -> exact router/pooler pair, without any synthetic spin.
        ao = torch.randn((m,8192),device=dev,generator=gen).to(torch.bfloat16)
        stream = torch.randn((m,8192),device=dev,generator=gen)
        def rfn(on):
            xb,rs,xr = state['attn_boundary'](stream,ao,branch_dtype=torch.bfloat16,router_dtype=torch.float32)
            r = _pdli_route(xb,xr,state) if on else _r2_pool_route(xb,xr,state)
            return (xb,rs,xr)+tuple(r)
        def rmut(seed):
            ao.normal_(generator=gen);stream.normal_(generator=gen)
        rok = False
        if 'r'+str(m) in plans:
            rok = _pdli_ab('norm_r2f_M'+str(m),{'parent':lambda:rfn(False),'pdl':lambda:rfn(True)},rmut,dev,t0)

        # One local expert per token for timing, plus all-nonlocal and dense-local correctness replays.
        xx = torch.randn((m,4096),device=dev,generator=gen).to(torch.bfloat16)
        ids = torch.full((m,8),-1,dtype=torch.int32,device=dev)
        ids[:,0] = 0
        tw = torch.full((m,8),.125,dtype=torch.float32,device=dev)
        def ufn(on):
            e = _experts(xx,tw,ids,state)
            u = _pdli_project(e,'unpooler',state) if on else _achita_dense_project(e,'unpooler',state,1024)
            return (e,u)
        def umut(seed):
            xx.normal_(generator=gen)
            ids.fill_(-1)
            if seed!=1:
                ids[:,0] = 0
            if seed==2:
                ids.copy_(torch.arange(8,dtype=torch.int32,device=dev).expand(m,8))
        uok = False
        if 'u'+str(m) in plans:
            # Restore the representative sparse-local routes after the adversarial replay probes.
            def uparent():return ufn(False)
            def updl():return ufn(True)
            if _okh_budget(t0,278.0):
                ok = _pdli_probe({'parent':uparent,'pdl':updl},umut,label='moe_unpool_M'+str(m))
                print('PDLI_BITWISE moe_unpool_M'+str(m),ok,'changing_routes=one,none,eight',flush=True)
                ids.fill_(-1);ids[:,0] = 0
                passes = []
                if ok:
                    for order in (('parent','pdl'),('pdl','parent')):
                        if not _okh_budget(t0,278.0):break
                        arms = {'parent':uparent,'pdl':updl}
                        med = _okh_time({lab:arms[lab] for lab in order},dev,reps=25)
                        pick = _okh_pick(med,'parent',.97)
                        passes.append(pick=='pdl')
                        print('PDLI_TIME moe_unpool_M'+str(m),'order='+','.join(order),
                              'parent_us=%.6f pdl_us=%.6f saving_us=%.6f ratio=%.6f pick=%s' %
                              (med['parent'],med['pdl'],med['parent']-med['pdl'],med['pdl']/med['parent'],pick),flush=True)
                uok = len(passes)==2 and all(passes)
        for sliding in (True,False):
            if not _okh_budget(t0,276.0):break
            data = _pdli_data(m,sliding,dev,gen)
            tag = 'M'+str(m)+'_'+('sliding' if sliding else 'full')
            cache = (data['k_cache'],data['v_cache'])
            def mutate(seed):
                data['attn_input'].normal_(generator=gen);data['stream'].normal_(generator=gen)
            def qa(on):
                q = _pdli_project(data['attn_input'],'qkv_gate',state) if on else _achita_dense_project(data['attn_input'],'qkv_gate',state)
                a = _pdli_attention(dict(data,qkv=q.reshape(1,m,-1)),on)
                return (q,a)
            qaok = False
            if 'q'+str(m) in plans and 'attention' not in _PDLI_ERRORS:
                qaok = _pdli_ab('qkv_attention_'+tag,{'parent':lambda:qa(False),'pdl':lambda:qa(True)},mutate,dev,t0,cache)
            qfixed = _achita_dense_project(data['attn_input'],'qkv_gate',state).reshape(1,m,-1)
            ad = dict(data,qkv=qfixed)
            def ofn(on):
                a = _native_attn_kernel_fn(ad).reshape(m,1024)
                o = _pdli_project(a,'o',state) if on else _achita_dense_project(a,'o',state)
                return (a,o)
            def omut(seed):qfixed.normal_(generator=gen)
            ook = False
            if 'o'+str(m) in plans:
                ook = _pdli_ab('attention_o_'+tag,{'parent':lambda:ofn(False),'pdl':lambda:ofn(True)},omut,dev,t0,cache)
            mask = int(qaok) | (int(ook)<<1) | (int(rok)<<2) | (int(uok)<<3)
            print('PDLI_COMPONENTS',tag,'qa=',qaok,'o=',ook,'r2f=',rok,'unpool=',uok,'candidate_mask=',mask,flush=True)
            if mask and _okh_budget(t0,278.0):
                ok = _pdli_ab('whole_layer_'+tag,{'parent':lambda:_pdli_parent_kernel_fn(data,state),
                    'pdl':lambda:_pdli_layer(data,state,mask)},mutate,dev,t0,cache)
                if ok:
                    state['pdli']['selected'][m,bool(sliding)] = mask
            print('PDLI_SELECT',tag,'mask=',state['pdli']['selected'].get((m,bool(sliding)),0),flush=True)


_pdli_parent_prepare = prepare
_pdli_parent_kernel_fn = kernel_fn


def prepare(weights):
    t0 = _okh_clock.perf_counter()
    state = _pdli_parent_prepare(weights)
    state['pdli'] = {'plans':{},'selected':{}}
    threads = [_pdli_build_thread,_pdli_attention_thread]+_PDLI_NORM_THREADS
    for thread in threads:
        thread.join(max(0.,274.-(_okh_clock.perf_counter()-t0)))
    if (any(t.is_alive() for t in threads)
        or not _okh_budget(t0,274.) or state.get('rf') is None):
        print('PDLI_SKIP_PREPARE',_okh_clock.perf_counter()-t0,'built=',sorted(_PDLI_EXT),flush=True)
        return state
    try:
        _pdli_select(state,t0)
    except Exception as e:
        state['pdli']['selected'] = {}
        print('PDLI_ERROR',repr(e)[-1500:],flush=True)
    torch.cuda.synchronize()
    print('PDLI_DONE elapsed_s=',_okh_clock.perf_counter()-t0,'selected=',state['pdli']['selected'],flush=True)
    return state


def kernel_fn(data,state):
    cfg = data['config']
    m = cfg['tokens']
    mask = state.get('pdli',{}).get('selected',{}).get((m,bool(cfg['window'])),0)
    if mask&1 and (cfg['batch']!=1 or cfg['q_len'] not in (1,8)):
        mask &= ~1
    if not mask:
        return _pdli_parent_kernel_fn(data,state)
    return _pdli_layer(data,state,mask)
