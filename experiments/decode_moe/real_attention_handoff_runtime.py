"""Fixed real-producer diagnostic; public Dense kernel/state remain retained."""

_EAC_NS={'torch':torch,'TypedDict':TypedDict,'math':math}
exec(compile(_EAC_NATIVE_PY,'frozen-native-attention','exec'),_EAC_NS)
_EAC_PARENT_EXT=None
_EAC_EARLY_EXT=None
_EAC_O_EXT=None
_EAC_ERRORS={}


def _eac_build_parent():
    global _EAC_PARENT_EXT
    try:_EAC_PARENT_EXT=_EAC_NS['_native_attn__module']()
    except Exception as e:_EAC_ERRORS['parent']=repr(e)[-1800:]


def _eac_build_early():
    global _EAC_EARLY_EXT
    try:
        from types import SimpleNamespace
        assert hashlib.sha256(_EAC_EARLY_CUDA.encode()).hexdigest()==_EAC_EARLY_SHA
        module=load_inline('okc_eac_a_'+_EAC_EARLY_SHA[:12],_EAC_EARLY_CPP,
            cuda_sources=_EAC_EARLY_CUDA,functions=['eac_run','eac_info'],
            extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','--expt-relaxed-constexpr'],verbose=False)
        _EAC_EARLY_EXT=SimpleNamespace(run=module.eac_run,query=module.eac_info)
    except Exception as e:_EAC_ERRORS['early']=repr(e)[-1800:]


def _eac_build_o():
    global _EAC_O_EXT
    try:
        assert hashlib.sha256(_EAC_O_CUDA.encode()).hexdigest()==_EAC_O_SHA
        _EAC_O_EXT=load_inline('okc_eac_o_'+_EAC_O_SHA[:12],_EAC_O_CPP,
            cuda_sources=_EAC_O_CUDA,functions=['eac_o8','eac_o4','eac_oattr'],
            extra_cuda_cflags=['-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True,with_pytorch_error_handling=False,verbose=False)
    except Exception as e:_EAC_ERRORS['projection']=repr(e)[-1800:]


_eac_threads=[_okh_bg(_eac_build_parent),_okh_bg(_eac_build_early),_okh_bg(_eac_build_o)]
_eac_parent_prepare=prepare
_eac_parent_kernel_fn=kernel_fn


def _eac_find(d,key,depth=0):
    if not isinstance(d,dict)or depth>6:return None
    if key in d:return d[key]
    for value in d.values():
        found=_eac_find(value,key,depth+1)
        if found is not None:return found
    return None


def _eac_bytes(arms,mutate,cache):
    captured={};passed={key:True for key in arms if key!='parent'};mismatches=[]
    for key,fn in arms.items():
        fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn()
        captured[key]=(graph,out)
    for seed in range(3):
        mutate(seed);reference=None
        for key,(graph,out)in captured.items():
            graph.replay();torch.cuda.synchronize()
            snap=tuple(x.clone()for x in out)+tuple(x.view(torch.uint8).clone()for x in cache)
            if key=='parent':reference=snap
            else:
                same=_pdli_equal(reference,snap);passed[key]=passed[key]and same
                if not same:
                    fields=[]
                    for i,(a,b)in enumerate(zip(reference,snap)):
                        count=int((a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8)).sum())
                        if count:fields.append([i,count])
                    mismatches.append(dict(arm=key,seed=seed,byte_mismatch_fields=fields))
    del captured
    return passed,mismatches


def prepare(weights):
    start=_okh_clock.perf_counter();state=_eac_parent_prepare(weights)
    for thread in _eac_threads:thread.join()
    report=dict(enables_nothing=True,timed_path='unchanged Dense parent',errors=dict(_EAC_ERRORS),
                resources={},byte_pass={},orders=[],gates={},budget_stop=False)
    try:
        if not _okh_budget(start,274.):report['budget_stop']=True
        elif _EAC_PARENT_EXT is None or _EAC_EARLY_EXT is None or _EAC_O_EXT is None:
            raise RuntimeError('EAC required module unavailable')
        else:
            dev=weights['router'].device;prop=torch.cuda.get_device_properties(dev)
            if prop.major!=9 or prop.multi_processor_count!=132:raise RuntimeError('EAC requires recorded H200 geometry')
            bundle=_eac_find(state,'bundle2')or{};plan=bundle.get('o',{})
            if tuple(map(tuple,plan.get('configs',())))!=((32,128,8,1,1,1),)or plan.get('packed')is None:
                raise RuntimeError('EAC exact O32 parent plan missing')
            if (32,'o',8192,1024)in state.get('achita_wg_choices',{}):raise RuntimeError('EAC parent WG override')
            ready=[]
            for stage in (8,4):
                query=getattr(_EAC_O_EXT,'eac_o'+str(stage));rc=int(query(0,0,0,0,1))
                attrs=[int(_EAC_O_EXT.eac_oattr(stage,i))for i in range(5)]
                report['resources'][str(stage)]=dict(query=rc,attrs=attrs)
                if rc<=0 or rc%10000 or attrs[0]<=0 or attrs[1]or attrs[2]or attrs[3]!=stage*24592 or attrs[4]<=0:
                    raise RuntimeError('EAC projection resource rejection: '+str((stage,rc,attrs)))
            gen=torch.Generator(device=dev).manual_seed(2026103033)
            data=_eac_data(32,True,dev,gen)
            data['qkv']=torch.randn((32,1,2304),device=dev,generator=gen).to(torch.bfloat16)
            def run(mode):
                module=_EAC_EARLY_EXT if mode.startswith('early')else _EAC_PARENT_EXT
                attention=_EAC_NS['_native_attn__launch'](data,_mod=module).reshape(32,1024)
                if mode=='parent':
                    out=_eac_parent_kernel_fn(dict(x=attention,config=dict(m=32,n=8192,k=1024,layer='o')),state)
                else:
                    out=torch.empty((32,8192),dtype=torch.bfloat16,device=dev)
                    stage=4 if mode=='early4'else 8
                    rc=getattr(_EAC_O_EXT,'eac_o'+str(stage))(attention.data_ptr(),plan['packed'].data_ptr(),out.data_ptr(),
                        torch.cuda.current_stream().cuda_stream,0)
                    if rc:raise RuntimeError('EAC O launch: '+str(rc))
                return attention,out
            def mutate(seed):
                data['qkv'].normal_(generator=gen)
                if seed==1:data['qkv'].zero_()
                elif seed==2:data['qkv'].mul_(8.)
            arms={key:(lambda key=key:run(key))for key in ('parent','late8','early8','early4')}
            run('early8');torch.cuda.synchronize()
            attrs=[int(_EAC_EARLY_EXT.query(i))for i in range(5)]
            report['resources']['early_attention']=attrs
            if attrs[0]<=0 or attrs[1]or attrs[2]or attrs[3]!=58368 or attrs[4]<=0:
                raise RuntimeError('EAC native Attention resource rejection: '+str(attrs))
            probes={'parent':arms['parent'],'parent_repeat':arms['parent'],
                    **{k:v for k,v in arms.items()if k!='parent'}}
            report['byte_pass'],report['mismatches']=_eac_bytes(probes,mutate,(data['k_cache'],data['v_cache']))
            print('EAC_BYTES',report['byte_pass'],flush=True)
            valid={'parent':arms['parent']}
            if report['byte_pass']['parent_repeat']:
                valid.update({k:arms[k]for k in arms if k!='parent'and report['byte_pass'][k]})
            else:report['errors']['reference']='Unchanged parent is not byte-stable; no performance admission.'
            mutate(0)
            if len(valid)>1:
                for order in (tuple(valid),tuple(reversed(tuple(valid)))):
                    if not _okh_budget(start,278.):report['budget_stop']=True;break
                    times=_okh_time({key:valid[key]for key in order},dev,reps=25)
                    report['orders'].append(dict(order=list(order),times_us=times));print('EAC_TIMES',order,times,flush=True)
                def beats(candidate,controls):
                    return len(report['orders'])==2 and all(all(candidate in row['times_us']and control in row['times_us']
                        and row['times_us'][candidate]<.97*row['times_us'][control]
                        and row['times_us'][control]-row['times_us'][candidate]>=1.5 for control in controls)
                        for row in report['orders'])
                report['gates']={'late8_real_transfer':beats('late8',['parent']),
                    'early8_causal_signal':beats('early8',['parent','late8']),
                    'early4_additional_schedule':beats('early4',['parent','late8','early8'])}
            torch.cuda.synchronize()
    except Exception as e:
        report['errors']['diagnostic']=repr(e)[-2200:];print('EAC_ERROR',report['errors']['diagnostic'],flush=True)
    report['prepare_s']=_okh_clock.perf_counter()-start
    import base64,zlib,json
    payload=base64.b64encode(zlib.compress(json.dumps(report,allow_nan=False).encode(),9)).decode();assert len(payload)<9000
    print('EAC_SUMMARY_Z64',payload,flush=True);print('EAC_DONE',report['prepare_s'],'enables_nothing=True',flush=True)
    return state
