VARIANTS2=('FS','NOB','NOFLUSH','NOA','NOACC','TL')
TRACE_STRIDE=160
_FS2_LOG_BYTES=0
_FS2_LOG_LIMIT=False


def log2(text,flush=True):
    """Hard cap diagnostic-owned output below10KB, with an explicit overflow marker."""
    global _FS2_LOG_BYTES,_FS2_LOG_LIMIT
    if _FS2_LOG_LIMIT:return
    size=len((text+'\n').encode())
    if _FS2_LOG_BYTES+size>9000:
        print('FS2_ERR log_budget_reached',flush=True)
        _FS2_LOG_BYTES+=len('FS2_ERR log_budget_reached\n')
        _FS2_LOG_LIMIT=True
        return
    print(text,flush=flush)
    _FS2_LOG_BYTES+=size


def compiler_errors(exc):
    lines=[s.strip() for s in str(exc).splitlines() if 'error' in s.lower()]
    # Keep diagnostics, not Torch's leading "Error building extension" wrapper.
    actual=[s for s in lines if not s.lower().startswith('error building extension')]
    return ' | '.join(actual or lines)[:600] or 'no compiler error lines; '+str(exc)[:180]


def error(where,exc):
    log2(f'FS2_ERR {where} {type(exc).__name__}: {compiler_errors(exc)}',flush=True)


def bundle_sources2(ns):
    units=[];decls=[];binds=[]
    info='''
std::vector<int64_t> fs_base_info(){cudaFuncAttributes a{};
cudaFuncGetAttributes(&a,okc::moe_fused<false>);
return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)okc_smem_bytes()};}
'''
    for k in ('TINY','TINY2'):
        defs=tuple(ns[k+'_DEFINES'])+('-DOKC_PRMTSC=1','-DOKC_DQ7=1')
        code='\n'.join('#define '+d[2:].replace('=',' ',1) for d in defs)+'\n'+ns['CUDA_SRC']+info
        decl=ns['CPP_SRC']+'\nstd::vector<int64_t> fs_base_info();\n'
        stem='fs2_'+k.lower()
        code=code.replace('namespace okc {','namespace '+stem+' {').replace('okc::',stem+'::')
        for old,new in [('okc_run',stem+'_run'),('okc_set_pdl',stem+'_pdl'),('fs_base_info',stem+'_info')]:
            code=code.replace(old+'(',new+'(');decl=decl.replace(old+'(',new+'(')
        decls.append(decl)
        for suffix in ('run','info'):binds.append(f'm.def("{k}_{suffix}", &{stem}_{suffix});')
        units.append((k,code))
    for v in VARIANTS2:
        # FS is byte-for-byte original device source. Instrument only other TUs.
        code=ns['FS_CUDA'] if v=='FS' else diagnostic_source(ns['FS_CUDA'])
        if v!='FS':code='#define FS_'+v+' 1\n'+code
        stem='fs2_'+v.lower()
        code=code.replace('namespace fs {','namespace '+stem+' {').replace('fs::',stem+'::')
        decl=ns['FS_CPP']
        for old in ('fs_run','fs_info','fs_set_pdl','okc_set_pdl'):
            code=code.replace(old+'(',stem+'_'+old+'(');decl=decl.replace(old+'(',stem+'_'+old+'(')
        decls.append(decl)
        for short in ('run','info'):binds.append(f'm.def("{v}_{short}", &{stem}_fs_{short});')
        units.append((v,code))
    cpp='#include <torch/extension.h>\n'+''.join(d.replace('#include <torch/extension.h>','') for d in decls)
    cpp+='\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){\n'+'\n'.join(binds)+'\n}\n'
    thin='#include <ATen/ATen.h>\n#include <vector>\nnamespace torch {using Tensor=at::Tensor;}'
    return cpp,[(k,c.replace('#include <torch/extension.h>',thin)) for k,c in units]


def build_bundle2(ns):
    from pathlib import Path
    from torch.utils.cpp_extension import load,_get_build_directory
    cpp,units=bundle_sources2(ns)
    tag=hashlib.sha256((cpp+''.join(c for _,c in units)).encode()).hexdigest()[:16]
    name='moe_fsdiag2_'+tag;directory=Path(_get_build_directory(name,verbose=False))
    files=[]
    for name,code in [('binding.cpp',cpp)]+[(k+'.cu',c) for k,c in units]:
        p=directory/name;p.write_text(code);files.append(str(p))
    # Deliberately no -std override: torch2.14's ATen requires C++20.
    return load(name='moe_fsdiag2_'+tag,sources=files,build_directory=str(directory),
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','-DOKC_NS=3','-DOKC_NSF=3'],verbose=False)


def timeline_summary2(trace):
    """Input [CTA,warp9,160]. Aggregate consumer observations, not producer exit."""
    z=trace.cpu().tolist();nw=8
    origin=min(w[0] for c in z for w in c[:nw] if w[0])
    def stats(vals):
        return '/'.join(f'{v:.2f}' for v in (min(vals),statistics.median(vals),max(vals))) if vals else 'NA'
    def col(c,i,earliest=False):
        a=[w[i] for w in c[:nw] if w[i]]
        return (min(a) if earliest else max(a)) if a else None
    def stamp(i,early=False):return stats([(v-origin)/1000 for c in z if (v:=col(c,i,early)) is not None])
    counts=[int(c[0][7]) for c in z]
    assert max(counts)<=64,'trace task overflow'
    assert max(int(c[0][10]) for c in z)<=8,'trace segment overflow'
    result=dict(tasks=stats(counts),entry=stamp(0,True),route=stamp(1),Wfull=stamp(2,True),
                flush0=stamp(144,True),flushN=stamp(4),barout=stamp(5),exit=stamp(6))
    segments=[]
    for s in range(max(int(c[0][10]) for c in z)):
        active=[c for c in z if int(c[0][10])>s]
        segments.append(f'{s}:'+stats([(col(c,144+2*s,True)-origin)/1000 for c in active])+','+
                        stats([(col(c,145+2*s)-origin)/1000 for c in active]))
    result['segments']=';'.join(segments)
    result['tail_active']=stats([(col(c,6)-col(c,5))/1000 for c in z if col(c,6)-col(c,5)>500])
    # Exact CTA task range metadata is validated, not dumped (compact logs).
    assert all(int(c[0][9])-int(c[0][8])==counts[i] for i,c in enumerate(z))
    task=[]
    for q in range(max(counts)):
        valid=[c for i,c in enumerate(z) if q<counts[i]]
        a=[col(c,16+2*q) for c in valid];b=[col(c,17+2*q) for c in valid]
        assert all(x is not None for x in a+b),'missing task stamps'
        task.append(f'{q}:'+stats([(x-origin)/1000 for x in a])+','+stats([(x-origin)/1000 for x in b]))
    # Per-CTA summed B interval includes W2 wait, MMA, accumulator and slot release.
    bsum=[];fsum=[];barwait=[]
    for c,n in zip(z,counts):
        bsum.append(sum(col(c,17+2*q)-col(c,16+2*q) for q in range(n))/1000)
        fsum.append(sum(col(c,145+2*s)-col(c,144+2*s,True) for s in range(int(c[0][10])))/1000)
        if col(c,4):barwait.append((col(c,5)-col(c,4))/1000)
    result.update(Bsum=stats(bsum),flushsum=stats(fsum),flush_to_bar=stats(barwait))
    return result,' '.join(task)


def run_all():
    global _FS2_LOG_BYTES,_FS2_LOG_LIMIT
    _FS2_LOG_BYTES=0;_FS2_LOG_LIMIT=False
    started=time.monotonic();dev=torch.device('cuda')
    if torch.cuda.get_device_capability(dev)!=(9,0):raise RuntimeError('Hopper required')
    raw=zlib.decompress(base64.b85decode(CANDIDATE_B85));assert hashlib.sha256(raw).hexdigest()==CANDIDATE_SHA
    ns={'__name__':'_fsdiag2_candidate'};exec(compile(raw,'<pinned FS1>','exec'),ns)
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
    log2(f'FS2_INFO device={torch.cuda.get_device_name(dev)!r} reps=40 flush_MiB=256 FS={CANDIDATE_SHA[:12]} TUs=8',flush=True)
    log2('FS2_NOTE NOB=no_B_math/acc/flush; NOFLUSH=red_add_to_volatile_shared_sink; NOA=no_A_math_volatile_x/weight_loads; NOACC=shared_y_add_to_register_checksum_keep_upper/MMA; all_keep_TMA/clear/barrier/finalizer; ablations_not_correct_outputs',flush=True)
    ready=queue.Queue()
    def worker():
        t=time.monotonic()
        try:ready.put((build_bundle2(ns),None,time.monotonic()-t))
        except Exception as e:ready.put((None,e,time.monotonic()-t))
    threading.Thread(target=worker,daemon=True).start()
    weights=generate_weights(dev);ns['_ext']=lambda d=None:None;ns['_wgs_on']=lambda:False
    state=ns['_moe_sel_original_prepare'](weights)
    bundle,exc,secs=ready.get(timeout=max(.001,235-(time.monotonic()-started)))
    if exc is not None:raise exc
    log2(f'FS2_BUILD seconds={secs:.2f}',flush=True)
    for v in ('TINY','TINY2')+VARIANTS2:log2(f'FS2_RES {v} {getattr(bundle,v+"_info")()}',flush=True)
    ns['_fs_ext']=lambda:types.SimpleNamespace(fs_run=bundle.FS_run)
    fs=ns['_fs_state'](weights,state);del weights
    states={v:dict(fs,ext=types.SimpleNamespace(fs_run=getattr(bundle,v+'_run')),
        scratch=[torch.zeros_like(s) for s in fs['scratch']]) for v in VARIANTS2}
    trace=torch.zeros((state['grid'],9,TRACE_STRIDE),dtype=torch.int64,device=dev)
    states['TL']['scratch'].append(trace)
    entries=[('R1',generate_input(1,'normal',2826763372427500438,dev)),
             ('R8',generate_input(8,'normal',2827019192515550467,dev))]
    refs={};calls={}
    for name,data in entries:
        k='TINY' if name=='R1' else 'TINY2'
        def base(data=data,k=k):
            out=torch.empty_like(data['hidden_states'])
            getattr(bundle,k+'_run')(state['w'],state['scratch'],data['hidden_states'],data['topk_ids'],data['topk_weights'],out,state['lo'],state['cnt'],state['grid'])
            return out
        calls[name,'BASE']=base;refs[name]=base().clone()
        counts=torch.bincount(data['topk_ids'][data['topk_ids']<32].flatten(),minlength=32).cpu().tolist()
        log2(f'FS2_CASE {name} cnt='+','.join(f'{e}:{n}' for e,n in enumerate(counts) if n),flush=True)
        for v in VARIANTS2:
            calls[name,v]=lambda d=data,v=v:ns['_fs_run'](d,state,states[v])
        for v in ('FS','TL'):
            m=ns['_fs_metrics'](calls[name,v](),refs[name])
            log2(f'FS2_CHECK {name} {v} L2={m["rel_L2"]:.3g} row={m["row_L2"]:.3g} ok={int(m["ok"])}',flush=True)
            if not m['ok']:raise RuntimeError('FS/TL correctness failed')
    flush=torch.zeros(256<<20,dtype=torch.uint8,device=dev);times={}
    log2('FS2_TL_NOTE us=min/med/max_over_CTAs_from_first_entry; task=q:A_done,B_done; Wfull=consumer_observed_after_clear/x; flush_to_bar_includes_fence; TL_perturbs_no_new_CTA_barriers',flush=True)
    order=('BASE','FS','NOB','NOFLUSH','NOA','NOACC')
    for rep in (1,2):
        for name,data in (entries if rep==1 else entries[::-1]):
            for v in (order if rep==1 else order[::-1]):
                if time.monotonic()-started>=235:raise RuntimeError('budget exhausted before full measurement set')
                med,low=bench(calls[name,v],flush,n=40);times[name,v,rep]=med
                log2(f'FS2_TIME {name} p={rep} {v} med={med:.3f} min={low:.3f}',flush=True)
            trace.zero_();flush.zero_();out=calls[name,'TL']();torch.cuda.synchronize()
            m=ns['_fs_metrics'](out,refs[name]);summary,tasks=timeline_summary2(trace)
            log2(f'FS2_TL {name} p={rep} '+' '.join(f'{k}={v}' for k,v in summary.items())+f' L2={m["rel_L2"]:.2g}',flush=True)
            log2(f'FS2_TASK {name} p={rep} '+tasks,flush=True)
    for name,_ in entries:
        for v in ('BASE','NOB','NOFLUSH','NOA','NOACC'):
            log2(f'FS2_RATIO {name} {v}/FS='+ '/'.join(f'{times[name,v,r]/times[name,"FS",r]:.4f}' for r in (1,2)),flush=True)
    log2(f'FS2_DONE seconds={time.monotonic()-started:.2f}',flush=True)
