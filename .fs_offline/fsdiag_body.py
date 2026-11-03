"""FS1 diagnostic body; assembled with pinned candidate and input/bench helpers."""
def build_bundle(ns):
    from pathlib import Path
    from torch.utils.cpp_extension import load, _get_build_directory
    units=[];declarations=[];bindings=[]
    info=r'''
std::vector<int64_t> fs_base_info(){
  cudaFuncAttributes a{};cudaFuncGetAttributes(&a,okc::moe_fused<false>);
  return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)okc_smem_bytes()};
}
'''
    for kind in ('TINY','TINY2'):
        defs=tuple(ns[kind+'_DEFINES'])+('-DOKC_PRMTSC=1','-DOKC_DQ7=1')
        code='\n'.join('#define '+d[2:].replace('=',' ',1) for d in defs)+'\n'+ns['CUDA_SRC']+info
        stem='fsdiag_'+kind.lower()
        code=code.replace('namespace okc {','namespace '+stem+' {').replace('okc::',stem+'::')
        decl=ns['CPP_SRC'].replace('#include <torch/extension.h>','')+'\nstd::vector<int64_t> fs_base_info();\n'
        for old,new in [('okc_run',stem+'_run'),('okc_set_pdl',stem+'_pdl'),('fs_base_info',stem+'_info')]:
            code=code.replace(old+'(',new+'(');decl=decl.replace(old+'(',new+'(')
        declarations.append(decl)
        bindings += [f'm.def("{kind}_run", &{stem}_run);',f'm.def("{kind}_info", &{stem}_info);']
        units.append((kind,code))
    units.append(('FS',ns['FS_CUDA']))
    declarations.append(ns['FS_CPP'].replace('#include <torch/extension.h>',''))
    for fn in ('fs_run','fs_info','okc_set_pdl','fs_set_pdl'):
        bindings.append(f'm.def("{fn}", &{fn});')
    cpp='#include <torch/extension.h>\n'+'\n'.join(declarations)+'\nPYBIND11_MODULE(TORCH_EXTENSION_NAME,m){\n'+'\n'.join(bindings)+'\n}\n'
    thin='#include <ATen/ATen.h>\n#include <vector>\nnamespace torch {using Tensor=at::Tensor;}'
    units=[(k,c.replace('#include <torch/extension.h>',thin)) for k,c in units]
    tag=hashlib.sha256((cpp+''.join(c for _,c in units)).encode()).hexdigest()[:16]
    name='moe_fsdiag_'+tag;directory=Path(_get_build_directory(name,verbose=False))
    files=[]
    for filename,code in [('binding.cpp',cpp)]+[(k+'.cu',c) for k,c in units]:
        path=directory/filename;path.write_text(code);files.append(str(path))
    return load(name=name,sources=files,build_directory=str(directory),
        # Inherit Torch's C++ standard: Torch2.14 ATen requires C++20.
        extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a','-lineinfo','-DOKC_NS=3','-DOKC_NSF=3'],verbose=False)


def _fs_error_lines(exc):
    """Bound compiler diagnostics, not the trailing ninja invocation."""
    lines = [s.strip() for s in str(exc).splitlines() if 'error' in s.lower()
             and not s.lstrip().lower().startswith('error building extension')]
    return ' | '.join(lines)[:600] or 'no compiler error lines captured'


def error(where,exc):
    print(f'FS_ERR {where} {type(exc).__name__}: {_fs_error_lines(exc)}',flush=True)


def run_all():
    started=time.monotonic();dev=torch.device('cuda')
    if torch.cuda.get_device_capability(dev)!=(9,0):raise RuntimeError('Hopper required')
    raw_source=zlib.decompress(base64.b85decode(CANDIDATE_B85))
    assert hashlib.sha256(raw_source).hexdigest()==CANDIDATE_SHA
    ns={'__name__':'_fsdiag_candidate'};exec(compile(raw_source,'<moe_fs1>','exec'),ns)
    os.environ['TORCH_CUDA_ARCH_LIST']='9.0a';os.environ['MAX_JOBS']='2'
    print(f'FS_INFO device={torch.cuda.get_device_name(dev)!r} reps=40 flush_MiB=256 BASE={BASE_SHA[:12]} FS={CANDIDATE_SHA[:12]}',flush=True)
    ready=queue.Queue()
    def worker():
        t=time.monotonic()
        try: ready.put((build_bundle(ns),None,time.monotonic()-t))
        except Exception as exc:ready.put((None,exc,time.monotonic()-t))
    threading.Thread(target=worker,daemon=True).start()
    weights=generate_weights(dev)
    # Pinned skinny builds, no MAIN selection/packing/build cost in the diagnostic.
    ns['_ext']=lambda defines=None:None
    ns['_wgs_on']=lambda:False
    state=ns['_moe_sel_original_prepare'](weights)
    bundle,exc,secs=ready.get(timeout=max(.001,235-(time.monotonic()-started)))
    if exc is not None:raise exc
    print(f'FS_BUILD diagnostic_TUs=3 seconds={secs:.2f}',flush=True)
    for k in ('TINY','TINY2'):
        print(f'FS_RES build=BASE_{k} regs_local_smem={getattr(bundle,k+"_info")()}',flush=True)
    print(f'FS_RES build=FS regs_low_high_local_low_high_smem={bundle.fs_info()}',flush=True)
    def selected(defines=None):
        for k in ('TINY','TINY2'):
            if tuple(defines)==tuple(ns[k+'_DEFINES'])+state['sfx']:
                return types.SimpleNamespace(okc_run=getattr(bundle,k+'_run'))
        raise AssertionError(('unexpected build',defines))
    ns['_ext']=selected;ns['_fs_ext']=lambda:bundle
    fs=ns['_fs_state'](weights,state)
    if not ns['_fs_probe'](weights,state,fs):
        print('FS_DIAG_FAIL prepare_probe; no performance claims',flush=True);return
    del weights
    entries=[]
    for M,seed,name in ((1,2826763372427500438,'R1'),(8,2827019192515550467,'R8'),(2,20261102,'S2'),(4,20261104,'S4')):
        data=generate_input(M,'normal',seed,dev)
        if name.startswith('S'):
            data['topk_ids'].fill_(32);data['topk_ids'][:,0]=7
        entries.append((name,data))
        ids=data['topk_ids'].cpu();cnt=torch.bincount(ids[(ids>=0)&(ids<32)].flatten(),minlength=32)
        print(f'FS_CASE case={name} seed={seed} cnt='+','.join(f'{e}:{n}' for e,n in enumerate(cnt.tolist()) if n),flush=True)
    zero=dict(entries[1][1]);zero['topk_ids']=torch.full_like(zero['topk_ids'],32)
    entries.append(('ZERO8',zero))
    flush=torch.zeros(256<<20,dtype=torch.uint8,device=dev)
    refs={};times={}
    for rep in (1,2):
        for name,data in (entries if rep==1 else entries[::-1]):
            variants=('FS',) if name=='ZERO8' else (('BASE','FS') if rep==1 else ('FS','BASE'))
            if name not in refs:
                refs[name]=torch.zeros_like(data['hidden_states']) if name=='ZERO8' else ns['_fs_parent_kernel'](data,state).clone()
            for variant in variants:
                if time.monotonic()-started>=235:
                    print(f'FS_ERR budget_skip case={name} pass={rep} var={variant}',flush=True);continue
                call=(lambda:ns['_fs_parent_kernel'](data,state)) if variant=='BASE' else (lambda:ns['_fs_run'](data,state,fs))
                med,low=bench(call,flush,n=40)
                got=call();m=ns['_fs_metrics'](got,refs[name])
                if name=='ZERO8':m['ok']=m['ok'] and bool((got==0).all())
                print(f'FS_TIME case={name} pass={rep} var={variant} med_us={med:.3f} min_us={low:.3f} '+
                      ' '.join(f'{k}={v:.6g}' if isinstance(v,float) else f'{k}={int(v)}' for k,v in m.items()),flush=True)
                times[name,variant,rep]=med
    for name,_ in entries:
        if all((name,v,r) in times for v in ('BASE','FS') for r in (1,2)):
            print(f'FS_SUM case={name} FS_BASE='+ '/'.join(f'{times[name,"FS",r]/times[name,"BASE",r]:.4f}' for r in (1,2)),flush=True)
    print(f'FS_DONE seconds={time.monotonic()-started:.2f}',flush=True)


_RAN=False
def prepare(weights):
    global _RAN
    if not _RAN:
        _RAN=True
        try:run_all()
        except Exception as exc:error('run_all',exc)
    return weights


def kernel_fn(data,weights):
    # Prefill-board diagnostic placeholder, not a submission implementation.
    t=data['config']['tokens'];dev=data['stream'].device
    return (torch.zeros((t,8192),dtype=torch.bfloat16,device=dev),
            torch.zeros((t,8192),dtype=torch.float32,device=dev),
            torch.zeros((t,8),dtype=torch.int32,device=dev))
