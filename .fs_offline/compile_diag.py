"""Offline device compilation of the generated three-TU diagnostic bundle."""
import os,sys,json
from pathlib import Path
os.environ['CUDA_CACHE_DISABLE']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[2]/'experiments/prepared'))
import local_nvrtc_compile as nv
flags=['--gpu-architecture=sm_90a','--std=c++17','--device-as-default-execution-space','--ptxas-options=-v',
       '--include-path='+str(nv.TOOL/'nvidia/cu13/include'),'--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl'),
       '-DOKC_NS=3','-DOKC_NSF=3']
results=[]
compiler=nv.NVRTC()
for name in ('TINY','TINY2','FS'):
    code=(ROOT/'.fs_offline/bundle'/(name+'.cu')).read_text().split('#include <ATen/ATen.h>')[0]
    if name!='FS':
        stem='fsdiag_'+name.lower()
        anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
        code=code.replace(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S,int pb,int s,int G);\n'+anchor)
        names=[f'&{stem}::moe_fused<false>',f'&{stem}::moe_fused<true>']
    else:names=['&fs::kernel<false>','&fs::kernel<true>']
    rc,log,cubin,secs=compiler.compile(code.encode(),names,flags,True)
    (ROOT/'.fs_offline'/('diag_'+name+'.log')).write_text(log)
    print(name,rc,secs,log,flush=True)
    if rc:raise SystemExit(rc)
    results.append(dict(name=name,rc=rc,seconds=secs,flags=flags))
(ROOT/'.fs_offline/diag_compile.json').write_text(json.dumps(results,indent=2))
