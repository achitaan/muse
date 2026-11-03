"""Device-only feasibility checks. OVER112 is deliberately NOT launchable."""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys
os.environ['CUDA_CACHE_DISABLE']='1';sys.dont_write_bytecode=True
from source import ROOT,BASE,L2,l3_source
OUT=ROOT/'.l3_offline'
sys.path.insert(0,str(ROOT.parents[2]/'experiments/prepared'))
import local_nvrtc_compile as nv
flags=['--gpu-architecture=sm_90a','--std=c++20','--generate-line-info','--device-as-default-execution-space','--ptxas-options=-v',
 '--include-path='+str(nv.TOOL/'nvidia/cu13/include'),'--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl'),
 '-DOKC_NS=3','-DOKC_NSF=3','-DOKC_PRMTSC=1','-DOKC_DQ7=1','-DOKC_NW=16','-DOKC_FAST1=0','-DOKC_EPI4=0','-DOKC_WGS=1']
l3=l3_source()
builds=[('BASE',BASE),('L2',L2),('RUNTIME96',l3),
 ('TWO96',l3.replace('return S.preissuedB != 0;','return true; // fixed two-slot wide-containing launch')),
 ('OVER112',l3.replace('__launch_bounds__(NTL, 1)','__maxnreg__(112)')),
 ('OVER255',l3.replace('__launch_bounds__(NTL, 1)','__maxnreg__(255)'))]
compiler=nv.NVRTC();results=[]
for name,src in builds:
    if len(sys.argv)>1 and name not in sys.argv[1:]:continue
    code=src.split('#include <torch/extension.h>')[0]
    if name=='BASE':
        anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
        code=code.replace(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S,int pb,int s,int G);\n'+anchor)
    (OUT/(name+'.cu')).write_text(code)
    rc,log,cubin,secs=compiler.compile(code.encode(),['&okc::moe_fused<false>','&okc::moe_fused<true>'],flags,True)
    (OUT/(name+'.log')).write_text(log)
    rows=[s for s in log.splitlines() if any(x in s for x in ('Used ','spill stores','stack frame','C75','error'))]
    print(name,rc,secs,'\n'.join(rows),flush=True)
    if rc:print(log);raise SystemExit(rc)
    p=OUT/(name+'.cubin');p.write_bytes(cubin)
    for arg,suffix in [('--dump-sass','sass'),('--dump-resource-usage','resources')]:
        (OUT/(name+'.'+suffix)).write_text(subprocess.check_output([str(nv.TOOL/'nvidia/cu13/bin/cuobjdump.exe'),arg,str(p)],text=True))
    results.append(dict(name=name,rc=rc,seconds=secs,flags=flags,sha256=hashlib.sha256(code.encode()).hexdigest(),resources=rows,
                       c7512='C7512' in log,launchable_register_budget=not name.startswith('OVER')))
(OUT/('compile_'+('_'.join(sys.argv[1:]) or 'all')+'.json')).write_text(json.dumps(results,indent=2))
