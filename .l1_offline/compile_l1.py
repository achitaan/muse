"""Device-only NVRTC/ptxas+SASS; no driver/context/launch."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys
os.environ['CUDA_CACHE_DISABLE']='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.l1_offline'
sys.path.insert(0,str(ROOT.parents[2]/'experiments/prepared'))
import local_nvrtc_compile as nv
def load(file):
    ns={};exec(compile(file.read_bytes(),str(file),'exec'),ns);return ns
base=load(ROOT/'experiments/boards/moe/moe_b13.py');cand=load(ROOT/'experiments/boards/moe/moe_l1.py')
flags=['--gpu-architecture=sm_90a','--std=c++20','--generate-line-info','--device-as-default-execution-space','--ptxas-options=-v',
       '--include-path='+str(nv.TOOL/'nvidia/cu13/include'),'--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl'),
       '-DOKC_NS=3','-DOKC_NSF=3','-DOKC_PRMTSC=1','-DOKC_DQ7=1']
builds=[('BASE',base,'MAIN',()),('L1',cand,'MAIN',()),('OFF',cand,'MAIN',('-DOKC_L1=0',))]
builds += [(k+'_BASE',base,k,()) for k in ('TINY','TINY2','MID','MAIN_FALLBACK')]
builds += [(k+'_L1',cand,k,()) for k in ('TINY','TINY2','MID','MAIN_FALLBACK')]
compiler=nv.NVRTC();results=[]
for name,ns,kind,extra in builds:
    if len(sys.argv)>1 and name not in sys.argv[1:]:continue
    src=ns['CUDA_SRC'].split('#include <torch/extension.h>')[0]
    if name.endswith('_BASE') or name=='BASE':
        anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
        src=src.replace(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S,int pb,int s,int G);\n'+anchor)
    opts=flags+list(ns[kind+'_DEFINES'])+list(extra)
    rc,log,cubin,secs=compiler.compile(src.encode(),['&okc::moe_fused<false>','&okc::moe_fused<true>'],opts,True)
    (OUT/(name+'.cu')).write_text(src);(OUT/(name+'.log')).write_text(log)
    rows=[s for s in log.splitlines() if any(x in s for x in ('Used ','spill stores','stack frame','C75','error'))]
    print(name,rc,secs,'\n'.join(rows),flush=True)
    if rc:print(log);raise SystemExit(rc)
    p=OUT/(name+'.cubin');p.write_bytes(cubin)
    for arg,suffix in [('--dump-sass','sass'),('--dump-resource-usage','resources')]:
        (OUT/(name+'.'+suffix)).write_text(subprocess.check_output([str(nv.TOOL/'nvidia/cu13/bin/cuobjdump.exe'),arg,str(p)],text=True))
    results.append(dict(name=name,seconds=secs,flags=opts,sha256=hashlib.sha256(src.encode()).hexdigest(),resources=rows))
(OUT/('results_'+('_'.join(sys.argv[1:]) or 'all')+'.json')).write_text(json.dumps(results,indent=2))
