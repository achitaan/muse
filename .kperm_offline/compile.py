"""Device-only sm90a NVRTC/ptxas checks. No CUDA driver/context/launch."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
os.environ['CUDA_CACHE_DISABLE']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.kperm_offline'
sys.path.insert(0,str(ROOT.parents[2]/'experiments/prepared'))
import local_nvrtc_compile as nv
def load(p):
    d={};exec(compile(p.read_bytes(),str(p),'exec'),d);return d
base=load(ROOT/'experiments/boards/moe/moe_b15.py');b16=load(ROOT/'experiments/boards/moe/moe_b16.py')
flags=['--gpu-architecture=sm_90a','--std=c++20','--generate-line-info','--device-as-default-execution-space','--ptxas-options=-v',
 '--include-path='+str(nv.TOOL/'nvidia/cu13/include'),'--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl'),
 '-DOKC_NS=3','-DOKC_NSF=3','-DOKC_PRMTSC=1','-DOKC_DQ7=1']
builds=[('BASE',base,'MAIN',()),('KPERM3',b16,'MAIN',()),('OFF',b16,'MAIN',('-DOKC_KPERM=0',))]
builds += [('PREDECODE',b16,'MAIN',())]
builds += [(k+'_'+v,ns,k,()) for k in ('TINY','TINY2','MID','MAIN_FALLBACK') for v,ns in (('BASE',base),('B16',b16))]
compiler=nv.NVRTC();result=[]
for name,ns,kind,extra in builds:
    if len(sys.argv)>1 and name not in sys.argv[1:]:continue
    src=ns['CUDA_SRC'].split('#include <torch/extension.h>')[0]
    if name=='PREDECODE':
        from predecode import predecode
        src=predecode(src)
    # Strict NVRTC declaration-order adaptation used equally for all builds;
    # actual published source/launcher otherwise retained.
    anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
    src=src.replace(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S,int pb,int s,int G);\n'+anchor)
    opts=flags+list(ns[kind+'_DEFINES'])+list(extra)
    rc,log,cubin,secs=compiler.compile(src.encode(),['&okc::moe_fused<false>','&okc::moe_fused<true>'],opts,True)
    (OUT/(name+'.cu')).write_text(src);(OUT/(name+'.log')).write_text(log)
    lines=[x for x in log.splitlines() if any(k in x for k in ('Used ','spill stores','stack frame','C75','error'))]
    print(name,rc,secs,'\n'.join(lines),flush=True)
    if rc:print(log);raise SystemExit(rc)
    p=OUT/(name+'.cubin');p.write_bytes(cubin)
    for arg,ext in [('--dump-sass','sass'),('--dump-resource-usage','resources')]:
        (OUT/(name+'.'+ext)).write_text(subprocess.check_output([str(nv.TOOL/'nvidia/cu13/bin/cuobjdump.exe'),arg,str(p)],text=True))
    result.append(dict(build=name,rc=rc,seconds=secs,flags=opts,resources=lines,device_sha256=hashlib.sha256(src.encode()).hexdigest(),c7512='C7512' in log))
(OUT/('compile_'+('_'.join(sys.argv[1:]) or 'all')+'.json')).write_text(json.dumps(result,indent=2))
