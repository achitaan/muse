"""CPU linkage/protocol checks and device-only NVRTC (no GPU execution)."""
import ast,base64,hashlib,importlib.util,json,os,re,subprocess,sys,zlib
from pathlib import Path
sys.dont_write_bytecode=True
os.environ['CUDA_CACHE_DISABLE']='1'
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.fs_offline/diag2';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('fd2',ROOT/'experiments/boards/diag/fsdiag2.py')
md=importlib.util.module_from_spec(spec);spec.loader.exec_module(md)
ns={};exec(compile(zlib.decompress(base64.b85decode(md.CANDIDATE_B85)),'<pinned>','exec'),ns)
cpp,units=md.bundle_sources2(ns)
(OUT/'binding.cpp').write_text(cpp)
for k,c in units:(OUT/(k+'.cu')).write_text(c)
for public,sym in re.findall(r'm.def\("([^"]+)", &(\w+)\)',cpp):
    assert sum(len(re.findall(r'(?:void|std::vector<int64_t>)\s+'+re.escape(sym)+r'\(',c)) for _,c in units)==1,(public,sym)
assert len(units)==8
assert 'std=c++17' not in ast.get_source_segment((ROOT/'experiments/boards/diag/fsdiag2.py').read_text(),
    next(n for n in ast.parse((ROOT/'experiments/boards/diag/fsdiag2.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='build_bundle2'))
original=ns['FS_CUDA'].split('#include <torch/extension.h>')[0]
fscode=dict(units)['FS'].split('#include <ATen/ATen.h>')[0].replace('namespace fs2_fs {','namespace fs {').replace('fs2_fs::','fs::')
assert original==fscode,'FS timed control is original device source'
assert md.diagnostic_source(ns['FS_CUDA']).count('CSYNC()')==ns['FS_CUDA'].count('CSYNC()')
for p,sha in json.loads((ROOT/'.fs_offline/frozen_diag2.json').read_text()).items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==sha,'frozen file changed'
print('PASS8-TU linkage; exact FS control; no new CTA barriers; frozen FS1/fsdiag unchanged')

import torch
# Known synthetic timestamps, not performance claims. Two CTAs differing by1us;
# producer exits late and must NOT contaminate consumer exit or per-task summary.
trace=torch.zeros((2,9,160),dtype=torch.int64)
for c in range(2):
    for w in range(9):
        off=100000+c*1000
        for i,t in {0:0,1:2000,2:4000,4:14000,5:15000,6:18000,16:7000,17:9000,18:10000,19:12000,144:12000,145:14000}.items():trace[c,w,i]=off+t
        trace[c,w,7:11]=torch.tensor([2,2*c,2*c+2,1])
    trace[c,8,6]=999999999
summary,tasks=md.timeline_summary2(trace)
assert summary['exit']=='18.00/18.50/19.00'
assert summary['Bsum']=='4.00/4.00/4.00'
assert summary['flushsum']=='2.00/2.00/2.00'
assert summary['segments']=='0:12.00/12.50/13.00,14.00/14.50/15.00'
assert tasks=='0:7.00/7.50/8.00,9.00/9.50/10.00 1:10.00/10.50/11.00,12.00/12.50/13.00'
err=RuntimeError("Error building extension: many commands\na.cu:5:2: error: C++20 required\n"+'ninja tail'*200)
assert 'C++20 required' in md.compiler_errors(err) and 'ninja tail' not in md.compiler_errors(err)
print('PASS timeline min/med/max/task-duration math, producer exclusion and600-char compiler logging')
if '--compile' not in sys.argv:raise SystemExit(0)
sys.path.insert(0,str(ROOT.parents[2]/'experiments/prepared'))
import local_nvrtc_compile as nv
flags=['--gpu-architecture=sm_90a','--std=c++20','--generate-line-info',
       '--device-as-default-execution-space','--ptxas-options=-v',
       '--include-path='+str(nv.TOOL/'nvidia/cu13/include'),'--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl'),
       '-DOKC_NS=3','-DOKC_NSF=3']
compiler=nv.NVRTC();results=[]
for k,code in units:
    code=code.split('#include <ATen/ATen.h>')[0]
    stem='fs2_'+k.lower()
    if k in ('TINY','TINY2'):
        anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
        code=code.replace(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S,int pb,int s,int G);\n'+anchor)
        names=[f'&{stem}::moe_fused<false>',f'&{stem}::moe_fused<true>']
    else:names=[f'&{stem}::kernel<false>',f'&{stem}::kernel<true>']
    rc,log,cubin,secs=compiler.compile(code.encode(),names,flags,True)
    (OUT/(k+'.log')).write_text(log)
    rows=[line for line in log.splitlines() if any(s in line for s in ('Used ','spill stores','stack frame','error'))]
    print(k,rc,secs,'\n'.join(rows),flush=True)
    if rc:print(log);raise SystemExit(rc)
    p=OUT/(k+'.cubin');p.write_bytes(cubin)
    sass=subprocess.check_output([str(nv.TOOL/'nvidia/cu13/bin/cuobjdump.exe'),'--dump-sass',str(p)],text=True)
    (OUT/(k+'.sass')).write_text(sass)
    results.append(dict(kind=k,seconds=secs,resource=rows,HMMA=len(re.findall(r'\bHMMA\.',sass)),
                        LDS=len(re.findall(r'\bLDS\b',sass)),STS=len(re.findall(r'\bSTS\b',sass)),
                        RED=len(re.findall(r'\bRED(?:G)?\.',sass))))
(OUT/'results.json').write_text(json.dumps(dict(flags=flags,results=results),indent=2))
r={x['kind']:x for x in results}
assert r['NOA']['HMMA']>0 and r['NOA']['HMMA']<r['FS']['HMMA']
assert r['NOB']['HMMA']>0 and r['NOB']['HMMA']<r['FS']['HMMA']
assert r['NOACC']['HMMA']==r['FS']['HMMA'],'NOACC must retain B math'
assert r['NOFLUSH']['HMMA']==r['FS']['HMMA'],'NOFLUSH must retain A/B math'
print('PASS all8 CUDA TUs/16 instantiations; ablations retain/remove intended MMA work; no GPU run')
