"""CPU-only payload/linkage/timing orchestration tests; optional device compile."""
from pathlib import Path
import ast,base64,contextlib,hashlib,importlib.util,io,json,os,re,subprocess,sys,types,zlib
from unittest.mock import patch
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.l1_offline/diag';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('md_l1',ROOT/'experiments/boards/diag/moediag_l1.py')
md=importlib.util.module_from_spec(spec);spec.loader.exec_module(md)
ns=md.incumbent_namespace()
for v,p in [('BASE','moe_b13.py'),('L1','moe_l1.py')]:
    raw=(ROOT/'experiments/boards/moe'/p).read_bytes()
    assert md.SOURCE_SHA[v]==hashlib.sha256(raw).hexdigest()
    code=ast.literal_eval(next(n.value for n in ast.parse(raw).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets)))
    assert zlib.decompress(base64.b85decode(md.CUDA_B85[v])).decode()==code
    assert md.CUDA_SHA[v]==hashlib.sha256(code.encode()).hexdigest()
assert zlib.decompress(base64.b85decode(md.INCUMBENT_B85))==(ROOT/'experiments/boards/moe/moe_b13.py').read_bytes()
assert md.variants_for('MAIN')==('BASE','L1')
def loader(**kw):
    cpp=Path(kw['sources'][0]).read_text();bodies=[Path(x).read_text() for x in kw['sources'][1:]]
    assert len(bodies)==2
    for public,sym in re.findall(r'm.def\("([^"]+)", &(\w+)\)',cpp):
        assert sum(len(re.findall(r'(?:void|std::vector<int64_t>) '+re.escape(sym)+r'\(',s)) for s in bodies)==1,(public,sym)
    assert 'setmaxnreg' not in bodies[0] and 'setmaxnreg' in bodies[1]
    assert all('#include <torch/extension.h>' not in s for s in bodies)
    assert not any(f.startswith('-std=') for f in kw['extra_cuda_cflags'])
    return kw
with patch('torch.utils.cpp_extension._get_build_directory',lambda *a,**k:str(OUT)),patch('torch.utils.cpp_extension.load',loader):
    md.build_kind(ns,ns['CUDA_SRC'],'MAIN')
print('PASS pinned moe_b13/L1 CUDA_B85/SHA/SOURCE_SHA and2-TU thin-host linkage, no std override')

# Preserve inherited40-repetition protocol: all cache flushes before event start.
events=[]
class Event:
    def __init__(self,**kw):pass
    def record(self):events.append('event')
    def synchronize(self):events.append('sync_event')
    def elapsed_time(self,other):return .01
class Graph:
    def replay(self):events.append('replay')
class Cuda:
    Event=Event;CUDAGraph=Graph
    @staticmethod
    def synchronize():events.append('sync')
    @staticmethod
    @contextlib.contextmanager
    def graph(g):events.append('capture');yield;events.append('captured')
class Flush:
    def zero_(self):events.append('flush')
with patch.object(md,'torch',types.SimpleNamespace(cuda=Cuda)):
    assert md.bench(lambda:events.append('call'),Flush(),n=40)==(10.,10.)
assert events.count('flush')==40 and events.count('replay')==41
assert events[events.index('flush'):]==['flush','event','replay','event','sync_event']*40
print('PASS40 cold-L2 graph repetitions, flush outside events, graph setup outside timing')

# Real routing transforms on CPU; timing/kernel outputs below are ONLY mocks.
data={m:md.generate_input(m,dist,seed,'cpu') for m,_,dist,seed in md.CASES if m in (32,128,256)}
def kernel(data,state):
    return _ext(tuple(MAIN_DEFINES)+state['sfx']).okc_run(data,state)
fake=dict(ns,kernel_fn=kernel)
fake['_moe_sel_original_prepare']=lambda w:dict(sfx=('-DOKC_PRMTSC=1',),scratch=[torch.zeros(1)],grid=132)
bundle=types.SimpleNamespace();calls=[]
for v in ('BASE','L1'):
    setattr(bundle,'okc_info_'+v,lambda v=v:[96,96,1024,1024,231168,0 if v=='L1' else 16,0 if v=='L1' else 16,640 if v=='L1' else 576])
    def run(d,s,v=v):
        calls.append(v);return torch.ones_like(d['hidden_states'])*(1. if v=='BASE' else 1.015625)
    setattr(bundle,'okc_run_'+v,run)
def bench(fn,flush,n):assert n==40;fn();return (45.,44.) if calls[-1]=='L1' else (50.,49.)
zeros=torch.zeros
def small_zeros(size,*args,**kw):
    kw.pop('device',None);return zeros(1 if isinstance(size,int) and size>10000 else size,*args,**kw)
with contextlib.ExitStack() as st:
    for obj,key,value in [(md,'incumbent_namespace',lambda:fake),(md,'build_kind',lambda *a:bundle),
        (md,'generate_weights',lambda dev:{}),(md,'generate_input',lambda m,dist,seed,dev:data[m]),
        (md,'bench',bench),(md,'over',lambda:False),(torch.cuda,'get_device_capability',lambda dev:(9,0)),
        (torch.cuda,'get_device_name',lambda dev:'MOCK_ONLY'),(torch,'zeros',small_zeros)]:
        st.enter_context(patch.object(obj,key,value))
    md.T0=md.time.monotonic();md._LOG_BYTES=0;md._LOG_LIMIT=False
    buf=io.StringIO();st.enter_context(contextlib.redirect_stdout(buf));md.run_all()
log=buf.getvalue();rows=[x for x in log.splitlines() if x.startswith('MD case=')]
assert len(rows)==24,log
names=['R32','R128','R128_NODUP','R256','R256_DUP3','R256_NODUP']
want=[(n,v,str(p)) for p in (1,2) for n in (names if p==1 else names[::-1]) for v in (('BASE','L1') if p==1 else ('L1','BASE'))]
got=[re.search(r'case=(\w+) var=(\w+) pass=(\d)',x).groups() for x in rows]
assert got==want,(got,want)
assert all('eq=0' in x and 'rel_L2=0.0156' in x for x in rows if 'var=L1' in x)
assert log.count('L1_BASE=0.9000/0.9000')==6 and 'MD_ZERO' not in log and 'MD_ERR' not in log
assert not md._LOG_LIMIT and len(log.encode())+1000<9400
(OUT/'mock.log').write_text(log)
print('PASS6 requested cases/24 timings, reverse order, equality/error reporting, zero disabled; mock bytes',len(log.encode()))

if '--compile' in sys.argv:
    os.environ['CUDA_CACHE_DISABLE']='1'
    sys.path.insert(0,str(ROOT.parents[2]/'experiments/prepared'))
    import local_nvrtc_compile as nv
    flags=['--gpu-architecture=sm_90a','--std=c++20','--generate-line-info','--device-as-default-execution-space','--ptxas-options=-v',
        '--include-path='+str(nv.TOOL/'nvidia/cu13/include'),'--include-path='+str(nv.TOOL/'nvidia/cu13/include/cccl'),'-DOKC_NS=3','-DOKC_NSF=3']
    compiler=nv.NVRTC();results=[]
    for v in ('BASE','L1'):
        src=(OUT/(v.lower()+'.cu')).read_text().split('#include <ATen/ATen.h>')[0]
        if v=='BASE':
            anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
            src=src.replace(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S,int pb,int s,int G);\n'+anchor)
        stem='md_MAIN_'+v
        rc,log,cubin,secs=compiler.compile(src.encode(),[f'&{stem}::moe_fused<false>',f'&{stem}::moe_fused<true>'],flags,True)
        (OUT/(v+'.log')).write_text(log)
        lines=[s for s in log.splitlines() if any(x in s for x in ('Used ','spill stores','stack frame','error'))]
        print('DIAG',v,rc,secs,'\n'.join(lines),flush=True)
        if rc:print(log);raise SystemExit(rc)
        results.append(dict(variant=v,rc=rc,seconds=secs,flags=flags,resources=lines))
    (OUT/'compile.json').write_text(json.dumps(results,indent=2))
