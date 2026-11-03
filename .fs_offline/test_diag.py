"""No GPU: exercise bundle generation and the inherited cold-L2 protocol."""
import ast,base64,contextlib,hashlib,importlib.util,io,re,sys,types,zlib
from pathlib import Path
from unittest.mock import patch
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fd',ROOT/'experiments/boards/diag/fsdiag.py')
md=importlib.util.module_from_spec(spec);spec.loader.exec_module(md)
ns={};exec(compile(zlib.decompress(base64.b85decode(md.CANDIDATE_B85)),'<candidate>','exec'),ns)
def getdir(name,verbose=False):
    p=ROOT/'.fs_offline/bundle';p.mkdir(exist_ok=True);return str(p)
def loader(**kw):
    files=list(map(Path,kw['sources']));assert len(files)==4
    cpp=files[0].read_text();bodies=[f.read_text() for f in files[1:]]
    for public,sym in re.findall(r'm.def\("([^"]+)", &(\w+)\)',cpp):
        assert sum(len(re.findall(r'(?:void|std::vector<int64_t>)\s+'+re.escape(sym)+r'\(',s)) for s in bodies)==1,(public,sym)
    for s in bodies:assert '#include <torch/extension.h>' not in s
    assert '#define OKC_TINY 1' in bodies[0] and '#define OKC_TINY 1' in bodies[1]
    assert 'namespace fsdiag_tiny {' in bodies[0] and 'namespace fsdiag_tiny2 {' in bodies[1]
    assert 'namespace fs {' in bodies[2]
    assert ns['FS_CUDA'].split('#include <torch/extension.h>')[0] in bodies[2]
    return kw
with patch('torch.utils.cpp_extension._get_build_directory',getdir),patch('torch.utils.cpp_extension.load',loader):
    md.build_bundle(ns)
print('PASS three-TU thin-host bundle: distinct symbols, exact FS device code, pinned BASE flags')

# Event protocol oracle: graph setup outside timing, exactly40 flush/replay samples.
events=[]
class Event:
    def __init__(self,enable_timing):pass
    def record(self):events.append('event')
    def synchronize(self):events.append('sync_event')
    def elapsed_time(self,other):return .01
class Graph:
    def replay(self):events.append('replay')
class Flush:
    def zero_(self):events.append('flush')
class CUDA:
    CUDAGraph=Graph;Event=Event
    @staticmethod
    def synchronize():events.append('sync')
    @staticmethod
    @contextlib.contextmanager
    def graph(g):events.append('capture_begin');yield;events.append('capture_end')
with patch.object(md,'torch',types.SimpleNamespace(cuda=CUDA)):
    med,low=md.bench(lambda:events.append('call'),Flush(),n=40)
assert med==low==10
assert events.count('flush')==40 and events.count('replay')==41
assert events.index('capture_end')<events.index('flush')
i=events.index('flush')
assert events[i:]==['flush','event','replay','event','sync_event']*40
print('PASS40 samples: L2 flush outside events, graph capture outside timing')

# Full orchestration mock: does not claim numerical/timing evidence.
raw_generate=md.generate_input
inputs={m:raw_generate(m,'normal',m,'cpu') for m in (1,2,4,8)}
logs=[];calls=[]
fake_ns=dict(ns)
state=dict(w=[torch.zeros(1)],sfx=('-DOKC_PRMTSC=1',),lo=0,cnt=32,grid=132)
fake_ns['_moe_sel_original_prepare']=lambda w:state
fake_ns['_fs_state']=lambda w,s:dict(packed=torch.zeros(1))
fake_ns['_fs_probe']=lambda w,s,f:True
fake_ns['_fs_parent_kernel']=lambda d,s:(calls.append(('BASE',d['hidden_states'].shape[0])) or torch.ones_like(d['hidden_states']))
def fsrun(d,s,f):
    calls.append(('FS',d['hidden_states'].shape[0]))
    return torch.zeros_like(d['hidden_states']) if bool((d['topk_ids']==32).all()) else torch.ones_like(d['hidden_states'])
fake_ns['_fs_run']=fsrun
# Keep real _fs_metrics on CPU, so errors/zero handling are exercised.
bundle=types.SimpleNamespace(fs_info=lambda:[151,168,0,80,218752],
    TINY_info=lambda:[144,0,214784],TINY2_info=lambda:[144,0,214784])
original_exec=exec
def inject(code,glob):glob.update(fake_ns)
original_zeros=torch.zeros
def zeros(size,*args,**kw):
    kw.pop('device',None)
    return original_zeros(1 if isinstance(size,int) and size>10000 else size,*args,**kw)
def bench(fn,flush,n):assert n==40;fn();return (10.,9.)
with contextlib.ExitStack() as st:
    for obj,key,val in [(md,'build_bundle',lambda n:bundle),(md,'generate_weights',lambda d:{}),
        (md,'generate_input',lambda m,dist,seed,dev:dict(inputs[m],topk_ids=inputs[m]['topk_ids'].clone())),
        (md,'bench',bench),(torch.cuda,'get_device_capability',lambda dev:(9,0)),
        (torch.cuda,'get_device_name',lambda dev:'MOCK_ONLY'),(torch,'zeros',zeros)]:
        st.enter_context(patch.object(obj,key,val))
    st.enter_context(patch.object(md,'exec',inject,create=True))
    out=io.StringIO();st.enter_context(contextlib.redirect_stdout(out));md.run_all()
text=out.getvalue();lines=[l for l in text.splitlines() if l.startswith('FS_TIME')]
assert len(lines)==18,text
seen=[re.search(r'case=(\w+) pass=(\d) var=(\w+)',l).groups() for l in lines]
names=['R1','R8','S2','S4','ZERO8']
expected=[(name,str(r),v) for r in (1,2) for name in (names if r==1 else names[::-1])
          for v in (('FS',) if name=='ZERO8' else (('BASE','FS') if r==1 else ('FS','BASE')))]
assert seen==expected
assert all('ok=1' in l for l in lines)
assert text.count('FS_SUM')==4 and 'FS_ERR' not in text
assert all(v=='FS' for name,r,v in seen if name=='ZERO8')
(ROOT/'.fs_offline/mock_diag.log').write_text(text)
print('PASS18 mocked timings, both orders, zero-only-FS,4 summaries; NOT GPU evidence')
