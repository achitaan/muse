"""CPU-only measurement protocol tests. Never imports the GPU submission.

The fake GPU models events, independent graph instances, capture side streams,
allocation/capture/launch-slot biases and full recomputation. These tests do
NOT establish H200 null ratios or the physical cause of the supplied bias.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import threading
from types import SimpleNamespace

import torch

ROOT=Path(__file__).resolve().parents[1]
CANDIDATE=ROOT/'candidates/dense_rotcodes_null.py'
PARENT=ROOT/'candidates/dense_rotcodes2.py'

PATHS={
 ('_make_chenwainuo','kernel_fn'):'torch.empty', ('_make_chenwainuo','_r6'):'torch.empty',
 ('_make_chenwainuo','_v12'):'torch.empty', ('_make_vjeux','_run_gq_family'):'torch.empty',
 ('_make_vjeux','_run_sk2'):'torch.empty', ('_make_vjeux','_run_skinny'):'torch.empty',
 ('_make_vjeux','_run_wide'):'torch.empty', ('_make_vjeux','_run_gemm'):'torch.empty',
 ('_make_vjeux','_run'):'torch.empty', ('_make_nb','_rt'):'torch.empty',
 ('_make_nb','_ol'):'torch.empty', ('_make_nb','_rt12'):'torch.empty',
 ('_make_ag','run'):'torch.empty', ('_v6_call',):'_v6_torch.empty',
 ('_dt_call',):'_dt_torch.empty', ('_rc_call',):'_dt_torch.empty', ('_r2_call',):'_dt_torch.empty',
}


def setup():
    tree,parent=ast.parse(CANDIDATE.read_text()),ast.parse(PARENT.read_text())
    start=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
               and any(isinstance(t,ast.Name) and t.id=='_PRUNE' for t in n.targets))
    prefix=copy.deepcopy(tree.body[start:start+len(parent.body)])
    class Restore(ast.NodeTransformer):
        def __init__(self):self.path=[];self.count=0
        def visit_FunctionDef(self,n):
            self.path.append(n.name);self.generic_visit(n);self.path.pop();return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Name) and n.func.id=='_ng_empty':
                assert tuple(self.path) in PATHS,tuple(self.path)
                n.func=ast.parse(PATHS[tuple(self.path)]).body[0].value;self.count+=1
            return n
    restore=Restore()
    normalized=restore.visit(ast.Module(body=prefix,type_ignores=[]))
    assert ast.dump(normalized)==ast.dump(parent)
    assert restore.count==24,restore.count
    ns=dict(_ng_contextlib=contextlib,_ng_threading=threading,_ng_torch=torch,
            _ng_math=math,_ng_stats=statistics,_dt_torch=torch,
            _DTBudget=type('Budget',(Exception,),{}),_dt_check=lambda d:None)
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith('_NG_') for root in node.targets for t in ast.walk(root)):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANDIDATE),'exec'),ns)
        if isinstance(node,ast.ClassDef) and node.name=='_NGBindingError':
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANDIDATE),'exec'),ns)
        if isinstance(node,ast.FunctionDef) and (node.name.startswith('_ng_') or node.name in ('_dt_errors','_dt_close')):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANDIDATE),'exec'),ns)
    # Only the final repaired definitions, not the earlier legacy versions.
    for name in ('_rc_measure','_r2_measure','_r2_pool_component','prepare'):
        node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANDIDATE),'exec'),ns)
    return tree,ns,restore.count


def static_gates(tree):
    active=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('_ng_')]
    forbidden={'empty_cache','_host_emptyCache','item','cpu','numpy','getframeinfo','stack','open'}
    for node in active:
        for x in ast.walk(node):
            if isinstance(x,ast.Call):
                name=x.func.attr if isinstance(x.func,ast.Attribute) else x.func.id if isinstance(x.func,ast.Name) else ''
                assert name not in forbidden,(node.name,name)
    phase=next(n for n in active if n.name=='_ng_phase')
    text=ast.unparse(phase)
    assert '.capture_begin()' in text and '.capture_end()' in text
    assert '.cuda.graph(' not in text
    for name in ('_rc_measure','_r2_measure'):
        node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        assert '_ng_evaluate' in ast.unparse(node) and '_dt_abba' not in ast.unparse(node)
    draw=ast.unparse(next(n for n in active if n.name=='_ng_draw'))
    assert draw.count('_ng_phase(')==3
    assert draw.index('pre = _ng_phase')<draw.index('candidate = _ng_phase')<draw.index('post = _ng_phase')
    empty=next(n for n in active if n.name=='_ng_empty')
    assert 'lease is None' in ast.unparse(empty)
    return True


def binding_gates(ns):
    empty=ns['_ng_empty'];borrow=ns['_ng_borrow'];err=ns['_NGBindingError']
    target=torch.empty((2,16),dtype=torch.bfloat16)
    a=empty((2,16),dtype=target.dtype,device='cpu');b=empty((2,16),dtype=target.dtype,device='cpu')
    assert a.data_ptr()!=b.data_ptr()
    with borrow(target) as lease:
        assert empty((2,16),dtype=target.dtype,device='cpu') is target
        assert lease['hits']==1
        try:empty((2,16),dtype=target.dtype,device='cpu')
        except err:pass
        else:raise AssertionError('second output not rejected')
        other=torch.empty_like(target)
        with borrow(other):assert empty((2,16),dtype=target.dtype,device='cpu') is other
        assert ns['_NG_TLS'].lease is lease
    assert getattr(ns['_NG_TLS'],'lease',None) is None
    try:
        with borrow(target):raise ValueError('injected')
    except ValueError:pass
    assert ns['_NG_TLS'].lease is None
    result=[]
    with borrow(target):
        def worker():result.append(empty((2,16),dtype=target.dtype,device='cpu'))
        thread=threading.Thread(target=worker);thread.start();thread.join()
        assert result[0].data_ptr()!=target.data_ptr()
    for dtype in (torch.bfloat16,torch.float32):
        for shape in ((1,32),(7,128),(32,256)):
            aligned=ns['_ng_aligned_empty'](shape,dtype,'cpu')
            assert aligned.is_contiguous() and aligned.data_ptr()%65536==0
    # Output options/shape mismatch cannot silently alter semantics.
    with borrow(target):
        assert empty((3,16),dtype=target.dtype,device='cpu').data_ptr()!=target.data_ptr()
    with borrow(target):
        try:empty((2,16),dtype=target.dtype,device='cpu',requires_grad=False)
        except err:pass
        else:raise AssertionError('allocation options ignored')
    return dict(production_fresh=True,nesting_and_exception_restore=True,thread_local=True,aligned_views=6)


def schedule_gates(ns):
    count=0
    for seed in range(16):
        capture=ns['_ng_capture_orders'](seed)
        assert capture.count(('A','B'))==capture.count(('B','A'))==2
        visits={(p,i,r):0 for p in range(4) for i in range(16) for r in 'AB'}
        for block in range(32):
            sequence=ns['_ng_schedule'](block,seed)
            assert len(sequence)==16
            for i,(panel,role) in enumerate(sequence):visits[panel,i,role]+=1
            for i in range(0,16,4):
                chunk=sequence[i:i+4]
                assert len({p for p,r in chunk})==1
                assert ''.join(r for p,r in chunk) in ('ABBA','BAAB')
        assert set(visits.values())=={4},visits
        count+=1
    return dict(seeds=count,every_panel_role_in_every_event_slot=4)


class Event:
    def __init__(self,backend):self.backend=backend;self.time=None
    def record(self):
        assert self.backend.capturing is None
        if self.backend.last=='flush':
            assert all(g.replays>=8 for g in self.backend.graphs[-8:]),'timing before symmetric warmup'
        self.time=self.backend.time;self.backend.last='event'
    def synchronize(self):self.backend.last='sync'
    def elapsed_time(self,end):return (end.time-self.time)/1000


class Stream:
    def __init__(self,backend):self.backend=backend
    def wait_stream(self,other):self.backend.stream_waits+=1


class Graph:
    def __init__(self,backend):
        self.backend=backend;self.uid=len(backend.graphs);self.ops=[];self.replays=0;backend.graphs.append(self)
    def capture_begin(self):
        assert self.backend.capturing is None
        self.backend.capturing=self;self.backend.begins+=1
    def capture_end(self):
        assert self.backend.capturing is self
        self.backend.capturing=None;self.backend.ends+=1
    def replay(self):
        assert self.backend.capturing is None
        self.replays+=1
        self.backend.run_graph(self)


class FakeCuda:
    def __init__(self,backend):
        self.backend=backend;self.memory=SimpleNamespace(get_allocator_backend=lambda:'native')
    def Event(self,**kw):return Event(self.backend)
    def CUDAGraph(self):return Graph(self.backend)
    def Stream(self):return Stream(self.backend)
    def current_stream(self):return self.backend.main_stream
    @contextlib.contextmanager
    def stream(self,s):
        old=self.backend.active_stream;self.backend.active_stream=s
        try:yield
        finally:self.backend.active_stream=old
    def synchronize(self):
        assert self.backend.capturing is None;self.backend.last='sync'
    def graph(self,*a,**kw):raise AssertionError('high-level graph context resets allocator caches')
    def empty_cache(self):raise AssertionError('allocator cache reset')


class FakeTorch:
    def __init__(self,backend):self.cuda=FakeCuda(backend)
    def __getattr__(self,name):return getattr(torch,name)


class Backend:
    def __init__(self,ns,candidate=9.,slot_bias=0.,capture_bias=0.,post_bias=0.,startup=0.,wrong=False,address_penalty=0.):
        self.ns=ns;self.time=0.;self.last='';self.capturing=None;self.graphs=[];self.begins=self.ends=0
        self.stream_waits=0;self.main_stream=Stream(self);self.active_stream=self.main_stream
        self.candidate=candidate;self.slot_bias=slot_bias;self.capture_bias=capture_bias
        self.post_bias=post_bias;self.startup=startup;self.wrong=wrong;self.flushed=0;self.replay_index=0
        self.address_penalty=address_penalty
    def flush_read(self,*args):
        assert self.capturing is None;self.time+=128.;self.flushed+=1;self.last='flush'
    def emit(self,tag,x,out,final=True):
        op=(tag,x,out,final)
        if self.capturing is not None:self.capturing.ops.append(op)
        else:
            self.time+=2. if tag=='pre' else 8. if tag=='base' else self.candidate
            self.execute(op)
    def execute(self,op):
        tag,x,out,final=op
        if final:out.copy_((x.float().sum(1,keepdim=True)+(16 if self.wrong and tag=='candidate' else 0)).expand_as(out).to(out.dtype))
    def run_graph(self,g):
        # Whole incumbent has TWO operations; candidates have one. Launch cost
        # is included once, exactly as in the real event-level gate.
        cost=4.+sum(2. if tag=='pre' else 8. if tag=='base' else self.candidate for tag,x,out,final in g.ops)
        if g.ops and ((g.ops[-1][2].data_ptr()>>16)&1):cost+=self.address_penalty
        phase=(g.uid//8)%3
        if g.uid%2==0:cost*=1.+self.capture_bias+(self.post_bias if phase==2 else 0.)
        if self.replay_index%4 in (0,3):cost*=1.+self.slot_bias
        if g.replays==1:cost+=self.startup
        self.time+=cost;self.replay_index+=1;self.last='graph'
        for op in g.ops:self.execute(op)
    def function(self,candidate=False,bind=True,capture_fail=False):
        def fn(data,state):
            x=data['x'];shape=(x.shape[0],32)
            out=self.ns['_ng_empty'](shape,dtype=torch.bfloat16,device=x.device) if bind else torch.empty(shape,dtype=torch.bfloat16)
            if not candidate:self.emit('pre',x,out,False)
            if capture_fail and self.capturing is not None:raise ValueError('injected capture failure')
            self.emit('candidate' if candidate else 'base',x,out)
            return out
        return fn


def install(ns,**kwargs):
    backend=Backend(ns,**kwargs)
    ns['_dt_torch']=FakeTorch(backend)
    ns['_K']={'chenwainuo':{'flushext':lambda:backend}}
    ns['_V6_K']={'o':64};ns['_v6_n']=lambda layer,m:32
    draw=[0]
    def inputs(m,k,device,gen):
        draw[0]+=1
        return [torch.full((m,k),float(draw[0]%5+1),dtype=torch.bfloat16)]
    ns['_dt_inputs']=inputs;ns['_dt_check']=lambda d:None
    ns['_NG_BLOCKS']=8  # complete cycles; shipping constant remains 32
    return backend


def evaluate(ns,backend,base=None,candidate=None):
    return ns['_ng_evaluate']('TEST','o',1,'rot',base or backend.function(),candidate or backend.function(True),
                              {},{'o':torch.empty(32,64)},None,None,None,float('inf'),.975)


def assert_graph_contract(result,backend):
    assert backend.begins==backend.ends==len(backend.graphs)
    assert backend.capturing is None and backend.active_stream is backend.main_stream
    assert len(backend.graphs)==24*len(result['records'])
    assert len({g.uid for g in backend.graphs})==len(backend.graphs)
    for draw,record in enumerate(result['records']):
        group=backend.graphs[draw*24:(draw+1)*24]
        assert len({op[2].data_ptr() for g in group for op in g.ops})==1
        assert all(op[2].data_ptr()==record['output_ptr'] for g in group for op in g.ops)
        assert all(op[1].data_ptr()==record['input_ptr'] for g in group for op in g.ops)
        assert record['input_ptr']%65536==record['output_ptr']%65536==0
        assert all(len(g.ops)==2 and all(op[0] in ('pre','base') for op in g.ops) for g in group[:8]+group[16:])
        assert sorted(len(g.ops) for g in group[8:16])==[1]*4+[2]*4
    def no_tensors(value):
        assert not isinstance(value,torch.Tensor)
        if isinstance(value,dict):
            for v in value.values():no_tensors(v)
        elif isinstance(value,(list,tuple)):
            for v in value:no_tensors(v)
    no_tensors(result)


def protocol_gates(ns):
    results=[]
    # Old unbalanced ABBA plus a five-percent edge/capture advantage admits a
    # SAME-KERNEL "win". Output-address equality and balanced slots remove the
    # former; a residual instance/capture bias is rejected, not normalized out.
    old_null=1/1.05
    assert old_null<.975
    b=install(ns,candidate=10.,address_penalty=.7)
    r=evaluate(ns,b)
    assert not r['keep'] and r['reason']=='NO_GAIN'
    assert all(ns['_ng_null_ok'](p) for q in r['records'] for p in (q['pre'],q['post']))
    assert_graph_contract(r,b)
    results.append(dict(case='different-address false win removed by exact output sharing',old_null=14./14.7,
                        new_null=r['records'][0]['pre']['median'],adopt=False))
    b=install(ns,candidate=10.,slot_bias=.05,startup=40.)
    r=evaluate(ns,b)
    assert not r['keep'] and r['reason']=='NO_GAIN'
    assert all(ns['_ng_null_ok'](p) for q in r['records'] for p in (q['pre'],q['post']))
    assert abs(r['records'][0]['candidate']['median']-1)<1e-9
    assert_graph_contract(r,b)
    results.append(dict(case='false old ABBA slot win removed',old_null=old_null,new_null=r['records'][0]['pre']['median'],adopt=r['keep']))
    b=install(ns,candidate=9.,slot_bias=.05,startup=40.)
    r=evaluate(ns,b);assert r['keep'] and len(r['records'])==3
    assert_graph_contract(r,b)
    results.append(dict(case='real gain survives slot bias and first-replay penalty',ratio=r['records'][0]['candidate']['median'],adopt=True))
    b=install(ns,candidate=8.,capture_bias=.05)
    r=evaluate(ns,b)
    assert not r['keep'] and r['reason']=='NULL_REJECT'
    # The crossover aggregate is 1, but raw panels disagree. Mandatory raw-panel
    # checks prevent mathematically averaging away the very bias being audited.
    pre=r['records'][0]['pre'];assert abs(pre['median']-1)<1e-9 and not ns['_ng_null_ok'](pre)
    assert_graph_contract(r,b)
    results.append(dict(case='residual capture bias rejected despite unit aggregate',aggregate=pre['median'],raw=[p['median'] for p in pre['panels']],adopt=False))
    b=install(ns,candidate=9.,post_bias=.04)
    r=evaluate(ns,b)
    assert not r['keep'] and r['reason']=='NULL_REJECT'
    assert ns['_ng_null_ok'](r['records'][0]['pre']) and not ns['_ng_null_ok'](r['records'][0]['post'])
    assert_graph_contract(r,b)
    results.append(dict(case='post-null failure vetoes a fast candidate',adopt=False))
    b=install(ns,candidate=9.9)
    r=evaluate(ns,b);assert not r['keep'] and r['reason']=='NO_GAIN'
    results.append(dict(case='sub-threshold genuine gain rejected',adopt=False))
    b=install(ns,candidate=8.,wrong=True)
    r=evaluate(ns,b);assert not r['keep'] and not r['records'] and 'accuracy' in r['reason']
    assert not b.graphs and ns['_NG_TLS'].lease is None
    results.append(dict(case='fast wrong result rejected before capture',adopt=False))
    b=install(ns,candidate=8.)
    r=evaluate(ns,b,base=b.function(bind=False))
    assert not r['keep'] and '_NGBindingError' in r['reason'] and not b.graphs
    results.append(dict(case='unbound complete incumbent skipped',adopt=False))
    b=install(ns,candidate=8.)
    r=evaluate(ns,b,candidate=b.function(True,capture_fail=True))
    assert not r['keep'] and 'capture failure' in r['reason']
    assert b.capturing is None and b.begins==b.ends and ns['_NG_TLS'].lease is None
    results.append(dict(case='capture failure clears stream and borrowing context',adopt=False))
    b=install(ns,candidate=8.)
    counter=[0]
    def deadline(d):
        counter[0]+=1
        if counter[0]>20:raise ns['_DTBudget']()
    ns['_dt_check']=deadline
    try:evaluate(ns,b)
    except ns['_DTBudget']:pass
    else:raise AssertionError('deadline admitted a partial design')
    assert b.capturing is None and b.begins==b.ends and ns['_NG_TLS'].lease is None
    results.append(dict(case='incomplete deadline design rejected and cleaned',adopt=False))
    ns['_dt_check']=lambda d:None
    return results


def policy_and_wiring_gates(tree,ns):
    def metric(value):
        return dict(median=value,paired=value,panels=[dict(median=value,paired=value,order=o) for o in ('AB','BA','BA','AB')],
                    orders={o:dict(median=value,paired=value) for o in ('AB','BA')})
    assert ns['_ng_null_ok'](metric(.995)) and ns['_ng_null_ok'](metric(1.005))
    assert not ns['_ng_null_ok'](metric(.99499)) and not ns['_ng_null_ok'](metric(1.00501))
    assert ns['_ng_candidate_ok'](metric(.975),.975) and not ns['_ng_candidate_ok'](metric(.97501),.975)
    assert not ns['_ng_null_ok'](metric(float('nan')))
    saved=ns['_ng_evaluate'];calls=[]
    ns.update(_RC_GAIN=.975,_R2_GAIN=.975,_rc_parent_kernel_fn=object(),_r2_parent_kernel_fn=object())
    try:
        def reject(*args):calls.append(args);return dict(keep=False,score=None)
        ns['_ng_evaluate']=reject
        assert ns['_rc_measure']('o',{},None,{},None,None,None,None,None) is False
        assert ns['_r2_measure']('o',8,dict(tag='cg8'),{},None,None,None,None,None) is None
        assert calls[0][0]=='RC' and calls[1][0]=='R2' and calls[0][4] is ns['_rc_parent_kernel_fn'] and calls[1][4] is ns['_r2_parent_kernel_fn']
    finally:ns['_ng_evaluate']=saved
    # Old preparation functions must resolve the repaired names dynamically.
    for name,call in (('_rc_measure','_rc_measure'),('_r2_measure','_r2_measure')):
        preparers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'
                   and any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id==call for c in ast.walk(n))]
        assert len(preparers)==1
    # No override may remain if preparation itself fails.
    target=torch.empty(1)
    def broken(weights):ns['_NG_TLS'].lease={'target':target,'hits':0};raise ValueError('prepare error')
    ns['_ng_parent_prepare']=broken
    with contextlib.redirect_stdout(io.StringIO()):
        try:ns['prepare']({})
        except ValueError:pass
        else:raise AssertionError('prepare error swallowed')
    assert ns['_NG_TLS'].lease is None
    return True


def main():
    torch.set_num_threads(4)
    assert CANDIDATE.read_text().startswith('# okc: problem=muse_bf16_proj\n') and b'\r' not in CANDIDATE.read_bytes()
    tree,ns,sites=setup()
    report=dict(candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),native_cuda_and_parent_logic_preserved=True,
                explicit_runtime_output_sites=sites,static_checks=static_gates(tree),binding=binding_gates(ns),
                scheduling=schedule_gates(ns),protocol=protocol_gates(ns),policy_wiring=policy_and_wiring_gates(tree,ns),
                gpu='NOT RUN: physical cause and H200 null band remain unverified; no GPU/OKC authorized')
    (ROOT/'results/rotcodes-null-offline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
