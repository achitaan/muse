"""CPU mock of the diagnostic driver; timing values are deliberately synthetic."""
import contextlib,importlib.util,io,json,sys,types
from pathlib import Path
from unittest.mock import patch
import torch
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('fd2',ROOT/'experiments/boards/diag/fsdiag2.py')
md=importlib.util.module_from_spec(sp);sp.loader.exec_module(md)
rawgen=md.generate_input
inputs={m:rawgen(m,'normal',m,'cpu') for m in (1,8)}
state=dict(w=[torch.zeros(1)],scratch=[torch.zeros(1)],lo=0,cnt=32,grid=132)
def run(data,state,fs):
    return fs['ext'].fs_run(None,None,fs['scratch'],data['hidden_states'],data['topk_ids'],data['topk_weights'],torch.empty_like(data['hidden_states']),0,132)
def met(a,b):return dict(rel_L2=0.,row_L2=0.,ok=torch.equal(a,b))
ns=dict(_moe_sel_original_prepare=lambda w:state,_fs_run=run,_fs_metrics=met,
        _fs_state=lambda w,s:dict(scratch=[torch.zeros(3),torch.zeros(2)],ext=None))
bundle=types.SimpleNamespace()
for v in ('TINY','TINY2')+md.VARIANTS2:
    setattr(bundle,v+'_info',lambda:[151,168,0,80,218752])
    def call(*args,v=v):
        if v in ('TINY','TINY2'):out=args[5]
        else:out=args[6]
        out.fill_(1)
        if v=='TL':
            trace=args[2][2]
            for c in range(132):
                for w in range(9):
                    off=100000+c*10
                    for i,t in {0:0,1:2000,2:4000,4:10000,5:11000,6:14000,16:6000,17:8000,144:8000,145:10000}.items():trace[c,w,i]=off+t
                    trace[c,w,7:11]=torch.tensor([1,c,c+1,1])
        return out
    setattr(bundle,v+'_run',call)
def inject(code,glob):glob.update(ns)
origzeros=torch.zeros
def zeros(size,*a,**kw):
    kw.pop('device',None)
    return origzeros(1 if isinstance(size,int) and size>1000 else size,*a,**kw)
reps=[]
def bench(fn,flush,n):reps.append(n);fn();return 10.,9.
with contextlib.ExitStack() as st:
    for obj,key,val in [(md,'build_bundle2',lambda n:bundle),(md,'generate_weights',lambda d:{}),
        (md,'generate_input',lambda m,dist,seed,dev:inputs[m]),(md,'bench',bench),
        (torch.cuda,'get_device_capability',lambda dev:(9,0)),(torch.cuda,'get_device_name',lambda dev:'MOCK'),
        (torch.cuda,'synchronize',lambda:None),(torch,'zeros',zeros)]:st.enter_context(patch.object(obj,key,val))
    st.enter_context(patch.object(md,'exec',inject,create=True))
    buf=io.StringIO();st.enter_context(contextlib.redirect_stdout(buf));md.run_all()
text=buf.getvalue();assert 'FS2_ERR' not in text,text
assert text.count('FS2_TIME')==24 and reps==[40]*24
assert text.count('FS2_TL R')==4 and text.count('FS2_TASK R')==4
lines=[line.split() for line in text.splitlines() if line.startswith('FS2_TIME')]
order=['BASE','FS','NOB','NOFLUSH','NOA','NOACC']
expected=[(name,f'p={r}',v) for r in (1,2) for name in (('R1','R8') if r==1 else ('R8','R1')) for v in (order if r==1 else order[::-1])]
assert [(s[1],s[2],s[3]) for s in lines]==expected
assert len(text.encode())+2000<9400  # reserve longer real per-task trace lines
assert not md._FS2_LOG_LIMIT
(ROOT/'.fs_offline/diag2/mock.log').write_text(text)
print('PASS24 timings*40reps, both orders,4 TL+task summaries; log bytes',len(text.encode()),'; CPU mock only')
# Output cap is an enforced bound, not only an estimated normal-run size.
with contextlib.redirect_stdout(io.StringIO()) as buf:
    md._FS2_LOG_BYTES=0;md._FS2_LOG_LIMIT=False
    for _ in range(100):md.log2('FS2_TEST '+'x'*550)
assert len(buf.getvalue().encode())<10000 and buf.getvalue().count('log_budget_reached')==1
assert md.compiler_errors(RuntimeError())=='no compiler error lines; '
print('PASS hard log bound and empty-exception formatting')
