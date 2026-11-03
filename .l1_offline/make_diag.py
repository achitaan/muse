"""moediag7 packaging/pipeline, two variants: exact moe_b13 BASE and L1."""
from pathlib import Path
import ast,base64,hashlib,zlib
ROOT=Path(__file__).resolve().parents[1]
t=(ROOT/'experiments/boards/diag/moediag7.py').read_text()
def assign(name,value):
    global t
    n=next(n for n in ast.parse(t).body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id==name for v in n.targets))
    lines=t.splitlines(True);t=''.join(lines[:n.lineno-1])+name+' = '+repr(value)+'\n'+''.join(lines[n.end_lineno:])
def function(name,new):
    global t
    n=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=t.splitlines(True);t=''.join(lines[:n.lineno-1])+new+'\n'+''.join(lines[n.end_lineno:])
raw={v:(ROOT/('experiments/boards/moe/'+p)).read_bytes() for v,p in [('BASE','moe_b13.py'),('L1','moe_l1.py')]}
cuda={v:ast.literal_eval(next(n.value for n in ast.parse(b).body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CUDA_SRC' for x in n.targets))).encode() for v,b in raw.items()}
assign('INCUMBENT_B85',base64.b85encode(zlib.compress(raw['BASE'],9)).decode())
assign('INCUMBENT_SHA256',hashlib.sha256(raw['BASE']).hexdigest())
assign('SOURCE_SHA',{v:hashlib.sha256(b).hexdigest() for v,b in raw.items()})
assign('CUDA_SHA',{v:hashlib.sha256(b).hexdigest() for v,b in cuda.items()})
assign('CUDA_B85',{v:base64.b85encode(zlib.compress(b,9)).decode() for v,b in cuda.items()})
assign('VARIANTS',('BASE','L1'))
function('variants_for','def variants_for(kind):\n    return ("BASE", "L1")\n')
function('error','''def error(where, exc):
    lines=[s.strip() for s in str(exc).splitlines() if 'error' in s.lower() and not s.lower().startswith('error building extension')]
    detail=' | '.join(lines)[:600] if lines else str(exc)[:200]
    log(f'MD_ERR {where} {type(exc).__name__}: {detail}')
''')
n=next(n for n in ast.parse(t).body if isinstance(n,ast.FunctionDef) and n.name=='run_all')
run=ast.get_source_segment(t,n)
run=run.replace('v=6','v=L1').replace(' HINT={SOURCE_SHA[\'HINT\'][:12]} ILV={SOURCE_SHA[\'ILV\'][:12]}',' L1={SOURCE_SHA[\'L1\'][:12]}')
run=run.replace('pass1=cases_forward_BASE_HINT_ILV pass2=cases_reverse_ILV_HINT_BASE;', 'pass1=cases_forward_BASE_L1 pass2=cases_reverse_L1_BASE;')
run=run.replace('same_units_ring_math; eq=bytes rel=elementwise_floor_1e-12; splitK_grouping_may_change',
                'same_units_K_MMA_order; L1=640threads_112consumer_32producer_stagecapture; eq=bytes rel=elementwise_floor_1e-12; zero_disabled')
run=run.replace("state=ns['prepare'](raw);del raw", "state=ns['_moe_sel_original_prepare'](raw);del raw\n    ns['_MOE_SEL'].clear()")
run=run.replace('TUs=3','TUs=2').replace("('BASE','HINT','ILV')","('BASE','L1')").replace("('ILV','HINT','BASE')","('L1','BASE')")
a=run.index('    for name,*_ in entries:');b=run.index('    if False:',a)
run=run[:a]+'''    for name,*_ in entries:
        if all((name,v,p) in times for v in ('BASE','L1') for p in (1,2)):
            r=[times[name,'L1',p]/times[name,'BASE',p] for p in (1,2)]
            log(f'MD_SUM case={name} L1_BASE={r[0]:.4f}/{r[1]:.4f}')
'''+run[b:]
run=run.replace("calls['ILV']", "calls['L1']").replace('var=ILV','var=L1').replace('BASE_HINT_not_run','BASE_not_run')
function('run_all',run)
t=t.replace('moediag_v6_', 'moediag_l1_')
n=ast.parse(t).body[0]
if isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str):
    lines=t.splitlines(True)
    t='''"""DIAGNOSTIC ONLY: moe_b13 BASE vs MAIN-only moe_l1.
R32/R128/R128_NODUP/R256/R256_DUP3/R256_NODUP,40 cold-L2 samples,
both case and variant orders, byte equality/relative L2. Zero routing disabled.
640 threads, producer WG32/consumer WGs112, full packed-stage early release.
Two CUDA TUs in one thin-host bundle, pinned CUDA_B85/CUDA_SHA/SOURCE_SHA.
No local GPU result is implied by offline resource or CPU protocol tests.
"""\n'''+''.join(lines[n.end_lineno:])
ast.parse(t)
p=ROOT/'experiments/boards/diag/moediag_l1.py';p.write_bytes(t.encode())
print(p,len(t.encode()),hashlib.sha256(t.encode()).hexdigest())
