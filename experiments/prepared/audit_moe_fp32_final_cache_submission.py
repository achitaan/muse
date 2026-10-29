"""Exact query/host binding, shape dispatch and evaluator identity; CPU only."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from moe_fp32_final_cache_evidence import PREFLIGHT,TRIAL,CUDA


def main():
    raw=(PREFLIGHT/'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    tree=ast.parse(raw)
    constants={t.id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
               for t in n.targets if isinstance(t,ast.Name) and t.id in ('_F32CACHE_QUERY','_F32CACHE_CUDA')}
    cuda,query=constants['_F32CACHE_CUDA'],constants['_F32CACHE_QUERY']
    assert hashlib.sha256(cuda.encode()).hexdigest()==CUDA
    assert 'namespace okc_f32cache {' in cuda and 'okc::' not in cuda
    assert query.count('okc_f32cache::')==9 and 'okc_epair' not in query
    assert query.count('okc_f32cache::moe_fused<false>')==3 and query.count('okc_f32cache::moe_fused<true>')==3
    assert 'std::vector<int64_t> okc_f32cache_info()' in query
    module=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_f32cache_module')
    text=ast.unparse(module)
    assert "functions=['okc_run', 'okc_set_pdl', 'okc_f32cache_info']" in text
    assert 'std::vector<int64_t> okc_f32cache_info();' in text and 'module.okc_set_pdl(0)' in text
    prep=ast.unparse([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'][-1])
    assert 'module.okc_f32cache_info()' in prep
    assert prep.index('_f32cache_resources_ok')<prep.index('_f32cache_case(')
    k=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'][-1]
    calls=[]
    def old(data,state):calls.append('old');return 'old'
    def new(data,state,module,scratch):
        assert module=='new-module' and scratch=='private-scratch'
        calls.append('new');return 'new'
    env=dict(_f32cache_original_kernel=old,_f32cache_run=new,MID_MAX_M=16)
    exec(compile(ast.Module([k],type_ignores=[]),'<dispatch-audit>','exec'),env)
    count=0
    for m in range(402):
        for enabled in (False,True):
            for prmt in (False,True):
                state=dict(_f32cache_enabled=enabled,sfx=('-DOKC_PRMTSC=1',) if prmt else (),
                           _f32cache_module='new-module',_f32cache_scratch='private-scratch')
                data=dict(hidden_states=SimpleNamespace(shape=(m,4096)))
                expected='new' if enabled and prmt and m>16 else 'old'
                calls.clear();assert env['kernel_fn'](data,state)==expected and calls==[expected];count+=1
    protected=json.loads((ROOT/'.tools/decoder-package-protected-sha256.json').read_bytes())
    assert len(protected)==44 and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in protected.items())
    report=dict(source_sha256=TRIAL,CUDA_sha256=CUDA,namespace_export_query_binding_exact=True,
                dispatch_cases_checked=count,all44protected_exact=True,query_namespace_references=9,
                fresh_outputs=True,separate_scratch=True,extra_weight_layouts=0,
                actual_host_NVCC_and_runtime_resources_verified=False,GPU_jobs=0)
    def freeze(name,data):
        p=PREFLIGHT/name
        assert not p.exists() or p.read_bytes()==data,('Frozen collision',name)
        p.write_bytes(data)
    freeze('host-audit.json',(json.dumps(report,indent=2)+'\n').encode())
    rows=[]
    for name in ('TASK.md','reference.py','checks.py','baseline.py','nvfp4.py','AGENTS.md'):
        pinned=(ROOT/'muse_moe_w4a16'/name).read_bytes()
        fresh=(ROOT/'.tools/fresh-moe-fp32-cache-20261029'/name).read_bytes()
        assert pinned==fresh,'Official identity changed: '+name
        h=hashlib.sha256(pinned).hexdigest();rows.append(dict(file=name,pinned_sha256=h,fresh_sha256=h,exact=True))
    identity=dict(problem='muse_moe_w4a16',files=rows,identity_verified_utc=datetime.now(timezone.utc).isoformat(),
                  fresh_download='Official okc init .tools/fresh-moe-fp32-cache-20261029; no GPU job')
    ip=PREFLIGHT/'evaluator-identity.json'
    if ip.exists():identity['identity_verified_utc']=json.loads(ip.read_bytes())['identity_verified_utc']
    freeze(ip.name,(json.dumps(identity,indent=2)+'\n').encode())
    print(json.dumps(report))


if __name__=='__main__':main()
