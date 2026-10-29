"""CPU-only host/interface/dispatch audit plus fresh public evaluator identity."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from moe_cluster_x_evidence import CUDA,PARENT,TRIAL,PREFLIGHT,pure_gates


def freeze(name,data):
    path=PREFLIGHT/name
    assert not path.exists() or path.read_bytes()==data, ('Frozen collision',name)
    path.write_bytes(data)


def main():
    raw=(PREFLIGHT/'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    tree=ast.parse(raw)
    cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='_CLUSTER_CUDA' for t in n.targets))
    assert hashlib.sha256(cuda.encode()).hexdigest()==CUDA
    assert 'namespace okc_cluster2 {' in cuda and 'okc::' not in cuda
    host=cuda.split('#include <torch/extension.h>',1)[1]
    assert host.count('okc_cluster2::moe_fused<false>')==4
    assert host.count('okc_cluster2::moe_fused<true>')==4
    assert 'std::vector<int64_t> okc_cluster2_info(int64_t grid)' in host
    assert 'g_cluster2_ready = fields_ok && cluster_support && cooperative_support && ca >= grid / 2 && cb >= grid / 2;' in host
    assert host.count('cudaLaunchKernelEx(')==2 and '<<<' not in host
    assert host.count('cudaLaunchAttributeClusterDimension')==host.count('cudaLaunchAttributeCooperative')==1
    assert 'Cluster2 device changed after prepare' in host and 'g_okc_pdl == 0' in host
    addon=(ROOT/'experiments/prepared/moe_cluster_x_addon.py').read_bytes()
    assert raw.endswith(addon)
    module=next(n for n in ast.parse(addon).body if isinstance(n,ast.FunctionDef) and n.name=='_cx_module')
    text=ast.unparse(module)
    assert "functions=['okc_run', 'okc_set_pdl', 'okc_cluster2_info']" in text
    assert 'std::vector<int64_t> okc_cluster2_info(int64_t grid);' in text
    prep=ast.unparse(next(n for n in ast.parse(addon).body if isinstance(n,ast.FunctionDef) and n.name=='prepare'))
    assert 'module.okc_cluster2_info(state[\'grid\'])' in prep
    assert prep.index('_cx_resources_ok')<prep.index('_cx_case(')
    dispatch=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'][-1]
    called=[]
    def old(data,state):
        called.append('old'); return 'old'
    def new(data,state,module,scratch):
        assert module=='new-module' and scratch=='separate-scratch'
        called.append('new'); return 'new'
    env=dict(_cx_original_kernel=old,_cx_run=new)
    exec(compile(ast.Module([dispatch],type_ignores=[]),'<dispatch-audit>','exec'),env)
    count=0
    for m in range(0,402):
        for enabled in (False,True):
            for prmt in (False,True):
                state=dict(_cx_enabled=enabled,sfx=('-DOKC_PRMTSC=1',) if prmt else (),
                           _cx_module='new-module',_cx_scratch='separate-scratch')
                data=dict(hidden_states=SimpleNamespace(shape=(m,4096)))
                expected='new' if enabled and prmt and m>=64 else 'old'
                called.clear()
                assert env['kernel_fn'](data,state)==expected and called==[expected]
                count+=1
    protected=json.loads((ROOT/'.tools/decoder-package-protected-sha256.json').read_bytes())
    assert len(protected)==44 and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in protected.items())
    report=dict(source_sha256=TRIAL,CUDA_sha256=CUDA,namespace_export_query_binding_exact=True,
                cooperative_cluster_required=True,no_plain_launch_fallback=True,
                dispatch_cases_checked=count,all44protected_exact=True,
                resource_vector_fields=21,actual_runtime_and_graph_validation=False,GPU_jobs=0)
    freeze('host-audit.json',(json.dumps(report,indent=2)+'\n').encode())
    rows=[]
    for name in ('TASK.md','reference.py','checks.py','baseline.py','nvfp4.py','AGENTS.md'):
        pinned=(ROOT/'muse_moe_w4a16'/name).read_bytes()
        fresh=(ROOT/'.tools/fresh-moe-cluster-x-20261029'/name).read_bytes()
        assert pinned==fresh, 'Evaluator changed: '+name
        h=hashlib.sha256(pinned).hexdigest()
        rows.append(dict(file=name,pinned_sha256=h,fresh_sha256=h,exact=True))
    identity=dict(problem='muse_moe_w4a16',files=rows,identity_verified_utc=datetime.now(timezone.utc).isoformat(),
                  fresh_download='Official okc init .tools/fresh-moe-cluster-x-20261029; no GPU')
    ip=PREFLIGHT/'evaluator-identity.json'
    if ip.exists():
        identity['identity_verified_utc']=json.loads(ip.read_bytes())['identity_verified_utc']
    freeze(ip.name,(json.dumps(identity,indent=2)+'\n').encode())
    print(json.dumps(report))


if __name__=='__main__':
    main()
