"""CPU wrapper fallback, signature and fault-injected full-result admission."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from decoder_cluster_norm_evidence import PACKET,PARENT,TRIAL,CHANGED,verify_source,verify_trial,bound_result,validate_packet


def freeze(path,obj):
    raw=(json.dumps(obj,indent=2)+'\n').encode();assert not path.exists() or path.read_bytes()==raw
    path.write_bytes(raw)


def host_proof():
    raw=(PACKET/'submission-v3/submission.py').read_bytes();report=verify_source(raw)
    compile(raw,'cluster_norm_candidate.py','exec')
    addon=ast.parse((ROOT/'experiments/prepared/decoder_cluster_norm_addon.py').read_bytes())
    definitions=[n for n in addon.body if isinstance(n,ast.ClassDef) and n.name=='_CNBoundary' or isinstance(n,ast.FunctionDef) and n.name=='_sfn_boundary']
    namespace=dict(torch=SimpleNamespace(bfloat16='bf16',float32='fp32'),HIDDEN=8192,UNPOOL_SLICE=1024,
        _cn_call=lambda ext,kind,stream,branch,cache:('cluster',kind,None),
        _cn_old_sfn=lambda *args:('original_sfn',None))
    exec(compile(ast.Module(body=definitions,type_ignores=[]),'CPU_wrapper_only','exec'),namespace)
    class Tensor:
        def __init__(self,m,n,dtype,contiguous=True):self.shape=(m,n);self.dtype=dtype;self.contiguous=contiguous
        def is_contiguous(self):return self.contiguous
    old=lambda *args,**kwargs:('original',None,None)
    wrapper=namespace['_CNBoundary'](old,None,{},False)
    fallbacks=0;selected=0
    for m in (0,1,3,8,9,16,31,32,33,64):
        branch=Tensor(m,8192,'bf16');stream=Tensor(m,8192,'fp32')
        value=wrapper(stream,branch,router_dtype='fp32' if m<=8 else None)
        assert value[0]==('cluster' if 1<=m<=32 else 'original')
        selected+=value[0]=='cluster';fallbacks+=value[0]=='original'
    for branch,stream,kw in (
        (Tensor(1,8192,'fp32'),Tensor(1,8192,'fp32'),{}),
        (Tensor(1,8192,'bf16',False),Tensor(1,8192,'fp32'),{}),
        (Tensor(1,8192,'bf16'),Tensor(1,8192,'fp32',False),{}),
        (Tensor(1,1024,'bf16'),Tensor(1,8192,'fp32'),{}),
        (Tensor(1,8192,'bf16'),Tensor(1,8192,'fp32'),dict(router_dtype='fp16')),
        (Tensor(1,8192,'bf16'),Tensor(1,8192,'fp32'),dict(branch_dtype='fp32')),
        (Tensor(1,8192,'bf16'),Tensor(1,8192,'fp32'),dict(unsupported=True))):
        assert wrapper(stream,branch,**kw)[0]=='original';fallbacks+=1
    fn=namespace['_sfn_boundary'];state={'cluster_norm':dict(ext=None,ffn={})}
    for m in (1,3,8,9,16,32):
        value=fn(Tensor(m,8192,'fp32'),Tensor(m,1024,'bf16'),state)
        assert value[0]==('cluster' if m<=8 else 'original_sfn')
        fallbacks+=m>8
    assert fn(Tensor(1,8192,'fp32'),Tensor(1,1024,'bf16'),{'cluster_norm':None})[0]=='original_sfn'
    report.update(candidate_python_compiles=True,unsupported_fallbacks_checked=fallbacks,
        eligible_attention_shapes_checked=selected,sliced_tail_above8_rejected=True,CPU_only=True,GPU_jobs=0)
    freeze(PACKET/'host-CPU-admission.json',report)


def recompute(result):
    loads={}
    for row in result['result']['records']:
        if row.get('event')=='benchmark':
            row['median_ns']=row['baseline_ns']/row['speedup'];w=row['case']['weight']
            a,b=loads.setdefault(row['case']['load'],(0.,0.))
            loads[row['case']['load']]=(a+w*row['baseline_ns'],b+w*row['median_ns'])
    result['result']['speedup']=math.prod(a/b for a,b in loads.values())**(1/6)


def fixture():
    r=copy.deepcopy(bound_result(PARENT));b=r['result'];pairs=[];combined=[]
    for m,kinds in ((1,(0,2)),(8,(0,2)),(16,(1,3)),(32,(1,3))):
        for kind in kinds:
            seed=20261029+m*10+kind
            samples=[dict(pair=i,order=[0,1] if (i+seed)%2==0 else [1,0],old_ns=4000,new_ns=3000,new_over_old=.75) for i in range(10)]
            pairs.append(dict(m=m,kind=kind,old_ns=4000,new_ns=3000,new_over_old=.75,samples=samples))
        combined.append(dict(m=m,old_ns=8000,new_ns=6000,net_saving_ns=2000,new_over_old=.75))
    marker=dict(CN_RESOURCES=[[k,32,0,72,32,132,1,4,256] for k in range(4)],
        CN_PROBES=dict(cases=20,changed_input_graph_replays=8),CN_PAIRS=pairs,
        CN_DECISION=dict(selected=True,combined=combined,statistical_significance_claim=False))
    b['log_tail']='\n'.join(k+' '+json.dumps(v) for k,v in marker.items())+'\n'
    for row in b['records']:
        if row.get('event')=='benchmark' and row['index'] in CHANGED:row['speedup']*=1.05
        if row.get('event')=='profile':
            kinds=(0,2) if row['index'] in (0,1,6,7) else (1,3)
            row['report']='\n'.join(f'void okc_cluster_norm::boundary<{k}>' for k in kinds) if row['index'] in CHANGED else 'sglang::muse_spark_norm_residual_kernel'
    recompute(r);return r


def change_marker(r,name,fn):
    rows=r['result']['log_tail'].splitlines()
    i=next(i for i,s in enumerate(rows) if s.startswith(name+' '))
    value=json.loads(rows[i][len(name)+1:]);fn(value);rows[i]=name+' '+json.dumps(value)
    r['result']['log_tail']='\n'.join(rows)+'\n'


def profile(r,i,value):next(x for x in r['result']['records'] if x.get('event')=='profile' and x['index']==i)['report']=value


def gains(r,affected,control=1.):
    prior={x['index']:x['speedup'] for x in bound_result(PARENT)['result']['records'] if x.get('event')=='benchmark'}
    for x in r['result']['records']:
        if x.get('event')=='benchmark':x['speedup']=prior[x['index']]*(affected if x['index'] in CHANGED else control)
    recompute(r)


def main():
    host_proof();positive=fixture();verify_trial(positive,quiet=True)
    mutants={
        'missing_resource':lambda r:r['result'].update(log_tail=''),
        'duplicate_resource':lambda r:r['result'].update(log_tail=r['result']['log_tail']*2),
        'local_memory':lambda r:change_marker(r,'CN_RESOURCES',lambda x:x[0].__setitem__(2,8)),
        'excess_registers':lambda r:change_marker(r,'CN_RESOURCES',lambda x:x[1].__setitem__(1,128)),
        'wrong_shared_bank':lambda r:change_marker(r,'CN_RESOURCES',lambda x:x[2].__setitem__(3,36)),
        'insufficient_clusters':lambda r:change_marker(r,'CN_RESOURCES',lambda x:x[3].__setitem__(4,16)),
        'unsupported_cluster_launch':lambda r:change_marker(r,'CN_RESOURCES',lambda x:x[0].__setitem__(6,0)),
        'wrong_cluster_geometry':lambda r:change_marker(r,'CN_RESOURCES',lambda x:x[0].__setitem__(7,2)),
        'missing_probe':lambda r:change_marker(r,'CN_PROBES',lambda x:x.update(cases=19)),
        'missing_replay':lambda r:change_marker(r,'CN_PROBES',lambda x:x.update(changed_input_graph_replays=0)),
        'family_not_selected':lambda r:change_marker(r,'CN_DECISION',lambda x:x.update(selected=False)),
        'missing_pair':lambda r:change_marker(r,'CN_PAIRS',lambda x:x.pop()),
        'pair_order_not_alternated':lambda r:change_marker(r,'CN_PAIRS',lambda x:x[0]['samples'][1].update(order=x[0]['samples'][0]['order'])),
        'fabricated_pair_median':lambda r:change_marker(r,'CN_PAIRS',lambda x:x[0].update(old_ns=9000)),
        'nonfinite_sample':lambda r:change_marker(r,'CN_PAIRS',lambda x:x[0]['samples'][0].update(old_ns=float('nan'))),
        'missing_actual_kernel':lambda r:profile(r,0,'sglang::muse_spark_norm_residual_kernel'),
        'partial_boundary_only':lambda r:profile(r,2,'okc_cluster_norm::boundary<1>'),
        'old_and_new_both':lambda r:profile(r,7,'okc_cluster_norm::boundary<0> okc_cluster_norm::boundary<2> _sfn_kernel'),
        'changed_control':lambda r:profile(r,8,'okc_cluster_norm::boundary<1>'),
        'control_backend_missing':lambda r:profile(r,10,'unknown backend'),
        'missing_test':lambda r:r['result'].update(records=[x for x in r['result']['records'] if not(x.get('event')=='test' and x['index']==12)]),
        'integrity_false':lambda r:r['result']['checks'].update(ok=False),
        'wrong_GPU':lambda r:r['result'].update(gpu='B200'),
        'no_gain':lambda r:gains(r,1.),
        'controls_explain_gain':lambda r:gains(r,1.02,1.03),
        'drifting_controls':lambda r:gains(r,1.15,1.10),
        'wrong_weighted_aggregate':lambda r:r['result'].update(speedup=99.),
    }
    rejected=[]
    for name,fn in mutants.items():
        r=copy.deepcopy(positive);fn(r)
        try:verify_trial(r,quiet=True)
        except (AssertionError,KeyError,ValueError,TypeError):rejected.append(name)
        else:raise AssertionError('Broken result fixture accepted:'+name)
    report=dict(source_sha256=TRIAL,CPU_only=True,GPU_jobs=0,positive_fixture_passed=True,
        rejected_mutants=rejected,synthetic_fixture_is_not_a_measurement=True,
        guard_sha256=hashlib.sha256((ROOT/'scripts/decoder_cluster_norm_evidence.py').read_bytes()).hexdigest())
    freeze(PACKET/'guard-CPU-admission.json',report)
    assert validate_packet()==(PACKET/'submission-v3/submission.py').read_bytes()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
