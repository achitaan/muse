"""Finite adversarial result fixtures for the exact fixed Dense source."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from dense_cluster_resources_evidence import PACKET,DRAFT,TRIAL,PARENT,verify_source,verify_trial,bound_result,validate_packet
from preflight_decoder_cluster_norm import freeze


def write(name,value):freeze(PACKET/name,(json.dumps(value,indent=2)+'\n').encode())


def recalculate(r):
    b=r['result'];loads={}
    for row in b['records']:
        if row.get('event')!='benchmark':continue
        row['median_ns']=row['baseline_ns']/row['speedup']
        a,c=loads.setdefault(row['case']['load'],[0.,0.]);w=row['case']['weight']
        loads[row['case']['load']]=[a+w*row['baseline_ns'],c+w*row['median_ns']]
    b['speedup']=math.prod(a/c for a,c in loads.values())**(1/6)
    b['speedup_ci']=[b['speedup']*.985,b['speedup']*1.015]


def gains(r,changed,controls=1.):
    prior={x['index']:x['speedup'] for x in bound_result(PARENT)['result']['records'] if x.get('event')=='benchmark'}
    for row in r['result']['records']:
        if row.get('event')=='benchmark':row['speedup']=prior[row['index']]*(changed if row['index'] in (20,22) else controls)
    recalculate(r)


def positive_fixture():
    r=copy.deepcopy(bound_result(PARENT));b=r['result'];evidence={}
    for layer,compared in (('qkv_gate',(33,64,65,96,127,128)),('pooler',(65,96,127,128))):
        tested=(33,64,65,127,128) if layer=='qkv_gate' else (65,96,127,128)
        record=dict(oracle=dict(ok=True,cases=[[m,p,.001,.0001,True,.002,.0002] for m in tested for p in (0,1)]),
            paired=dict(ok=True,cases=[]))
        for m in compared:
            for p in (0,1):
                old=[12345.123456789013+i for i in range(7)];new=[x*(.95 if m==128 else .99) for x in old]
                record['paired']['cases'].append(dict(m=m,pattern=p,previous_ns=old,candidate_ns=new,
                    ratio=statistics.median(n/o for n,o in zip(new,old))))
        evidence[layer]=record
    native=[[0,90,0,0,132,36,230544,160,3,108,230544,1,1],
            [1,120,0,0,132,64,215184,160,2,128,215184,1,1]]
    b['log_tail']='DENSE_CLUSTER_RESOURCES '+json.dumps(native)+'\nDENSE_CLUSTER_EVIDENCE '+json.dumps(evidence,separators=(',',':'))+"\nDENSE_CLUSTER_CHOICES ('pooler', 'qkv_gate')\n"
    for row in b['records']:
        if row.get('event')=='profile' and row['index'] in (20,22):
            row['report']='gqx_kernel<128, 1, 8, 3>' if row['index']==20 else 'gq_kernel<128, 1, 8, 2>'
    gains(r,1.5);return r


def change_marker(r,name,fn):
    lines=r['result']['log_tail'].splitlines();i=next(i for i,s in enumerate(lines) if s.startswith(name+' '))
    value=json.loads(lines[i][len(name)+1:]);fn(value);lines[i]=name+' '+json.dumps(value,separators=(',',':'))
    r['result']['log_tail']='\n'.join(lines)+'\n'


def choices(r,value):
    lines=r['result']['log_tail'].splitlines();i=next(i for i,s in enumerate(lines) if s.startswith('DENSE_CLUSTER_CHOICES '))
    lines[i]='DENSE_CLUSTER_CHOICES '+repr(value);r['result']['log_tail']='\n'.join(lines)+'\n'


def profile(r,index,text):next(x for x in r['result']['records'] if x.get('event')=='profile' and x['index']==index)['report']=text


def row_regression(r):
    gains(r,3.)
    row=next(x for x in r['result']['records'] if x.get('event')=='benchmark' and x['index']==20)
    old=next(x for x in bound_result(PARENT)['result']['records'] if x.get('event')=='benchmark' and x['index']==20)
    row['speedup']=old['speedup']*.97;recalculate(r)


def wrong_aggregation(r):
    b=r['result'];b['speedup']=math.prod(x['speedup'] for x in b['records'] if x.get('event')=='benchmark')**(1/30)
    b['speedup_ci']=[b['speedup']*.985,b['speedup']*1.015]


def main():
    proof=verify_source((PACKET/'submission.py').read_bytes());compile((PACKET/'submission.py').read_bytes(),'Dense_resource_candidate','exec')
    write('source-equivalence-v2.json',proof)
    positive=positive_fixture();summary=verify_trial(positive,quiet=True)
    mutants={
        'missing_resources':lambda r:r['result'].update(log_tail=''),
        'duplicate_resources':lambda r:r['result'].update(log_tail=r['result']['log_tail']*2),
        'missing_family':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v.pop()),
        'malformed_fields':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].pop()),
        'boolean_kind':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(0,False)),
        'wrong_kind':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[1].__setitem__(0,0)),
        'too_many_registers':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(1,256)),
        'local_memory':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[1].__setitem__(2,8)),
        'static_shared':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(3,16)),
        'wrong_GPU_size':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(4,131)),
        'qkv_multiple_waves':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(5,35)),
        'pooler_multiple_waves':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[1].__setitem__(5,63)),
        'wrong_dynamic_shared':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(6,230543)),
        'wrong_threads':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(7,256)),
        'wrong_cluster':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(8,4)),
        'wrong_grid':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(9,107)),
        'dynamic_optin_too_small':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(10,230543)),
        'no_resident_block':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[0].__setitem__(11,0)),
        'no_cluster_support':lambda r:change_marker(r,'DENSE_CLUSTER_RESOURCES',lambda v:v[1].__setitem__(12,0)),
        'no_choices':lambda r:choices(r,()),
        'unknown_choice':lambda r:choices(r,('router',)),
        'duplicate_choice':lambda r:choices(r,('pooler','pooler')),
        'oracle_failed':lambda r:change_marker(r,'DENSE_CLUSTER_EVIDENCE',lambda v:v['qkv_gate']['oracle'].update(ok=False)),
        'wrong_worst_row':lambda r:change_marker(r,'DENSE_CLUSTER_EVIDENCE',lambda v:v['pooler']['oracle']['cases'][0].__setitem__(5,.03)),
        'missing_nearby_case':lambda r:change_marker(r,'DENSE_CLUSTER_EVIDENCE',lambda v:v['qkv_gate']['paired']['cases'].pop()),
        'missing_pair_sample':lambda r:change_marker(r,'DENSE_CLUSTER_EVIDENCE',lambda v:v['qkv_gate']['paired']['cases'][0]['previous_ns'].pop()),
        'nonfinite_pair':lambda r:change_marker(r,'DENSE_CLUSTER_EVIDENCE',lambda v:v['pooler']['paired']['cases'][0]['candidate_ns'].__setitem__(0,float('nan'))),
        'forged_pair_median':lambda r:change_marker(r,'DENSE_CLUSTER_EVIDENCE',lambda v:v['qkv_gate']['paired']['cases'][0].update(ratio=.5)),
        'missing_selected_backend':lambda r:profile(r,20,'old cublas'),
        'unexpected_control_backend':lambda r:profile(r,0,'gqx_kernel<128,1,8,3>'),
        'missing_profile':lambda r:r['result'].update(records=[x for x in r['result']['records'] if not(x.get('event')=='profile' and x['index']==29)]),
        'missing_test':lambda r:r['result'].update(records=[x for x in r['result']['records'] if not(x.get('event')=='test' and x['index']==8)]),
        'missing_benchmark':lambda r:r['result'].update(records=[x for x in r['result']['records'] if not(x.get('event')=='benchmark' and x['index']==29)]),
        'integrity_false':lambda r:r['result']['checks'].update(ok=False),
        'integrity_flag':lambda r:r['result']['checks'].update(flagged=['synthetic flag']),
        'wrong_GPU':lambda r:r['result'].update(gpu='B200'),
        'inconsistent_CI':lambda r:r['result'].update(speedup_ci=[.1,.2]),
        'wrong_row_latency':lambda r:next(x for x in r['result']['records'] if x.get('event')=='benchmark').update(median_ns=1.),
        'wrong_row_weight':lambda r:next(x for x in r['result']['records'] if x.get('event')=='benchmark')['case'].update(weight=1),
        'wrong_30row_geomean':wrong_aggregation,
        'no_gain':lambda r:gains(r,1.),
        'controls_explain_projection_gain':lambda r:gains(r,1.04,1.05),
        'controls_explain_whole_track_gain':lambda r:gains(r,1.18,1.02),
        'drifting_controls':lambda r:gains(r,2.,1.10),
        'selected_projection_regression':row_regression,
    }
    rejected=[]
    for name,mutate in mutants.items():
        r=copy.deepcopy(positive);mutate(r)
        try:verify_trial(r,quiet=True)
        except (AssertionError,KeyError,TypeError,ValueError):rejected.append(name)
        else:raise AssertionError('Broken fixture accepted:'+name)
    prior=(ROOT/'scripts/verify_dense_cluster_evidence.py').read_text()
    node=next(n for n in ast.parse(prior).body if isinstance(n,ast.FunctionDef) and n.name=='verify_oracle')
    report=dict(source_sha256=TRIAL,CPU_only=True,GPU_jobs=0,positive_fixture_passed=True,
        synthetic_fixture_is_not_measurement=True,rejected_mutants=rejected,
        prior_oracle_source_sha256=hashlib.sha256(ast.get_source_segment(prior,node).encode()).hexdigest(),
        guard_sha256=hashlib.sha256((ROOT/'scripts/dense_cluster_resources_evidence.py').read_bytes()).hexdigest(),
        synthetic_weighted_gain=summary['aggregate_ratio'])
    write('guard-CPU-admission-v2.json',report)
    assert validate_packet()==(PACKET/'submission.py').read_bytes()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
