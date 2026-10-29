"""Synthetic result fixtures verify rejection gates; no performance evidence."""
import base64
import copy
import json
import math
from pathlib import Path
import sys
import zlib

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from decoder_cooperative_evidence import PACKET,PARENT,TRIAL,CHANGED,bound_result,verify_trial


def fixture():
    result=copy.deepcopy(bound_result(PARENT));body=result['result']
    logs=[]
    for family in (1,8):
        for fmt in (0,1):
            q={str(i):v for i,v in enumerate([1,48,0,0,1 if family==1 else 4,132,1,16384 if family==1 else 4096],1)}
            logs.append('CB_RESOURCES '+json.dumps(dict(family=family,fmt=fmt,query=q)))
            cases=[dict(m=m,fixture=k,errors=[[0.,0.]]*4)
                   for m in ((1,) if family==1 else range(2,9)) for k in ('random','zero','outlier','tiny')]
            cases.extend(dict(m=family,fixture=f'fresh_graph_replay_{i}',errors=[[0.,0.]]*4) for i in range(4))
            payload=base64.b64encode(zlib.compress(json.dumps(cases).encode())).decode()
            logs.append('CB_PROBES_Z '+json.dumps(dict(family=family,fmt=fmt,cases=len(cases),payload=payload)))
        samples=[dict(pair=i,order=[0,1] if i%2==0 else [1,0],old=100.,new=80.) for i in range(16)]
        logs.append('CB_PAIR '+json.dumps(dict(family=family,fmt=1,gain=1.25,wins=16,keep=True,samples=samples)))
    logs.append('CB_SELECTED [1, 8]')
    body['log_tail']='\n'.join(logs)
    for record in body['records']:
        if record['event']=='benchmark':
            if record['index'] in CHANGED: record['speedup']*=1.06
            record['median_ns']=record['baseline_ns']/record['speedup']
        if record['event']=='profile' and record['index'] in CHANGED:
            record['report']+='\ncb_smallm2_gemv' if record['index']<2 else '\ncb_v18s_gemv'
    loads={}
    for r in body['records']:
        if r['event']=='benchmark':
            a,b=loads.setdefault(r['case']['load'],[0.,0.]);w=r['case']['weight']
            loads[r['case']['load']]=[a+w*r['baseline_ns'],b+w*r['median_ns']]
    body['speedup']=math.prod(a/b for a,b in loads.values())**(1/6)
    return result


def mutate_marker(r,prefix,fn):
    lines=r['result']['log_tail'].splitlines()
    for i,line in enumerate(lines):
        if line.startswith(prefix+' '):
            value=json.loads(line[len(prefix)+1:]);fn(value)
            lines[i]=prefix+' '+json.dumps(value);break
    r['result']['log_tail']='\n'.join(lines)


def main():
    positive=fixture();verify_trial(positive,quiet=True)
    def no_last_test(r): r['result']['records']=[x for x in r['result']['records'] if not(x['event']=='test' and x['index']==12)]
    def no_changed_profile(r):
        for row in r['result']['records']:
            if row['event']=='profile' and row['index']==0: row['report']='retained only'
    def missing_replay(r):
        def change(p):
            cases=json.loads(zlib.decompress(base64.b64decode(p['payload'])))
            cases[-1]['fixture']='random'
            p['payload']=base64.b64encode(zlib.compress(json.dumps(cases).encode())).decode()
        mutate_marker(r,'CB_PROBES_Z',change)
    def wrong_norm(r):
        def change(p):
            cases=json.loads(zlib.decompress(base64.b64decode(p['payload'])))
            cases[0]['errors'][1]=[.01,.01]
            p['payload']=base64.b64encode(zlib.compress(json.dumps(cases).encode())).decode()
        mutate_marker(r,'CB_PROBES_Z',change)
    def fallback_only(r): mutate_marker(r,'CB_SELECTED',lambda x:x.clear())
    def unknown_case(r):
        next(x for x in r['result']['records'] if x['event']=='test')['case']['ctx']=12345
    def no_affected_gain(r):
        for row in r['result']['records']:
            if row['event']=='benchmark' and row['index'] in CHANGED:
                row['speedup']/=1.06;row['median_ns']=row['baseline_ns']/row['speedup']
        r['result']['speedup']=bound_result(PARENT)['result']['speedup']
    mutants={
      'missing_test':no_last_test,'new_path_missing':no_changed_profile,'replay_not_fresh':missing_replay,
      'numerical_boundary_wrong':wrong_norm,'empty_selection':fallback_only,'wrong_case_identity':unknown_case,
      'no_affected_gain':no_affected_gain,
      'integrity_false':lambda r:r['result']['checks'].update(ok=False),
      'local_memory_nonzero':lambda r:mutate_marker(r,'CB_RESOURCES',lambda x:x['query'].update({'3':16})),
      'residency_too_small':lambda r:mutate_marker(r,'CB_RESOURCES',lambda x:x['query'].update({'6':1})),
      'cooperative_unsupported':lambda r:mutate_marker(r,'CB_RESOURCES',lambda x:x['query'].update({'7':0})),
      'component_tie':lambda r:mutate_marker(r,'CB_PAIR',lambda x:x.update(gain=1.0)),
      'paired_order_wrong':lambda r:mutate_marker(r,'CB_PAIR',lambda x:x['samples'][1].update(order=[0,1])),
      'aggregate_not_weighted':lambda r:r['result'].update(speedup=99.),
      'wrong_device':lambda r:r['result'].update(gpu='B200'),
    }
    rejected=[]
    for name,mutation in mutants.items():
        candidate=copy.deepcopy(positive);mutation(candidate)
        try: verify_trial(candidate,quiet=True)
        except (AssertionError,ValueError,KeyError,TypeError): rejected.append(name)
        else: raise AssertionError('Guard accepted broken fixture: '+name)
    report=dict(source_sha256=TRIAL,CPU_only=True,positive_fixture_passed=True,
        rejected_mutants=rejected,GPU_jobs=0,synthetic_fixture_is_not_measurement=True)
    path=PACKET/'guard-CPU-admission.json';data=(json.dumps(report,indent=2)+'\n').encode()
    assert not path.exists() or path.read_bytes()==data
    path.write_bytes(data);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
