"""Fault-injected result fixtures; none is a GPU timing measurement."""
import copy
import json
import math
from pathlib import Path
import sys
import hashlib

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from attention_two_score_evidence import PACKET,TRIAL,PARENT,CHANGED,bound_result,verify_trial,validate_packet


def recompute(result):
    loads={}
    for row in result['result']['records']:
        if row['event']=='benchmark':
            row['median_ns']=row['baseline_ns']/row['speedup']
            a,b=loads.setdefault(row['case']['load'],[0.,0.]);w=row['case']['weight']
            loads[row['case']['load']]=[a+w*row['baseline_ns'],b+w*row['median_ns']]
    result['result']['speedup']=math.prod(a/b for a,b in loads.values())**(1/6)


def fixture():
    result=copy.deepcopy(bound_result(PARENT));body=result['result']
    body['log_tail']='ATTN_TWO_SCORE_RESOURCES 80 0 0 2 168 0 0 1 114688 132 384 4 32 232\n'
    for row in body['records']:
        if row['event']=='benchmark' and row['index'] in CHANGED:row['speedup']*=1.1
        if row['event']=='profile':
            row['report']='void attn_kernel_two_score<8,true,0>' if row['index'] in CHANGED else 'void attn_kernel_fp8pv<1,true,0>'
    recompute(result)
    return result


def resource(result,index,value):
    fields=result['result']['log_tail'].split();fields[index+1]=str(value)
    result['result']['log_tail']=' '.join(fields)+'\n'


def profile(result,index,value):
    next(r for r in result['result']['records'] if r['event']=='profile' and r['index']==index)['report']=value


def set_gains(result,affected,controls=1.):
    old={r['index']:r['speedup'] for r in bound_result(PARENT)['result']['records'] if r['event']=='benchmark'}
    for row in result['result']['records']:
        if row['event']=='benchmark':
            gain=affected.get(row['index'],controls)
            row['speedup']=old[row['index']]*gain
    recompute(result)


def main():
    positive=fixture();verify_trial(positive,verbose=False)
    truncated=copy.deepcopy(positive)
    for row in truncated['result']['records']:
        if row['event']=='profile' and row['index'] in CHANGED:
            row['report']=row['report'].replace('<8,true,0>','<8,true,0...')
    verify_trial(truncated,verbose=False)
    mutations={
      'missing_resource':lambda r:r['result'].update(log_tail=''),
      'duplicate_resource':lambda r:r['result'].update(log_tail=r['result']['log_tail']*2),
      'old_occupancy_unproven':lambda r:resource(r,3,1),
      'wrong_register_pool':lambda r:resource(r,4,160),
      'new_local_memory':lambda r:resource(r,5,8),
      'new_static_memory':lambda r:resource(r,6,1024),
      'wrong_new_occupancy':lambda r:resource(r,7,2),
      'wrong_dynamic_memory':lambda r:resource(r,8,112640),
      'wrong_SM_count':lambda r:resource(r,9,80),
      'wrong_threads':lambda r:resource(r,10,256),
      'wrong_ring_depth':lambda r:resource(r,11,3),
      'wrong_producer_budget':lambda r:resource(r,12,56),
      'wrong_consumer_budget':lambda r:resource(r,13,152),
      'missing_pipeline':lambda r:profile(r,9,'attn_kernel_fp8pv<8,true,0>'),
      'duplicate_old_new':lambda r:profile(r,9,'attn_kernel_two_score<8,true,0> attn_kernel_fp8pv<8,true,0>'),
      'rejected_finalizer':lambda r:profile(r,11,'attn_kernel_two_score<8,true,0> attn_external_finalize8'),
      'changed_control':lambda r:profile(r,8,'attn_kernel_two_score<8,true,0>'),
      'missing_control':lambda r:profile(r,0,'unrelated fallback'),
      'missing_test':lambda r:r['result'].update(records=[x for x in r['result']['records'] if not(x['event']=='test' and x['index']==15)]),
      'integrity_false':lambda r:r['result']['checks'].update(ok=False),
      'wrong_GPU':lambda r:r['result'].update(gpu='B200'),
      'no_gain':lambda r:set_gains(r,{9:1.,11:1.}),
      'controls_explain_gain':lambda r:set_gains(r,{9:1.03,11:1.03},1.04),
      'weak_case':lambda r:set_gains(r,{9:1.1,11:1.005}),
      'wrong_weighted_aggregate':lambda r:r['result'].update(speedup=99.),
    }
    rejected=[]
    for name,mutation in mutations.items():
        result=copy.deepcopy(positive);mutation(result)
        try:verify_trial(result,verbose=False)
        except (AssertionError,ValueError,KeyError,TypeError):rejected.append(name)
        else:raise AssertionError('Broken fixture accepted: '+name)
    report=dict(revision=1,source_sha256=TRIAL,CPU_only=True,GPU_jobs=0,positive_fixture_passed=True,
                source_bound_actual_truncated_spelling_passed=True,
                guard_sha256=hashlib.sha256((ROOT/'scripts/attention_two_score_evidence.py').read_bytes()).hexdigest(),
                rejected_mutants=rejected,synthetic_fixture_is_not_measurement=True)
    path=PACKET/'guard-CPU-audit.json';data=(json.dumps(report,indent=2)+'\n').encode()
    assert not path.exists() or path.read_bytes()==data
    path.write_bytes(data)
    assert validate_packet()==(PACKET/'admission-v3/submission.py').read_bytes()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
