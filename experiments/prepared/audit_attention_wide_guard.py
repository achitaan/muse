"""Synthetic result-guard regression cases; none are performance measurements."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from attention_wide_evidence import PARENT,TRIAL,PACKET,bound_result,full_result,verify_trial


def positive(affected=1.12,control=1.):
    result=deepcopy(bound_result(PARENT));body=result['result']
    result['id']='synthetic-guard-fixture-not-a-job'
    body['log_tail']='ATTN_WIDE_RESOURCES 128 0 0 2 84992 83968 128 2 128 128 200 128 132\n'
    loads={}
    for row in body['records']:
        if row['event']=='benchmark':
            factor=affected if row['index'] in (3,5) else control
            row['median_ns']/=factor
            row['speedup']=row['baseline_ns']/row['median_ns']
            a,b=loads.setdefault(row['case']['load'],[0.,0.]);w=row['case']['weight']
            loads[row['case']['load']]=[a+w*row['baseline_ns'],b+w*row['median_ns']]
        elif row['event']=='profile':
            i=row['index'];mode='1,true' if i in (3,5) else '1,false' if i<6 else '8,true'
            row['report']=f'void attn_kernel_wide_l1<{mode},0>(Params)'
    body['speedup']=math.prod(a/b for a,b in loads.values())**(1/6)
    return result


def main():
    base=positive();verify_trial(base,verbose=False)
    changes={}
    def add(name,fn):
        result=deepcopy(base);fn(result);changes[name]=result
    add('missing_test',lambda r:r['result']['records'].pop(next(i for i,x in enumerate(r['result']['records']) if x['event']=='test')))
    add('integrity_failed',lambda r:r['result']['checks'].update(ok=False))
    add('wrong_GPU',lambda r:r['result'].update(gpu='H100'))
    add('missing_resource',lambda r:r['result'].update(log_tail=''))
    add('ambiguous_resource',lambda r:r['result'].update(log_tail=r['result']['log_tail']*2))
    add('local_spills',lambda r:r['result'].update(log_tail=r['result']['log_tail'].replace('128 0 0 2','128 8 0 2')))
    add('one_resident_block',lambda r:r['result'].update(log_tail=r['result']['log_tail'].replace('128 0 0 2','128 0 0 1')))
    add('wrong_shared_size',lambda r:r['result'].update(log_tail=r['result']['log_tail'].replace('84992','84993')))
    add('old_tile_geometry',lambda r:r['result'].update(log_tail=r['result']['log_tail'].replace('83968 128 2','83968 64 3')))
    add('missing_profile',lambda r:r['result']['records'].pop(next(i for i,x in enumerate(r['result']['records']) if x['event']=='profile')))
    add('old_backend',lambda r:[x.update(report='void attn_kernel_fp8pv<1,true,0>(Params)') for x in r['result']['records'] if x['event']=='profile' and x['index']==3])
    add('forged_weight',lambda r:next(x for x in r['result']['records'] if x['event']=='benchmark')['case'].update(weight=60))
    add('forged_aggregate',lambda r:r['result'].update(speedup=r['result']['speedup']*1.2))
    changes['weak_gain']=positive(1.01)
    changes['global_drift_only']=positive(1.12,1.12)
    rejected=[]
    for name,result in changes.items():
        try:verify_trial(result,verbose=False)
        except (AssertionError,KeyError,ValueError,TypeError):rejected.append(name)
        else:raise AssertionError('Guard accepted mutant: '+name)
    report=dict(source_sha256=TRIAL,positive_fixture_passed=True,positive_fixture_synthetic=True,
                rejected_mutants=rejected,GPU_jobs=0,
                fixed_gates=dict(affected_gm_gt=1.04,normalized_affected_gt=1.04,
                                 each_normalized_affected_ge=1.02,control_range=[.95,1.05],track_ratio_gt=1.015))
    data=(json.dumps(report,indent=2)+'\n').encode();path=PACKET/'guard-cpu-audit.json'
    assert not path.exists() or path.read_bytes()==data,'Frozen guard collision'
    path.write_bytes(data)
    print(json.dumps(report))


if __name__=='__main__':main()
