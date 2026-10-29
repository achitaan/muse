"""Artificial fixtures verify rejection coverage; no GPU jobs or measurements."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from moe_cluster_x_evidence import CUDA,PARENT,TRIAL,PREFLIGHT,SHAPES,bound_result,verify_trial,pure_gates


def main():
    cases=[]
    for m,mode in [(m,'normal') for m in SHAPES]+[(128,'skewed')]:
        old=[100000.123456789+i*64 for i in range(10)]
        cases.append(dict(m=m,mode=mode,synthetic=True,seed=20261030+m+(10000 if mode=='skewed' else 0),
                          old_ns=old,new_ns=[v*.95 for v in old],arm_orders=[[0,1]]*5+[[1,0]]*5,
                          median_ratio=.95,relative=0.,worst=0.,
                          changed_input_graph_replays=[[0.,0.],[0.,0.]] if m in (128,385) and mode=='normal' else []))
    analytic=[dict(m=m,mode='remote',passed=True) for m in (64,128,384,385)]
    analytic.append(dict(m=128,mode='duplicate',passed=True,relative=0.,worst=0.))
    e=dict(cuda_sha256=CUDA,synthetic_only=True,independent_jobs=1,repeats_are_not_independent=True,
           selected_main=True,resources=[96,96,0,0,0,0,231168,576,1,1,66,66,2,1,1,2,1,1,132,1,20261029],grid=132,
           cases=cases,analytic=analytic,extra_weight_layouts=0,separate_scratch=True,
           original_small_and_ineligible_scale_paths=True,minimum_timed_M=64,prepare_s=240.)
    baseline=bound_result(PARENT)
    fixture=copy.deepcopy(baseline)
    fixture['id']='CPU-FIXTURE-NOT-A-JOB'
    fixture['result']['speedup']*=1.05**(1/3)
    for r in fixture['result']['records']:
        if r.get('event')=='benchmark' and r['index'] in (4,5):
            r['speedup']*=1.05
        elif r.get('event')=='profile' and r['index'] in (4,5):
            r['report']='void okc_cluster2::moe_fused<false>(params) CPU fixture only'
    marker='CLUSTER_X_EVIDENCE '
    fixture['result']['log_tail']=marker+json.dumps(e,separators=(',',':'))
    assert len(fixture['result']['log_tail'])<14500
    with contextlib.redirect_stdout(io.StringIO()):
        verify_trial(fixture)
    mutations=[]
    def changed(name,edit):
        b=copy.deepcopy(fixture); ee=copy.deepcopy(e); edit(ee)
        b['result']['log_tail']=marker+json.dumps(ee)
        mutations.append((name,b))
    changed('empty_choice',lambda x:x.update(selected_main=False))
    changed('wrong_CUDA',lambda x:x.update(cuda_sha256='0'*64))
    changed('actual_local_memory',lambda x:x['resources'].__setitem__(2,8))
    changed('insufficient_cluster_residency',lambda x:x['resources'].__setitem__(10,65))
    changed('cooperative_unsupported',lambda x:x['resources'].__setitem__(9,0))
    changed('wrong_cluster_dimension',lambda x:x['resources'].__setitem__(12,1))
    changed('wrong_grid_binding',lambda x:x['resources'].__setitem__(18,130))
    changed('not_admitted',lambda x:x['resources'].__setitem__(19,0))
    changed('missing_nearby',lambda x:x['cases'].pop())
    changed('forged_pairs',lambda x:x['cases'][0].update(median_ratio=.8))
    changed('missing_changed_graph',lambda x:next(c for c in x['cases'] if c['m']==385).update(changed_input_graph_replays=[]))
    changed('bad_numerical_error',lambda x:x['cases'][8].update(worst=.03))
    changed('bad_remote_zero',lambda x:x['analytic'][0].update(passed=False))
    changed('bad_duplicate_error',lambda x:x['analytic'][-1].update(relative=.01))
    changed('preparation_over_budget',lambda x:x.update(prepare_s=301))
    b=copy.deepcopy(fixture); b['result']['log_tail']+='\n'+b['result']['log_tail']; mutations.append(('ambiguous_marker',b))
    b=copy.deepcopy(fixture)
    next(r for r in b['result']['records'] if r.get('event')=='profile' and r['index']==4)['report']='old backend'
    mutations.append(('new_backend_absent',b))
    b=copy.deepcopy(fixture)
    next(r for r in b['result']['records'] if r.get('event')=='profile' and r['index']==2)['report']='okc_cluster2::moe_fused<false>'
    mutations.append(('M32_control_changed',b))
    b=copy.deepcopy(fixture);b['result']['speedup']=baseline['result']['speedup']*1.01
    mutations.append(('full_track_gain_missing',b))
    b=copy.deepcopy(fixture)
    for r in b['result']['records']:
        if r.get('event')=='benchmark' and r['index'] in (0,1,2,3):r['speedup']*=1.04
    mutations.append(('control_normalized_gain_missing',b))
    b=copy.deepcopy(fixture);b['result']['checks']['ok']=False;mutations.append(('integrity_failed',b))
    b=copy.deepcopy(fixture);b['result']['evaluator']['files']['problems/muse_moe_w4a16/reference.py']='0'*64
    mutations.append(('evaluator_changed',b))
    rejected=[]
    for name,b in mutations:
        try:
            with contextlib.redirect_stdout(io.StringIO()):verify_trial(b)
        except (AssertionError,KeyError,TypeError,ValueError):rejected.append(name)
        else:raise AssertionError(('Malformed fixture accepted',name))
    # Runtime decision must reject the same physical capacity defect before launch.
    gates=pure_gates()
    assert gates['_cx_decision'](cases,e['resources'],132,analytic)
    bad=e['resources'].copy();bad[11]=65
    assert not gates['_cx_decision'](cases,bad,132,analytic)
    report=dict(source_sha256=TRIAL,positive_fixture_passed=True,rejected=rejected,
                fixtures_are_not_measurements=True,GPU_jobs=0,evidence_fixture_bytes=len(fixture['result']['log_tail']))
    target=PREFLIGHT/'guard-cpu-audit.json';data=(json.dumps(report,indent=2)+'\n').encode()
    assert not target.exists() or target.read_bytes()==data,'Frozen guard collision'
    target.write_bytes(data)
    print(json.dumps(report))


if __name__=='__main__':
    main()
