"""Review and archive the existing terminal job; no API calls or kernel launches."""
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from moe_cluster_x_evidence import TRIAL,PARENT,CUDA,directory,full_result,verify_trial,pure_gates


def main():
    folder=directory(TRIAL)
    raw=(folder/'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    launch=json.loads((folder/'launch.json').read_bytes())
    result=json.loads((folder/'result.json').read_bytes())
    assert result['id']==launch.get('resolved_job_id',launch['job_id']) and launch['source_sha256']==TRIAL
    assert result['status'] in ('done','failed','rejected'),'Follow the existing live handle'
    body=result.get('result') or {}
    try:full_result(result);full=True
    except (AssertionError,KeyError,TypeError,ValueError):full=False
    marker='CLUSTER_X_EVIDENCE '
    lines=[s[len(marker):] for s in body.get('log_tail','').splitlines() if s.startswith(marker)]
    e=json.loads(lines[0]) if len(lines)==1 else None
    assert e and e['cuda_sha256']==CUDA
    resources_admitted=pure_gates()['_cx_resources_ok'](e['resources'],e['grid'])
    rows=[dict(m=c['m'],mode=c['mode'],new_over_old_median=c['median_ratio'],relative=c['relative'],worst=c['worst'],
               changed_input_graph_replays=c['changed_input_graph_replays']) for c in e['cases']]
    try:
        with contextlib.redirect_stdout(io.StringIO()):summary=verify_trial(result)
        material,rejection=True,None
    except (AssertionError,KeyError,TypeError,ValueError) as exc:
        material,summary,rejection=False,None,type(exc).__name__+': '+str(exc)
    profiles={r['index']:r['report'] for r in body.get('records',[]) if r.get('event')=='profile'}
    report=dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),job=result['id'],source_sha256=TRIAL,
                CUDA_sha256=CUDA,baseline_sha256=PARENT,status=result['status'],
                full5tests_6benchmarks_integrity=full,official_speedup=body.get('speedup'),
                selected_main=e['selected_main'],resources=e['resources'],resources_admitted=resources_admitted,
                prepare_s=e['prepare_s'],synthetic_cases=rows,analytic=e['analytic'],
                synthetic_global_max=max(c['relative'] for c in rows),synthetic_worst_row_max=max(c['worst'] for c in rows),
                changing_input_graph_replays=sum(len(c['changed_input_graph_replays']) for c in rows),
                official_profile_indices_with_new_namespace=[i for i,s in profiles.items() if 'okc_cluster2::' in s],
                official_speedup_is_new_path_evidence=bool(full and e['selected_main'] and all('okc_cluster2::moe_fused<' in profiles.get(i,'') for i in (4,5))),
                material_guard_passed=material,material_summary=summary,rejection=rejection,
                reason='Normal128/256 paired regressions and M385 nearby loss; no new-path full-suite timing',
                logical_request_halving_is_not_measured_bandwidth_or_speedup=True,
                statistical_significance_claim=False,scored_submissions_used=0,decode_transfer=False,
                derived_MoE_checks_left=13,derived_total_checks_left=91,
                next_new_check_not_before_utc='2026-10-29T09:53:13.031328+00:00',
                recommended_action='Retain33114871; close this fixed cluster design; no noise retry or parameter sweep')
    out=ROOT/'results/moe-cluster-x-check-20261029'
    out.mkdir(exist_ok=True)
    def freeze(name,data):
        p=out/name
        p.parent.mkdir(parents=True,exist_ok=True)
        assert not p.exists() or p.read_bytes()==data,('Frozen archive collision',name)
        p.write_bytes(data)
    for name in ('submission.py','launch.json','launch-intent.json','quota-before-launch.json','jobs-before-launch.json','result.json'):
        freeze(name,(folder/name).read_bytes())
    for name in ('events.jsonl','run.json'):
        freeze('controller/'+name,(ROOT/'autoresearch-results'/name).read_bytes())
    for f in (ROOT/'autoresearch-results/logs').glob('*.json'):
        freeze('controller/logs/'+f.name,f.read_bytes())
    for i,s in profiles.items():freeze('profiles/'+str(i)+'.txt',s.encode())
    if (out/'review.json').exists():report['reviewed_utc']=json.loads((out/'review.json').read_bytes())['reviewed_utc']
    freeze('review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:report[k] for k in ('job','full5tests_6benchmarks_integrity','official_speedup','selected_main','resources_admitted','prepare_s','changing_input_graph_replays','material_guard_passed','derived_MoE_checks_left','recommended_action')}))


if __name__=='__main__':main()
