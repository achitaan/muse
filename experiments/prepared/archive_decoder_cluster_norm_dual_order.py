"""Archive a terminal numerical success/performance rejection without spin."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime,timedelta,timezone

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from decoder_cluster_norm_dual_order_evidence import TRIAL,PARENT,full_result,markers,verify_trial
from preflight_decoder_cluster_norm import freeze


def main():
    src=ROOT/'.decoder-cluster-norm-dual-order-measurements'/TRIAL
    raw=json.loads((src/'result.json').read_bytes());body,rows,loads=full_result(raw)
    assert raw['id']=='545f0b524917423b' and raw['status']=='done'
    log=body['log_tail'];resources=markers(log,'CN_RESOURCES')
    assert resources==[[[k,56,0,16,248,132,1,4,64] for k in range(4)]]
    assert markers(log,'CN_PROBES')==[dict(cases=36,changed_input_graph_replays=8)]
    decisions=markers(log,'CN_DECISION');assert len(decisions)==1 and decisions[0]['selected'] is False
    pairs=markers(log,'CN_PAIRS');assert len(pairs)==1 and len(pairs[0])==8
    assert all(p['new_over_old']>1.6 for p in pairs[0])
    assert all('okc_cluster_norm_dual_order::' not in r.get('report','') for r in body['records'] if r.get('event')=='profile')
    try:verify_trial(raw,quiet=True)
    except AssertionError as exc:guard_error=str(exc)
    else:raise AssertionError('Unselected losing family passed the complete-layer guard')
    assert guard_error=='Cluster family was not selected'
    protected=json.loads((ROOT/'.tools/decoder-package-protected-sha256.json').read_bytes())
    restored={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    assert restored==protected
    out=ROOT/'results/decoder-cluster-norm-dual-order-check-20261029'
    for name in ('submission.py','launch.json','launch-intent.json','result.json','jobs-before-launch.json','quota-before-launch.json'):
        freeze(out/name,(src/name).read_bytes())
    for p in (src/'profiles').glob('*'):
        if p.is_file():freeze(out/'profiles'/p.name,p.read_bytes())
    archive=ROOT/'autoresearch-results/archive/20261029-171355'
    for p in archive.rglob('*'):
        if p.is_file():freeze(out/'controller'/p.relative_to(archive),p.read_bytes())
    for name in ('decoder_cluster_norm_dual_order_evidence.py','measure_decoder_cluster_norm_dual_order.py','verify_decoder_cluster_norm_dual_order_evidence.py'):
        freeze(out/'guards'/name,(ROOT/'scripts'/name).read_bytes())
    freeze(out/'quota-after.json',(ROOT/'.tools/decoder-cluster-norm-dual-order-quota-after.json').read_bytes())
    freeze(out/'protected-source-restoration.json',(json.dumps(restored,indent=2)+'\n').encode())
    events=[json.loads(line) for line in (archive/'events.jsonl').read_text().splitlines()]
    trial=next(e for e in events if e['event']=='iteration')
    quota=json.loads((out/'quota-after.json').read_bytes())
    counts={p['name']:p['limits'] for p in quota['response']['problems']}
    launch=json.loads((src/'launch.json').read_bytes())
    next_check=datetime.fromisoformat(launch['submitted_utc'])+timedelta(minutes=60)
    report=dict(source_sha256=TRIAL,job_id=raw['id'],outcome='rejected: all eight paired native boundaries slower; no family selected',
        native_resources=resources[0],synthetic_numerical_cases=36,changed_input_graph_replays=8,
        full_test_count=13,full_benchmark_count=12,integrity_passed=True,
        full_suite_used_retained_fallback=True,new_native_whole_layer_integration_not_validated=True,
        fallback_point_speedup=body['speedup'],fallback_speedup_ci=body['speedup_ci'],
        paired_components=pairs[0],combined_boundaries=decisions[0]['combined'],
        full_layer_guard_error=guard_error,controller_guard_not_run_because_metric_not_improved=trial['guard']=='not_run',
        independent_offline_guard_rejected=True,scored_submission=False,retained_source_sha256=PARENT,
        trial_commit=trial['trial_commit'],revert_commit=trial['revert_commit'],controller=trial['run_id'],
        archive='20261029-171355',all44protected_sources_restored=True,
        quota_snapshot_utc=quota['observed_utc'],quotas=counts,
        next_new_check_not_before_utc=next_check.isoformat(),
        decision='Close this boundary-only cluster design. No parameter/occupancy retry is justified by the current evidence. Prepare the existing fixed Dense candidate with native resource binding before considering another paced check.')
    # Preserve the initial report, which used server creation time89ms earlier.
    # The pacing driver enforces the later local submitted timestamp.
    freeze(out/'review-v2.json',(json.dumps(report,indent=2)+'\n').encode())
    path=ROOT/'.tools/decoder-cluster-norm-dual-order-active-session.json'
    state=json.loads(path.read_text());state.update(active=False,terminal=True,outcome='discard',controller_archive='20261029-171355',restored_source=PARENT)
    path.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('paired_components','quotas')},indent=2))
    print('checks_left', {k:v['checks']['left'] for k,v in counts.items()})


if __name__=='__main__':main()
