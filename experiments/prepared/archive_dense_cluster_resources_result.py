"""Preserve the complete fixed Dense rejection and restored verified sources."""
from datetime import datetime,timedelta
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from dense_cluster_resources_evidence import TRIAL,PARENT,full_result,marker,resources,verify_trial
from preflight_decoder_cluster_norm import freeze


def main():
    src=ROOT/'.dense-cluster-resource-measurements'/TRIAL
    r=json.loads((src/'result.json').read_bytes());body,rows,loads=full_result(r)
    assert r['id']=='f026c3b14a864be0' and r['status']=='done'
    log=body['log_tail'];native=resources(log)
    assert ast.literal_eval(marker(log,'DENSE_CLUSTER_CHOICES'))==()
    evidence=json.loads(marker(log,'DENSE_CLUSTER_EVIDENCE'))
    assert set(evidence)=={'qkv_gate','pooler'}
    assert all(v['oracle']['ok'] and v['paired']['ok'] is False for v in evidence.values())
    assert all(c['ratio']>1 for v in evidence.values() for c in v['paired']['cases'] if c['m']==128)
    assert all('gq_kernel<' not in x['report'] and 'gqx_kernel<' not in x['report'] for x in body['records'] if x.get('event')=='profile')
    try:verify_trial(r,quiet=True)
    except AssertionError:pass
    else:raise AssertionError('Empty family was accepted')
    protected=json.loads((ROOT/'.tools/decoder-package-protected-sha256.json').read_bytes())
    actual={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected};assert actual==protected
    out=ROOT/'results/dense-cluster-resources-check-20261029'
    for name in ('submission.py','launch.json','launch-intent.json','result.json','jobs-before-launch.json','quota-before-launch.json'):
        freeze(out/name,(src/name).read_bytes())
    for p in (src/'profiles').glob('*'):
        if p.is_file():freeze(out/'profiles'/p.name,p.read_bytes())
    archive=ROOT/'autoresearch-results/archive/20261029-181018'
    for p in archive.rglob('*'):
        if p.is_file():freeze(out/'controller'/p.relative_to(archive),p.read_bytes())
    for name in ('dense_cluster_resources_evidence.py','measure_dense_cluster_resources.py','verify_dense_cluster_resources_evidence.py'):
        freeze(out/'guards'/name,(ROOT/'scripts'/name).read_bytes())
    freeze(out/'quota-after.json',(ROOT/'.tools/dense-cluster-resources-quota-after.json').read_bytes())
    freeze(out/'protected-source-restoration.json',(json.dumps(actual,indent=2)+'\n').encode())
    events=[json.loads(s) for s in (archive/'events.jsonl').read_text().splitlines()]
    trial=next(v for v in events if v['event']=='iteration');launch=json.loads((src/'launch.json').read_bytes())
    setup=next(v for v in body['records'] if v.get('event')=='setup')
    quota=json.loads((out/'quota-after.json').read_bytes())
    report=dict(source_sha256=TRIAL,job_id=r['id'],outcome='rejected: both fixed families lose on scored M128; no choices',
        native_resources=native,numerical_oracle_cases={k:len(v['oracle']['cases']) for k,v in evidence.items()},
        paired_cases={k:v['paired']['cases'] for k,v in evidence.items()},
        full_test_count=9,full_benchmark_count=30,profile_count=30,integrity_passed=True,prepare_s=setup['prepare_s'],
        full_suite_used_retained_fallback=True,new_native_scored_integration_not_validated=True,
        fallback_point_speedup=body['speedup'],fallback_speedup_ci=body['speedup_ci'],
        nearby_unscored_gains_are_not_retained_score_improvements=True,scored_submission=False,
        independent_guard_rejected_empty_choices=True,controller_guard=trial['guard'],retained_source_sha256=PARENT,
        trial_commit=trial['trial_commit'],revert_commit=trial['revert_commit'],controller=trial['run_id'],archive=archive.name,
        all44protected_hashes_restored=True,quota_snapshot_utc=quota['observed_utc'],
        quotas={p['name']:p['limits'] for p in quota['response']['problems']},
        next_new_check_not_before_utc=(datetime.fromisoformat(launch['submitted_utc'])+timedelta(minutes=60)).isoformat(),
        decision='Close the two fixed M128 families. Do not rescore, rerun for noise, or retain unscored-neighbor-only gains as progress toward top50.')
    freeze(out/'review.json',(json.dumps(report,indent=2)+'\n').encode())
    state_path=ROOT/'.tools/dense-cluster-resources-active-session.json';state=json.loads(state_path.read_text())
    state.update(active=False,terminal=True,outcome='discard',controller_archive=archive.name,restored_source=PARENT)
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('paired_cases','quotas')},indent=2))


if __name__=='__main__':main()
