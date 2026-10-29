"""Preserve the terminal check and offline parser repair; never submits work."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from attention_external_evidence import TRIAL,PARENT,PACKET,directory,bound_result,full_result,verify_trial,validate_packet


def freeze(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==data,'Frozen result collision: '+str(path)
    path.write_bytes(data)


def main():
    result=bound_result(TRIAL);body,rows=full_result(result)
    assert result['id']=='4c2b701a8d284b71' and result['mode']=='check'
    assert validate_packet()==(directory(TRIAL)/'submission.py').read_bytes()
    protected=json.loads((ROOT/'.tools/decoder-package-protected-sha256.json').read_bytes())
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in protected.items())
    assert hashlib.sha256((ROOT/'candidates/attention_active.py').read_bytes()).hexdigest()==PARENT
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:verify_trial(result)
        except AssertionError as exc:failure=str(exc)
        else:raise AssertionError('This fixed trial must remain rejected')
    assert failure=='Insufficient affected/control gain'
    performance=json.loads(captured.getvalue().splitlines()[0])
    out=ROOT/'results/attention-external-finalize-check-20261029';source=directory(TRIAL)
    for path in source.rglob('*'):
        if path.is_file():freeze(out/path.relative_to(source),path.read_bytes())
    archive=ROOT/'autoresearch-results/archive/20261029-135849'
    for path in archive.rglob('*'):
        if path.is_file():freeze(out/'controller'/path.relative_to(archive),path.read_bytes())
    for path in (PACKET/'post-result-guard-repair').iterdir():
        if path.is_file():freeze(out/'guard-parser-repair'/path.name,path.read_bytes())
    for path in (ROOT/'scripts/attention_external_evidence.py',PACKET/'guard-CPU-audit-v2.json'):
        freeze(out/'guard-parser-repair'/path.name,path.read_bytes())
    freeze(out/'decoder-transfer-inspection.json',(ROOT/'.tools/attention-external-decoder-transfer-inspection.json').read_bytes())
    clarified=ROOT/'.tools/attention-external-decoder-transfer-inspection-v2.json'
    if clarified.exists():freeze(out/'decoder-transfer-inspection-v2.json',clarified.read_bytes())
    freeze(out/'overall-API-snapshot.json',(ROOT/'.tools/attention-external-overall-snapshot.json').read_bytes())
    freeze(out/'restored-source-proof.json',(json.dumps(dict(all_protected_hashes_restored=True,count=len(protected),hashes=protected),indent=2)+'\n').encode())
    quotas=json.loads((source/'quota-before-launch.json').read_bytes())
    expected={p['name']:dict(checks=p['limits']['checks']['left']-(1 if p['name']=='muse_attention_decode' else 0),
                            scores=p['limits']['scored']['left']) for p in quotas['response']['problems']}
    events=[json.loads(line) for line in (archive/'events.jsonl').read_text().splitlines()]
    iteration=next(e for e in events if e['event']=='iteration')
    report=dict(job_id=result['id'],source_sha256=TRIAL,CUDA_sha256='c02c053a7afcab82fb1d09df7b74361855d6c24c6b664633db3dec675d06eeb7',
        full_tests=16,full_benchmarks=12,integrity_ok=body['checks']['ok'],GPU=body['gpu'],
        actual_new_main_and_finalizer_execute_in_cases=[9,11],all_ten_control_profiles_retain_original_backend=True,
        source_resource_and_execution_gates_pass_after_name_parser_repair=True,
        speedup=body['speedup'],speedup_ci=body['speedup_ci'],performance=performance,
        original_controller_failure='Deferred main path absent: exact name expectation missed official fixed-width16... spelling',
        offline_parser_repair='Accept only the fixed source-bound16... spelling in addition to full MODE16 name; all source, resource, execution and gain criteria retained',
        offline_fixed_guard_failure=failure,
        failed_precommitted_performance_requirements=['normalized affected1.015549232 <1.02','normalized case11 1.006150219 <1.01'],
        CPU_guard_revision2_negative_fixtures=22,new_GPU_jobs_for_parser_repair=0,
        controller_run_id='d945eea56dd94d74985d825e895b9ca2',trial_commit=iteration['trial_commit'],
        revert_commit=iteration['revert_commit'],controller_archive='20261029-135849',
        retained_source_sha256=PARENT,retained_speedup=full_result(bound_result(PARENT))[0]['speedup'],
        all_44_protected_hashes_restored=True,new_scored_submissions=0,
        derived_quotas_after_accepted_check=expected,
        quota_scope='Fresh pre-launch API minus this accepted check; not a new post-result API snapshot',
        next_new_check_not_before_utc='2026-10-29T14:50:51.779806+00:00',
        decision='Retain9e; close this fixed external-finalizer design without score, noise rerun, batch selector or Decode transfer',
        limitations='Raw aggregate and affected point gains do not establish a retained speedup; controls also moved. No statistical-significance claim. Overall API reports rank only, no aggregate speedup.')
    freeze(out/'review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
