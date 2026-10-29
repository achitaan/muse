"""Preserve the terminal failed pipeline trial without relaunching or scoring."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from attention_two_score_evidence import TRIAL,PARENT,PACKET,directory,read,full_result,verify_trial
from build_attention_two_score_pipeline import freeze


def main():
    source=directory(TRIAL); result=read(source/'result.json')
    body,rows=full_result(result)
    assert result['id']=='4e8836d4ff934fce'
    assert hashlib.sha256((source/'submission.py').read_bytes()).hexdigest()==TRIAL
    transcript=io.StringIO()
    with contextlib.redirect_stdout(transcript):
        try:verify_trial(result)
        except AssertionError as exc:rejection=str(exc)
        else:raise AssertionError('This archive expects the fixed performance rejection')
    assert rejection=='Insufficient affected/control gain'
    summary=json.loads(transcript.getvalue().strip())
    out=ROOT/'results/attention-two-score-pipeline-check-20261029'
    for name in ('submission.py','launch.json','result.json','launch-intent.json','quota-before-launch.json','jobs-before-launch.json'):
        freeze(out/name,(source/name).read_bytes())
    archive=ROOT/'autoresearch-results/archive/20261029-150254'
    for path in archive.rglob('*'):
        if path.is_file():freeze(out/'controller'/path.relative_to(archive),path.read_bytes())
    for name in ('scripts/attention_two_score_evidence.py','scripts/measure_attention_two_score.py',
                 'scripts/verify_attention_two_score_evidence.py','experiments/prepared/audit_attention_two_score_guard_CPU.py'):
        freeze(out/'guards'/Path(name).name,(ROOT/name).read_bytes())
    for row in body['records']:
        if row.get('event')=='profile':freeze(out/'profiles'/f"case{row['index']:02d}.txt",(row['report']+'\n').encode())
    protected=read(ROOT/'.tools/decoder-package-protected-sha256.json')
    actual={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in protected}
    assert actual==protected
    freeze(out/'protected-source-restoration.json',(json.dumps(actual,indent=2)+'\n').encode())
    summary.update(job_id=result['id'],full_tests=16,full_benchmarks=12,integrity=True,
        speedup=body['speedup'],speedup_ci=body['speedup_ci'],
        actual_new_paths=(9,11),native_resource_admission_passed=True,
        fixed_guard_rejection=rejection,retained_source_sha256=PARENT,
        scored_submission=False,rerun=False,posthoc_selector=False,Decode_transfer=False,
        trial_commit='fe4a829',revert_commit='04a32b6',controller_id='9a755016ade344fba9fe38b40b74a58f',
        controller_archive='20261029-150254',all_protected_hashes_restored=len(protected),
        next_new_check_not_before_utc='2026-10-29T15:55:34.212782+00:00',
        mechanism_conclusion='Two score banks, legal redistribution and zero spills were insufficient. The one-CTA grid and changed schedule lose materially as a complete configuration; no measured attribution to an isolated stage.',
        top50_goal_complete=False)
    freeze(out/'review.json',(json.dumps(summary,indent=2)+'\n').encode())
    session=read(ROOT/'.tools/attention-two-score-active-session.json')
    session.update(active=False,terminal=True,outcome='discard',controller_archive='20261029-150254')
    (ROOT/'.tools/attention-two-score-active-session.json').write_text(json.dumps(session,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
