"""Archive the one terminal source-bound wide-tile check; never submits work."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from attention_wide_evidence import TRIAL,PARENT,directory,bound_result,full_result,verify_trial


def freeze(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists() or path.read_bytes()==data,'Frozen result collision: '+str(path)
    path.write_bytes(data)


def main():
    result=bound_result(TRIAL);body,rows=full_result(result)
    assert result['id']=='fa78c34851144383'
    assert hashlib.sha256((ROOT/'candidates/attention_active.py').read_bytes()).hexdigest()==PARENT
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured):
        try:verify_trial(result)
        except AssertionError as exc:failure=str(exc)
        else:raise AssertionError('Expected fixed rejected result')
    assert failure=='Insufficient affected/control gain'
    summary=json.loads(captured.getvalue().splitlines()[0])
    report=dict(job_id=result['id'],source_sha256=TRIAL,mode=result['mode'],
                full_tests=16,full_benchmarks=12,integrity_ok=body['checks']['ok'],
                GPU=body['gpu'],speedup=body['speedup'],speedup_ci=body.get('speedup_ci'),
                source_resource_and_actual_path_gates_passed=True,
                fixed_guard_failure=failure,performance=summary,
                retained_source_sha256=PARENT,retained_speedup=full_result(bound_result(PARENT))[0]['speedup'],
                controller_run_id='78405b391e1c46c9b3975b942f66e0f3',
                trial_commit='2f910f7',revert_commit='b500997fada55fcb45ff86c13d22c230198c6721',
                controller_archive='20261029-112814',new_scored_submissions=0,
                decision='Close this fixed128-key/two-stage design; retain9e; no noise retry or Decode transfer.',
                limits='Aggregate point improvement does not establish a new-kernel gain; unchanged controls also moved. No statistical-significance claim.')
    out=ROOT/'results/attention-l1-wide-tile-check-20261029'
    source=directory(TRIAL)
    for path in source.rglob('*'):
        if path.is_file():freeze(out/path.relative_to(source),path.read_bytes())
    archive=ROOT/'autoresearch-results/archive/20261029-112814'
    assert archive.is_dir()
    for path in archive.rglob('*'):
        if path.is_file():freeze(out/'controller'/path.relative_to(archive),path.read_bytes())
    proof=ROOT/'.tools/attention-wide-live-admission-proof.json'
    freeze(out/'live-source-proof.json',proof.read_bytes())
    freeze(out/'review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report))


if __name__=='__main__':main()
