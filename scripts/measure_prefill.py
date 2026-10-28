"""Freeze and measure the prefill candidate using the official check service.

The resumed overall-ranking objective has no time deadline. The immutable source
hash identifies one evaluation; existing job handles are always resumed.
"""
import hashlib
import json
import sys
import time

from official import ROOT, LINUX_ROOT, cli


def main():
    raw = (ROOT / 'candidates/prefill_active.py').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    directory = ROOT / '.prefill-measurements' / digest
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / 'submission.py'
    if frozen.exists() and frozen.read_bytes() != raw:
        raise RuntimeError('Frozen source hash collision')
    frozen.write_bytes(raw)
    launch_path = directory / 'launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
    else:
        # A failed upload is not automatically retried: reconcile official jobs
        # before invoking this command again if no launch record was returned.
        launch = cli(['check', 'muse_prefill_layer',
                      f'{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}', '--no-wait', '--json'])
        launch.update(source_sha256=digest)
        launch_path.write_text(json.dumps(launch, indent=2))
    job = launch['job_id']
    result_path = directory / 'result.json'
    saved = json.loads(result_path.read_text()) if result_path.exists() else None
    previous = None
    while True:
        try:
            if saved and saved.get('status') in ('done', 'failed', 'rejected'):
                result, saved = saved, None
            else:
                result = cli(['status', job, '--json'])
        except Exception as exc:
            print(f'Observation failed for {job}: {type(exc).__name__}; retaining handle', file=sys.stderr, flush=True)
            time.sleep(20)
            continue
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        status = result['status']
        if previous != status:
            print(f'Prefill job {job}: {status}; source {digest}', file=sys.stderr, flush=True)
            previous = status
        if status == 'duplicate' and result.get('duplicate_of'):
            job = result['duplicate_of']
            continue
        if status in ('done', 'failed', 'rejected', 'duplicate'):
            body = result.get('result') or {}
            records = body.get('records', [])
            tests = [r for r in records if r.get('event') == 'test']
            benches = [r for r in records if r.get('event') == 'benchmark']
            # The complete official task contains 5 tests and 8 scored cases;
            # validate named counts against the pinned task before initialization.
            valid = (body.get('passed') is True and len(tests) == 5 and len(benches) == 8
                     and all(r.get('status') == 'pass' for r in tests + benches)
                     and body.get('checks', {}).get('ok') is True)
            print(json.dumps({'speedup': float(body['speedup']) if valid else 0.0,
                              'correct': valid, 'job': job, 'source_sha256': digest,
                              'tests': len(tests), 'benchmarks': len(benches),
                              'interval': body.get('speedup_ci'), 'error': body.get('error')}), flush=True)
            return
        time.sleep(45)


if __name__ == '__main__':
    main()
