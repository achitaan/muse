"""Authoritative, resumable scored H200 measurement for the foreground experiment loop.

The source and full official evidence are frozen under ignored .measurements/.
No credential is read or printed. Correctness is enforced by the official judge.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
import time

from official import ROOT, LINUX_ROOT, DEADLINE, cli


def main():
    source = ROOT / 'candidates' / 'active.py'
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    directory = ROOT / '.scored-measurements' / digest
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / 'submission.py'
    if frozen.exists() and frozen.read_bytes() != raw:
        raise RuntimeError('Frozen payload differs from its content hash')
    frozen.write_bytes(raw)
    launch_path = directory / 'launch.json'
    if launch_path.exists():
        launch = json.loads(launch_path.read_text())
    else:
        if datetime.now(timezone.utc) >= DEADLINE:
            raise RuntimeError('4 AM Toronto deadline reached; no new evaluation')
        launch = cli(['submit', 'muse_bf16_proj',
                      f'{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}', '--no-wait', '--json'])
        launch.update(local_sha256=digest, source='candidates/active.py')
        launch_path.write_text(json.dumps(launch, indent=2))
    job = launch['job_id']
    previous = None
    while True:
        try:
            result = cli(['status', job, '--json'])
        except Exception as exc:
            print(f'Observation failed for existing job {job}: {type(exc).__name__}; retrying', file=sys.stderr, flush=True)
            time.sleep(30)
            continue
        (directory / 'result.json').write_text(json.dumps(result, indent=2))
        status = result['status']
        if status != previous:
            print(f'H200 job {job}: {status}; payload {digest}', file=sys.stderr, flush=True)
            previous = status
        if status in {'done', 'failed', 'rejected', 'duplicate'}:
            if status == 'duplicate' and result.get('duplicate_of'):
                job = result['duplicate_of']
                continue
            body = result.get('result') or {}
            records = body.get('records', [])
            tests = [r for r in records if r.get('event') == 'test']
            benches = [r for r in records if r.get('event') == 'benchmark']
            valid = (body.get('passed') is True and len(tests) == 9 and len(benches) == 30
                     and all(r.get('status') == 'pass' for r in tests + benches)
                     and body.get('checks', {}).get('ok') is True)
            metric = float(body['speedup']) if valid else 0.0
            print(json.dumps({'speedup': metric, 'correct': valid, 'job': job,
                              'sha256': digest, 'interval': body.get('speedup_ci'),
                              'error': body.get('error')}), flush=True)
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
