"""One full Dense check, cached exact scored baseline, and persistent job handles.

New launches obey the user's evaluation window, pacing and quota reserve.
Observation failures retain the handle and never re-upload.
"""
from datetime import datetime, timezone
import json
import sys
import time

from official import LINUX_ROOT, cli
from dense_cluster_evidence import ROOT, PARENT, source, directory, bound_result, full_result
from evaluation_pacing import authorize_check, record_check


def save(path, value):
    path.write_bytes(json.dumps(value, indent=2).encode('utf8'))


def main():
    raw, digest = source()
    folder = directory(digest)
    if digest == PARENT:
        # An exact scored run supplies the complete suite, not a fresh baseline job.
        result = bound_result(digest)
    else:
        folder.mkdir(parents=True, exist_ok=True)
        frozen = folder / 'submission.py'
        if frozen.exists() and frozen.read_bytes() != raw:
            raise RuntimeError('Frozen source collision')
        frozen.write_bytes(raw)
        launch_path = folder / 'launch.json'
        if launch_path.exists():
            launch = json.loads(launch_path.read_bytes())
            assert launch['source_sha256'] == digest
        else:
            quotas = cli(['problems', '--json'])
            save(folder / 'quota-before-launch.json',
                 {'utc': datetime.now(timezone.utc).isoformat(), 'response': quotas})
            problem = next(p for p in quotas['problems'] if p['name'] == 'muse_bf16_proj')
            if problem['limits']['checks']['left'] <= 0:
                raise RuntimeError('Dense daily check quota exhausted; no evaluation launched')
            authorize_check(problem)
            # After an unknown upload outcome, reconcile official jobs before any retry.
            launch = cli(['check', 'muse_bf16_proj',
                          f'{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}', '--no-wait', '--json'])
            launch.update(source_sha256=digest, submitted_utc=datetime.now(timezone.utc).isoformat())
            save(launch_path, launch)
            record_check(launch, 'muse_bf16_proj')
        job = launch.get('resolved_job_id', launch['job_id'])
        previous = None
        visited = {job}
        while True:
            try:
                result = cli(['status', job, '--json'])
            except Exception as exc:
                print(f'Observation failed for {job}: {type(exc).__name__}; retaining handle', file=sys.stderr, flush=True)
                time.sleep(20)
                continue
            save(folder / 'result.json', result)
            status = result['status']
            if previous != status:
                print(f'Dense job {job}: {status}; source {digest}', file=sys.stderr, flush=True)
                previous = status
            if status == 'duplicate' and result.get('duplicate_of'):
                job = result['duplicate_of']
                if job in visited:
                    raise RuntimeError('Duplicate cycle; retain handles and inspect official state')
                visited.add(job)
                launch['resolved_job_id'] = job
                save(launch_path, launch)
                continue
            if status in ('done', 'failed', 'rejected', 'duplicate'):
                break
            time.sleep(45)
    body = result.get('result') or {}
    try:
        full_result(result)
        valid = True
    except (AssertionError, KeyError, TypeError, ValueError):
        valid = False
    print(json.dumps({'speedup': float(body['speedup']) if valid else 0.0,
                      'correct': valid, 'job': result.get('id'), 'source_sha256': digest,
                      'interval': body.get('speedup_ci'), 'error': body.get('error')}), flush=True)


if __name__ == '__main__':
    main()
