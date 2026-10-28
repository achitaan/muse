"""One quota-gated check; exact cached baseline and resumable immutable payload."""
from datetime import datetime, timezone
import json
import sys
import time

from official import LINUX_ROOT, cli
from moe_skinny16_evidence import ROOT, PARENT, source, directory, bound_result, full_result, read_json


def save(path, value):
    path.write_bytes(json.dumps(value, indent=2).encode())


def main():
    raw, digest = source()
    if digest == PARENT:
        result = bound_result(digest)
    else:
        folder = directory(digest)
        folder.mkdir(parents=True, exist_ok=True)
        frozen = folder / 'submission.py'
        assert not frozen.exists() or frozen.read_bytes() == raw
        frozen.write_bytes(raw)
        launch_path = folder / 'launch.json'
        if launch_path.exists():
            launch = read_json(launch_path)
            assert launch['source_sha256'] == digest
        else:
            quotas = cli(['problems', '--json'])
            save(folder / 'quota-before-launch.json',
                 {'utc': datetime.now(timezone.utc).isoformat(), 'response': quotas})
            problem = next(x for x in quotas['problems'] if x['name'] == 'muse_moe_w4a16')
            if problem['limits']['checks']['left'] <= 0:
                raise RuntimeError('MoE check quota exhausted; no GPU evaluation launched')
            # Reconcile jobs after any unknown upload outcome before trying again.
            launch = cli(['check', 'muse_moe_w4a16',
                          f'{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}', '--no-wait', '--json'])
            launch.update(source_sha256=digest, submitted_utc=datetime.now(timezone.utc).isoformat())
            save(launch_path, launch)
        job = launch.get('resolved_job_id', launch['job_id'])
        visited, previous = {job}, None
        result_path = folder / 'result.json'
        saved = read_json(result_path) if result_path.exists() else None
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
            save(result_path, result)
            status = result['status']
            if status != previous:
                print(f'MoE job {job}: {status}; source {digest}', file=sys.stderr, flush=True)
                previous = status
            if status == 'duplicate' and result.get('duplicate_of'):
                job = result['duplicate_of']
                assert job not in visited, 'Duplicate cycle; retain handle and inspect'
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
    print(json.dumps(dict(speedup=float(body['speedup']) if valid else 0.0, correct=valid,
                          job=result.get('id'), source_sha256=digest,
                          interval=body.get('speedup_ci'), error=body.get('error'))), flush=True)


if __name__ == '__main__':
    main()
