"""One paced check of a frozen candidate; cached baseline, persistent job handle."""
from datetime import datetime, timezone
import json
import sys
import time
from official import cli, LINUX_ROOT
from evaluation_pacing import check_window, PacingWait, authorize_check, record_check
from decoder_cluster_norm_ordered_evidence import ROOT, PARENT, source, directory, bound_result, full_result, validate_packet, read


def save(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def main():
    raw, sha = source()
    if sha == PARENT:
        result = bound_result(sha)
    else:
        assert raw == validate_packet()
        admission=read(ROOT/'results/decoder-cluster-norm-ordered-preflight-20261029/launch-admission.json')
        assert admission['source_sha256']==sha and admission['GPU_admitted'],admission['reason']
        folder = directory(sha)
        folder.mkdir(parents=True, exist_ok=True)
        frozen = folder / 'submission.py'
        assert not frozen.exists() or frozen.read_bytes() == raw
        frozen.write_bytes(raw)
        launch_path = folder / 'launch.json'
        if launch_path.exists():
            launch = read(launch_path)
            assert launch['source_sha256'] == sha
        else:
            announced = False
            while True:
                try:
                    check_window()
                    break
                except PacingWait:
                    if not announced:
                        print('Waiting locally for the authorized check slot; no API polling.', file=sys.stderr, flush=True)
                        announced = True
                    time.sleep(30)
            if (folder / 'launch-intent.json').exists():
                raise RuntimeError('Unknown prior upload outcome: reconcile official jobs before any new upload')
            jobs = cli(['jobs', '--json'])
            save(folder / 'jobs-before-launch.json', jobs)
            assert isinstance(jobs, list)
            active = [r for r in jobs if r.get('status') not in
                      ('done','failed','rejected','duplicate','cancelled','timeout')]
            if active:
                raise RuntimeError('An official job is still live; follow its handle before another launch')
            quotas = cli(['problems', '--json'])
            save(folder / 'quota-before-launch.json', dict(utc=datetime.now(timezone.utc).isoformat(), response=quotas))
            problem = next(p for p in quotas['problems'] if p['name'] == 'muse_decode_layer')
            authorize_check(problem)
            save(folder / 'launch-intent.json', dict(source_sha256=sha, utc=datetime.now(timezone.utc).isoformat()))
            launch = cli(['check', 'muse_decode_layer', f'{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}', '--no-wait', '--json'])
            launch.update(source_sha256=sha, submitted_utc=datetime.now(timezone.utc).isoformat())
            save(launch_path, launch)
            record_check(launch, 'muse_decode_layer')
        job = launch.get('resolved_job_id', launch['job_id'])
        visited, previous = {job}, None
        result_path = folder / 'result.json'
        cached = read(result_path) if result_path.exists() else None
        while True:
            try:
                result = cached if cached and cached.get('status') in ('done', 'failed', 'rejected') else cli(['status', job, '--json'])
                cached = None
            except Exception as exc:
                print(f'Observation failed for {job}: {type(exc).__name__}; retaining handle', file=sys.stderr, flush=True)
                time.sleep(20)
                continue
            save(result_path, result)
            if result['status'] != previous:
                previous = result['status']
                print(f'Decode ordered cluster-norm check {job}: {previous}', file=sys.stderr, flush=True)
            if previous == 'duplicate' and result.get('duplicate_of'):
                job = result['duplicate_of']
                assert job not in visited
                visited.add(job); launch['resolved_job_id'] = job; save(launch_path, launch)
                continue
            if previous in ('done', 'failed', 'rejected', 'duplicate'):
                break
            time.sleep(45)
    body = result.get('result') or {}
    try:
        full_result(result)
        valid = True
    except (AssertionError, KeyError, TypeError, ValueError):
        valid = False
    print(json.dumps(dict(speedup=float(body['speedup']) if valid else 0.0,
                          correct=valid, job=result.get('id'), source_sha256=sha,
                          error=body.get('error'))), flush=True)


if __name__ == '__main__':
    main()
