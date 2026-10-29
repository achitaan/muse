"""One paced official check of a frozen Decode candidate from this lane.

Uses the main checkout's authenticated okc binary and its shared pacing ledger,
so the 60-minute spacing, reserve and window rules in EVALUATION_POLICY.json
apply across both lanes. Never reads or prints credentials.

usage: python scripts/measure_decode_lane.py <candidate.py> <result-dir> [check|submit] [problem]
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

LANE = Path(__file__).resolve().parents[1]
MAIN = Path(r'C:\Users\Achita\Documents\ChatGPT\Muse 2')
OKC = '/mnt/c/Users/Achita/Documents/ChatGPT/Muse 2/.tools/okc'
LEDGER = MAIN / '.tools/evaluation-pacing-launches.jsonl'
POLICY = MAIN / 'EVALUATION_POLICY.json'
TERMINAL = ('done', 'failed', 'rejected', 'duplicate', 'cancelled', 'timeout')


def linux(path):
    p = Path(path).resolve().as_posix()
    return '/mnt/' + p[0].lower() + p[2:]


def cli(args, timeout=90):
    out = subprocess.run(['wsl', '-d', 'Ubuntu', '--', OKC, *args], cwd=MAIN, capture_output=True,
                         text=True, encoding='utf-8', timeout=timeout)
    if out.returncode:
        raise RuntimeError(out.stderr.strip() or out.stdout.strip())
    return json.loads(out.stdout)


def save(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def pacing():
    policy = json.loads(POLICY.read_bytes())
    while True:
        now = datetime.now(timezone.utc)
        if policy.get('checks_authorized') is not True:
            raise RuntimeError('GPU checks are not authorized')
        left = (datetime.fromisoformat(policy['window_end_utc']) - now).total_seconds() / 60
        if left <= policy['minimum_minutes_remaining_before_new_check']:
            raise RuntimeError('Evaluation window is closed or too short for another check')
        rows = [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]
        last = max(datetime.fromisoformat(r['submitted_utc']) for r in rows)
        waited = (now - last).total_seconds() / 60
        if waited >= policy['minimum_minutes_between_new_checks']:
            return policy
        print(f'waiting locally for the paced slot ({waited:.1f} min since last accepted launch); no API calls', flush=True)
        time.sleep(min(60.0, (policy['minimum_minutes_between_new_checks'] - waited) * 60 + 2))


def main():
    source, folder = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    action = sys.argv[3] if len(sys.argv) > 3 else 'check'
    problem_name = sys.argv[4] if len(sys.argv) > 4 else 'muse_decode_layer'
    assert action in ('check', 'submit')
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / 'submission.py'
    assert not frozen.exists() or frozen.read_bytes() == raw, 'a different candidate is already frozen here'
    frozen.write_bytes(raw)
    launch_path = folder / ('launch.json' if action == 'check' else 'submit-launch.json')
    intent_path = folder / ('launch-intent.json' if action == 'check' else 'submit-intent.json')
    result_path = folder / ('result.json' if action == 'check' else 'submit-result.json')
    if launch_path.exists():
        launch = json.loads(launch_path.read_bytes())
        assert launch['source_sha256'] == sha
    else:
        if intent_path.exists():
            raise RuntimeError('Unknown prior upload outcome: reconcile official jobs before any new upload')
        policy = pacing()
        jobs = cli(['jobs', '--json'])
        save(folder / f'jobs-before-{action}.json', jobs)
        live = [r for r in jobs if r.get('status') not in TERMINAL]
        if live:
            raise RuntimeError('An official job is still live')
        quotas = cli(['problems', '--json'])
        save(folder / f'quota-before-{action}.json', dict(utc=datetime.now(timezone.utc).isoformat(), response=quotas))
        problem = next(p for p in quotas['problems'] if p['name'] == problem_name)
        if action == 'check' and problem['limits']['checks']['left'] <= policy['minimum_checks_reserved_per_track']:
            raise RuntimeError(f'{problem_name} check reserve reached')
        if action == 'submit' and problem['limits']['scored']['left'] <= policy['minimum_scores_reserved_per_track']:
            raise RuntimeError(f'{problem_name} scored reserve reached')
        save(intent_path, dict(source_sha256=sha, action=action, utc=datetime.now(timezone.utc).isoformat()))
        launch = cli([action, problem_name, linux(frozen), '--no-wait', '--json'])
        launch.update(source_sha256=sha, submitted_utc=datetime.now(timezone.utc).isoformat())
        save(launch_path, launch)
        with LEDGER.open('a', encoding='utf-8') as out:
            out.write(json.dumps(dict(problem=problem_name, job_id=launch['job_id'], source_sha256=sha,
                                      submitted_utc=launch['submitted_utc'], action=action)) + '\n')
        print(f'{action} launched {launch["job_id"]} for {sha}', flush=True)
    job = launch.get('resolved_job_id', launch['job_id'])
    previous = None
    while True:
        try:
            result = cli(['status', job, '--json'])
        except Exception as exc:
            print(f'observation failed for {job}: {type(exc).__name__}', file=sys.stderr, flush=True)
            time.sleep(30)
            continue
        save(result_path, result)
        if result.get('status') != previous:
            previous = result.get('status')
            print(f'{datetime.now(timezone.utc).strftime("%H:%M:%S")} {job}: {previous}', flush=True)
        if previous == 'duplicate' and result.get('duplicate_of'):
            job = result['duplicate_of']
            launch['resolved_job_id'] = job
            save(launch_path, launch)
            continue
        if previous in TERMINAL:
            break
        time.sleep(40)
    body = result.get('result') or {}
    print(json.dumps(dict(job=job, status=previous, speedup=body.get('speedup'), passed=body.get('passed'),
                          error=body.get('error'), source_sha256=sha)), flush=True)


if __name__ == '__main__':
    main()
