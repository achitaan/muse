"""Conserve evaluation quota under the user's current GPU window."""
from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / '.tools/evaluation-pacing-launches.jsonl'


def authorize_check(problem):
    policy = json.loads((ROOT / 'EVALUATION_POLICY.json').read_bytes())
    now = datetime.now(timezone.utc)
    if policy.get('checks_authorized') is not True:
        raise RuntimeError('GPU checks are not authorized')
    if now >= datetime.fromisoformat(policy['window_end_utc']):
        raise RuntimeError('User evaluation window has ended; no new GPU check')
    reserve = policy['minimum_checks_reserved_per_track']
    if problem['limits']['checks']['left'] <= reserve:
        raise RuntimeError(f'Keeping the last {reserve} checks in reserve for this track')
    launches = [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    if launches:
        last = max(datetime.fromisoformat(row['submitted_utc']) for row in launches)
        elapsed = (now - last).total_seconds() / 60
        if elapsed < policy['minimum_minutes_between_new_checks']:
            raise RuntimeError('Pacing reserve: continue CPU work before another new check')
    return policy


def record_check(launch, problem):
    row = dict(problem=problem, job_id=launch['job_id'], source_sha256=launch['source_sha256'],
               submitted_utc=launch['submitted_utc'], action='check')
    with LEDGER.open('a', encoding='utf-8') as output:
        output.write(json.dumps(row) + '\n')
