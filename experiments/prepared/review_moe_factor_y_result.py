"""Review a saved terminal factor trial without API calls or source changes."""
import contextlib
import hashlib
import io
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
from moe_factor_y_evidence import TRIAL, PARENT, directory, full_result, verify_trial

archive = ROOT / 'results/moe-factor-y-20261029'
folder = archive if (archive / 'result.json').exists() else directory(TRIAL)
assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == TRIAL
launch = json.loads((folder / 'launch.json').read_bytes())
result = json.loads((folder / 'result.json').read_bytes())
assert launch['source_sha256'] == TRIAL
assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
assert result['status'] in ('done', 'failed', 'rejected'), 'Follow the existing live job'
body = result.get('result') or {}
marker = 'FACTOR_Y_EVIDENCE '
lines = [s[len(marker):] for s in body.get('log_tail', '').splitlines() if s.startswith(marker)]
evidence = json.loads(lines[0]) if len(lines) == 1 else None
try:
    full_result(result)
    full_suite = True
except (AssertionError, KeyError, TypeError, ValueError):
    full_suite = False
stream = io.StringIO()
try:
    with contextlib.redirect_stdout(stream):
        summary = verify_trial(result)
    material_gain, rejection = True, None
except (AssertionError, KeyError, TypeError, ValueError) as exc:
    summary = None
    material_gain, rejection = False, type(exc).__name__ + ': ' + str(exc)
profiles = {x['index']: x['report'] for x in body.get('records', []) if x.get('event') == 'profile'}
rows = []
for case in (evidence or {}).get('cases', []):
    rows.append(dict(m=case['m'], mode=case['mode'], old_median_ns=statistics.median(case['old_ns']),
                     new_median_ns=statistics.median(case['new_ns']), median_ratio=case['median_ratio'],
                     relative=case['relative'], worst=case['worst']))
report = dict(reviewed_utc=datetime.now(timezone.utc).isoformat(), job=result['id'],
              source_sha256=TRIAL, baseline_sha256=PARENT, status=result['status'],
              full5tests_6benchmarks_integrity=full_suite, official_speedup=body.get('speedup'),
              selected_main=(evidence or {}).get('selected_main'),
              resources=(evidence or {}).get('resources'), skip=(evidence or {}).get('skip'),
              prepare_s=(evidence or {}).get('prepare_s'), marker_count=len(lines),
              profiles_using_factor=[i for i, s in profiles.items() if 'okc_factor_y::moe_fused<' in s],
              paired_synthetic_cases=rows, material_guard_passed=material_gain,
              material_summary=summary, rejection=rejection,
              official_speedup_is_new_path_evidence=bool(material_gain),
              statistical_significance_claim=False,
              recommended_action='Review for scored publication and Decode transfer' if material_gain else 'Retain33114871; no score, no transfer, no unmotivated retry')
target = folder / 'review.json'
if target.exists():
    existing = json.loads(target.read_bytes())
    report['reviewed_utc'] = existing['reviewed_utc']
    assert report == existing, 'Frozen review differs; inspect before replacing evidence'
else:
    target.write_bytes((json.dumps(report, indent=2) + '\n').encode())
print(json.dumps(report, indent=2))
