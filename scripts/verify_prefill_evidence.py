"""Require complete official evidence for the current Prefill source, without rerunning it."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
digest = hashlib.sha256((root / 'candidates/prefill_active.py').read_bytes()).hexdigest()
result = json.loads((root / '.prefill-measurements' / digest / 'result.json').read_text())
body = result.get('result') or {}
records = body.get('records', [])
tests = [r for r in records if r.get('event') == 'test']
benches = [r for r in records if r.get('event') == 'benchmark']
assert result['status'] == 'done' and body.get('passed') is True
assert len(tests) == 5 and len(benches) == 8
assert all(r.get('status') == 'pass' for r in tests + benches)
assert body.get('checks', {}).get('ok') is True
print('Full 5-test/8-benchmark Prefill evidence matches the current source hash.')
