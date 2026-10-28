"""Exact-source full MoE validation and guarded FP32 SiLU diagnostics."""
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.argv = [str(ROOT / 'scripts/verify_moe_precision.py')]
try:
    runpy.run_path(str(ROOT / 'scripts/verify_moe_precision.py'), run_name='__main__')
except SystemExit as e:
    assert e.code in (0, None)
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
TRIAL = 'f57377770e5124b8fac8f5d70d7e47d82d69a7c4988516f564700d4cf65ceddd'
raw = (ROOT / 'candidates/moe_active.py').read_bytes()
sha = hashlib.sha256(raw).hexdigest()
assert sha in (PARENT, TRIAL)
result = json.loads((ROOT / '.moe-measurements' / sha / 'result.json').read_bytes())
b = result.get('result') or {}
tests = [r for r in b.get('records', []) if r.get('event') == 'test']
benches = [r for r in b.get('records', []) if r.get('event') == 'benchmark']
assert result['status'] == 'done' and b.get('passed') is True
assert len(tests) == 5 and len(benches) == 6 and all(r.get('status') == 'pass' for r in tests + benches)
assert b.get('checks', {}).get('ok') is True
if sha == TRIAL:
    messages = [s.split('MOE_SILU_AUDIT', 1)[1].strip() for s in b.get('log_tail', '').splitlines() if 'MOE_SILU_AUDIT' in s]
    assert len(messages) == 1, 'Require full-domain native GPU activation diagnostic'
    diagnostic = ast.literal_eval(messages[0])
    assert diagnostic['count'] == 65280 and diagnostic['all_finite'] is True and diagnostic['cutoff'] == -4
    assert 0 <= diagnostic['scaled_error'] < 2e-4 and 0 <= diagnostic['bf16_stage_l2'] < .001
    old = json.loads((ROOT / '.moe-measurements' / PARENT / 'result.json').read_bytes())['result']
    assert b['speedup'] > old['speedup'] * 1.015, 'No material whole-MoE gain'
    previous = {r['index']: r['speedup'] for r in old['records'] if r.get('event') == 'benchmark'}
    ratios = {r['index']: r['speedup'] / previous[r['index']] for r in benches}
    assert min(ratios.values()) >= .985, ('Shape regression', ratios)
    print(json.dumps(dict(silu_audit=diagnostic, per_case_speedup_ratios=ratios)))
print('Exact-source full5/6/integrity and BF16/FP32 contracts verified.')
