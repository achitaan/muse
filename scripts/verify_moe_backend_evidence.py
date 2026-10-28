"""Require full coverage and an actual backend change for the selector trial."""
import ast
import hashlib
import json
from pathlib import Path
import re
import runpy

root = Path(__file__).resolve().parents[1]
try:
    runpy.run_path(str(root / 'scripts/verify_moe_precision.py'), run_name='__main__')
except SystemExit as exc:
    assert exc.code in (None, 0)
raw = (root / 'candidates/moe_active.py').read_bytes()
digest = hashlib.sha256(raw).hexdigest()
result = json.loads((root / '.moe-measurements' / digest / 'result.json').read_text(encoding='utf-8'))
body = result.get('result') or {}
assert result['status'] == 'done' and body.get('passed') and body.get('checks', {}).get('ok')
for event, count in [('test', 5), ('benchmark', 6)]:
    records = [r for r in body['records'] if r.get('event') == event]
    assert len(records) == count and all(r.get('status') == 'pass' for r in records)
tree = ast.parse(raw)
if any(isinstance(n, ast.FunctionDef) and n.name == '_wgs32_cuda_source' for n in tree.body):
    enabled = re.findall(r'^MOE_WGS32_ENABLED (True|False)$', body.get('log_tail', ''), flags=re.MULTILINE)
    assert 'True' in enabled, 'Require evidence that the four-tile WGS module was enabled'
    resources = re.findall(r'^MOE_WGS32_RESOURCES (\([^\n]*?\)) (\([^\n]*?\))$', body.get('log_tail', ''), flags=re.MULTILINE)
    assert resources, 'Missing real-kernel resource attributes'
    old, new = map(ast.literal_eval, resources[-1])
    assert len(old) == len(new) == 7 and all(isinstance(v, int) and v >= 0 for v in old + new)
    assert new[6] < old[6], 'The specialized module must have the smaller shared-memory layout'
    print('Four-tile WGS enabled; original/specialized resources:', old, new)
if any(isinstance(n, ast.FunctionDef) and n.name == '_s6_eligible' for n in tree.body):
    matches = re.findall(r'^MOE_SCALE6_WGS (True|False)$', body.get('log_tail', ''), flags=re.MULTILINE)
    assert 'True' in matches, 'Require evidence that the compressed WGS scale path was enabled'
    definitions = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'MAIN_DEFINES' for t in n.targets)))
    assert '-DOKC_WGS=1' in definitions
    print('Lossless six-bit WGS scale path was enabled in the full official check.')
if any(isinstance(n, ast.FunctionDef) and n.name == '_bkt_select' for n in tree.body):
    matches = re.findall(r'^MOE_BACKEND_CHOICES (.+)$', body.get('log_tail', ''), flags=re.MULTILINE)
    assert matches, 'Missing backend selection evidence'
    selected = ast.literal_eval(matches[-1])
    assert isinstance(selected, tuple) and selected and set(selected) <= {16, 32}, \
        'No alternate backend selected: retain the existing simpler source instead of identical timed compute'
    print('Alternate backend selected for buckets:', selected)
print('Complete MoE evidence matches the current source.')
