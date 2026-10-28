"""Require full official decoder evidence and retain its FP32 router contract."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = (root / 'candidates/decode_active.py').read_bytes()
digest = hashlib.sha256(source).hexdigest()
result = json.loads((root / '.decode-measurements' / digest / 'result.json').read_text())
body = result.get('result') or {}
records = body.get('records', [])
tests = [r for r in records if r.get('event') == 'test']
benches = [r for r in records if r.get('event') == 'benchmark']
assert result['status'] == 'done' and body.get('passed') is True
assert len(tests) == 13 and len(benches) == 12
assert all(r.get('status') == 'pass' for r in tests + benches)
assert body.get('checks', {}).get('ok') is True
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
text = ast.unparse(fn)
assert "F.linear(router_input, state['router'])" in text
assert 'router_dtype=None if split_router else torch.float32' in text
assert 'split_router = tokens > ROUTER_FP32_MAX_TOKENS' in text
limits = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == 'ROUTER_FP32_MAX_TOKENS' for t in n.targets)]
assert limits == [8]
print('Full decoder evidence matches current source; FP32 router path is preserved.')
