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
if any(isinstance(n, ast.FunctionDef) and n.name == '_sp_select' for n in tree.body):
    selections = [line.split('SMALL_PROJ_CHOICES', 1)[1].strip()
                  for line in body.get('log_tail', '').splitlines()
                  if 'SMALL_PROJ_CHOICES' in line]
    assert len(selections) == 1, 'Require explicit evidence of projection selection'
    choices = ast.literal_eval(selections[0])
    assert isinstance(choices, tuple) and choices, 'Unselected projection trials change no timed compute'
    assert len(set(choices)) == len(choices)
    output_only = any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant) and n.value.value == 16
                      and any(isinstance(t, ast.Name) and t.id == '_SP_TOKR' for t in n.targets) for n in tree.body)
    allowed = {('o', 16)} if output_only else {(layer, tokens) for layer in ('pooler', 'qkv_gate', 'unpooler') for tokens in (1, 8)}
    assert all(key in allowed for key in choices)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
text = ast.unparse(fn)
assert "F.linear(router_input, state['router'])" in text
assert 'router_dtype=None if split_router else torch.float32' in text
assert 'split_router = tokens > ROUTER_FP32_MAX_TOKENS' in text
limits = [n.value.value for n in tree.body if isinstance(n, ast.Assign)
          and any(isinstance(t, ast.Name) and t.id == 'ROUTER_FP32_MAX_TOKENS' for t in n.targets)]
assert limits == [8]
small_ffn = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sfn_kernel']
if small_ffn:
    assert len(small_ffn) == 1
    kernel = ast.unparse(small_ffn[0])
    assert '(branch * post_weight).to(_sfn_tl.bfloat16).to(_sfn_tl.float32)' in kernel
    assert '(post_input * post_inv).to(_sfn_tl.bfloat16).to(_sfn_tl.float32)' in kernel
    assert '/ H + PostEps' in kernel and '/ H + PreEps' in kernel
    assert 'updated.to(' not in kernel, 'Residual and pre-norm input must remain FP32'
    assert '_sfn_tl.store(NewStream + row * H + col, updated)' in kernel
    assert 'col < Slice' in kernel
print('Full decoder evidence matches current source; FP32 router path is preserved.')
