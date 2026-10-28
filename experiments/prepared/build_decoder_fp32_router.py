"""Integrate the full-FP32 small-token router into the checked fusion draft."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
source = (root / 'results/resumed-decoder-phase4-20261028/bd3fad49244f5ce6f1d9ce598a69593dcd9c81bd7dc3113c5bd5387db981a848/submission.py').read_text(encoding='utf-8')
tree = ast.parse(source)
base_dump = ast.dump(tree)
base_sha = hashlib.sha256(source.encode()).hexdigest()
header = ''.join(source.splitlines(keepends=True)[:tree.body[0].lineno - 1])
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
contexts = [n for n in ast.walk(fn) if isinstance(n, ast.With)
            and any(ast.unparse(i.context_expr) == 'torch.cuda.stream(side_stream)' for i in n.items)]
assert len(contexts) == 1
context = contexts[0]
old = context.body
branch = ast.parse('''
if not split_router and tokens > 0 and state['rf'] is not None:
    topk_weights, topk_ids = _rf_route(router_input, state['rf'])
else:
    pass
''').body[0]
branch.orelse = old
context.body = [branch]
# Prove that expanding the new branch's fallback restores the entire old AST.
context.body = old
assert ast.dump(tree) == base_dump
context.body = [branch]

addon = ast.parse((root / 'experiments/prepared/fp32_router_addon.py').read_text(encoding='utf-8'))
if isinstance(addon.body[0], ast.Expr): addon.body.pop(0)
cuda = next(n.value.value for n in addon.body if isinstance(n, ast.Assign)
            and any(isinstance(a, ast.Name) and a.id == '_RF_CUDA' for a in n.targets))
assert '__fmaf_rn' in cuda and 'mma.' not in cuda and 'bfloat16' not in cuda
module_tag = 'okc_router_fp32_' + hashlib.sha256(cuda.encode()).hexdigest()[:16] + '_90a'
for n in ast.walk(addon):
    if isinstance(n, ast.Constant) and n.value == 'okc_router_fp32_v1_90a': n.value = module_tag
tree.body.extend(addon.body)
ast.fix_missing_locations(tree)
lines = []
for line in ast.unparse(tree).splitlines():
    n = len(line) - len(line.lstrip(' '))
    assert n % 4 == 0
    lines.append(' ' * (n // 4) + line[n:])
text = header + '# Additional local change: full FP32 router/top-8 fusion for 1..8 tokens; no activation or weight quantization.\n' + '\n'.join(lines) + '\n'
compile(text, '<decoder-fp32-router>', 'exec')
assert len(text.encode()) < 1_000_000
out = root / '.tools/decoder-fp32-router-variant.py'
out.write_bytes(text.encode())
audit = {'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
         'base_sha256': base_sha, 'packaged_bytes': len(text.encode()),
         'enabled_tokens': '1..8', 'router_inputs_weights_accumulation': 'FP32',
         'grid': '16 row tiles x 8 K splits = 128 blocks',
         'partial_storage': '8 x 8 x 256 FP32 values; every consumed entry rewritten',
         'counter': 'uint32 advances 128 per invocation; last = low seven bits all one',
         'fallback_ast_identical': True, 'gpu_status': 'unmeasured'}
(root / '.tools/decoder-fp32-router-source-audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
print(json.dumps(audit))
