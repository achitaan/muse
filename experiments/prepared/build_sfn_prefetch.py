"""Move independent FFN loads early, without changing arithmetic statements."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sha = '427cebf148cb15201b82ce25f38e311047884f22ada1f1b220b35acfe73ec6f5'
raw = (root / 'results/resumed-decoder-phase6-20261028' / sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == sha
source = raw.decode('utf-8')
tree = ast.parse(source)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sfn_kernel')
old = ast.get_source_segment(source, fn)
loads = [
    '    stream = _sfn_tl.load(Stream + row * StreamStride + col).to(_sfn_tl.float32)\n',
    '    alpha = _sfn_tl.load(Alpha + col).to(_sfn_tl.float32)\n',
    '    beta = _sfn_tl.load(Beta + col).to(_sfn_tl.float32)\n',
    '    pre_weight = _sfn_tl.load(PreWeight + col).to(_sfn_tl.float32)\n',
]
new = old
for line in loads:
    assert new.count(line) == 1
    new = new.replace(line, '')
anchor = '    post_input = (branch * post_weight).to(_sfn_tl.bfloat16).to(_sfn_tl.float32)\n'
assert new.count(anchor) == 1
new = new.replace(anchor, ''.join(loads) + anchor)
a, b = ast.parse(old).body[0], ast.parse(new).body[0]
assert sorted(ast.dump(n) for n in a.body) == sorted(ast.dump(n) for n in b.body)
text = source.replace(old, new, 1)
compile(text, '<sfn-prefetch>', 'exec')
out = root / '.tools/decoder-sfn-prefetch-variant.py'
out.write_bytes(text.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
          'base_sha256': sha, 'change': 'read-only stream, gate and pre-norm weight loads moved before first reduction',
          'kernel_statements_identical_except_order': True,
          'arithmetic_launch_geometry_and_other_functions_unchanged': True, 'gpu_status': 'unmeasured'}
(root / '.tools/decoder-sfn-prefetch-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
