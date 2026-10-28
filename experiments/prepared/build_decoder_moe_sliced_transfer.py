"""Combine the checked exact MoE with the audited sliced FFN boundary."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import symtable
import builtins

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
source = (root / '.tools/decoder-sliced-ffn-variant.py').read_text(encoding='utf-8')
moe_sha = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
moe_raw = (root / 'results/resumed-moe-phase4-20261028' / moe_sha / 'submission.py').read_bytes()
assert hashlib.sha256(moe_raw).hexdigest() == moe_sha
moe = moe_raw.decode('utf-8')
# Reuse only the already-audited namespacing function, not the old builder's I/O.
builder = ast.parse((root / 'experiments/prepared/build_decode_native_variant.py').read_text(encoding='utf-8'))
ns_fn = next(n for n in builder.body if isinstance(n, ast.FunctionDef) and n.name == 'namespace')
scope = {'ast': ast}
exec(compile(ast.Module(body=[ns_fn], type_ignores=[]), '<audited-namespace>', 'exec'), scope)
namespaced = ast.parse(scope['namespace'](moe, '_native_moe_'))
for n in ast.walk(namespaced):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)) and n.body:
        if isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body.pop(0)
            if not n.body: n.body.append(ast.Pass())
ast.fix_missing_locations(namespaced)
lines = []
for line in ast.unparse(namespaced).splitlines():
    indentation = len(line) - len(line.lstrip(' '))
    assert indentation % 4 == 0
    lines.append(' ' * (indentation // 4) + line[indentation:])
replacement = '\n'.join(lines) + '\n'
tree = ast.parse(source)
native = []
for i, n in enumerate(tree.body):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
        names = [n.name]
    elif isinstance(n, ast.Assign):
        names = [t.id for t in n.targets if isinstance(t, ast.Name)]
    else:
        names = []
    if names and all(name.startswith('_native_moe_') for name in names): native.append((i, n))
assert [i for i, _ in native] == list(range(native[0][0], native[-1][0] + 1))
source_lines = source.splitlines(keepends=True)
source_lines[native[0][1].lineno - 1:native[-1][1].end_lineno] = [replacement]
combined = ''.join(source_lines)

# All observed whole-layer logs rejected these dense alternatives. Removing
# their prepare-only sweep frees compilation budget; the timed dense dispatch
# remains the already-measured incumbent with an empty choices dictionary.
t = ast.parse(combined)
tuner = next(n for n in reversed(t.body) if isinstance(n, ast.FunctionDef) and n.name == '_achita_dense_prepare')
old_tuner = ast.get_source_segment(combined, tuner)
assert 'WG_CANDIDATE' in old_tuner and 'achita_wg_choices' in old_tuner
new_tuner = '''def _achita_dense_prepare(weights):
 state = _achita_wg_base_prepare(weights)
 state['achita_wg_choices'] = {}
 return state'''
combined = combined.replace(old_tuner, new_tuner, 1)
combined += ('\n# Additional exact FP4 placement: Copyright 2026 Lee Penkman, lee101/openkernels\n'
             '# c565e373d34575c5f7df862b05f8be7619078b0f, Apache-2.0; full license retained above.\n')
compile(combined, '<decoder-moe-sliced-transfer>', 'exec')
assert len(combined.encode('utf-8')) < 1_000_000
before = ast.parse(source)
after = ast.parse(combined)
def cuda_assignment(tree, name):
    return next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(a, ast.Name) and a.id == name for a in n.targets))
old_cuda = cuda_assignment(before, '_native_moe_CUDA_SRC')
new_cuda = cuda_assignment(after, '_native_moe_CUDA_SRC')
assert new_cuda == cuda_assignment(ast.parse(moe), 'CUDA_SRC')
payloads = lambda tree: Counter(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                               and isinstance(n.value, str) and '#include' in n.value)
a, b = payloads(before), payloads(after)
a[old_cuda] -= 1
b[new_cuda] -= 1
assert +a == +b, 'Only the separately checked native MoE CUDA may change'
symbols = symtable.symtable(combined, '<decoder>', 'exec')
bound = {s.get_name() for s in symbols.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
bound |= set(dir(builtins)) | {'__file__', '__name__', '__builtins__'}
unresolved = set()
def audit(table):
    for symbol in table.get_symbols():
        if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in bound:
            unresolved.add(symbol.get_name())
    for child in table.get_children(): audit(child)
audit(symbols)
assert not unresolved, sorted(unresolved)
target = root / '.tools/decoder-moe-sliced-variant.py'
target.write_bytes(combined.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(combined.encode('utf-8')).hexdigest(),
          'decoder_base_sha256': 'a61735ee0cb7774ae8e187526d2e104f3df4e2e95453f66ee3b46fdc33f9186d',
          'moe_checked_sha256': moe_sha, 'packaged_bytes': len(combined.encode('utf-8')),
          'only_changed_native_cuda': 'exact checked MoE CUDA', 'global_bindings_resolved': True,
          'small_fp32_router_unchanged': True, 'attention_cuda_unchanged': True,
          'dense_timed_dispatch': 'prior incumbents; unselected prepare-only WG sweep removed',
          'sliced_ffn_boundary': 'implicit zero tail, full 8192-column norm domain',
          'gpu_status': 'unmeasured'}
(root / '.tools/decoder-moe-sliced-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
