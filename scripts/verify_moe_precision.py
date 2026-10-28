"""Protect the FP32 small-M norm contract verified against pinned SGLang.

This structural check supplements the official full GPU correctness tests.
It does not establish GPU correctness or speed.
"""
import ast
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tree = ast.parse((root / 'candidates/moe_active.py').read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_achita_moe_small_finalize']
assert len(functions) == 1, 'Expected one small-M finalizer'
fn = functions[0]
for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
    if isinstance(call.func, ast.Attribute) and call.func.attr == 'to':
        assert len(call.args) == 1 and ast.unparse(call.args[0]) == 'tl.float32', \
            'Norm intermediates must remain FP32 through the final store'
assignments = {target.id: node.value for node in fn.body if isinstance(node, ast.Assign)
               for target in node.targets if isinstance(target, ast.Name)}
assert ast.unparse(assignments['scaled']) == 'values * norm[None, :]'
assert ast.unparse(assignments['normalized']) == 'scaled * inv[:, None]'
assert 'normalized * weights[:, None]' in ast.unparse(assignments['result'])
print('FP32 norm-intermediate contract verified; GPU validation remains authoritative.')
