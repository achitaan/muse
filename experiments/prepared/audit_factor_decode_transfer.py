"""Map exact retained MoE code into Decode; no integration or GPU work."""
import ast
import copy
import hashlib
import json
import operator
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
paths = ('best/moe_verified.py', 'best/decoder_verified.py')
raws = [(ROOT / p).read_bytes() for p in paths]
expected = ('331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50',
            'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988')
assert tuple(hashlib.sha256(r).hexdigest() for r in raws) == expected
trees = list(map(ast.parse, raws))


class Normalize(ast.NodeTransformer):
    @staticmethod
    def name(s):
        return s.removeprefix('_native_moe_')

    def visit_Name(self, node):
        node.id = self.name(node.id)
        return node

    def visit_FunctionDef(self, node):
        node.name = self.name(node.name)
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
        return node


def functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def literals(tree):
    found = {}
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.FloorDiv: operator.floordiv, ast.LShift: operator.lshift}
    def value(node):
        if isinstance(node, ast.Name):
            return found[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](value(node.left), value(node.right))
        return ast.literal_eval(node)
    for n in tree.body:
        if isinstance(n, ast.Assign):
            try:
                v = value(n.value)
            except (KeyError, ValueError, TypeError):
                continue
            for target in n.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = v
    return found


a, b = list(map(literals, trees))
assert a['CUDA_SRC'] == b['_native_moe_CUDA_SRC']
assert a['CPP_SRC'] == b['_native_moe_CPP_SRC']
constants = ('MAIN_DEFINES', 'MAIN_FALLBACK_DEFINES', 'TINY_DEFINES', 'TINY2_DEFINES',
             'MID_DEFINES', 'MAIN_NS', 'DQ7', 'PDL', 'HIDDEN', 'CHUNK_TOKENS',
             'EMAX', 'WGS_FIXF', 'WGS_XROWS', 'TINY_MAX_M', 'TINY2_MAX_M', 'MID_MAX_M')
for name in constants:
    assert a[name] == b['_native_moe_' + name], name
fa, fb = list(map(functions, trees))
equal, different = [], []
names = ('_scales_prmt_exact', '_stored_pos', '_split_nibbles', '_scatter_code_bits',
         '_pack_codes', '_pack_scales', '_fat_order', '_unit_order', '_pack_codes_nat',
         '_wgs_copy', '_main_defines', '_wgs_on', '_make_scratch')
for name in names:
    x = Normalize().visit(copy.deepcopy(fa[name]))
    y = Normalize().visit(copy.deepcopy(fb['_native_moe_' + name]))
    (equal if ast.dump(x, include_attributes=False) == ast.dump(y, include_attributes=False) else different).append(name)
assert not different, different
report = dict(moe_sha256=expected[0], decode_sha256=expected[1], cpu_only=True,
              exact_shared_cuda_bytes=len(a['CUDA_SRC'].encode()),
              exact_shared_cuda_sha256=hashlib.sha256(a['CUDA_SRC'].encode()).hexdigest(),
              exact_CPP_SRC=True, equal_dispatch_constants=list(constants),
              namespace_normalized_AST_equal_helpers=equal,
              integration_built=False, gpu_verified=False,
              next_condition='Standalone retained gain first; then replace eligible MAIN build only, preserve small/ineligible paths, and validate alternating-shape scratch reuse and whole-layer numerics/performance')
(ROOT / '.tools/factor-decode-transfer-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
print(json.dumps(report, indent=2))
