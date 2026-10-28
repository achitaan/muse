"""Compare embedded repair/guards to the independently audited standalone path."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_attention_three_cta import ROOT,literal
sys.path.insert(0,str(ROOT/'scripts'))
from verify_attention_precision import verify

raw=(ROOT/'.tools/decoder-segment-repair-variant.py').read_bytes();verify(raw,'_native_attn_')
tree=ast.parse(raw)
cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_native_attn_CUDA_SRC' for t in n.targets))
standalone=(ROOT/'.tools/attention-complete-segment-variant.py').read_bytes();stree=ast.parse(standalone)
scuda=literal(standalone)[1]
# Remove only the earlier standalone FP8PV transformation. It is deliberately
# excluded from Decode because the timing screen did not justify transfer.
edits=json.loads((ROOT/'.tools/attention-fp8pv-source-audit.json').read_bytes())['edits']
for e in reversed(edits):
    assert scuda.count(e['after'])==e['count']
    scuda=scuda.replace(e['after'],e['before'])
assert scuda==cuda
class Unprefix(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id.startswith('_native_attn_'):node.id=node.id[len('_native_attn_'):]
        return node
    def visit_FunctionDef(self,node):
        if node.name.startswith('_native_attn_'):node.name=node.name[len('_native_attn_'):]
        return self.generic_visit(node)
class FlattenAnd(ast.NodeTransformer):
    def visit_BoolOp(self,node):
        self.generic_visit(node)
        if isinstance(node.op,ast.And):
            node.values=[x for v in node.values for x in (v.values if isinstance(v,ast.BoolOp) and isinstance(v.op,ast.And) else [v])]
        return node
def fn(t,name):return copy.deepcopy(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name==name))
for name in ('_fits','_fallback'):
    a=Unprefix().visit(fn(tree,'_native_attn_'+name));b=fn(stree,name)
    for n in (a,b):
        if isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) and isinstance(n.body[0].value.value,str):n.body.pop(0)
    assert ast.dump(FlattenAnd().visit(a))==ast.dump(FlattenAnd().visit(b)),name
report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),cuda_repair_identical_after_removing_standalone_fp8pv=True,
            fits_guard_ast_identical=True,fallback_ast_identical=True,bf16_guard=True,
            reused_cpu_evidence='Standalone156native partitions/28835segments;66guard cases; B65L8 fallback FP64 oracle',
            numerical_dtype_router_and_residual_unchanged=True,bytes=len(raw),gpu_verified=False)
(ROOT/'.tools/decoder-segment-repair-cpu-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report,indent=2))
