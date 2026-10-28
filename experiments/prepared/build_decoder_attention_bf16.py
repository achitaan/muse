"""Transfer the exact corrected attention CUDA and fallback into Decode09."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
parent='09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67'
source=ROOT/'.decode-measurements'/parent/'submission.py'
if not source.exists():source=ROOT/'best/decoder_verified.py'
old=source.read_bytes()
assert hashlib.sha256(old).hexdigest()==parent
attention=(ROOT/'.tools/attention-bf16-boundaries-variant.py').read_bytes()
old_attention=(ROOT/'.attention-measurements/b56e07b1dbc930ef8fd4adaeba108e6d17b62a0f390d68c024b98d3cd608aee4/submission.py').read_bytes()
trees=[ast.parse(s) for s in (old,attention,old_attention)]
def assign(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def function(tree,name):
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
old_node=assign(trees[0],'_native_attn_CUDA_SRC')
new_node=assign(trees[1],'CUDA_SRC')
assert ast.literal_eval(old_node.value)==ast.literal_eval(assign(trees[2],'CUDA_SRC').value)
class Rename(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id in ('_rms','_rope'): node.id='_native_attn_'+node.id
        return node
    def visit_FunctionDef(self,node):
        node.name='_native_attn_'+node.name
        return self.generic_visit(node)
old_fallback=function(trees[0],'_native_attn__fallback')
renamed_old=Rename().visit(copy.deepcopy(function(trees[2],'_fallback')))
# Existing unparse strips function docstrings and uses compact whitespace.
if isinstance(renamed_old.body[0],ast.Expr) and isinstance(renamed_old.body[0].value,ast.Constant):renamed_old.body.pop(0)
assert ast.dump(renamed_old)==ast.dump(old_fallback)
new_fallback=Rename().visit(copy.deepcopy(function(trees[1],'_fallback')))
new_fallback.body.pop(0)
text=old.decode('utf8');lines=text.splitlines(keepends=True)
def offsets(node):
    return sum(map(len,lines[:node.lineno-1]))+node.col_offset,sum(map(len,lines[:node.end_lineno-1]))+node.end_col_offset
edits=[(*offsets(old_node.value),repr(ast.literal_eval(new_node.value))),(*offsets(old_fallback),ast.unparse(new_fallback))]
for start,end,replacement in sorted(edits,reverse=True):text=text[:start]+replacement+text[end:]
text=text.replace('# Local changes: decoder integration/cache lifetime, strict BF16 MoE boundaries,',
                  '# Local changes: integration/cache lifetime, strict BF16 MoE/attention boundaries,')
text=text.replace('# and all fallback computation retained. Shared scratch requires sequential replay.',
                  '# retained; attention fallback restores BF16 boundaries. Sequential scratch replay.')
text=text.replace('# Python metadata/formatting compacted for 1 MB cap; CUDA literals unchanged.',
                  '# Python metadata compacted for 1 MB; attention CUDA restores BF16 boundaries.')
out_tree=ast.parse(text)
out_cuda=assign(out_tree,'_native_attn_CUDA_SRC')
out_fallback=function(out_tree,'_native_attn__fallback')
assert ast.literal_eval(out_cuda.value)==ast.literal_eval(new_node.value)
assert ast.dump(out_fallback)==ast.dump(new_fallback)
out_tree.body[out_tree.body.index(out_cuda)]=old_node
out_tree.body[out_tree.body.index(out_fallback)]=old_fallback
assert ast.dump(out_tree)==ast.dump(trees[0]),'Unrelated Decode computation changed'
out=ROOT/'.tools/decoder-attention-bf16-variant.py';out.write_bytes(text.encode('utf8'))
report=dict(source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,parent_sha256=parent,attention_sha256=hashlib.sha256(attention).hexdigest(),attention_cuda_exact=True,fallback_ast_exact=True,all_other_ast_unchanged=True,router='original full FP32 small router; rejected Redux excluded',gpu_verified=False)
assert report['bytes']<1000000
(ROOT/'.tools/decoder-attention-bf16-source-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
