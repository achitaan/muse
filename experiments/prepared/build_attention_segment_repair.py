"""Correctness-only fallback if three-CTA performance loses; no GPU run here."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
sys.path.insert(0,str(ROOT/'experiments/prepared'))
from build_attention_three_cta import parent_source,literal,BASE
raw=parent_source();source=raw.decode();tree=ast.parse(source);node,old=literal(source)
new=old;edits=[]
for a,b in [('int seg[F ? MAXSEG : 1][3];','int seg[F ? ((L == 1) ? MAXB : MAXSEG) : 1][3];'),
            ('S.cum[b] < ceg && n < MAXSEG','S.cum[b] < ceg && n < MAXB')]:
    assert new.count(a)==1;new=new.replace(a,b);edits.append(dict(before=a,after=b,count=1))
inverse=new
for e in reversed(edits):inverse=inverse.replace(e['after'],e['before'])
assert inverse==old
fits=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fits')
newfits=copy.deepcopy(fits)
bound=ast.parse("cfg['batch'] * (data['page_table'].shape[1] * cfg['page_size'] + cfg['q_len'] + 600) < INDEX_LIMIT",mode='eval').body
newfits.body[-1].value=ast.BoolOp(op=ast.And(),values=[newfits.body[-1].value,bound])
lines=source.splitlines(keepends=True)
def offsets(n):return sum(map(len,lines[:n.lineno-1]))+n.col_offset,sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
replacements=[(*offsets(node.value),repr(new)),(*offsets(fits),ast.unparse(newfits))]
result=source
for start,end,value in sorted(replacements,reverse=True):result=result[:start]+value+result[end:]
resulttree=ast.parse(result)
for n in (node,fits):
    name=n.targets[0].id if isinstance(n,ast.Assign) else n.name
    idx=next(i for i,t in enumerate(resulttree.body) if (isinstance(t,ast.Assign) and any(isinstance(a,ast.Name) and a.id==name for a in t.targets)) or (isinstance(t,ast.FunctionDef) and t.name==name))
    resulttree.body[idx]=n
assert ast.dump(resulttree)==ast.dump(tree)
out=ROOT/'.tools/attention-segment-repair-variant.py';out.write_bytes(result.encode())
report=dict(parent_sha256=BASE,source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,
            inverse_cuda_exact=True,all_other_ast_unchanged=True,changed_python_functions=['_fits'],
            math_registers_grid_unchanged=True,edits=edits,gpu_verified=False)
(ROOT/'.tools/attention-segment-repair-source-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report,indent=2))
