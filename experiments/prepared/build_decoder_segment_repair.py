"""Port only the required coverage repair, retaining Decode's FP16 QK/PV."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
BASE='1fb5d05d7ed5ed6ee490ee0e5ba3b612b24895448b4c86d4f298620d6ae32432'
paths=[ROOT/'.decode-measurements'/BASE/'submission.py',
       ROOT/'results/decoder-attention-bf16-boundaries-20261028/publication'/BASE/'submission.py',
       ROOT/'best/decoder_verified.py']
raw=next(p.read_bytes() for p in paths if p.is_file());assert hashlib.sha256(raw).hexdigest()==BASE
tree=ast.parse(raw)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_native_attn_CUDA_SRC' for t in n.targets))
old=ast.literal_eval(node.value);new=old;edits=[]
for before,after in [('int seg[F ? MAXSEG : 1][3];','int seg[F ? ((L == 1) ? MAXB : MAXSEG) : 1][3];'),
                     ('S.cum[b] < ceg && n < MAXSEG','S.cum[b] < ceg && n < MAXB')]:
    assert new.count(before)==1
    new=new.replace(before,after);edits.append(dict(before=before,after=after,count=1))
assert 'mma_f8pv(' not in new and 'static constexpr int MINB = 2;' in new
reverse=new
for e in reversed(edits):reverse=reverse.replace(e['after'],e['before'])
assert reverse==old
fits=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_native_attn__fits')
fits2=copy.deepcopy(fits)
fits2.body[-1].value=ast.BoolOp(op=ast.And(),values=[fits2.body[-1].value,
    ast.parse("cfg['batch'] * (data['page_table'].shape[1] * cfg['page_size'] + cfg['q_len'] + 600) < _native_attn_INDEX_LIMIT",mode='eval').body,
    ast.parse("cfg['q_len'] != 8 or cfg['batch'] <= 64",mode='eval').body])
lines=raw.splitlines(keepends=True)
def offsets(n):return sum(map(len,lines[:n.lineno-1]))+n.col_offset,sum(map(len,lines[:n.end_lineno-1]))+n.end_col_offset
result=raw
for start,end,value in sorted([(*offsets(node.value),repr(new).encode()),(*offsets(fits),ast.unparse(fits2).encode())],reverse=True):
    result=result[:start]+value+result[end:]
updated=ast.parse(result)
for n in (node,fits):
    name=n.targets[0].id if isinstance(n,ast.Assign) else n.name
    i=next(i for i,t in enumerate(updated.body) if (isinstance(t,ast.Assign) and any(isinstance(x,ast.Name) and x.id==name for x in t.targets)) or (isinstance(t,ast.FunctionDef) and t.name==name))
    updated.body[i]=n
assert ast.dump(updated)==ast.dump(tree)
assert len(result)<1000000
out=ROOT/'.tools/decoder-segment-repair-variant.py';out.write_bytes(result)
report=dict(parent_sha256=BASE,source_sha256=hashlib.sha256(result).hexdigest(),bytes=len(result),
            inverse_cuda_exact=True,all_other_ast_unchanged=True,changed_python_functions=['_native_attn__fits'],
            fp16_attention_qk_pv_retained=True,moe_dense_router_residual_unchanged=True,edits=edits,gpu_verified=False)
(ROOT/'.tools/decoder-segment-repair-source-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps({k:v for k,v in report.items() if k!='edits'},indent=2))
