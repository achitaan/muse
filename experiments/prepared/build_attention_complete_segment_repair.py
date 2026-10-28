"""Complete segment-domain guard; native CUDA is the exact checked c050 source."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from build_attention_three_cta import ROOT,literal

BASE='c05090e240b22b6e2e20f644cfaf48aff969d0b0418e448c4ca369ce7fcb8ba9'
paths=[ROOT/'.attention-measurements'/BASE/'submission.py',ROOT/'results/attention-segment-repair-20261028/check'/BASE/'submission.py',ROOT/'.tools/attention-segment-repair-variant.py']
raw=next(p.read_bytes() for p in paths if p.is_file())
assert hashlib.sha256(raw).hexdigest()==BASE
text=raw.decode();tree=ast.parse(text)
fits=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fits')
updated=copy.deepcopy(fits)
updated.body[-1].value=ast.BoolOp(op=ast.And(),values=[updated.body[-1].value,
    ast.parse("cfg['q_len'] != 8 or cfg['batch'] <= 64",mode='eval').body])
lines=text.splitlines(keepends=True)
start=sum(map(len,lines[:fits.lineno-1]))+fits.col_offset
end=sum(map(len,lines[:fits.end_lineno-1]))+fits.end_col_offset
result=text[:start]+ast.unparse(updated)+text[end:]
assert literal(result)[1]==literal(text)[1]
outtree=ast.parse(result)
outfit=next(n for n in outtree.body if isinstance(n,ast.FunctionDef) and n.name=='_fits')
outtree.body[outtree.body.index(outfit)]=fits
assert ast.dump(outtree)==ast.dump(tree)
out=ROOT/'.tools/attention-complete-segment-variant.py';out.write_bytes(result.encode())
report=dict(parent_sha256=BASE,source_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),bytes=out.stat().st_size,
            native_cuda_exact=True,all_other_ast_unchanged=True,changed_functions=['_fits'],
            fallback_guard='L8 batch>64 uses the existing general path',scored_evaluation_pending=True)
(ROOT/'.tools/attention-complete-segment-source-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report,indent=2))
