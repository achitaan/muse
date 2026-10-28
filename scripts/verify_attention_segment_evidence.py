"""Exact-source full coverage plus required partition and index-space repair."""
import ast
import hashlib
import json
from pathlib import Path
from verify_attention_precision import verify

ROOT=Path(__file__).resolve().parents[1]
raw=(ROOT/'candidates/attention_active.py').read_bytes();verify(raw)
digest=hashlib.sha256(raw).hexdigest()
tree=ast.parse(raw)
cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
assert 'int seg[F ? ((L == 1) ? MAXB : MAXSEG) : 1][3];' in cuda
assert 'S.cum[b] < ceg && n < MAXB' in cuda
assert 'static constexpr int MINB = 2;' in cuda and 'ATTN_PV3_RESOURCES' not in cuda
fits=ast.unparse(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fits'))
assert "cfg['batch'] * (data['page_table'].shape[1] * cfg['page_size'] + cfg['q_len'] + 600) < INDEX_LIMIT" in fits
r=json.loads((ROOT/'.attention-measurements'/digest/'result.json').read_bytes());b=r['result']
tests=[x for x in b['records'] if x.get('event')=='test'];benches=[x for x in b['records'] if x.get('event')=='benchmark']
assert r['status']=='done' and b['passed'] and b['checks']['ok']
assert len(tests)==16 and len(benches)==12 and all(x['status']=='pass' for x in tests+benches)
print('Full16/12/integrity for exact repaired source. Segment capacity512, int32 shape guard, original2CTA configuration.')
