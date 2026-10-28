"""Require full evidence, safe actual occupancy and gains on changed loads."""
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import runpy

ROOT=Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT/'scripts/verify_moe_backend_evidence.py'),run_name='__main__')
raw=(ROOT/'candidates/moe_active.py').read_bytes();tree=ast.parse(raw)
literal={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
if 'W2G_CUDA' not in literal:raise SystemExit(0) # cached retained baseline
cuda=literal['W2G_CUDA']
assert cuda.count('achita_marlin_out2(')==12
assert '__bfloat1622float2(__hmul2_rn(acc, __bfloat162bfloat162(gs)))' in cuda
assert 'float* ybuf' in cuda and 'st_keep_f32(p.ybuf' in cuda
assert '__launch_bounds__(NTL, 2)' in cuda
d=hashlib.sha256(raw).hexdigest();b=json.loads((ROOT/'.moe-measurements'/d/'result.json').read_text())['result']
log=b.get('log_tail','')
assert re.findall(r'^MOE_W2G_ACTIVE (True|False)$',log,re.MULTILINE)==['True']
assert re.findall(r'^MOE_W2G_ZERO (True|False)$',log,re.MULTILINE)==['True']
resources=re.findall(r'^MOE_W2G_RESOURCES (\([^\n]*\))$',log,re.MULTILINE)
assert len(resources)==1
values=ast.literal_eval(resources[0]);assert len(values)==13
assert values[0]==320 and values[2]>0 and 2*values[2]<=values[11]
assert min(values[5],values[8])>=2,'Two-CTA resource advantage must exist on the real GPU'
# Report local-memory bytes; admission is based on actual occupancy and
# changed-load timings, rather than assuming every spill necessarily loses.
assert values[3]<=96 and values[6]<=96
assert values[-1]>=2*values[9]
profiles={r['index']:r.get('report','') for r in b['records'] if r.get('event')=='profile'}
assert all('okc_w2g::moe_fused<' in profiles.get(i,'') for i in (2,4,5)),'Require actual changed M32/128/256 paths'
base=json.loads((ROOT/'.moe-measurements/331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50/result.json').read_text())['result']
old={r['index']:r['speedup'] for r in base['records'] if r.get('event')=='benchmark'}
new={r['index']:r['speedup'] for r in b['records'] if r.get('event')=='benchmark'}
ratios=[new[i]/old[i] for i in (2,4,5)]
assert min(ratios)>=.99,'Do not conceal a changed-load regression with other aggregate gains'
affected=math.prod(ratios)**(1/3)
assert affected>1.02,'Require a material changed-load gain from the architectural rewrite'
print('WGS2 real resources:',values,'changed-load geometric ratio:',affected)
