"""Admit FP8Q only with full source-bound coverage and material changed-path gain."""
import hashlib
import json
import math
from pathlib import Path
from verify_attention_precision import verify

ROOT=Path(__file__).resolve().parents[1]
BASE='24782695e507801dcf4e753349a97de723d99e3463b2aac0c67a645ff53a884a'
raw=(ROOT/'candidates/attention_active.py').read_bytes()
digest=hashlib.sha256(raw).hexdigest()
verify(raw)
result=json.loads((ROOT/'.attention-measurements'/digest/'result.json').read_bytes())
b=result['result']
tests=[r for r in b['records'] if r.get('event')=='test']
benches=[r for r in b['records'] if r.get('event')=='benchmark']
assert result['status']=='done' and b['passed'] and b['checks']['ok']
assert len(tests)==16 and len(benches)==12
assert all(r['status']=='pass' for r in tests+benches)
if digest==BASE:
    print('Cached corrected baseline: full16/12/integrity.')
else:
    assert b['gpu']=='H200'
    profiles={r['index']:r.get('report','') for r in b['records'] if r.get('event')=='profile'}
    assert all('attn_kernel_fp8q<1,' in profiles.get(i,'') for i in range(6))
    base=json.loads((ROOT/'.attention-measurements'/BASE/'result.json').read_bytes())['result']
    old={r['index']:r['speedup'] for r in base['records'] if r.get('event')=='benchmark'}
    ratios=[r['speedup']/old[r['index']] for r in benches if r['index']<6]
    changed=math.prod(ratios)**(1/6)
    assert min(ratios)>=.98,('Changed-path regression',ratios)
    assert changed>1.025,('Require material changed-path gain',changed)
    assert b['speedup']/base['speedup']>1.015,'Require aggregate gain beyond the minimum interval floor'
    print('FP8Q full16/12/integrity; actual six new L1 paths; changed geometric gain:',changed,'ratios:',ratios)
