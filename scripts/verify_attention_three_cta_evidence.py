"""Require real three-CTA resources and gains beyond unchanged controls."""
import hashlib
import json
import math
from pathlib import Path
import re
from verify_attention_precision import verify

ROOT=Path(__file__).resolve().parents[1]
BASE='a310b988ce0eaccb7ff5db7016dc041ec900019528bc8c136e34a278c3fd9372'
raw=(ROOT/'candidates/attention_active.py').read_bytes();verify(raw)
digest=hashlib.sha256(raw).hexdigest()
r=json.loads((ROOT/'.attention-measurements'/digest/'result.json').read_bytes());b=r['result']
tests=[x for x in b['records'] if x['event']=='test'];benches=[x for x in b['records'] if x['event']=='benchmark']
assert r['status']=='done' and b['passed'] and b['checks']['ok']
assert len(tests)==16 and len(benches)==12 and all(x['status']=='pass' for x in tests+benches)
if digest==BASE:
    print('Cached FP8PV baseline: full16/12/integrity.')
else:
    assert b['gpu']=='H200'
    lines=re.findall(r'^ATTN_PV3_RESOURCES ([^\r\n]+)',b.get('log_tail',''),re.MULTILINE)
    assert len(lines)==1
    values=tuple(map(int,lines[0].split()));assert len(values)==11
    regs,local,occ,dynamic,size,minb,average,consumer,sms,rsm,ssm=values
    assert occ>=3 and dynamic==67584 and size==66560
    assert (minb,average,consumer)==(3,80,104)
    assert 3*dynamic<=ssm and 3*(128*consumer+128*56)<=rsm
    profiles={x['index']:x.get('report','') for x in b['records'] if x['event']=='profile'}
    assert all('attn_kernel_pv3<1, true,' in profiles.get(i,'') for i in (3,5))
    base=json.loads((ROOT/'.attention-measurements'/BASE/'result.json').read_bytes())['result']
    old={x['index']:x['speedup'] for x in base['records'] if x['event']=='benchmark'}
    ratios={x['index']:x['speedup']/old[x['index']] for x in benches}
    affected=math.sqrt(ratios[3]*ratios[5])
    controls=math.prod(v for i,v in ratios.items() if i not in (3,5))**.1
    assert min(ratios[3],ratios[5])>=.99
    assert affected/controls>1.03,('Require material gain beyond controls',affected,controls)
    assert b['speedup']/base['speedup']>1.015
    print('Actual resources:',values,'changed ratio:',affected,'controls:',controls,'normalized:',affected/controls)
