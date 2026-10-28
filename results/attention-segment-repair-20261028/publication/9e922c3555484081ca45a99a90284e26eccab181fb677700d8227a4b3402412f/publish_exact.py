"""One full scored validation of the complete repair; native CUDA already checked."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
sys.path.insert(0,str(ROOT/'scripts'))
from official import cli,LINUX_ROOT
SHA='9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
PARENT='c05090e240b22b6e2e20f644cfaf48aff969d0b0418e448c4ca369ce7fcb8ba9'
raw=(ROOT/'.tools/attention-complete-segment-variant.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==SHA
parentdir=ROOT/'.attention-measurements'/PARENT
b=json.loads((parentdir/'result.json').read_bytes())['result']
assert b['passed'] and b['checks']['ok']
assert sum(r['event']=='test' for r in b['records'])==16 and sum(r['event']=='benchmark' for r in b['records'])==12
def cuda(source):return next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
assert cuda(raw)==cuda((parentdir/'submission.py').read_bytes())
audit=json.loads((ROOT/'.tools/attention-complete-segment-cpu-audit.json').read_bytes());assert audit['source_sha256']==SHA and audit['guard_cases']==66
d=ROOT/'.attention-measurements'/SHA;d.mkdir(exist_ok=True)
frozen=d/'submission.py'
if frozen.exists():assert frozen.read_bytes()==raw
else:frozen.write_bytes(raw)
launchpath=d/'scored-launch.json'
if launchpath.exists():launch=json.loads(launchpath.read_bytes())
else:
    launch=cli(['submit','muse_attention_decode',f'{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}','--no-wait','--json'])
    launch['source_sha256']=SHA;launchpath.write_bytes(json.dumps(launch,indent=2).encode())
job=launch['job_id'];print('Complete repair scored job',job,flush=True)
last=None
while True:
    try:r=cli(['status',job,'--json'])
    except Exception as e:
        print('Observation failed; retaining',job,type(e).__name__,flush=True);time.sleep(20);continue
    (d/'scored-result.json').write_bytes(json.dumps(r,indent=2).encode())
    if r['status']!=last:print('Status',r['status'],flush=True);last=r['status']
    if r['status']=='duplicate' and r.get('duplicate_of'):job=r['duplicate_of'];continue
    if r['status'] in ('done','failed','rejected','duplicate'):
        b=r.get('result') or {};records=b.get('records',[])
        print(json.dumps(dict(job=job,source_sha256=SHA,tests=sum(x['event']=='test' for x in records),
                         benchmarks=sum(x['event']=='benchmark' for x in records),
                         **{k:b.get(k) for k in ('passed','speedup','speedup_ci','checks','error')}),indent=2),flush=True)
        break
    time.sleep(45)
