"""One scored confirmation, exact checked bytes; resume its persisted handle."""
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
sys.path.insert(0,str(ROOT/'scripts'))
from official import cli,LINUX_ROOT
sha='a310b988ce0eaccb7ff5db7016dc041ec900019528bc8c136e34a278c3fd9372'
d=ROOT/'.attention-measurements'/sha
raw=(d/'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()==sha
b=json.loads((d/'result.json').read_bytes())['result']
assert b['passed'] and b['checks']['ok']
assert sum(r['event']=='test' for r in b['records'])==16
assert sum(r['event']=='benchmark' for r in b['records'])==12
launch_path=d/'scored-launch.json'
if launch_path.exists():launch=json.loads(launch_path.read_bytes())
else:
    launch=cli(['submit','muse_attention_decode',f'{LINUX_ROOT}/{(d/"submission.py").relative_to(ROOT).as_posix()}', '--no-wait','--json'])
    launch['source_sha256']=sha
    launch_path.write_bytes(json.dumps(launch,indent=2).encode())
job=launch['job_id'];print('Scored confirmation:',job,flush=True)
previous=None
while True:
    try:r=cli(['status',job,'--json'])
    except Exception as e:
        print('Observation failed; keep handle',job,type(e).__name__,flush=True)
        time.sleep(20);continue
    (d/'scored-result.json').write_bytes(json.dumps(r,indent=2).encode())
    if r['status']!=previous:print('Status',r['status'],flush=True);previous=r['status']
    if r['status']=='duplicate' and r.get('duplicate_of'):
        job=r['duplicate_of'];continue
    if r['status'] in ('done','failed','rejected','duplicate'):
        b=r.get('result') or {}
        print(json.dumps({k:b.get(k) for k in ('passed','speedup','speedup_ci','checks','error')},indent=2),flush=True)
        break
    time.sleep(45)
