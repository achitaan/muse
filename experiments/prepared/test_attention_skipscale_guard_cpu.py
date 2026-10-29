"""Synthetic result fixtures test the admission guard; no GPU job or API call."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from attention_skipscale_evidence import PARENT,TRIAL,PREFLIGHT,bound_result,verify_trial,CHANGED

fixture = copy.deepcopy(bound_result(PARENT))
fixture['id'] = 'cpu_fixture_not_a_gpu_job'
body = fixture['result']
body['log_tail'] = 'ATTN_SKIP_RESOURCES 128 0 2 67584 66560 2 128 200 132\n'


def aggregate(result):
    totals = {}
    for row in result['result']['records']:
        if row.get('event') != 'benchmark':continue
        a,b = totals.setdefault(row['case']['load'],(0,0));w=row['case']['weight']
        row['median_ns'] = row['baseline_ns']/row['speedup']
        totals[row['case']['load']] = (a+w*row['baseline_ns'],b+w*row['median_ns'])
    result['result']['speedup'] = math.prod(a/b for a,b in totals.values())**(1/6)


for row in body['records']:
    if row.get('event') == 'benchmark' and row['index'] in CHANGED:
        row['speedup'] *= 1.10
    elif row.get('event') == 'profile':
        i=row['index']
        row['report'] = ('attn_kernel_skipscale<1,true,0>' if i in CHANGED else
                         'attn_kernel_skipscale<1,false,1>' if i<6 else
                         'attn_kernel_skipscale<8,false,1>')
aggregate(fixture)
with contextlib.redirect_stdout(io.StringIO()):verify_trial(fixture)
mutants=[]
for name,marker in (
    ('register_budget','129 0 2 67584 66560 2 128 200 132'),
    ('local_storage','128 8 2 67584 66560 2 128 200 132'),
    ('residency','128 0 1 67584 66560 2 128 200 132'),
    ('shared_memory','128 0 2 67585 66560 2 128 200 132')):
    bad=copy.deepcopy(fixture);bad['result']['log_tail']='ATTN_SKIP_RESOURCES '+marker;mutants.append((name,bad))
bad=copy.deepcopy(fixture);bad['result']['log_tail']='';mutants.append(('missing_resources',bad))
bad=copy.deepcopy(fixture);bad['result']['log_tail']*=2;mutants.append(('ambiguous_resources',bad))
bad=copy.deepcopy(fixture)
next(r for r in bad['result']['records'] if r.get('event')=='profile' and r['index']==3)['report']='attn_kernel_fp8pv<1,true,0>'
mutants.append(('new_backend_absent',bad))
bad=copy.deepcopy(fixture)
next(r for r in bad['result']['records'] if r.get('event')=='profile' and r['index']==0)['report']='attn_kernel_skipscale<1,true,0>'
mutants.append(('small_control_backend_changed',bad))
bad=copy.deepcopy(fixture);bad['result']['checks']['ok']=False;mutants.append(('integrity_failed',bad))
bad=copy.deepcopy(fixture)
bad['result']['records']=[r for r in bad['result']['records'] if not (r.get('event')=='test' and r['index']==15)]
mutants.append(('missing_test',bad))
bad=copy.deepcopy(fixture)
for row in bad['result']['records']:
    if row.get('event')=='benchmark' and row['index'] not in CHANGED:row['speedup']*=1.08
aggregate(bad);mutants.append(('control_drift',bad))
bad=copy.deepcopy(fixture)
for row in bad['result']['records']:
    if row.get('event')=='benchmark' and row['index'] in CHANGED:row['speedup']/=1.10/1.01
aggregate(bad);mutants.append(('insufficient_affected_gain',bad))
bad=copy.deepcopy(fixture);bad['result']['speedup']*=1.1;mutants.append(('forged_unweighted_aggregate',bad))
rejected=[]
for name,bad in mutants:
    try:
        with contextlib.redirect_stdout(io.StringIO()):verify_trial(bad)
    except (AssertionError,ValueError):rejected.append(name)
    else:raise AssertionError(('Invalid result fixture admitted',name))
report=dict(source_sha256=TRIAL,positive_fixture_passed=True,rejected=rejected,
            fixtures_are_not_measurements=True,no_gpu_jobs=True,no_api_calls=True)
target=PREFLIGHT/'guard-cpu-audit.json';data=(json.dumps(report,indent=2)+'\n').encode()
assert not target.exists() or target.read_bytes()==data,'Frozen guard audit collision'
target.write_bytes(data)
print(json.dumps(report,indent=2))
