"""Review only the saved terminal job, never launch or query a GPU job."""
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from attention_skipscale_evidence import TRIAL,PARENT,CHANGED,CONTROLS,directory,full_result,verify_trial,bound_result

archive=ROOT/'results/attention-skipscale-20261029'
folder=archive if (archive/'result.json').exists() else directory(TRIAL)
assert hashlib.sha256((folder/'submission.py').read_bytes()).hexdigest()==TRIAL
launch=json.loads((folder/'launch.json').read_bytes())
result=json.loads((folder/'result.json').read_bytes())
assert launch['source_sha256']==TRIAL and result['id']==launch.get('resolved_job_id',launch['job_id'])
assert result['status'] in ('done','failed','rejected'),'Keep following the existing live handle'
body=result.get('result') or {}
try:
    full_result(result);complete=True
except (AssertionError,KeyError,ValueError,TypeError):complete=False
sink=io.StringIO()
try:
    with contextlib.redirect_stdout(sink):summary=verify_trial(result)
    admitted,rejection=True,None
except (AssertionError,KeyError,ValueError,TypeError) as exc:
    summary=None;admitted=False;rejection=type(exc).__name__+': '+str(exc)
lines=[s for s in body.get('log_tail','').splitlines() if s.startswith('ATTN_SKIP_RESOURCES ')]
resources=[int(v) for v in lines[0].split()[1:]] if len(lines)==1 else None
profiles={r['index']:r.get('report','').replace(' ','') for r in body.get('records',[]) if r.get('event')=='profile'}
new_cases=[i for i,s in profiles.items() if 'attn_kernel_skipscale<1,true' in s]
old_body,old_rows=full_result(bound_result(PARENT));old={r['index']:r for r in old_rows}
rows=[]
for r in body.get('records',[]):
    if r.get('event')=='benchmark' and r.get('status')=='pass':
        ratio=r['speedup']/old[r['index']]['speedup']
        rows.append(dict(index=r['index'],case=r['case'],speedup=r['speedup'],
                         old_source_speedup=old[r['index']]['speedup'],relative_speedup=ratio,
                         candidate_median_ns=r['median_ns'],old_source_median_ns=old[r['index']]['median_ns']))
manifest=json.loads((ROOT/'results/attention-skipscale-preflight-20261029/evaluator-identity.json').read_bytes())
identities={}
for f in manifest['files']:
    if f['file'] in ('baseline.py','reference.py','checks.py'):
        key='challenge/checks.py' if f['file']=='checks.py' else 'problems/muse_attention_decode/'+f['file']
        identities[f['file']]=body.get('evaluator',{}).get('files',{}).get(key)==f['pinned_sha256']
assert not admitted or all(identities.values()),'Evaluator identity differs'
report=dict(format_parser_repair_applied=True,scientific_thresholds_unchanged=True,evaluator_file_identities=identities,reviewed_utc=datetime.now(timezone.utc).isoformat(),job=result['id'],source_sha256=TRIAL,
            parent_sha256=PARENT,status=result['status'],full16tests_12benchmarks_integrity=complete,
            official_speedup=body.get('speedup'),official_interval=body.get('speedup_ci'),
            resources=resources,resource_marker_count=len(lines),new_flat_L1_profile_cases=new_cases,
            new_path_full_suite_verified=bool(complete and set(new_cases)==set(CHANGED)),
            benchmark_comparisons=rows,cross_job_times_are_not_paired_samples=True,
            material_guard_passed=admitted,material_summary=summary,rejection=rejection,
            statistical_significance_claim=False,
            next_action='Review exact-source publication and narrowly bound Decode transfer' if admitted else 'Retain9e922c35; no score, transfer or unmotivated retry')
target=folder/'review.json'
if target.exists():
    previous=json.loads(target.read_bytes());report['reviewed_utc']=previous['reviewed_utc']
    assert previous==report,'Frozen review differs'
else:target.write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps(report,indent=2))
