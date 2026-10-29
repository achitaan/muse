"""Archive and interpret one existing terminal FP32-cache job, never launch it."""
import contextlib
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from moe_fp32_final_cache_evidence import TRIAL,PARENT,CUDA,directory,full_result,verify_trial


def main():
    folder=directory(TRIAL)
    assert hashlib.sha256((folder/'submission.py').read_bytes()).hexdigest()==TRIAL
    launch=json.loads((folder/'launch.json').read_bytes())
    result=json.loads((folder/'result.json').read_bytes())
    assert result['id']==launch.get('resolved_job_id',launch['job_id'])
    assert launch['source_sha256']==TRIAL
    assert result['status'] in ('done','failed','rejected'),'Follow the existing live job'
    body=result.get('result') or {}
    try:full_result(result);full=True
    except (AssertionError,KeyError,TypeError,ValueError):full=False
    marker='F32_CACHE_EVIDENCE '
    lines=[s[len(marker):] for s in body.get('log_tail','').splitlines() if s.startswith(marker)]
    e=json.loads(lines[0]) if len(lines)==1 else None
    if e:assert e['cuda_sha256']==CUDA
    profiles={r['index']:r['report'] for r in body.get('records',[]) if r.get('event')=='profile'}
    rows=[dict(m=c['m'],mode=c['mode'],new_over_old_median=c['median_ratio'],relative=c['relative'],worst=c['worst'],
               changed_input_graph_replays=c['changed_input_graph_replays']) for c in (e or {}).get('cases',[])]
    try:
        with contextlib.redirect_stdout(io.StringIO()):summary=verify_trial(result)
        material,rejection=True,None
    except (AssertionError,KeyError,TypeError,ValueError) as exc:
        material,summary,rejection=False,None,type(exc).__name__+': '+str(exc)
    report=dict(reviewed_utc=datetime.now(timezone.utc).isoformat(),job=result['id'],source_sha256=TRIAL,
                CUDA_sha256=CUDA,baseline_sha256=PARENT,status=result['status'],
                full5tests_6benchmarks_integrity=full,official_speedup=body.get('speedup'),
                selected_main=(e or {}).get('selected_main'),resources=(e or {}).get('resources'),
                skip=(e or {}).get('skip'),prepare_s=(e or {}).get('prepare_s'),synthetic_cases=rows,
                analytic=(e or {}).get('analytic'),evidence_markers=len(lines),
                changing_input_graph_replays=sum(len(c['changed_input_graph_replays']) for c in rows),
                official_profile_indices_with_new_namespace=[i for i,s in profiles.items() if 'okc_f32cache::' in s],
                official_speedup_is_new_path_evidence=bool(full and (e or {}).get('selected_main') is True and all('okc_f32cache::moe_fused<' in profiles.get(i,'') for i in (2,4,5))),
                material_guard_passed=material,material_summary=summary,rejection=rejection,
                statistical_significance_claim=False,scored_submissions_used=0,decode_transfer=False,
                derived_MoE_checks_left=12,derived_total_checks_left=90,
                next_new_check_not_before_utc='2026-10-29T10:59:18.187359+00:00',
                recommended_action='Review for scored publication and exact Decode transfer' if material else 'Retain33114871; no score or Decode transfer; close fixed cache design unless a bounded diagnosed defect exists')
    out=ROOT/'results/moe-fp32-final-cache-check-20261029';out.mkdir(exist_ok=True)
    def freeze(name,data):
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True)
        assert not p.exists() or p.read_bytes()==data,('Frozen archive collision',name)
        p.write_bytes(data)
    for name in ('submission.py','launch.json','launch-intent.json','quota-before-launch.json','jobs-before-launch.json','result.json'):
        freeze(name,(folder/name).read_bytes())
    for name in ('events.jsonl','run.json'):freeze('controller/'+name,(ROOT/'autoresearch-results'/name).read_bytes())
    for f in (ROOT/'autoresearch-results/logs').glob('*.json'):freeze('controller/logs/'+f.name,f.read_bytes())
    for i,s in profiles.items():freeze('profiles/'+str(i)+'.txt',s.encode())
    if (out/'review.json').exists():report['reviewed_utc']=json.loads((out/'review.json').read_bytes())['reviewed_utc']
    freeze('review.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report))


if __name__=='__main__':main()
