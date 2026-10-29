"""Fixed source, whole-layer coverage, resource, execution and retention gates."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import zlib

ROOT=Path(__file__).resolve().parents[1]
PARENT='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
TRIAL='ad7b55cfdf01408842510fca48eef235b95e73fcd9d721b73ce418d9f561ea92'
CUDA='51180052ffd10dfb11ab1bb747c11192922c7effdd2669b4389185184028e4b1'
PACKET=ROOT/'results/decoder-cooperative-boundary-preflight-20261029'
CHANGED=(0,1,6,7)
CONTROLS=tuple(i for i in range(12) if i not in CHANGED)


def read(path): return json.loads(path.read_bytes())


def source():
    raw=(ROOT/'candidates/decode_active.py').read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT,TRIAL),('Unreviewed source',sha)
    return raw,sha


def directory(sha):
    assert sha in (PARENT,TRIAL)
    return ROOT/'results/decoder-segment-repair-20261028' if sha==PARENT else ROOT/'.decoder-cooperative-measurements'/sha


def bound_result(sha):
    folder=directory(sha)
    assert hashlib.sha256((folder/'submission.py').read_bytes()).hexdigest()==sha
    launch,result=read(folder/'launch.json'),read(folder/'result.json')
    assert launch['source_sha256']==sha
    assert result['id']==launch.get('resolved_job_id',launch['job_id'])
    return result


def validate_packet():
    folder=PACKET/'v3'
    raw=(folder/'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    binding=read(folder/'binding.json')
    assert binding['source_sha256']==TRIAL and binding['CUDA_sha256']==CUDA
    assert binding['exact_inverse_restores_packaged_parent'] and binding['all_other_original_module_nodes_unchanged']
    assert binding['scope_tokens']==[1,8] and binding['host_pybind_header_explicit']
    cuda=(folder/'kernel.cu').read_bytes()
    assert hashlib.sha256(cuda).hexdigest()==CUDA
    tree=ast.parse(raw)
    definitions={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign)
                 and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    expr=definitions['_CB_CUDA']
    assert isinstance(expr,ast.Call) and expr.func.id=='_zsrc' and not expr.keywords and len(expr.args)==1
    assert zlib.decompress(base64.b64decode(ast.literal_eval(expr.args[0]),validate=True))==cuda
    cpp=ast.literal_eval(definitions['_CB_CPP'])
    assert hashlib.sha256(cpp.encode()).hexdigest()==binding['CPP_sha256']
    assert '#include <pybind11/pybind11.h>' in cpp
    screen=read(ROOT/'results/decoder-cooperative-boundary-screen-20261029/screen.json')
    model=read(ROOT/'results/decoder-cooperative-boundary-screen-20261029/model-v2.json')
    assert screen['baseline_sha256']==PARENT and screen['inverse_edit_restores_selected_GEMV_device_prefix']
    assert model['numerical_cases']==40 and model['finalizer_row_ownership_cases']==36
    assert all(model['mutants_detected'].values()) and model['post_eps']==1e-8 and model['pre_eps']==1e-5
    identity=read(PACKET/'evaluator-identity.json')
    assert identity['problem']=='muse_decode_layer'
    assert {r['file'] for r in identity['files']}=={'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md','nvfp4.py'}
    assert all(r['exact'] and r['pinned_sha256']==r['fresh_sha256'] for r in identity['files'])
    assert all(hashlib.sha256((ROOT/'muse_decode_layer'/r['file']).read_bytes()).hexdigest()==r['pinned_sha256'] for r in identity['files'])
    cpu=read(PACKET/'host-CPU-admission.json')
    assert cpu['source_sha256']==TRIAL and cpu['query_and_launch_arity']==20 and cpu['unsupported_layout_rejected']
    assert cpu['candidate_python_compiles'] and cpu['CPP_declaration_matches_host']
    guard=read(PACKET/'guard-CPU-admission.json')
    assert guard['source_sha256']==TRIAL and guard['positive_fixture_passed'] and len(guard['rejected_mutants'])==15
    return raw


def case_identity(case): return {k:v for k,v in case.items() if k!='seed'}


def full_result(result):
    body=result.get('result') or {}
    assert result['status']=='done' and result['board']=='muse_decode_layer'
    assert body.get('passed') is True and body.get('gpu')=='H200' and body.get('checks',{}).get('ok') is True
    assert math.isfinite(body['speedup']) and body['speedup']>0
    reference=read(directory(PARENT)/'result.json')['result']
    rows_by_event={}
    for event,count in (('test',13),('benchmark',12)):
        rows=sorted((r for r in body['records'] if r.get('event')==event),key=lambda r:r['index'])
        expected=sorted((r for r in reference['records'] if r.get('event')==event),key=lambda r:r['index'])
        assert [r['index'] for r in rows]==list(range(count))
        assert all(r.get('status')=='pass' for r in rows)
        assert [case_identity(r['case']) for r in rows]==[case_identity(r['case']) for r in expected]
        rows_by_event[event]=rows
    rows=rows_by_event['benchmark'];loads={}
    assert all(math.isfinite(r['speedup']) and r['speedup']>0 and r['median_ns']>0 and r['baseline_ns']>0 for r in rows)
    for row in rows:
        load=row['case']['load'];w=row['case']['weight']
        assert w==(46 if row['case']['layer']=='sliding' else 16)
        a,b=loads.setdefault(load,(0.,0.))
        loads[load]=(a+w*row['baseline_ns'],b+w*row['median_ns'])
    assert len(loads)==6
    assert math.isclose(math.prod(a/b for a,b in loads.values())**(1/6),body['speedup'],rel_tol=1e-9)
    return body,rows,loads


def markers(text,prefix):
    return [json.loads(line[len(prefix)+1:]) for line in text.splitlines() if line.startswith(prefix+' ')]


def verify_trial(result,quiet=False):
    body,rows,loads=full_result(result)
    log=body.get('log_tail','')
    selected=markers(log,'CB_SELECTED')
    assert selected==[[1,8]],'Both fixed families must really be selected'
    resources=markers(log,'CB_RESOURCES')
    assert {(r['family'],r['fmt']) for r in resources}=={(1,0),(1,1),(8,0),(8,1)} and len(resources)==4
    for r in resources:
        q=r['query'];m=r['family'];grid,nt=(128,1024) if m==1 else (512,256)
        assert q['1']==q['7']==1 and q['3']==q['4']==0
        assert 0<q['2']<=64 and q['5']>=(1 if m==1 else 4) and grid<=q['5']*q['6']
        assert q['8']==(16384 if m==1 else 4096)
    probes=markers(log,'CB_PROBES_Z')
    assert {(p['family'],p['fmt']) for p in probes}=={(1,0),(1,1),(8,0),(8,1)} and len(probes)==4
    for p in probes:
        cases=json.loads(zlib.decompress(base64.b64decode(p['payload'],validate=True)))
        assert len(cases)==p['cases']==(8 if p['family']==1 else 32)
        assert sum(c['fixture'].startswith('fresh_graph_replay_') for c in cases)==4
        assert {c['m'] for c in cases}==({1} if p['family']==1 else set(range(2,9)))
        expected={(m,f) for m in ((1,) if p['family']==1 else range(2,9))
                  for f in ('random','zero','outlier','tiny')}
        expected|={(p['family'],f'fresh_graph_replay_{i}') for i in range(4)}
        assert {(c['m'],c['fixture']) for c in cases}==expected
        for case in cases:
            assert len(case['errors'])==4 and case['errors'][3]==[0.,0.]
            for i,values in enumerate(case['errors'][:3]):
                assert len(values)==2 and all(math.isfinite(v) and 0<=v<=(1e-3 if i==0 else 2e-5) for v in values)
    pairs=markers(log,'CB_PAIR')
    assert {p['family'] for p in pairs}=={1,8} and len(pairs)==2
    for p in pairs:
        samples=p['samples'];assert len(samples)==16 and p['keep'] is True
        assert [s['pair'] for s in samples]==list(range(16))
        assert all(s['order']==([0,1] if s['pair']%2==0 else [1,0]) for s in samples)
        assert all(s['old']>0 and s['new']>0 for s in samples)
        ratios=[s['old']/s['new'] for s in samples]
        gm=math.prod(ratios)**(1/16)
        assert math.isclose(gm,p['gain'],rel_tol=1e-12) and gm>=1.08
        assert p['wins']==sum(r>1 for r in ratios)>=12
    profiles={r['index']:r.get('report','') for r in body['records'] if r.get('event')=='profile'}
    assert set(profiles)==set(range(12))
    assert all('cb_smallm2_gemv' in profiles[i] for i in (0,1))
    assert all('cb_v18s_gemv' in profiles[i] for i in (6,7))
    assert all('cb_smallm2_gemv' not in profiles[i] and 'cb_v18s_gemv' not in profiles[i] for i in CONTROLS)
    old,old_rows,old_loads=full_result(bound_result(PARENT))
    prior={r['index']:r['speedup'] for r in old_rows}
    ratios={r['index']:r['speedup']/prior[r['index']] for r in rows}
    gain_by_load={load:(a/b)/(old_loads[load][0]/old_loads[load][1]) for load,(a,b) in loads.items()}
    affected_loads={r['case']['load'] for r in rows if r['index'] in CHANGED}
    assert len(affected_loads)==2
    controls=math.prod(v for k,v in gain_by_load.items() if k not in affected_loads)**.25
    affected=math.prod(ratios[i] for i in CHANGED)**.25
    track=math.prod(gain_by_load.values())**(1/6)
    summary=dict(affected_case_geomean=affected,control_weighted_load_geomean=controls,
        aggregate_ratio=body['speedup']/old['speedup'],reconstructed_track_ratio=track,
        control_normalized_track_gain=track/controls,per_case_ratios=ratios,weighted_load_gains=gain_by_load,
        statistical_significance_claim=False,resources=resources,component_pairs=pairs)
    if not quiet: print(json.dumps(summary),flush=True)
    assert .95<=controls<=1.05,'Controls drifted; retain baseline'
    assert affected>=1.005,'No material affected-case gain'
    assert track/controls>=1.003,'No material control-normalized weighted track gain'
    assert all(gain_by_load[k]/controls>=.995 for k in affected_loads),'Affected load regressed'
    return summary
