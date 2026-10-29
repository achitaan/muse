"""Complete-layer, native resource, actual-path and weighted gain gates."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import zlib

from decoder_cooperative_evidence import full_result, bound_result as retained_result, markers

ROOT=Path(__file__).resolve().parents[1]
PARENT='d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
PACKAGED='307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072'
TRIAL='f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf'
CUDA='c84debd7016155722eda211709914721b254f8383d091e9154a72f97807e87c6'
PACKET=ROOT/'results/decoder-cluster-norm-preflight-20261029'
CHANGED=tuple(range(8));CONTROLS=(8,9,10,11)


def read(path):return json.loads(path.read_bytes())


def source():
    raw=(ROOT/'candidates/decode_active.py').read_bytes();sha=hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT,TRIAL),('Unreviewed source',sha)
    return raw,sha


def directory(sha):
    assert sha in (PARENT,TRIAL)
    return ROOT/'results/decoder-segment-repair-20261028' if sha==PARENT else ROOT/'.decoder-cluster-norm-measurements'/sha


def bound_result(sha):
    if sha==PARENT:return retained_result(sha)
    folder=directory(sha)
    assert hashlib.sha256((folder/'submission.py').read_bytes()).hexdigest()==sha
    launch,result=read(folder/'launch.json'),read(folder/'result.json')
    assert launch['source_sha256']==sha and result['id']==launch.get('resolved_job_id',launch['job_id'])
    return result


def verify_source(raw):
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    packaged=(ROOT/'results/decoder-unused-rope-prune-20261029/submission.py').read_bytes()
    assert hashlib.sha256(packaged).hexdigest()==PACKAGED and raw.startswith(packaged)
    parent_tree=ast.parse(packaged);tree=ast.parse(raw)
    assert ast.dump(ast.Module(body=tree.body[:len(parent_tree.body)],type_ignores=[]))==ast.dump(parent_tree)
    constants={n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    expr=constants['_CN_CUDA']
    assert isinstance(expr,ast.Call) and isinstance(expr.func,ast.Name) and expr.func.id=='_zsrc'
    cuda=zlib.decompress(base64.b64decode(ast.literal_eval(expr.args[0]),validate=True))
    assert hashlib.sha256(cuda).hexdigest()==CUDA
    native=(ROOT/'experiments/prepared/decoder_cluster_norm.cu').read_bytes()
    host=(ROOT/'experiments/prepared/decoder_cluster_norm_host.cu').read_bytes()
    assert cuda==native+b'\n'+host
    assert hashlib.sha256(native).hexdigest()=='49e61e6a225b9a9ec245aeeb41554ff8b717c06cf8053d387dc1ddd6eaa72653'
    assert cuda==(PACKET/'submission-v3/kernel.cu').read_bytes()
    cpp=ast.literal_eval(constants['_CN_CPP']);text=host.decode()
    assert '#include <pybind11/pybind11.h>' in cpp
    a=text.index('void cn_launch(');b=text.index(' {\n#define CN_ARGS',a)
    assert text[a:b]+';' in cpp
    launch=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='cn_launch')
    assert len(launch.args)==16 and not launch.keywords
    # Source hash binds these operations; assertions also make the required
    # precision/lifetime invariants visible to this independent verifier.
    device=native.decode()
    assert 'float partial[2][NT/32]' in device and device.count('cluster.sync();')==2
    assert 'rank*PART+threadIdx.x' in device and 'PART=2048' in device and 'H=8192' in device
    assert '__float2bfloat16_rn(__fmul_rn(b,w))' in device
    assert '__float2bfloat16_rn(__fmul_rn(values[i],post))' in device
    assert '__fadd_rn(residual,branch)' in device and 'p.router[output_base+col]=value' in device
    return dict(source_sha256=TRIAL,exact_packaged_prefix=True,compiled_device_source_bound=True,
        CPP_host_signature_equal=True,query_launch_arity=16,required_BF16_and_FP32_boundaries_preserved=True)


def validate_packet():
    raw=(PACKET/'submission-v3/submission.py').read_bytes();verify_source(raw)
    cpu=read(PACKET/'CPU-model.json');compiler=read(PACKET/'v2/binding.json')
    assert cpu['parent_source_sha256']==compiler['parent_source_sha256']==PARENT
    assert cpu['CUDA_sha256']==compiler['CUDA_sha256']
    assert cpu['full_row_ownership_columns']==8192 and cpu['DSM_random_schedules']==256
    assert len(cpu['numerical_cases'])==60 and all(cpu['hazards_detected'].values()) and cpu['wrong_1024_domain_detected']
    assert compiler['all_zero_stack_and_spills'] and compiler['offline_register_counts']==[32]*4
    identity=read(PACKET/'evaluator-identity.json')
    assert identity['problem']=='muse_decode_layer'
    assert {r['file'] for r in identity['files']}=={'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md','nvfp4.py'}
    for r in identity['files']:
        assert r['exact'] and r['pinned_sha256']==r['fresh_sha256']
        assert hashlib.sha256((ROOT/'muse_decode_layer'/r['file']).read_bytes()).hexdigest()==r['pinned_sha256']
    admission=read(PACKET/'host-CPU-admission.json')
    assert admission['source_sha256']==TRIAL and admission['unsupported_fallbacks_checked']>=6
    guard=read(PACKET/'guard-CPU-admission.json')
    assert guard['source_sha256']==TRIAL and guard['positive_fixture_passed'] and len(guard['rejected_mutants'])>=20
    assert guard['guard_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return raw


def verify_trial(result,quiet=False):
    body,rows,loads=full_result(result);log=body.get('log_tail','')
    resources=markers(log,'CN_RESOURCES');assert len(resources)==1
    resources=resources[0];assert len(resources)==4
    for kind,r in enumerate(resources):
        assert len(r)==9 and r[0]==kind and 0<r[1]<=64 and r[2:4]==[0,72]
        assert r[4]>=32 and r[5]>=132 and r[6:]==[1,4,256]
    assert markers(log,'CN_PROBES')==[dict(cases=20,changed_input_graph_replays=8)]
    pairs=markers(log,'CN_PAIRS');decisions=markers(log,'CN_DECISION')
    assert len(pairs)==len(decisions)==1 and decisions[0]['selected'] is True,'Cluster family was not selected'
    pairs=pairs[0]
    assert len(pairs)==8 and {(p['m'],p['kind']) for p in pairs}=={(1,0),(1,2),(8,0),(8,2),(16,1),(16,3),(32,1),(32,3)}
    for p in pairs:
        samples=p['samples'];assert len(samples)==10
        seed=20261029+p['m']*10+p['kind']
        assert [s['pair'] for s in samples]==list(range(10))
        assert all(s['order']==([0,1] if (s['pair']+seed)%2==0 else [1,0]) for s in samples)
        assert all(math.isfinite(s['old_ns']) and math.isfinite(s['new_ns']) and min(s['old_ns'],s['new_ns'])>0 for s in samples)
        old=statistics.median(s['old_ns'] for s in samples);new=statistics.median(s['new_ns'] for s in samples)
        assert math.isclose(p['old_ns'],old,rel_tol=1e-12) and math.isclose(p['new_ns'],new,rel_tol=1e-12)
        assert math.isclose(p['new_over_old'],new/old,rel_tol=1e-12) and new/old<=1.02
    combined=decisions[0]['combined'];assert len(combined)==4 and {p['m'] for p in combined}=={1,8,16,32}
    for m in (1,8,16,32):
        selected=[p for p in pairs if p['m']==m]
        old=sum(p['old_ns'] for p in selected);new=sum(p['new_ns'] for p in selected)
        assert new<=old/1.03 and old-new>=1000,'Insufficient complete-boundary payoff'
        c=next(p for p in combined if p['m']==m)
        assert math.isclose(c['old_ns'],old,rel_tol=1e-12) and math.isclose(c['new_ns'],new,rel_tol=1e-12)
        assert math.isclose(c['net_saving_ns'],old-new,rel_tol=1e-12)
    profiles={r['index']:r.get('report','').replace(' ','') for r in body['records'] if r.get('event')=='profile'}
    assert set(profiles)==set(range(12))
    for i in CHANGED:
        kinds=(0,2) if i in (0,1,6,7) else (1,3)
        assert all(f'okc_cluster_norm::boundary<{k}>' in profiles[i] for k in kinds),'New boundary did not execute'
        assert 'muse_spark_norm_residual_kernel' not in profiles[i] and '_sfn_kernel' not in profiles[i],'Old boundary also executed'
    for i in CONTROLS:
        assert 'okc_cluster_norm::boundary<' not in profiles[i] and 'muse_spark_norm_residual_kernel' in profiles[i]
    old,old_rows,old_loads=full_result(bound_result(PARENT))
    per_case={r['index']:r['speedup']/next(a['speedup'] for a in old_rows if a['index']==r['index']) for r in rows}
    gains={k:(a/b)/(old_loads[k][0]/old_loads[k][1]) for k,(a,b) in loads.items()}
    affected_loads={r['case']['load'] for r in rows if r['index'] in CHANGED};assert len(affected_loads)==4
    controls=math.prod(v for k,v in gains.items() if k not in affected_loads)**.5
    affected=math.prod(gains[k] for k in affected_loads)**.25
    track=math.prod(gains.values())**(1/6)
    summary=dict(affected_weighted_load_geomean=affected,control_weighted_load_geomean=controls,
        reconstructed_track_ratio=track,control_normalized_track_gain=track/controls,
        weighted_load_gains=gains,per_case_ratios=per_case,statistical_significance_claim=False)
    if not quiet:print(json.dumps(summary),flush=True)
    assert .95<=controls<=1.05,'Controls drifted; retain baseline'
    assert affected>=1.005 and affected/controls>=1.005,'Insufficient affected/control gain'
    assert track>=1.003 and track/controls>=1.003,'Insufficient complete weighted track gain'
    assert all(gains[k]/controls>=.995 for k in affected_loads),'Affected load regressed'
    assert all(per_case[i]/controls>=.98 for i in CHANGED),'Changed case regressed'
    return summary
