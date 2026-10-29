"""Fixed two-load source, execution, replay and material-performance gates."""
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics
from moe_skinny16_evidence import full_result, bound_result as previous_result

ROOT = Path(__file__).resolve().parents[1]
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
TRIAL = 'b7043df79affee252e20165932f6a80b862ec91a8094145b93639989554dbebe'
CUDA = '2701a25e854caef3a46c1ae2002bd59932b2b056a05a1c0d2cf498dc5f8b6114'
PREFLIGHT = ROOT / 'results/moe-cluster-x-submission-preflight-20261029'
SHAPES = (1,4,8,16,31,32,33,63,64,65,127,128,129,255,256,257,384,385)


def read(path):
    return json.loads(path.read_bytes())


def source():
    raw = (ROOT / 'candidates/moe_active.py').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT, TRIAL), ('Unreviewed source', sha)
    return raw, sha


def directory(sha):
    return ROOT / '.moe-measurements' / sha


def bound_result(sha):
    if sha == PARENT:
        return previous_result(PARENT)
    folder = directory(sha)
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == sha
    launch, result = read(folder / 'launch.json'), read(folder / 'result.json')
    assert launch['source_sha256'] == sha
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result


def pure_gates():
    tree = ast.parse((PREFLIGHT / 'submission.py').read_bytes())
    names = {'_cx_resources_ok', '_cx_decision'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == 2
    env = dict(_cx_math=math, _cx_stats=statistics, _CLUSTER_SHAPES=SHAPES)
    exec(compile(ast.Module(nodes, type_ignores=[]), '<frozen-pure-cluster-gates>', 'exec'), env)
    return env


def validate_packet():
    raw = (PREFLIGHT / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRIAL
    assert raw.startswith((ROOT / 'best/moe_verified.py').read_bytes())
    binding = read(PREFLIGHT / 'source-binding.json')
    assert binding['source_sha256'] == TRIAL and binding['CUDA_sha256'] == CUDA
    assert binding['parent_is_exact_prefix'] and binding['parent_AST_unchanged']
    for name in ('host-audit.json', 'guard-cpu-audit.json', 'evaluator-identity.json'):
        assert (PREFLIGHT / name).exists(), 'Missing admission: ' + name
    host, guard, identity = [read(PREFLIGHT / name) for name in ('host-audit.json','guard-cpu-audit.json','evaluator-identity.json')]
    assert host['source_sha256'] == guard['source_sha256'] == TRIAL
    assert host['CUDA_sha256'] == CUDA and host['namespace_export_query_binding_exact']
    assert host['cooperative_cluster_required'] and host['no_plain_launch_fallback']
    assert host['dispatch_cases_checked'] >= 12 and host['all44protected_exact']
    assert guard['positive_fixture_passed'] and len(guard['rejected']) >= 15
    assert {f['file'] for f in identity['files']} == {'TASK.md','reference.py','checks.py','baseline.py','nvfp4.py','AGENTS.md'}
    assert all(f['exact'] and f['pinned_sha256'] == f['fresh_sha256'] for f in identity['files'])
    return raw


def verify_probe_evidence(e):
    assert e['cuda_sha256'] == CUDA and e['selected_main'] is True and not e.get('skip')
    assert e['synthetic_only'] and e['independent_jobs'] == 1 and e['repeats_are_not_independent']
    assert e['extra_weight_layouts'] == 0 and e['separate_scratch'] and e['minimum_timed_M'] == 64
    assert e['original_small_and_ineligible_scale_paths'] and 0 < e['prepare_s'] <= 300
    assert pure_gates()['_cx_decision'](e['cases'],e['resources'],e['grid'],e['analytic'])
    for c in e['cases']:
        assert c['synthetic'] and c['seed'] == 20261030+c['m']+(10000 if c['mode']=='skewed' else 0)
    assert all(a['passed'] is True for a in e['analytic'])
    assert 0 <= e['analytic'][-1]['relative'] <= .005 and 0 <= e['analytic'][-1]['worst'] <= .02


def verify_trial(result):
    body, benches = full_result(result)
    identity = read(PREFLIGHT / 'evaluator-identity.json')
    files = body.get('evaluator',{}).get('files',{})
    for f in identity['files']:
        if f['file'] in ('baseline.py','reference.py','checks.py','nvfp4.py'):
            prefix = 'challenge/' if f['file'] in ('checks.py','nvfp4.py') else 'problems/muse_moe_w4a16/'
            assert files.get(prefix+f['file']) == f['pinned_sha256'], 'Evaluator changed: '+f['file']
    marker='CLUSTER_X_EVIDENCE '
    lines=[s[len(marker):] for s in body.get('log_tail','').splitlines() if s.startswith(marker)]
    assert len(lines)==1, 'Missing/ambiguous cluster evidence'
    verify_probe_evidence(json.loads(lines[0]))
    profiles={r['index']:r['report'] for r in body['records'] if r.get('event')=='profile'}
    assert set(profiles)==set(range(6))
    assert all('okc_cluster2::moe_fused<' in profiles[i] for i in (4,5)), 'New backend absent'
    assert all('okc_cluster2::' not in profiles[i] for i in (0,1,2,3)), 'Control backend changed'
    old_body,old=full_result(bound_result(PARENT))
    ratios={r['index']:r['speedup']/old[r['index']]['speedup'] for r in benches}
    affected=math.prod(ratios[i] for i in (4,5))**.5
    controls=math.prod(ratios[i] for i in (0,1,2,3))**.25
    summary=dict(affected_geomean=affected,controls_geomean=controls,normalized_affected=affected/controls,
                 aggregate_ratio=body['speedup']/old_body['speedup'],per_case_ratios=ratios,
                 affected_indices=[4,5],control_indices=[0,1,2,3],statistical_significance_claim=False)
    print(json.dumps(summary))
    assert affected>1.03 and affected/controls>1.03, 'Insufficient affected/control gain'
    assert all(ratios[i]/controls>=.99 for i in (4,5)), 'Affected load regression'
    assert .95<=controls<=1.05, 'Controls drifted'
    assert body['speedup']>old_body['speedup']*1.015, 'Insufficient full-track gain'
    return summary
