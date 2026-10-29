"""Fixed source identity and material-gain gates for original-FP32 finalizer reuse."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from moe_skinny16_evidence import full_result

ROOT = Path(__file__).resolve().parents[1]
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
TRIAL = 'f200e6e061ae3ed5533dcce30274bbd75d3f56725bdb6b58abdc8b9c50279656'
CUDA = '403bd12392a617b28ef14f5f92c0a8892c465b8ed851773b2ffe4d036d233733'
PREFLIGHT = ROOT / 'results/moe-fp32-final-cache-preflight-20261029'
SHAPES = (17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 384, 385)


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
    folder = (ROOT / 'results/resumed-moe-phase4-20261028' / sha) if sha == PARENT else directory(sha)
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == sha
    launch, result = read(folder / 'launch.json'), read(folder / 'result.json')
    assert launch['source_sha256'] == sha
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result


def validate_packet(require_guard=True):
    raw=(PREFLIGHT/'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    parent=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==PARENT and raw.startswith(parent)
    bind=read(PREFLIGHT/'source-binding.json')
    assert bind['source_sha256']==TRIAL and bind['CUDA_sha256']==CUDA
    assert bind['parent_is_exact_prefix'] and bind['parent_AST_unchanged']
    cpu=read(PREFLIGHT/'cpu-and-compiler-audit.json')
    assert cpu['full_CUDA_sha256']==CUDA and cpu['finite_finalizer_cases']==72
    assert cpu['CPU_same_bits_with_identical_FP32_arithmetic'] and cpu['all_nonlocal_zero_and_NaN_poison_masking']
    assert set(cpu['rejected_mutants'])=={'load_invalid','premature_BF16','wrong_route'}
    assert cpu['vector_load_instructions_removed_each_BIG']==16 and cpu['spills_and_matrix_wait_static_counts_equal']
    host=read(PREFLIGHT/'host-audit.json')
    assert host['source_sha256']==TRIAL and host['CUDA_sha256']==CUDA
    assert host['namespace_export_query_binding_exact'] and host['dispatch_cases_checked']==1608
    assert host['all44protected_exact']
    identity=read(PREFLIGHT/'evaluator-identity.json')
    assert {f['file'] for f in identity['files']}=={'TASK.md','reference.py','checks.py','baseline.py','nvfp4.py','AGENTS.md'}
    assert all(f['exact'] and f['pinned_sha256']==f['fresh_sha256'] for f in identity['files'])
    if require_guard:
        guard=read(PREFLIGHT/'guard-cpu-audit.json')
        assert guard['source_sha256']==TRIAL and guard['positive_fixture_passed'] and len(guard['rejected'])>=12
    return raw


def verify_probe_evidence(evidence):
    assert evidence['cuda_sha256'] == CUDA and evidence['selected_main'] is True and not evidence.get('skip')
    assert evidence['synthetic_only'] and evidence['independent_jobs'] == 1 and evidence['repeats_are_not_independent']
    assert evidence['extra_weight_layouts'] == 0 and evidence['separate_scratch']
    assert evidence['original_small_and_ineligible_scale_paths'] and 0 < evidence['prepare_s'] <= 300
    resources = evidence['resources']
    assert len(resources) == 10 and all(isinstance(v, int) for v in resources)
    assert all(0 < v <= 112 for v in resources[:2]) and resources[2:6] == [0, 0, 0, 0]
    assert resources[6] == 231168 and min(resources[7:9]) >= 1 and resources[9] == 576
    assert 0 < evidence['grid'] <= 160
    cases = evidence['cases']
    assert [(c['m'], c['mode']) for c in cases] == [(m, 'normal') for m in SHAPES] + [(128, 'skewed')]
    for case in cases:
        assert case['synthetic'] and case['seed'] == 20261030 + case['m'] + (10000 if case['mode'] == 'skewed' else 0)
        old, new = case['old_ns'], case['new_ns']
        assert len(old) == len(new) == 10 and all(math.isfinite(v) and v > 0 for v in old + new)
        orders = [tuple(o) for o in case['arm_orders']]
        assert orders.count((0, 1)) == orders.count((1, 0)) == 5
        ratio = statistics.median(n / o for o, n in zip(old, new))
        assert math.isclose(ratio, case['median_ratio'], rel_tol=1e-10)
        assert ratio <= 1.02 and (case['m'] not in (32, 128, 256) or case['mode'] != 'normal' or ratio < .97)
        assert 0 <= case['relative'] <= .005 and 0 <= case['worst'] <= .02
    assert len(cases[2]['changed_input_graph_replays']) == 2
    for relative, worst in cases[2]['changed_input_graph_replays']:
        assert 0 <= relative <= .005 and 0 <= worst <= .02
    analytic = evidence['analytic']
    assert [(a['m'], a['mode']) for a in analytic] == [(m, 'remote') for m in (17, 33, 384, 385)] + [(33, 'duplicate')]
    assert all(a['passed'] is True for a in analytic)
    assert 0 <= analytic[-1]['relative'] <= .005 and 0 <= analytic[-1]['worst'] <= .02


def verify_trial(result):
    body, benches = full_result(result)
    evaluator_files = body.get('evaluator', {}).get('files', {})
    identity = read(PREFLIGHT / 'evaluator-identity.json')
    for f in identity['files']:
        if f['file'] in ('baseline.py', 'reference.py', 'checks.py', 'nvfp4.py'):
            prefix = 'challenge/' if f['file'] in ('checks.py', 'nvfp4.py') else 'problems/muse_moe_w4a16/'
            assert evaluator_files.get(prefix + f['file']) == f['pinned_sha256'], 'Evaluator file changed: ' + f['file']
    marker = 'F32_CACHE_EVIDENCE '
    lines = [line[len(marker):] for line in body.get('log_tail', '').splitlines() if line.startswith(marker)]
    assert len(lines) == 1, 'Missing or ambiguous FP32 final-cache evidence'
    evidence = json.loads(lines[0])
    verify_probe_evidence(evidence)
    profiles = {x['index']: x['report'] for x in body['records'] if x.get('event') == 'profile'}
    assert set(profiles) == set(range(6))
    assert all('okc_f32cache::moe_fused<' in profiles[i] for i in (2, 4, 5)), 'New backend absent in affected profiles'
    assert all('okc_f32cache::' not in profiles[i] for i in (0, 1, 3)), 'Small path changed'
    baseline, old = full_result(bound_result(PARENT))
    ratios = {b['index']: b['speedup'] / old[b['index']]['speedup'] for b in benches}
    affected = math.prod(ratios[i] for i in (2, 4, 5)) ** (1 / 3)
    controls = math.prod(ratios[i] for i in (0, 1, 3)) ** (1 / 3)
    summary = dict(affected_geomean=affected, controls_geomean=controls,
                   normalized_affected=affected / controls,
                   aggregate_ratio=body['speedup'] / baseline['speedup'],
                   per_case_ratios=ratios, statistical_significance_claim=False)
    print(json.dumps(summary))
    assert affected > 1.03 and affected / controls > 1.03
    assert all(ratios[i] / controls >= .99 for i in (2, 4, 5))
    assert .95 <= controls <= 1.05, 'Controls drifted; retain baseline'
    assert body['speedup'] > baseline['speedup'] * 1.015, 'Insufficient complete-track gain'
    return summary
