"""Fixed source identity and material-gain gates for the paired epilogue store trial."""
import hashlib
import json
import math
from pathlib import Path
import statistics
from moe_skinny16_evidence import full_result

ROOT = Path(__file__).resolve().parents[1]
PARENT = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
TRIAL = 'e8e3086d34a30de0283e37f2f40d3b1d9742313203fc624c1720bc6a6badced8'
CUDA = '9d6c435420eb3f43d9fa2cbbea343d1414a557ee883140e67d97718b2b0e3e5d'
PREFLIGHT = ROOT / 'results/moe-epair-store-preflight-20261029'
SHAPES = (17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 384, 385)


def read(path):
    return json.loads(path.read_bytes())


def source():
    raw = (ROOT / 'candidates/moe_active.py').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT, TRIAL), ('Unreviewed source', sha)
    return raw, sha


def directory(sha):
    return ROOT / '.moe-epair-measurements' / sha


def bound_result(sha):
    folder = (ROOT / 'results/resumed-moe-phase4-20261028' / sha) if sha == PARENT else directory(sha)
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == sha
    launch, result = read(folder / 'launch.json'), read(folder / 'result.json')
    assert launch['source_sha256'] == sha
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result


def validate_packet():
    raw = (PREFLIGHT / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRIAL
    assert hashlib.sha256((PREFLIGHT / 'epair.cu').read_bytes()).hexdigest() == CUDA
    parent = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT and raw.startswith(parent)
    source_audit, cpu, host = [read(PREFLIGHT / name) for name in ('source-audit.json', 'cpu-audit.json', 'host-audit.json')]
    assert source_audit['source_sha256'] == host['source_sha256'] == TRIAL
    assert source_audit['epair_cuda_sha256'] == cpu['cuda_sha256'] == host['cuda_sha256'] == CUDA
    assert cpu['unique_halfword_writes'] == 8519680 and cpu['raw_bf16_pattern_pairs'] == 65536
    assert cpu['counts'] == list(range(1, 65)) and cpu['all4096columns']
    assert cpu['complete_reference_layout_equal'] and cpu['padding_untouched']
    assert cpu['uint32_stores_aligned'] and cpu['unconditional_full_warp_shuffle']
    assert cpu['phase_B_unchanged'] and cpu['parent_prefix_and_inverse_exact']
    assert set(cpu['rejected_mutants']) == {'wrong_lane_bit', 'swapped_columns', 'padding_writer'}
    assert source_audit['original_cuda_inverse_exact'] and source_audit['scratch_layout_unchanged']
    assert host['dispatch_cases'] == 100 and len(host['selection_rejections']) == 11
    assert host['fresh_outputs'] and host['separate_scratch_binding'] and host['evidence_fixture_bytes'] < 13000
    identity = read(PREFLIGHT / 'evaluator-identity.json')
    assert identity['problem'] == 'muse_moe_w4a16'
    assert {x['file'] for x in identity['files']} == {'TASK.md', 'reference.py', 'checks.py', 'baseline.py', 'nvfp4.py', 'AGENTS.md'}
    assert all(x['exact'] and x['pinned_sha256'] == x['fresh_sha256'] for x in identity['files'])
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
    marker = 'EPAIR_STORE_EVIDENCE '
    lines = [line[len(marker):] for line in body.get('log_tail', '').splitlines() if line.startswith(marker)]
    assert len(lines) == 1, 'Missing or ambiguous paired epilogue store evidence'
    evidence = json.loads(lines[0])
    verify_probe_evidence(evidence)
    profiles = {x['index']: x['report'] for x in body['records'] if x.get('event') == 'profile'}
    assert set(profiles) == set(range(6))
    assert all('okc_epair::moe_fused<' in profiles[i] for i in (2, 4, 5)), 'New backend absent in affected profiles'
    assert all('okc_epair::' not in profiles[i] for i in (0, 1, 3)), 'Small path changed'
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
