"""Fixed source, coverage, resource, actual-backend and material-gain gates."""
import ast
import hashlib
import json
import math
from pathlib import Path
from verify_attention_precision import verify as verify_precision

ROOT = Path(__file__).resolve().parents[1]
PARENT = '9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f'
TRIAL = 'ed810c7abd8a2c852959609cbe9977ae562c61b85f57f5dce93bafc3b8925dcd'
CUDA = 'd9637b769cc6af0e032da8373e18f792598dd611f3800ff6a308a2678e7c55bf'
PREFLIGHT = ROOT / 'results/attention-skipscale-preflight-20261029'
CHANGED = (3, 5)
CONTROLS = tuple(i for i in range(12) if i not in CHANGED)


def read(path):
    return json.loads(path.read_bytes())


def source():
    raw = (ROOT / 'candidates/attention_active.py').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT, TRIAL), ('Unreviewed source', sha)
    return raw, sha


def directory(sha):
    return (ROOT / 'results/attention-segment-repair-20261028/publication' / sha) if sha == PARENT else ROOT / '.attention-skipscale-measurements' / sha


def bound_result(sha):
    folder = directory(sha)
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == sha
    launch = read(folder / 'launch.json')
    result = read(folder / 'result.json')
    assert launch['source_sha256'] == sha
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result


def validate_packet():
    raw = (PREFLIGHT / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRIAL
    assert hashlib.sha256((PREFLIGHT / 'kernel.cu').read_bytes()).hexdigest() == CUDA
    meta, cpu, identity = [read(PREFLIGHT / n) for n in ('source-audit.json','cpu-audit.json','evaluator-identity.json')]
    assert meta['source_sha256'] == cpu['source_sha256'] == TRIAL
    assert meta['cuda_sha256'] == cpu['cuda_sha256'] == CUDA
    assert meta['original_cuda_inverse_exact'] and meta['same_decode_arithmetic_anchors'] == 3
    assert cpu['online_state_bits_equal'] and cpu['online_state_iterations'] == 503
    assert cpu['finite_FP32_identity_products'] == cpu['finite_FP32_add_vs_exact_sum_screens'] == 261153
    assert cpu['warp_quad_vote_masks'] == 256 and cpu['all_Smem_field_layouts_unchanged']
    assert cpu['dynamic_shared_bytes']['L1_flat1'] == 67584
    guard = read(PREFLIGHT / 'guard-cpu-audit.json')
    assert guard['source_sha256'] == TRIAL and guard['positive_fixture_passed'] and len(guard['rejected']) == 13
    assert identity['problem'] == 'muse_attention_decode'
    assert {x['file'] for x in identity['files']} == {'TASK.md','reference.py','checks.py','baseline.py','AGENTS.md'}
    assert all(x['exact'] and x['pinned_sha256'] == x['fresh_sha256'] for x in identity['files'])
    verify_precision(raw)
    return raw


def case_identity(case):
    return {k:v for k,v in case.items() if k != 'seed'}


def full_result(result):
    body = result.get('result') or {}
    assert result['status'] == 'done' and result['board'] == 'muse_attention_decode'
    assert body.get('passed') is True and body.get('gpu') == 'H200' and body.get('checks',{}).get('ok') is True
    assert math.isfinite(body['speedup']) and body['speedup'] > 0
    reference = read(directory(PARENT) / 'result.json')['result']
    actual = {}
    for event, count in (('test',16),('benchmark',12)):
        rows = sorted((r for r in body['records'] if r.get('event') == event),key=lambda r:r['index'])
        expected = sorted((r for r in reference['records'] if r.get('event') == event),key=lambda r:r['index'])
        assert [r['index'] for r in rows] == list(range(count))
        assert all(r.get('status') == 'pass' for r in rows)
        assert [case_identity(r['case']) for r in rows] == [case_identity(r['case']) for r in expected]
        actual[event] = rows
    benches = actual['benchmark']
    assert all(math.isfinite(r['speedup']) and r['speedup'] > 0 and r['median_ns'] > 0 and r['baseline_ns'] > 0 for r in benches)
    loads = {}
    for r in benches:
        baseline, candidate = loads.setdefault(r['case']['load'],[0.,0.])
        w = r['case']['weight']
        assert w == (46 if r['case']['layer'] == 'sliding' else 16)
        loads[r['case']['load']] = [baseline+w*r['baseline_ns'],candidate+w*r['median_ns']]
    assert len(loads) == 6
    reconstructed = math.prod(a/b for a,b in loads.values())**(1/6)
    assert math.isclose(reconstructed,body['speedup'],rel_tol=1e-9), 'Weighted-load reconstruction mismatch'
    return body, benches


def verify_trial(result):
    body, benches = full_result(result)
    lines = [s.split() for s in body.get('log_tail','').splitlines() if s.startswith('ATTN_SKIP_RESOURCES ')]
    assert len(lines) == 1, 'Missing or ambiguous resource marker'
    resources = [int(x) for x in lines[0][1:]]
    assert len(resources) == 9 and 0 < resources[0] <= 128 and resources[1] == 0
    assert resources[2] >= 2 and resources[3:8] == [67584,66560,2,128,200]
    assert 0 < resources[8] <= 512
    profiles = {r['index']:r.get('report','').replace(' ','') for r in body['records'] if r.get('event') == 'profile'}
    assert set(profiles) == set(range(12))
    assert all('attn_kernel_skipscale<1,true' in profiles[i] for i in CHANGED), 'New flat L1 backend absent'
    assert all('attn_kernel_skipscale<1,false' in profiles[i] for i in (0,1,2,4)), 'Nonflat L1 backend changed'
    assert all('attn_kernel_skipscale<8,' in profiles[i] for i in range(6,12)), 'L8 backend changed'
    old, old_rows = full_result(bound_result(PARENT))
    previous = {r['index']:r['speedup'] for r in old_rows}
    ratios = {r['index']:r['speedup']/previous[r['index']] for r in benches}
    changed = math.prod(ratios[i] for i in CHANGED)**(1/len(CHANGED))
    controls = math.prod(ratios[i] for i in CONTROLS)**(1/len(CONTROLS))
    summary = dict(affected_geomean=changed,controls_geomean=controls,normalized_affected=changed/controls,
                   aggregate_ratio=body['speedup']/old['speedup'],per_case_ratios=ratios,
                   resources=resources,statistical_significance_claim=False)
    print(json.dumps(summary))
    assert changed > 1.04 and changed/controls > 1.04, 'Insufficient affected/control gain'
    assert all(ratios[i]/controls >= 1.02 for i in CHANGED), 'Affected-case regression or weak gain'
    assert .95 <= controls <= 1.05, 'Controls drifted; retain baseline'
    assert body['speedup'] > old['speedup']*1.015, 'Insufficient complete-track gain'
    return summary
