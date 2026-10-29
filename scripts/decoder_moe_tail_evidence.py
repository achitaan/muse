"""Source-bound, complete Decode and actual-new-path retention gates."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics
import zlib

from decoder_cooperative_evidence import full_result, bound_result as parent_result, markers

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
TRIAL = '1ed5ba7b1c87ae0142bf0d52fba08e7938c1be9dcc5d3200d388fabee15c951f'
CUDA = '95abcfa8857ca584e03c86a5ba1ac583a26b6abde4ff13b5c4fcafe5a2052091'
PACKET = ROOT / 'results/decoder-moe-tail-preflight-20261029'
CHANGED = (0, 1, 6, 7)


def read(path): return json.loads(path.read_bytes())


def source():
    raw = (ROOT / 'candidates/decode_active.py').read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha in (PARENT, TRIAL), ('unreviewed source', sha)
    return raw, sha


def directory(sha):
    assert sha in (PARENT, TRIAL)
    return ROOT / 'results/decoder-segment-repair-20261028' if sha == PARENT else ROOT / '.decoder-moe-tail-measurements' / sha


def bound_result(sha):
    if sha == PARENT: return parent_result(sha)
    folder = directory(sha)
    assert hashlib.sha256((folder / 'submission.py').read_bytes()).hexdigest() == sha
    launch, result = read(folder / 'launch.json'), read(folder / 'result.json')
    assert launch['source_sha256'] == sha
    assert result['id'] == launch.get('resolved_job_id', launch['job_id'])
    return result


def validate_packet():
    raw = (PACKET / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == TRIAL
    binding = read(PACKET / 'submission-binding.json')
    assert binding['source_sha256'] == TRIAL and binding['CUDA_sha256'] == CUDA
    assert binding['exact_root_inverse'] and binding['complete_other_packaged_source_preserved']
    packaged = (ROOT / 'results/decoder-unused-rope-prune-20261029/submission.py').read_bytes()
    constants_begin = raw.index(b'\n_DT_CUDA = ')
    assert raw[:constants_begin].decode().replace(binding['root_new'], binding['root_old']).encode() == packaged
    tree = ast.parse(raw)
    values = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
              and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    expr = values['_DT_CUDA']
    assert isinstance(expr, ast.Call) and expr.func.id == '_zsrc' and len(expr.args) == 1
    cuda = zlib.decompress(base64.b64decode(ast.literal_eval(expr.args[0]), validate=True))
    assert cuda == (PACKET / 'prototype.cu').read_bytes()
    assert hashlib.sha256(cuda).hexdigest() == CUDA
    cpp = ast.literal_eval(values['_DT_CPP'])
    assert hashlib.sha256(cpp.encode()).hexdigest() == binding['CPP_sha256']
    for name in ('CPU-model.json', 'host-CPU-admission.json'):
        row = read(PACKET / name)
        assert row['native_CUDA_sha256'] == CUDA
    model = read(PACKET / 'CPU-model.json')
    assert model['bitwise_weight_values'] == 4194304 and model['replay_schedules'] == 2048
    assert len(model['mutants_rejected']) == 3 and len(model['independent_boundary_cases']) == 30
    assert model['norm_domain'] == 8192 and model['unpooler_full_K'] == 4096
    host = read(PACKET / 'host-CPU-admission.json')
    assert host['source_sha256'] == TRIAL and host['all_passed']
    identity = read(PACKET / 'evaluator-identity.json')
    assert {v['file'] for v in identity['files']} == {'TASK.md','AGENTS.md','reference.py','checks.py','baseline.py','nvfp4.py'}
    assert all(v['exact'] and v['pinned_sha256'] == v['fresh_sha256'] for v in identity['files'])
    for v in identity['files']:
        assert hashlib.sha256((ROOT / 'muse_decode_layer' / v['file']).read_bytes()).hexdigest() == v['pinned_sha256']
    admission = read(PACKET / 'launch-admission.json')
    assert admission['source_sha256'] == TRIAL and admission['GPU_admitted']
    return raw


def verify_report(report):
    assert report['cuda'] == CUDA and report['source_tag'] == CUDA[:12]
    assert report['selected'] is True and report['weight_permutation_bitwise'] is True
    assert 'setup_error' not in report
    assert [r['family'] for r in report['resources']] == [0, 1]
    for r in report['resources']:
        values = r['values']; assert len(values) == 28
        for offset in (0,14):
            q = values[offset:offset+14]
            assert q[0] == offset // 14 and q[1:6] == [0]*5
            assert 0 < q[6] <= 160 and q[7] == 0 and 0 <= q[8] <= 1024
            assert 0 < q[9] and q[9] + q[8] <= 232448
            assert q[10] == 132 and q[11] >= 1 and q[12] == 1 and q[13] == 288
    rows = report['numerical']
    assert len(rows) == 24
    assert {(r['m'],r['kind']) for r in rows} == {(m,k) for m in (1,2,3,4,7,8)
                                                               for k in ('normal','all_nonlocal','duplicate','outlier')}
    for r in rows:
        assert len(r['dense_oracle']) == 2
        assert all(math.isfinite(v) and 0 <= v <= lim for v,lim in zip(r['dense_oracle'],(.0005,.002)))
    assert len(report['comparison_maxima']) == 4
    for index, pair in enumerate(report['comparison_maxima']):
        assert len(pair) == 2
        assert all(math.isfinite(v) and 0 <= v <= lim for v,lim in zip(pair, (.001,.003) if index == 2 else (.002,.004)))
    replay = report['replay']; assert len(replay) == 16
    expected_m = (1,8,3,7,2,4,8,1)
    assert [(r['cycle'],r['index'],r['m']) for r in replay] == [(c,i,m) for c in range(2) for i,m in enumerate(expected_m)]
    for r in replay:
        assert len(r['max_error']) == 2 and all(math.isfinite(v) and 0 <= v <= lim for v,lim in zip(r['max_error'],(.002,.004)))
    pairs = report['paired']; assert len(pairs) == 8
    assert {(r['m'],r['seed']) for r in pairs} == {(m,s) for m in (1,8,3,7) for s in (20261029,20261030)}
    for r in pairs:
        samples = r['samples']; assert len(samples) == 10
        assert r['sample_schema'] == ['pair','first_arm','old_ns','new_ns']
        for i,s in enumerate(samples):
            assert len(s) == 4 and s[:2] == [i,(i+r['seed'])%2]
            assert all(math.isfinite(v) and v > 0 for v in s[2:])
        ratio = statistics.median(s[3]/s[2] for s in samples)
        saved = statistics.median(s[2]-s[3] for s in samples)
        assert math.isclose(ratio,r['new_over_old'],rel_tol=1e-12)
        assert math.isclose(saved,r['saved_ns'],rel_tol=1e-12,abs_tol=1e-6)
        if r['m'] in (1,8): assert ratio < .97 and saved >= 4600
        else: assert ratio <= 1.02
    return report


def verify_trial(result, quiet=False):
    body, rows, loads = full_result(result)
    ci = body['speedup_ci']
    assert len(ci) == 2 and all(math.isfinite(v) and v > 0 for v in ci)
    assert ci[0] <= body['speedup'] <= ci[1]
    reports = markers(body.get('log_tail',''),'DECODER_TAIL')
    assert len(reports) == 1
    verify_report(reports[0])
    profiles = {r['index']:r['report'] for r in body['records'] if r.get('event') == 'profile'}
    assert set(profiles) == set(range(12))
    for i in range(12):
        if i in CHANGED:
            assert 'okc_tail::moe_fused' in profiles[i]
            assert '_sfn_kernel' not in profiles[i] and 'smtrace_gemv' not in profiles[i]
        else:
            assert 'okc_tail::' not in profiles[i] and 'okc::moe_fused' in profiles[i]
    old, old_rows, old_loads = full_result(parent_result(PARENT))
    old_by_index = {r['index']:r for r in old_rows}
    affected_loads = {r['case']['load'] for r in rows if r['index'] in CHANGED}
    assert len(affected_loads) == 2
    gains = {name:(a/b)/(old_loads[name][0]/old_loads[name][1]) for name,(a,b) in loads.items()}
    controls = math.prod(g for name,g in gains.items() if name not in affected_loads)**.25
    track = body['speedup']/old['speedup']
    assert math.isclose(track, math.prod(gains.values())**(1/6), rel_tol=1e-9)
    assert .95 <= controls <= 1.05
    assert track > 1.015 and track/controls > 1.015
    assert all(gains[name]/controls >= 1.02 for name in affected_loads)
    assert all(r['speedup']/old_by_index[r['index']]['speedup']/controls >= .995 for r in rows if r['index'] in CHANGED)
    summary = dict(aggregate_gain=track, weighted_control_gain=controls,
                   normalized_aggregate_gain=track/controls, weighted_load_gains=gains,
                   source_sha256=TRIAL, statistical_significance_claim=False)
    if not quiet: print(json.dumps(summary), flush=True)
    return summary
