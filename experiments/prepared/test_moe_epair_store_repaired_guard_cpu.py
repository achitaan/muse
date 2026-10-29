"""Guard fixtures are not GPU jobs or measurements."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
from moe_epair_store_repaired_evidence import (CUDA, TRIAL, PREFLIGHT, SHAPES, PARENT, bound_result,
                                   validate_packet, verify_trial)

validate_packet()
cases = []
for m, mode in [(m, 'normal') for m in SHAPES] + [(128, 'skewed')]:
    old = [100000.123456789 + i * 64 for i in range(10)]
    cases.append(dict(m=m, mode=mode, synthetic=True, seed=20261030 + m + (10000 if mode == 'skewed' else 0),
                      old_ns=old, new_ns=[v * .95 for v in old],
                      arm_orders=[[0, 1]] * 5 + [[1, 0]] * 5,
                      median_ratio=.95, relative=0., worst=0.,
                      changed_input_graph_replays=[[0., 0.], [0., 0.]] if m == 32 else []))
analytic = [dict(m=m, mode='remote', passed=True) for m in (17, 33, 384, 385)]
analytic.append(dict(m=33, mode='duplicate', passed=True, relative=0., worst=0.))
evidence = dict(cuda_sha256=CUDA, synthetic_only=True, independent_jobs=1, repeats_are_not_independent=True,
                selected_main=True, resources=[96, 96, 0, 0, 0, 0, 231168, 1, 1, 576], grid=132,
                cases=cases, analytic=analytic, extra_weight_layouts=0, separate_scratch=True,
                original_small_and_ineligible_scale_paths=True, prepare_s=240.)
baseline = bound_result(PARENT)
fixture = copy.deepcopy(baseline)
fixture['id'] = 'CPU-FIXTURE-NOT-A-JOB'
fixture['result']['speedup'] *= 1.05 ** .5
for record in fixture['result']['records']:
    if record.get('event') == 'benchmark' and record['index'] in (2, 4, 5):
        record['speedup'] *= 1.05
    elif record.get('event') == 'profile' and record['index'] in (2, 4, 5):
        record['report'] = 'void okc_epair::moe_fused<false>(params) CPU fixture only'
fixture['result']['log_tail'] = 'EPAIR_STORE_EVIDENCE ' + json.dumps(evidence)
with contextlib.redirect_stdout(io.StringIO()):
    verify_trial(fixture)
mutations = []
bad = copy.deepcopy(fixture); e = copy.deepcopy(evidence); e['selected_main'] = False
bad['result']['log_tail'] = 'EPAIR_STORE_EVIDENCE ' + json.dumps(e); mutations.append(('empty_choice', bad))
bad = copy.deepcopy(fixture); e = copy.deepcopy(evidence); e['resources'][2] = 8
bad['result']['log_tail'] = 'EPAIR_STORE_EVIDENCE ' + json.dumps(e); mutations.append(('local_storage', bad))
bad = copy.deepcopy(fixture); e = copy.deepcopy(evidence); e['cuda_sha256'] = '0' * 64
bad['result']['log_tail'] = 'EPAIR_STORE_EVIDENCE ' + json.dumps(e); mutations.append(('wrong_cuda', bad))
bad = copy.deepcopy(fixture); e = copy.deepcopy(evidence); e['cases'][0]['median_ratio'] = .8
bad['result']['log_tail'] = 'EPAIR_STORE_EVIDENCE ' + json.dumps(e); mutations.append(('forged_pairs', bad))
bad = copy.deepcopy(fixture); e = copy.deepcopy(evidence); e['cases'].pop()
bad['result']['log_tail'] = 'EPAIR_STORE_EVIDENCE ' + json.dumps(e); mutations.append(('missing_nearby_case', bad))
bad = copy.deepcopy(fixture); bad['result']['log_tail'] += '\n' + bad['result']['log_tail']; mutations.append(('ambiguous_marker', bad))
bad = copy.deepcopy(fixture)
next(x for x in bad['result']['records'] if x.get('event') == 'profile' and x['index'] == 2)['report'] = 'original backend'
mutations.append(('new_backend_absent', bad))
bad = copy.deepcopy(fixture)
next(x for x in bad['result']['records'] if x.get('event') == 'profile' and x['index'] == 0)['report'] = 'okc_epair::moe_fused<false>'
mutations.append(('small_backend_changed', bad))
bad = copy.deepcopy(fixture); bad['result']['speedup'] = baseline['result']['speedup'] * 1.01
mutations.append(('complete_track_gain_missing', bad))
bad = copy.deepcopy(fixture)
for x in bad['result']['records']:
    if x.get('event') == 'benchmark' and x['index'] in (0, 1, 3):
        x['speedup'] *= 1.04
mutations.append(('control_normalized_gain_missing', bad))
bad = copy.deepcopy(fixture); bad['result']['checks']['ok'] = False; mutations.append(('integrity_failed', bad))
bad = copy.deepcopy(fixture)
bad['result']['evaluator']['files']['problems/muse_moe_w4a16/reference.py'] = '0' * 64
mutations.append(('evaluator_identity_changed', bad))
rejected = []
for name, bad in mutations:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            verify_trial(bad)
    except AssertionError:
        rejected.append(name)
    else:
        raise AssertionError(('Malformed fixture accepted', name))
report = dict(source_sha256=TRIAL, positive_fixture_passed=True, rejected=rejected,
              no_gpu_jobs=True, no_api_calls=True, fixtures_are_not_measurements=True)
target = PREFLIGHT / 'guard-cpu-audit-v2.json'; data = (json.dumps(report, indent=2) + '\n').encode()
if target.exists():
    assert target.read_bytes() == data, 'Frozen audit collision'
else:
    target.write_bytes(data)
print(json.dumps(report, indent=2))
