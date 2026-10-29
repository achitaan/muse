"""CPU fixtures test diagnostic validation, not GPU performance or correctness."""
import copy
import json
from pathlib import Path
import runpy
import sys
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
from check_moe_phase_diagnostic import verify_diagnostics, PREFLIGHT, TRIAL

ns = runpy.run_path(str(ROOT / 'experiments/prepared/moe_phase_diagnostic_addon.py'),
                    init_globals={'prepare': None, 'kernel_fn': None, 'torch': torch})
trace = [[10 ** 12 + 100 + i, 10 ** 12 + 110 + i, 10 ** 12 + 200 + 2 * i,
          10 ** 12 + 300 + 3 * i, 10 ** 12 + 1000 + i, 10 ** 12 + 1100 + 4 * i,
          10 ** 12 + 130 + i, 10 ** 12 + 120 + i] for i in range(132)]
cases = []
for m in (32, 128, 256):
    ids = torch.arange(8, dtype=torch.int32).expand(m, 8).contiguous()
    metadata = ns['_phase_metadata']({'topk_ids': ids}, 132)
    old = [100000.0 + i * 64 for i in range(10)]
    new = [v * 1.01 for v in old]
    cases.append(dict(m=m, seed=20261029 + m, synthetic=True, metadata=metadata,
                      paired_original_ns=old, paired_instrumented_ns=new,
                      arm_orders=[[0, 1]] * 5 + [[1, 0]] * 5, overhead_median_ratio=1.01,
                      relative_to_incumbent=0.0, worst_row_to_incumbent=0.0,
                      timing_interpretation_allowed=True,
                      trace=ns['_phase_pack_trace'](trace, list(range(132))),
                      summary=ns['_phase_trace_summary'](trace, metadata)))
evidence = dict(synthetic_only=True, independent_jobs=1, repeats_are_not_independent=True,
                incumbent_timed_source_unchanged=True, resources=[96, 96, 0, 0, 0, 0, 231168, 1, 1, 576],
                resource_matches_historical_incumbent=True, cases=cases)
tests = [dict(event='test', index=i, status='pass', case=dict(bs=m, sharding='ep8',
          **({'dist': 'skewed'} if i == 3 else {}))) for i, m in enumerate((4, 33, 257, 128, 1))]
benches = [dict(event='benchmark', index=i, status='pass', speedup=1.0,
           case=dict(bs=m, load=load, sharding='ep8')) for i, (m, load) in enumerate(
           [(1, 'decode 1'), (16, 'decode 16'), (32, 'decode 32'), (8, 'verify 1'), (128, 'verify 16'), (256, 'verify 32')])]


def result(record):
    return dict(id='CPU-FIXTURE-NOT-A-JOB', status='done', board='muse_moe_w4a16', result=dict(
        passed=True, gpu='H200', speedup=1.0, checks={'ok': True},
        records=[dict(event='setup', status='pass', prepare_s=200.0)] + tests + benches,
        log_tail='MOE_PHASE_EVIDENCE ' + json.dumps(record, separators=(',', ':'))))


assert verify_diagnostics(result(evidence))['complete']
mutations = []
bad = copy.deepcopy(evidence); bad['cases'].pop(); mutations.append(('missing_shape', bad))
bad = copy.deepcopy(evidence); bad['cases'][0]['arm_orders'][0] = [1, 0]; mutations.append(('unbalanced_order', bad))
bad = copy.deepcopy(evidence); bad['cases'][0]['overhead_median_ratio'] = 1.2; mutations.append(('forged_overhead', bad))
bad = copy.deepcopy(evidence); bad['cases'][0]['metadata']['jobs'] += 1; mutations.append(('inconsistent_jobs', bad))
bad = copy.deepcopy(evidence)
bad['cases'][0]['trace']['deltas_u32le_zlib_b64'] = ns['_phase_b64'].b64encode(
    ns['_phase_zlib'].compress(bytes(132 * 8 * 4))).decode()
mutations.append(('altered_clock_payload', bad))
rejected = []
for name, bad in mutations:
    try:
        verdict = verify_diagnostics(result(bad))
    except AssertionError:
        rejected.append(name)
    else:
        assert verdict['complete'] is False, ('Bad fixture accepted', name)
        rejected.append(name)
report = dict(source_sha256=TRIAL, positive_cpu_fixture=True, malformed_fixtures_rejected=rejected,
              no_gpu_calls=True, fixture_is_not_a_measurement=True)
data = (json.dumps(report, indent=2) + '\n').encode()
target = PREFLIGHT / 'guard-cpu-audit.json'
if target.exists():
    assert target.read_bytes() == data
else:
    target.write_bytes(data)
print(json.dumps(report, indent=2))
