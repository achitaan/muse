"""One resumable diagnostic check; never score or promote this unchanged timed path."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import time
import zlib

from evaluation_pacing import authorize_check, check_window, record_check, PacingWait
from official import LINUX_ROOT, cli
from moe_skinny16_evidence import full_result

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / 'results/moe-phase-diagnostic-db0f-preflight-20261029'
TRIAL = 'db0f6481654d41790e09b309a9f28fdf95e79d86a4f1b0a00183b18be26cba0b'


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_bytes((json.dumps(value, indent=2) + '\n').encode())


def verify_diagnostics(result):
    body, benches = full_result(result)
    lines = [line[len('MOE_PHASE_EVIDENCE '):] for line in body.get('log_tail', '').splitlines()
             if line.startswith('MOE_PHASE_EVIDENCE ')]
    if len(lines) != 1:
        return dict(complete=False, reason='No unique diagnostic marker; inspect preparation budget/compilation')
    evidence = json.loads(lines[0])
    assert evidence['synthetic_only'] and evidence['independent_jobs'] == 1
    assert evidence['repeats_are_not_independent'] and evidence['incumbent_timed_source_unchanged']
    resources = evidence['resources']
    assert len(resources) == 10 and all(isinstance(v, int) and v >= 0 for v in resources)
    admission = resources[:2] == [96, 96] and resources[2:4] == [0, 0] and resources[7:9] == [1, 1]
    assert evidence['resource_matches_historical_incumbent'] == admission
    cases = evidence['cases']
    if [c['m'] for c in cases] != [32, 128, 256]:
        return dict(complete=False, reason='Preparation budget did not cover all three diagnostic shapes', evidence=evidence)
    for case in cases:
        assert case['synthetic'] is True and case['seed'] == 20261029 + case['m']
        old, new = case['paired_original_ns'], case['paired_instrumented_ns']
        assert len(old) == len(new) == 10 and all(math.isfinite(v) and v > 0 for v in old + new)
        assert case['arm_orders'].count([0, 1]) == case['arm_orders'].count([1, 0]) == 5
        ratio = statistics.median(n / o for o, n in zip(old, new))
        assert math.isclose(ratio, case['overhead_median_ratio'], rel_tol=1e-10)
        assert 0 <= case['relative_to_incumbent'] <= .005 and 0 <= case['worst_row_to_incumbent'] <= .02
        trace = case['trace']; indices = trace['cta_indices']; grid = trace['full_grid']
        assert isinstance(grid, int) and 0 < grid <= 160
        meta = case['metadata']
        assert len(meta['counts']) == 32 and all(isinstance(v, int) and 0 <= v <= case['m'] for v in meta['counts'])
        assert meta['local_routes'] == sum(meta['counts']) and meta['active_experts'] == sum(v > 0 for v in meta['counts'])
        assert 0 <= meta['zero_local_tokens'] <= case['m']
        expected_jobs = sum((v + 63) // 64 if v > 8 else int(v > 0) for v in meta['counts'])
        assert meta['jobs'] == expected_jobs
        assert len(meta['phases']) == 2
        for phase, units in zip(meta['phases'], (32, 16)):
            counts = phase['stage_counts']
            assert len(counts) == grid and all(isinstance(v, int) and v >= 0 for v in counts)
            assert sum(counts) == expected_jobs * units * 16
            boundaries, total = [], 0
            for count in counts[:-1]:
                total += count; boundaries.append(total)
            assert phase['split_units'] == len({s // 16 for s in boundaries if s % 16})
        assert indices == sorted(set(indices)) and all(0 <= i < grid for i in indices)
        data = zlib.decompress(base64.b64decode(trace['deltas_u32le_zlib_b64'], validate=True))
        assert len(data) == len(indices) * 8 * 4
        values = struct.unpack('<' + 'I' * (len(data) // 4), data)
        rows = [[values[i * 8 + c] + trace['column_origins_ns'][c] for c in range(8)] for i in range(len(indices))]
        order = [0, 1, 7, 6, 2, 3, 4, 5]
        assert all(all(row[a] > 0 and row[a] <= row[b] for a, b in zip(order, order[1:])) for row in rows)
        summary = case['summary']
        assert summary['trace_span_ns'] == max(row[5] for row in rows) - trace['column_origins_ns'][0]
        assert summary['final_tail_after_latest_B_ns'] == max(row[5] for row in rows) - max(row[3] for row in rows)
        expected = .95 <= ratio <= 1.05 and evidence['resource_matches_historical_incumbent']
        assert case['timing_interpretation_allowed'] == expected
    return dict(complete=True, evidence=evidence, performance_improvement_claim=False,
                official_suite=dict(tests=5, benchmarks=len(benches), integrity=True),
                interpretation='Synthetic instrumented traces only; summed spans are not critical-path fractions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Permit the single new check if pacing allows')
    parser.add_argument('--wait-for-slot', action='store_true', help='Wait locally for the authorized pacing slot; no API polling while waiting')
    args = parser.parse_args()
    source = (PREFLIGHT / 'submission.py').read_bytes()
    audit = read(PREFLIGHT / 'source-audit.json')
    digest = hashlib.sha256(source).hexdigest()
    assert digest == audit['source_sha256'] == TRIAL
    assert read(PREFLIGHT / 'cpu-audit.json')['source_sha256'] == digest
    parent = (ROOT / 'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == audit['parent_sha256'] and source.startswith(parent)
    identity = read(PREFLIGHT / 'evaluator-identity.json')
    assert identity['problem'] == 'muse_moe_w4a16'
    assert {r['file'] for r in identity['files']} == {'TASK.md', 'reference.py', 'checks.py', 'baseline.py', 'nvfp4.py', 'AGENTS.md'}
    assert all(r['exact'] and r['pinned_sha256'] == r['fresh_sha256'] for r in identity['files'])
    folder = ROOT / '.moe-phase-measurements' / digest
    folder.mkdir(parents=True, exist_ok=True)
    frozen = folder / 'submission.py'
    assert not frozen.exists() or frozen.read_bytes() == source
    frozen.write_bytes(source)
    launch_path = folder / 'launch.json'
    if launch_path.exists():
        launch = read(launch_path)
        assert launch['source_sha256'] == digest
    elif not args.launch:
        print(json.dumps(dict(ready=True, source_sha256=digest, launched=False, purpose='diagnostic only')))
        return
    else:
        announced_wait = False
        while True:
            try:
                check_window()  # Check time before any API calls.
                break
            except PacingWait:
                if not args.wait_for_slot:
                    raise
                if not announced_wait:
                    print('Waiting for the authorized GPU pacing slot; no API polling during this wait.', flush=True)
                    announced_wait = True
                time.sleep(30)
        if (folder / 'launch-intent.json').exists():
            raise RuntimeError('Prior launch outcome unknown: reconcile official jobs before any new upload')
        quotas = cli(['problems', '--json'])
        save(folder / 'quota-before-launch.json', dict(utc=datetime.now(timezone.utc).isoformat(), response=quotas))
        problem = next(p for p in quotas['problems'] if p['name'] == 'muse_moe_w4a16')
        authorize_check(problem)
        save(folder / 'launch-intent.json', dict(source_sha256=digest, utc=datetime.now(timezone.utc).isoformat()))
        launch = cli(['check', 'muse_moe_w4a16', f'{LINUX_ROOT}/{frozen.relative_to(ROOT).as_posix()}', '--no-wait', '--json'])
        launch.update(source_sha256=digest, submitted_utc=datetime.now(timezone.utc).isoformat())
        save(launch_path, launch)
        record_check(launch, 'muse_moe_w4a16')
    job, previous = launch.get('resolved_job_id', launch['job_id']), None
    visited = {job}
    result_path = folder / 'result.json'
    saved = read(result_path) if result_path.exists() else None
    while True:
        try:
            result = saved if saved and saved.get('status') in ('done', 'failed', 'rejected') else cli(['status', job, '--json'])
            saved = None
        except Exception as exc:
            print(f'Observation failed for {job}: {type(exc).__name__}; retaining handle', file=sys.stderr, flush=True)
            time.sleep(20)
            continue
        save(result_path, result)
        if result['status'] != previous:
            previous = result['status']
            print(f'MoE diagnostic {job}: {previous}', file=sys.stderr, flush=True)
        if previous == 'duplicate' and result.get('duplicate_of'):
            job = result['duplicate_of']
            assert job not in visited
            visited.add(job); launch['resolved_job_id'] = job; save(launch_path, launch)
            continue
        if previous in ('done', 'failed', 'rejected', 'duplicate'):
            break
        time.sleep(45)
    try:
        verdict = verify_diagnostics(result)
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        verdict = dict(complete=False, reason=type(exc).__name__, error=(result.get('result') or {}).get('error'))
    save(folder / 'diagnostic-verdict.json', verdict)
    print(json.dumps(dict(job=job, status=result['status'], diagnostic_complete=verdict['complete'],
                          scored_submissions_used=0, performance_improvement_claim=False,
                          error=(result.get('result') or {}).get('error'))), flush=True)


if __name__ == '__main__':
    main()
