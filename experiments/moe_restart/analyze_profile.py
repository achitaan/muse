"""Reproduce the v3 diagnosis from COMPLETE judge lines; no GPU or Torch.

Usage: python3 experiments/moe_restart/analyze_profile.py LOG [--output JSON]
The v3 log has no globaltimer or clock-rate sample. Conversion at --sm-ghz
(default 1.98) and aggregate bandwidth using max CTA duration are explicitly
proxies, not invented profiler measurements. M16 phases cannot be recovered.
"""
import argparse
import json
from pathlib import Path

PHASES = ('route', 'gather_sync', 'A', 'A_sync', 'B', 'B_sync', 'finish')
WA, WB = 18_874_368, 9_437_184


def fields(line):
    return dict(word.split('=', 1) for word in line.split() if '=' in word)


def pipeline_model(shifting, latency=800, compute=240, look=4):
    """Illustration ONLY: FIFO MOVs touch future LDG results too early."""
    ready = [latency] * look
    elapsed = 0
    for k in range(64):
        slot = k % look
        elapsed = max(elapsed, max(ready) if shifting else ready[slot])
        if k + look < 64:
            ready[slot] = elapsed + latency
        elapsed += compute
    return elapsed


def analyze(text, ghz=1.98):
    records = []
    for line in text.splitlines():
        if not line.startswith('RSTREAM_PROFILE '):
            continue
        f = fields(line)
        m, active = int(f['M']), int(f['active'])
        if m not in (8, 16, 32):
            continue
        rows, grid = (64 if m <= 8 else 128), int(f['ctas'])
        phases = {p: {s: int(f[f'{p}_cycles_{s}']) for s in ('med', 'max')} for p in PHASES}
        a_cycles = phases['A']['max']
        b_cycles = phases['B']['max']
        sm_share = 4.4e12 / 132 / (ghz * 1e9)
        records.append(dict(
            M=m, mode=f['mode'], active=active, grid=grid, rows=rows,
            profile_event_us=float(f['us']), phases=phases,
            phase_median_us_at_assumed_clock={p: phases[p]['med']/(ghz*1000) for p in PHASES},
            A_weight_bytes=active*WA, B_weight_bytes=active*WB,
            A_bytes_per_cycle_per_SM_proxy=active*WA/(132*a_cycles),
            B_bytes_per_cycle_per_SM_proxy=active*WB/(132*b_cycles),
            A_bytes_per_cycle_per_CTA_proxy=active*WA/(grid*a_cycles),
            A_fraction_of_HBM_share_proxy=(active*WA/(132*a_cycles))/sm_share,
            unique_weight_floor_us=(active*(WA+WB))/4.4e6,
            A_full_wait_max=int(f['A_full_wait_cycles_max']),
            A_work_max=int(f['A_work_cycles_max']),
            A_producer_empty_wait_max=int(f['A_empty_wait_cycles_max']),
        ))
    gates = [fields(l) for l in text.splitlines() if l.startswith('RSTREAM_GATE ')]
    return dict(
        assumptions=dict(sm_ghz=ghz, SMs=132, HBM_TBps=4.4,
                         HBM_bytes_per_cycle_per_SM=4.4e12/132/(ghz*1e9),
                         phase_medians_are_not_additive_wall_time=True,
                         CTA_phase_max_is_not_globally_aligned_phase_envelope=True,
                         incumbent_phase_times_not_in_log=True),
        missing_small_M_phase_samples=sorted({8,16,32}-{r['M'] for r in records}),
        gates=gates, profiles=records,
        illustrative_pipeline=dict(latency_cycles=800, compute_cycles_per_K64=240, look=4,
                                   shifting_cycles=pipeline_model(True),
                                   cyclic_cycles=pipeline_model(False),
                                   not_a_measured_speedup=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    parser.add_argument('--sm-ghz', type=float, default=1.98)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = analyze(args.log.read_text(), args.sm_ghz)
    data = json.dumps(report, indent=2)+'\n'
    if args.output:
        args.output.write_text(data)
    print(data, end='')
