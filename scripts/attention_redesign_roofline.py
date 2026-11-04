"""Reproduce the scored board and screen a fresh QK/P/PV kernel decomposition.

This is an analytical byte lower bound, NOT GPU timing or kernel validation.
Assumptions are explicit: E4M3 K/V, Hkv=2, Hq=16, D=64, L=8, 4.8 TB/s.
The completed job does not expose actual seq_lens: 74k/80k/86k are sensitivity
points, not recovered input lengths. MiB cache credit is intentionally generous.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'results/attention_redesign_baseline.json').read_text())
ROWS = DATA['records']
BY_KEY = {(r['batch'], r['q_len'], r['layer']): r for r in ROWS}
HBM_BYTES_PER_US = 4.8e6  # decimal 4.8 TB/s, not TiB/s
P_CACHE_CREDIT = 64 * 1024**2  # allow ALL 64 MiB to shelter P; no K/V pressure
TARGETS = ((16, 8, 'full'), (32, 8, 'full'))


def board_speedup(replacements_us=None):
    """Exact six-load rule: GM((46*S_ref+16*F_ref)/(46*S+16*F))."""
    replacements_us = replacements_us or {}
    speeds = []
    for length in (1, 8):
        for batch in (1, 16, 32):
            pair = [BY_KEY[batch, length, layer] for layer in ('sliding', 'full')]
            reference = sum(r['weight'] * r['baseline_ns'] / 1000 for r in pair)
            candidate = sum(r['weight'] * replacements_us.get(
                (batch, length, r['layer']), r['median_ns'] / 1000) for r in pair)
            speeds.append(reference / candidate)
    return math.exp(sum(map(math.log, speeds)) / len(speeds))


def full_target_ratio_for_gain(gain):
    lo, hi = 0., 1.
    for _ in range(80):
        ratio = (lo + hi) / 2
        times = {k: BY_KEY[k]['median_ns'] / 1000 * ratio for k in TARGETS}
        if board_speedup(times) / DATA['speedup'] >= gain:
            lo = ratio
        else:
            hi = ratio
    return (lo + hi) / 2


def traffic(batch, mean_cached_tokens, length=8, tile_keys=64):
    tokens = batch * mean_cached_tokens
    # Both allocations, every KV head, each dimension, one byte per FP8 value.
    # GQA reuses those bytes: do NOT multiply KV by q_len or Hq/Hkv.
    kv = tokens * 2 * 2 * 64
    p = tokens * length * 16
    # One FP32 log2 offset per query row and key tile. Kernel B obtains the
    # denominator from P (or a ones PV column), so no l metadata is required.
    m = batch * math.ceil(mean_cached_tokens / tile_keys) * length * 16 * 4
    # Two sequential kernels cannot both keep a >L2 P matrix resident. Give
    # each write and read the ENTIRE cache as a benefit, ignoring even m traffic.
    p_hbm_min = 2 * max(0, p - P_CACHE_CREDIT)
    return dict(kv=kv, p=p, m_write_read=2*m,
                fused_floor_us=kv / HBM_BYTES_PER_US,
                fresh_no_cache_us=(kv + 2*p) / HBM_BYTES_PER_US,
                fresh_optimistic_us=(kv + p_hbm_min) / HBM_BYTES_PER_US)


class RooflineTests(unittest.TestCase):
    def test_reproduces_completed_score_and_score_us(self):
        # Source: completed scored job 8f53cd76d828427c, not a prediction.
        self.assertAlmostEqual(board_speedup(), DATA['speedup'], places=12)
        self.assertAlmostEqual(DATA['baseline_us'] / board_speedup(), DATA['score_us'], places=9)

    def test_no_gqa_or_verify_multiplier_on_cache_bytes(self):
        # Analytic: 16 requests * 80k cached rows * K/V * 2 heads * 64 FP8 B.
        self.assertEqual(traffic(16, 80000)['kv'], 327680000)
        self.assertEqual(traffic(32, 80000)['kv'], 655360000)
        self.assertEqual(traffic(16, 80000, length=1)['kv'], traffic(16, 80000)['kv'])
        self.assertEqual(traffic(16, 80000)['p'], 163840000)

    def test_three_percent_board_needs_10_8067_percent_on_both_targets(self):
        ratio = full_target_ratio_for_gain(1.03)
        self.assertAlmostEqual(ratio, .8919330302322836, places=10)
        new = {k: BY_KEY[k]['median_ns'] / 1000 * ratio for k in TARGETS}
        self.assertAlmostEqual(board_speedup(new) / DATA['speedup'], 1.03, places=12)

    def test_fresh_design_fails_even_with_64_mib_cache_and_no_metadata_math_latency(self):
        for mean in (74000, 80000, 86000):
            new = {(batch, 8, 'full'): traffic(batch, mean)['fresh_optimistic_us'] for batch in (16, 32)}
            # Analytic lower bound, not a timing claim. Both target shapes lose
            # even at 74k with a larger-than-actual dedicated P cache credit.
            for key, bound in new.items():
                self.assertGreater(bound, BY_KEY[key]['median_ns'] / 1000)
            self.assertLess(board_speedup(new), DATA['speedup'])

    def test_do_not_claim_absolute_hbm_floor(self):
        ideal = {(batch, 8, 'full'): traffic(batch, 80000)['fused_floor_us'] for batch in (16, 32)}
        # A different perfectly fused design still has theoretical room. The
        # rejected materialization design is NOT a proof of global optimality.
        self.assertGreater(board_speedup(ideal) / DATA['speedup'], 1.03)


def report():
    total = sum(r['weight'] * r['median_ns'] for r in ROWS)
    print('completed job:', DATA['job_id'], 'reproduced speedup:', board_speedup())
    print('case          layer    candidate_us  SGLang_x  weighted_time_share')
    for r in ROWS:
        share = 100 * r['weight'] * r['median_ns'] / total
        print(f"{r['load']:12} {r['layer']:8} {r['median_ns']/1000:11.4f} {r['speedup']:9.4f} {share:10.3f}%")
    ratio = full_target_ratio_for_gain(1.03)
    print(f'both verify-full targets require {(1-ratio)*100:.6f}% lower time for +3% board')
    for mean in (74000, 80000, 86000):
        new = {}
        for batch in (16, 32):
            t = traffic(batch, mean)
            new[batch, 8, 'full'] = t['fresh_optimistic_us']
            print(f"B={batch} N={mean} KV={t['kv']/1e6:.3f}MB P={t['p']/1e6:.3f}MB "
                  f"ideal_fused={t['fused_floor_us']:.3f}us fresh_with_64MiB={t['fresh_optimistic_us']:.3f}us")
        gain = board_speedup(new) / DATA['speedup']
        print(f'forced fresh board upper bound: {board_speedup(new):.6f} ({100*(gain-1):+.3f}%)')
    print('BUILD GATE: FAIL; zero target gains even before metadata, indexing, math or extra launch latency')
    print('With incumbent fallback: 2.653624x unchanged; no kernel built or checked.')
    ideal = {(batch, 8, 'full'): traffic(batch, 80000)['fused_floor_us'] for batch in (16, 32)}
    print(f'Not an absolute-floor proof: ideal fused HBM ceiling {board_speedup(ideal):.6f}x')


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RooflineTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    report()
