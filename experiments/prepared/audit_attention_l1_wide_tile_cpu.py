"""Ownership, metadata bounds and mathematical screens; not GPU validation."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from build_attention_l1_wide_tile import OUT, ROOT


def quantize_positive_e4m3(x):
    bits = np.arange(127)
    exponent = bits >> 3
    mantissa = bits & 7
    table = np.where(exponent == 0, mantissa * 2.0**-9,
                     (1 + mantissa / 8) * np.exp2(exponent - 7)).astype(np.float32)
    hi = np.minimum(np.searchsorted(table, x), 126)
    lo = np.maximum(hi - 1, 0)
    dl, dh = x - table[lo], table[hi] - x
    choose_hi = (dh < dl) | ((dh == dl) & ((hi & 1) == 0))
    return table[np.where(choose_hi, hi, lo)]


def stream_attention(scores, values, tile):
    """FP32 model with the retained256-scaled FP8 PV probability boundary.

    NumPy reductions are mathematical proxies, not CUDA MMA accumulation order.
    """
    states = []
    for half in range(2):
        maximum = np.float32(-np.inf)
        denominator = np.float32(0)
        numerator = np.zeros(values.shape[1], np.float32)
        for first in range(0, len(scores), tile):
            lo, hi = first + half * (tile // 2), min(len(scores), first + (half + 1) * (tile // 2))
            if lo >= hi:
                continue
            s = scores[lo:hi]
            new_maximum = np.maximum(maximum, s.max())
            alpha = np.exp2(maximum - new_maximum)
            probability = np.exp2(s - new_maximum).astype(np.float32) * np.float32(256)
            denominator = np.float32(denominator * alpha + probability.sum(dtype=np.float32))
            numerator *= alpha
            rounded = quantize_positive_e4m3(probability)
            for chunk in range(0, len(s), 32):
                numerator += np.einsum('i,ij->j', rounded[chunk:chunk+32], values[lo+chunk:min(hi,lo+chunk+32)], dtype=np.float32)
            maximum = new_maximum
        states.append((maximum, denominator, numerator))
    maximum = max(s[0] for s in states)
    denominator = np.float32(0)
    numerator = np.zeros(values.shape[1], np.float32)
    for m, d, n in states:
        factor = np.float32(0) if not np.isfinite(m) else np.exp2(m - maximum)
        denominator += d * factor
        numerator += n * factor
    return numerator / denominator


def main():
    binding = json.loads((OUT / 'binding.json').read_text())
    assert binding['draft_sha256'] == hashlib.sha256((OUT / 'submission.py').read_bytes()).hexdigest()
    cases = 0
    for tile in (64, 128):
        # Producer: one unique16-byte chunk per row and one of8chunks.
        writes = Counter((row, chunk) for q in range(128)
                         for k in range(tile // 16)
                         for row, chunk in [(q // 8 + 16*k, q & 7)])
        assert len(writes) == tile * 8 and set(writes.values()) == {1}
        # QK and PV consume every key once per KV head, in the same half.
        for head in (0, 1):
            qk = Counter((tile//2)*half + 8*nt + 2*tig + e
                         for half in (0,1) for nt in range(tile//16)
                         for tig in range(4) for e in (0,1))
            pv = Counter((tile//2)*half + 32*piece + r
                         for half in (0,1) for piece in range(tile//64) for r in range(32))
            assert qk == pv == Counter(range(tile))
            cases += 1
        # Aligned page windows: at most3 preceding entries, rounded transaction
        # end; all valid positions lie inside the allocated pidx row.
        capacity = 144 if tile == 128 else 80
        for page_shift in range(11):
            for offset in range(4):
                fp = 12 + offset
                first = fp << page_shift
                base = fp & -4
                last = (fp + tile + 3) & -4
                assert (last - base) <= capacity
                for valid in (1, tile-1, tile):
                    indices = [((first+r) >> page_shift) - base for r in range(valid)]
                    assert min(indices) >= 0 and max(indices) < last-base
                    cases += 1
        # Ring ownership and phase transitions over many generations.
        stages = 2 if tile == 128 else 3
        for initial in range(13):
            for count in (0,1,2,3,7,31,129):
                occupied = {}
                consumed = 0
                for issued in range(count):
                    if len(occupied) == stages:
                        slot = (initial + consumed) % stages
                        assert occupied.pop(slot) == initial + consumed
                        consumed += 1
                    slot = (initial + issued) % stages
                    assert slot not in occupied
                    occupied[slot] = initial + issued
                while consumed < count:
                    assert occupied.pop((initial+consumed) % stages) == initial+consumed
                    consumed += 1
                assert not occupied
                cases += 1
    rng = np.random.default_rng(20261029)
    numerical = []
    for n in (1,31,32,33,63,64,65,127,128,129,257,2049,8193,80001):
        for sigma in (0.25, 1., 8.):
            scores = (rng.normal(size=n) * sigma).astype(np.float32)
            values = rng.normal(size=(n,16)).astype(np.float32)
            # Include coherent values, avoiding a cancellation-only screen.
            if sigma == 0.25:
                values += np.float32(0.5)
            reference_p = np.exp2(scores.astype(np.float64) - scores.max())
            oracle = (reference_p @ values.astype(np.float64)) / reference_p.sum()
            baseline = stream_attention(scores, values, 64)
            wide = stream_attention(scores, values, 128)
            norm = max(float(np.linalg.norm(oracle)), 1e-12)
            old_error = float(np.linalg.norm(baseline-oracle))/norm
            new_error = float(np.linalg.norm(wide-oracle))/norm
            difference = float(np.linalg.norm(wide-baseline))/norm
            assert np.isfinite(wide).all()
            numerical.append(dict(keys=n, score_sigma=sigma, baseline_oracle_relative_L2=old_error,
                                  wide_oracle_relative_L2=new_error, wide_baseline_relative_L2=difference))
    report = dict(draft_sha256=binding['draft_sha256'], ownership_and_bounds_cases=cases,
                  numerical_proxy_cases=numerical, GPU_jobs=0, GPU_correctness_verified=False,
                  probability_rounding='Positive finite E4M3 nearest-even, retained scale256',
                  limitations=['No CUDA MMA/exp2/rounding-order emulation or official input distribution claim',
                               'The ring model checks finite sequential ownership, not real barrier memory ordering or deadlock freedom',
                               'Full cache writes, norms, gate and BF16 outputs still require the official GPU suite'])
    data = (json.dumps(report, indent=2) + '\n').encode()
    path = OUT / 'cpu-audit.json'
    assert not path.exists() or path.read_bytes() == data, 'Frozen CPU audit collision'
    path.write_bytes(data)
    print(json.dumps(dict(ownership_and_bounds_cases=cases, numerical_proxy_cases=len(numerical),
                          max_wide_oracle_relative_L2=max(x['wide_oracle_relative_L2'] for x in numerical),
                          max_wide_baseline_relative_L2=max(x['wide_baseline_relative_L2'] for x in numerical), GPU_jobs=0)))


if __name__ == '__main__':
    main()
