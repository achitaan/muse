"""Scalar contribution model only: no GEMM, kernel or GPU validation."""
from pathlib import Path
import hashlib
import json
import numpy as np

OUT = Path(__file__).resolve().parents[2] / 'results/decode-moe-plan-20261030/norm-register'


def addends_sum(a, tail, flush=True):
    jobs = len(a)
    v = a.copy()
    if flush:
        v[(v.view(np.uint32) & 0x7f800000) == 0] = 0
    contributions = [[] for _ in range(jobs)]
    for c in range(132):
        lo, hi = 256 * jobs * c // 132, 256 * jobs * (c + 1) // 132
        stop = hi - min(tail, hi - lo)
        for j in range(jobs):
            cnt = 1 + j % 8
            x, y = max(lo, 256 * j) - 256 * j, min(stop, 256 * (j + 1)) - 256 * j
            if x < y:
                part = np.cumsum(v[j, :cnt, x:y], axis=1, dtype=np.float32)[:, -1]
                part[(part.view(np.uint32) & 0x7f800000) == 0] = 0
                contributions[j].append(part)
        for unit in range(stop, hi):
            j, rb = divmod(unit, 256)
            contributions[j].append(v[j, :1 + j % 8, rb])
    result = np.zeros((jobs, 8), dtype=np.float32)
    for j, parts in enumerate(contributions):
        if parts:
            result[j, :1 + j % 8] = np.cumsum(np.stack(parts[::-1]), axis=0, dtype=np.float32)[-1]
    return result


def main():
    cases, maximum = 0, 0.
    for jobs in (1, 2, 4, 8, 16, 32):
        for seed in (17, 37, 71):
            rng = np.random.default_rng(seed)
            for pattern in ('normal', 'wide', 'subnormal', 'zero'):
                shape = (jobs, 8, 256)
                if pattern == 'normal': a = (4 * rng.random(shape)).astype(np.float32)
                elif pattern == 'wide': a = np.exp2(rng.integers(-140, 21, shape)).astype(np.float32)
                elif pattern == 'subnormal': a = np.full(shape, np.float32(2. ** -130))
                else: a = np.zeros(shape, dtype=np.float32)
                ftz = a.copy()
                ftz[(ftz.view(np.uint32) & 0x7f800000) == 0] = 0
                old = np.zeros((jobs, 8), dtype=np.float32)
                exact = np.zeros((jobs, 8), dtype=np.float64)
                for j in range(jobs):
                    cnt = 1 + j % 8
                    old[j, :cnt] = np.cumsum(ftz[j, :cnt], axis=1, dtype=np.float32)[:, -1]
                    exact[j, :cnt] = ftz[j, :cnt].astype(np.float64).sum(axis=1)
                    a[j, cnt:] = np.nan  # invalid routed entries must not enter arithmetic
                for tail in (1, 2):
                    new = addends_sum(a, tail)
                    assert np.isfinite(new).all()
                    err = float(np.linalg.norm(new.astype(np.float64) - old) / max(np.linalg.norm(exact), 1e-30))
                    assert err <= 1e-5
                    maximum = max(maximum, err)
                    cases += 1
    tiny = np.full((32, 8, 256), np.float32(2. ** -130))
    assert np.all(addends_sum(tiny, 1) == 0)
    assert np.any(addends_sum(tiny, 1, flush=False) > 0)
    report = dict(CUDA_sha256=hashlib.sha256((OUT / 'candidate.cu').read_bytes()).hexdigest(),
                  cases=cases, maximum_relative_change=maximum,
                  missing_FTZ_mutant_detected=True, invalid_slots_poisoned=True, GPU_calls=0,
                  limitation='Positive addends and one legal arrival order only; no CUDA, GEMM, race or output-bound proof.')
    (OUT / 'numerical-cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
