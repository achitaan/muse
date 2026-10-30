"""Independent finalizer oracle and mutation screen; no CUDA/device calls.

The oracle is the RMSNorm/weighted-sum formula in pinned MoE TASK.md,
applied to already scaled FP32 ybuf. CPU arithmetic does not emulate CUDA
rsqrt/FMA exactly and cannot establish the GPU byte or variance gates.
"""

from pathlib import Path
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/decode-moe-plan-20261030/tiny-norm-whole'
SOURCE_SHA = '690ef087234197c1b91c7cb99161dcceb56d3f2c671986613c90c716d9d05bf0'
H = 4096


def bf16(x):
    """Round finite FP32 to BF16 with IEEE round-to-nearest/ties-to-even."""
    a = np.asarray(x, dtype=np.float32).copy()
    assert np.isfinite(a).all()
    bits = a.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16
    return (rounded.astype(np.uint32) << 16).view(np.float32)


def fixture(m, seed, pattern, distribution):
    rng = np.random.default_rng(seed)
    mask = np.zeros((m, 8), dtype=bool)
    if pattern == 'one':
        mask[:, 0] = True
    elif pattern == 'skew':
        mask = rng.random((m, 8)) < np.linspace(.1, .9, m)[:, None]
    elif pattern == 'all':
        mask[:] = True
    elif pattern == 'holes':
        mask[:, (1, 3, 7)] = True
    else:
        assert pattern == 'none'
    n = int(mask.sum())
    pos = np.full((m, 8), -1, dtype=np.int32)
    pos[mask] = rng.permutation(n)
    # Poison unused rows: multiply-by-zero is not a valid substitute for masking.
    y = np.full((m * 8 + 1, H), np.nan, dtype=np.float32)
    if distribution == 'normal':
        values = rng.standard_normal((n, H)).astype(np.float32)
    elif distribution == 'wide':
        values = (rng.standard_normal((n, H)) * np.exp2(rng.integers(-12, 13, (n, H)))).astype(np.float32)
    elif distribution == 'uniform':
        values = np.broadcast_to(np.exp2(rng.integers(-8, 9, (n, 1))), (n, H)).astype(np.float32).copy()
    else:
        assert distribution == 'zero'
        values = np.zeros((n, H), dtype=np.float32)
    y[:n] = values
    weights = rng.random((m, 8)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    return y, pos, weights, n


def oracle(y, pos, weights):
    # Independent mathematical oracle: dense [token,slot,column] gathering,
    # FP64 norm/sum, then one output BF16 materialization. Source: MoE TASK.md.
    gathered = np.zeros((*pos.shape, H), dtype=np.float64)
    mask = pos >= 0
    gathered[mask] = y[pos[mask]].astype(np.float64)
    sums = np.einsum('msh,msh->ms', gathered, gathered)
    scaled = gathered / np.sqrt(sums[..., None] / H + 1e-8)
    return bf16(np.einsum('ms,msh->mh', weights.astype(np.float64), scaled)), sums


def modeled_finalizer(y, pos, weights, mutant=None):
    n = int((pos >= 0).sum())
    rows = y[:n].reshape(n, 256, 16)
    partial = np.zeros((n, 256), dtype=np.float32)
    for c in range(16):
        partial += rows[:, :, c] * rows[:, :, c]
    warp = partial.reshape(n, 8, 32)
    lanes = np.arange(32)
    for offset in (16, 8, 4, 2, 1):
        warp = warp + warp[:, :, lanes ^ offset]
    sums = np.zeros(n, dtype=np.float32)
    for w in range(7 if mutant == 'missing_warp' else 8):
        sums += warp[:, w, 0]
    accum = np.zeros((pos.shape[0], H), dtype=np.float32)
    observed = np.zeros(pos.shape, dtype=np.float32)
    for t in range(pos.shape[0]):
        for slot in range(8):
            q = pos[t, slot]
            if q < 0:
                if mutant == 'nan_times_zero':
                    accum[t] += y[-1] * np.float32(0)
                continue
            observed[t, slot] = sums[q]
            denom = np.float32(2048 if mutant == 'wrong_domain' else H)
            inv = np.float32(1) / np.sqrt(sums[q] / denom + np.float32(1e-8))
            scale = inv if mutant == 'missing_weight' else np.float32(weights[t, slot] * inv)
            accum[t] += scale * y[q]
    if mutant == 'store_stride':
        accum[:, H // 2:] = 0
    if not np.isfinite(accum).all():
        return accum, observed
    return bf16(accum), observed


def errors(out, ref):
    diff = (out.astype(np.float64) - ref.astype(np.float64))
    norms = np.linalg.norm(ref.astype(np.float64), axis=1)
    floor = max(.1 * float(norms.mean()), 1e-12)
    return float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-12)), float(np.max(np.linalg.norm(diff, axis=1) / np.maximum(norms, floor)))


def main():
    source = ROOT / 'experiments/decode_moe/moe_tiny_norm_whole.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_SHA
    # Analytical BF16 ties: one halfway value rounds down to even, the next up.
    assert bf16([1 + 2**-8, 1 + 3 * 2**-8]).tolist() == [1., 1. + 2**-6]
    maxima = dict(norm_relative=0., output_relative=0., worst_row=0.)
    cases = 0
    for m in range(1, 9):
        for seed in (11, 29, 57):
            for pattern in ('none', 'one', 'skew', 'all', 'holes'):
                for distribution in ('normal', 'wide', 'uniform', 'zero'):
                    y, pos, weights, n = fixture(m, seed, pattern, distribution)
                    ref, ss = oracle(y, pos, weights)
                    actual, observed = modeled_finalizer(y, pos, weights)
                    assert np.isfinite(actual).all()
                    nr = float(np.linalg.norm(observed.astype(np.float64) - ss) / max(np.linalg.norm(ss), 1e-12))
                    rel, row = errors(actual, ref)
                    # Gates derive from the pinned task's 1.5%/7.5% output
                    # bounds and the authorized 1e-5 norm-sum bound. Applying
                    # them to this restricted CPU audit is not GPU validation.
                    assert nr <= 1e-5 and rel <= .015 and row <= .075, (m, seed, pattern, distribution, nr, rel, row)
                    if n == 0:
                        assert np.array_equal(actual.view(np.uint32), np.zeros_like(actual).view(np.uint32))
                    maxima = {k:max(maxima[k], v) for k,v in zip(maxima, (nr, rel, row))}
                    cases += 1
    y, pos, weights, _ = fixture(3, 713, 'holes', 'normal')
    ref, _ = oracle(y, pos, weights)
    rejected = []
    for mutant in ('missing_warp', 'wrong_domain', 'nan_times_zero', 'missing_weight', 'store_stride'):
        out, _ = modeled_finalizer(y, pos, weights, mutant)
        assert not np.isfinite(out).all() or errors(out, ref)[0] > .015
        rejected.append(mutant)
    record = dict(source_sha256=SOURCE_SHA, cases=cases, maxima=maxima,
                  mutants_rejected=rejected, GPU_calls=0,
                  oracle='Pinned MoE TASK.md RMSNorm/weighted-sum formula on scaled FP32 ybuf, independent FP64 dense evaluation.',
                  limitations=['No GEMM/SwiGLU or routing-kernel simulation.', 'CPU FMA/rsqrt behavior differs from CUDA.', 'No GPU byte/variance, replay, residency or latency claim.'])
    (OUT/'numerical-cpu-audit.json').write_bytes((json.dumps(record, indent=2)+'\n').encode())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
