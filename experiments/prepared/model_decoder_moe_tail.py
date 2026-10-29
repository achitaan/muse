"""Independent finite layout, publication and arithmetic checks; no GPU calls."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from model_decoder_cluster_norm import bf16
from model_decoder_cluster_norm_dual_order import tiny_sum
from model_decoder_cluster_norm_ordered import compute
from diagnose_decoder_cluster_norm_reduction import xor_sum

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
OUT = ROOT / '.tools/decoder-moe-tail-20261029'


def weight_addresses():
    rb, ki, warp, lane, word, half = np.ogrid[:64, :32, :8, :32, :4, :2]
    # PTX's A fragment: words0/2 own row g; words1/3 own row g+8;
    # words2/3 own the upper K8. All eight K512 partitions are disjoint.
    row = rb * 16 + (lane >> 2) + (word & 1) * 8
    col = warp * 512 + ki * 16 + (lane & 3) * 2 + (word >> 1) * 8 + half
    return (row * 4096 + col).astype(np.int64)


def sum_cuda(x):
    # A separate implementation of the emitted CUDA loop, compared below to
    # the earlier exact retained-Triton model, not to itself.
    v = (x * x).astype(np.float32).reshape(-1, 32, 256).copy()
    for step in (1, 2, 4, 8, 16):
        for i in range(0, 32, step * 2):
            v[:, i] = (v[:, i] + v[:, i + step]).astype(np.float32)
    warp = xor_sum(v[:, 0].reshape(-1, 8, 32))[:, :, 0]
    red = np.zeros((len(x), 32), np.float32)
    red[:, :8] = warp
    return xor_sum(red)[:, 0]


def publication_schedule(seed, mutant=None):
    rng = random.Random(seed)
    ready = [0] * 8
    second = [0, 0]
    gen = 0
    for launch, rows in enumerate((1, 8, 3, 7, 2, 4, 8, 1)):
        second[(gen + 1) & 1] = 0
        state = ['publish' if b < 4 * rows else 'wait' for b in range(132)]
        written = [set() for _ in range(rows)]
        consumed = set()
        while any(s != 'done' for s in state):
            actors = []
            for b, s in enumerate(state):
                if s == 'done': continue
                if s == 'wait' and b < 64 and min(ready[:rows]) < 4: continue
                if s == 'join' and second[gen & 1] < 132 and mutant != 'omit_second_join': continue
                actors.append(b)
            assert actors, 'publication deadlock'
            b = rng.choice(actors)
            s = state[b]
            if s == 'publish':
                tk, seg = divmod(b, 4)
                if mutant == 'publish_before_store':
                    ready[tk] += 1
                    state[b] = 'late_store'
                else:
                    written[tk].add(seg)
                    ready[tk] += 1
                    state[b] = 'wait'
            elif s == 'late_store':
                written[b // 4].add(b % 4)
                state[b] = 'wait'
            elif s == 'wait':
                if b < 64:
                    assert all(len(v) == 4 for v in written), 'read before BF16 segment publication'
                    consumed.add(b)
                second[gen & 1] += 1
                state[b] = 'join'
            else:
                if b < rows:
                    assert len(consumed) == 64, 'boundary/reset before all projection consumers'
                    if mutant != 'omit_ready_clear': ready[b] = 0
                state[b] = 'done'
        assert all(x == 0 for x in ready[:rows]), 'stale readiness at replay exit'
        assert second[gen & 1] == 132
        gen += 1
    return 8


def main():
    binding = json.loads((OUT / 'binding.json').read_bytes())
    assert hashlib.sha256((OUT / 'prototype.cu').read_bytes()).hexdigest() == binding['prototype_CUDA_sha256']
    addresses = weight_addresses()
    assert addresses.shape == (64, 32, 8, 32, 4, 2)
    assert np.array_equal(np.sort(addresses.ravel()), np.arange(1024 * 4096))
    # Bit identities include signed zeros, subnormals, infinity and NaN payloads.
    # This proves a byte permutation, independently of any floating-point oracle.
    bits = np.random.default_rng(20261029).integers(0, 65536, 1024 * 4096, dtype=np.uint16)
    packed = bits[addresses]
    restored = np.empty_like(bits)
    restored[addresses.ravel()] = packed.ravel()
    assert np.array_equal(bits, restored)
    ownership = []
    for rows in range(1, 9):
        dest = []
        for rb in range(64):
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for h in range(2):
                    if 2 * t + h < rows:
                        dest.extend(((2 * t + h) * 1024 + rb * 16 + g,
                                     (2 * t + h) * 1024 + rb * 16 + g + 8))
        assert sorted(dest) == list(range(rows * 1024))
        ownership.append(dict(rows=rows, output_writes=len(dest), exactly_once=True))
    # B-fragment ownership reconstructs K16 x 8 exactly for every warp/step.
    bd = [(2 * (lane & 3) + 8 * word + h, lane >> 2)
          for lane in range(32) for word in range(2) for h in range(2)]
    assert sorted(bd) == [(k, t) for k in range(16) for t in range(8)]
    schedules = sum(publication_schedule(seed) for seed in range(256))
    mutants = []
    for mutant in ('publish_before_store', 'omit_second_join', 'omit_ready_clear'):
        for seed in range(256):
            try: publication_schedule(seed, mutant)
            except AssertionError:
                mutants.append(mutant)
                break
        else: raise AssertionError('mutation escaped: ' + mutant)
    rng = np.random.default_rng(20261029)
    numerical = []
    for rows in (1, 2, 3, 4, 7, 8):
        for fixture in ('random', 'zero_branch', 'zero_stream', 'outlier', 'small'):
            branch = bf16(rng.normal(size=(rows, 1024)).astype(np.float32))
            stream = rng.normal(size=(rows, 8192)).astype(np.float32)
            post = bf16(rng.normal(1, .25, size=(1, 8192)).astype(np.float32))
            pre = bf16(rng.normal(1, .25, size=(1, 8192)).astype(np.float32))
            alpha = rng.uniform(.25, 1, size=(1, 8192)).astype(np.float32)
            beta = rng.uniform(.25, 1, size=(1, 8192)).astype(np.float32)
            if fixture == 'zero_branch': branch[:] = 0
            if fixture == 'zero_stream': stream[:] = 0
            if fixture == 'outlier': branch[:, 17] = 32; stream[:, 34] = -32
            if fixture == 'small': branch = bf16(branch * np.float32(1e-4)); stream *= np.float32(1e-4)
            args = (branch, stream, post, pre, alpha, beta, True)
            actual = compute(*args, sum_cuda)
            expected = compute(*args, tiny_sum)
            assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(actual, expected))
            numerical.append(dict(rows=rows, fixture=fixture, boundary_CPU_bitwise_equal=True))
    # Analytic unpooler oracles: impulse selects one exact original BF16
    # column; dyadic outer products have an exactly representable dot product.
    a = bf16(rng.normal(size=(16, 4096)).astype(np.float32))
    for rows in range(1, 9):
        x = np.zeros((rows, 4096), np.float32)
        for tk in range(rows): x[tk, (tk * 521 + 7) % 4096] = np.float32(2 ** (tk % 3 - 1))
        partial = [x[:, k:k+512] @ a[:, k:k+512].T for k in range(0, 4096, 512)]
        value = np.zeros((rows, 16), np.float32)
        for p in partial: value = (value + p).astype(np.float32)
        ideal = x.astype(np.float64) @ a.astype(np.float64).T
        assert np.array_equal(value, ideal.astype(np.float32))
        assert np.array_equal(bf16(value), bf16(ideal))
    report = dict(CPU_only=True, GPU_jobs=0, native_CUDA_sha256=binding['prototype_CUDA_sha256'],
                  bitwise_weight_values=bits.size, weight_representation='BF16 retained, 8 MiB lossless layout',
                  exactly_once_output_ownership=ownership, B_fragment_values=len(bd),
                  replay_schedules=schedules, random_schedules=256, mutants_rejected=mutants,
                  independent_boundary_cases=numerical, analytic_impulse_GEMM_cases=8,
                  sequential_scratch_required=True, private_tail_scratch_required=True,
                  unpooler_full_K=4096, unpooler_output_N=1024, norm_domain=8192,
                  limits='Finite CPU models, no GPU execution or proof of hardware rsqrt, host ABI, residency or profitable timing')
    (OUT / 'CPU-model.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
    print(json.dumps({k:v for k,v in report.items() if k not in ('independent_boundary_cases','exactly_once_output_ownership')}, indent=2))


if __name__ == '__main__':
    main()
