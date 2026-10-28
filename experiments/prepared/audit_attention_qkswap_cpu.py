"""PTX-derived LDMATRIX/MMA operand, shared transpose and numerical audit.

Independent oracle: direct K[16,64] @ Q[8,64].T. Register assignments follow
NVIDIA PTX m16n8k16 figures82/84/86 and ldmatrix figure107. No GPU evidence.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_attention_precision import verify

torch.set_num_threads(2)
torch.manual_seed(20261028)
source = (ROOT / '.tools/attention-qkswap.py').read_bytes()
audit = json.loads((ROOT / '.tools/attention-qkswap-source-audit.json').read_bytes())
assert hashlib.sha256(source).hexdigest() == audit['source_sha256']
parent = (ROOT / '.attention-measurements' / audit['parent_sha256'] / 'submission.py').read_bytes()
assert hashlib.sha256(parent).hexdigest() == audit['parent_sha256']
verify(source)
new_tree, old_tree = ast.parse(source), ast.parse(parent)
def cu_node(t):
    return next(n for n in t.body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'CUDA_SRC' for x in n.targets))
node, old_node = cu_node(new_tree), cu_node(old_tree)
cuda = ast.literal_eval(node.value)
rev = cuda
for e in reversed(audit['edits']):
    assert rev.count(e['after']) == 1
    rev = rev.replace(e['after'], e['before'])
assert rev == ast.literal_eval(old_node.value)
node.value = old_node.value
assert ast.dump(new_tree, include_attributes=False) == ast.dump(old_tree, include_attributes=False)
assert cuda.count('csync<WS, NT>();\n  float* qk_transpose') == 1
assert 'float* qk_transpose = reinterpret_cast<float*>(S.kvq) + warp * 128;' in cuda
assert 'kvq[(L == 1) ? L * 16 * 64 * 2 : 18432]' in cuda


def qstage(q):
    stage = np.empty((8, 64), dtype=q.dtype)
    for p in range(32):
        m = (p & ~3) + ((p & 1) << 1) + ((p & 2) >> 1)
        stage[:, 2 * m:2 * m + 2] = q[:, 2 * p:2 * p + 2]
    return stage


def kstage(k):
    stage = np.empty(64 * 128, dtype=k.dtype)
    for key in range(64):
        for hk in range(2):
            for dim in range(64):
                address = key * 128 + (((hk * 4 + (dim >> 4)) ^ (key & 7)) << 4) + (dim & 15)
                stage[address] = k[key, hk, dim]
    return stage


def operands(stage, qs, hk, wsub, kt, ks):
    a, b = np.empty((16, 16), stage.dtype), np.empty((16, 8), qs.dtype)
    ac, bc = np.zeros_like(a, dtype=int), np.zeros_like(b, dtype=int)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        # LDMATRIX.x2: address-provider lanes0..15; four consumers per row.
        # Each raw register contains four FP8 bytes, before lossless f16 expansion.
        raw = []
        for matrix in range(2):
            provider = g + 8 * matrix
            key = 32 * wsub + 16 * kt + provider
            addr = key * 128 + (((hk * 4 + ks) ^ (key & 7)) << 4)
            assert addr % 16 == 0 and 0 <= addr <= len(stage) - 16
            raw.append(stage[addr + 4 * t:addr + 4 * t + 4])
        # Actual helper call: cvt2(kl),cvt2(kh),cvt2h(kl),cvt2h(kh).
        frags = (raw[0][:2], raw[1][:2], raw[0][2:], raw[1][2:])
        for reg, values in enumerate(frags):
            row = g + 8 * (reg & 1)
            col = 2 * t + 8 * (reg >> 1)
            a[row, col:col + 2] = values
            ac[row, col:col + 2] += 1
        pair0 = 8 * ks + 2 * t
        for reg, p in enumerate((pair0, pair0 + 1)):
            m = (p & ~3) + ((p & 1) << 1) + ((p & 2) >> 1)
            col = 2 * t + 8 * reg
            b[col:col + 2, g] = qs[g, 2 * m:2 * m + 2]
            bc[col:col + 2, g] += 1
    assert (ac == 1).all() and (bc == 1).all()
    return a, b


def transpose(c):
    sm = np.empty(128, c.dtype)
    written = np.zeros(128, int)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(4):
            key, head = g + 8 * (i >> 1), 2 * t + (i & 1)
            address = (head * 16 + key) ^ (t * 8)
            sm[address] = c[key, head]
            written[address] += 1
    assert (written == 1).all()
    result = np.empty((32, 2, 2), c.dtype)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for h in range(2):
            ix = (g * 16 + 8 * h + 2 * t) ^ ((g >> 1) * 8)
            assert ix % 2 == 0 and 0 <= ix < 127
            result[lane, h] = sm[ix:ix + 2]
            assert np.array_equal(result[lane, h], c[8 * h + 2 * t:8 * h + 2 * t + 2, g])
    return result


rng = np.random.default_rng(20261028)
matrices = 0
for trial in range(8):
    # Integer arithmetic makes operand coverage and complete QK equality exact.
    k = rng.integers(-16, 16, (64, 2, 64), dtype=np.int64)
    q = rng.integers(-16, 16, (2, 8, 64), dtype=np.int64)
    stage = kstage(k)
    for hk in range(2):
        qs = qstage(q[hk])
        for wsub in range(2):
            for kt in range(2):
                c = np.zeros((16, 8), dtype=np.int64)
                for ks in range(4):
                    a, b = operands(stage, qs, hk, wsub, kt, ks)
                    perm = np.array([4 * t + e for h in range(2) for t in range(4) for e in range(2)]) + np.repeat([0, 2], 8)
                    assert sorted(perm.tolist()) == list(range(16))
                    base = 32 * wsub + 16 * kt
                    assert np.array_equal(a, k[base:base + 16, hk][:, 16 * ks + perm])
                    assert np.array_equal(b, q[hk][:, 16 * ks + perm].T)
                    c += a @ b
                assert np.array_equal(c, k[base:base + 16, hk] @ q[hk].T)
                transpose(c)
                matrices += 1

# All FP8 byte values survive pair grouping; no byte is omitted or duplicated.
bytes_ = np.arange(256, dtype=np.uint8).reshape(64, 4)
assert np.array_equal(np.concatenate((bytes_[:, :2], bytes_[:, 2:]), axis=1), bytes_)
# Bank model: each scalar write hits every bank once. Each half-warp's float2
# read covers every bank once, the ideal two128Btransactions for a warp.
for reg in range(4):
    banks = []
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        key, head = g + 8 * (reg >> 1), 2 * t + (reg & 1)
        banks.append(((head * 16 + key) ^ (t * 8)) % 32)
    assert len(set(banks)) == 32
for half in range(2):
    for lane_start in (0, 16):
        banks = []
        for lane in range(lane_start, lane_start + 16):
            g, t = lane >> 2, lane & 3
            ix = (g * 16 + half * 8 + 2 * t) ^ ((g >> 1) * 8)
            banks.extend((ix % 32, (ix + 1) % 32))
        assert len(set(banks)) == 32
slots = [set(range(w * 512, (w + 1) * 512)) for w in range(4)]
assert len(set.union(*slots)) == 2048 and sum(map(len, slots)) == 2048

# Numerical screen: same FP16 queries and exact FP8->FP16 cache values,
# FP32 sequential sums in the candidate's K permutation vs independent FP64.
errors = []
for trial in range(16):
    q = torch.randn(8, 64).half().float().numpy()
    k = (torch.randn(32, 64) * (1 if trial < 8 else 8)).to(torch.float8_e4m3fn).half().float().numpy()
    perm = np.array([4 * t + e + 2 * h for ks in range(4) for h in range(2) for t in range(4) for e in range(2)]) + np.repeat(np.arange(4) * 16, 16)
    assert sorted(perm.tolist()) == list(range(64))
    got = np.zeros((32, 8), np.float32)
    for col in perm:
        got = (got + k[:, col:col + 1] * q[:, col]).astype(np.float32)
    want = k.astype(np.float64) @ q.astype(np.float64).T
    errors.append(float(np.linalg.norm(got - want) / np.linalg.norm(want)))
assert max(errors) < 1e-6
report = dict(source_sha256=audit['source_sha256'], seed=20261028,
              exact_int64_complete_matrix_oracles=matrices, operand_k_coverage_exact=True,
              ld_matrix_x2_and_original_swizzle_addresses=True, inherited_q_word_order_inverted=True,
              warp_transpose_bijective=True, warp_private_bytes=2048, modeled_bank_conflicts='none beyond ideal transactions',
              query_to_transpose_reuse_order='one128consumer barrier; perwarp write/read/reuse syncwarp',
              max_fp32_accum_relative_l2_vs_fp64=max(errors), numerical_cases=len(errors),
              fp32_model_not_hardware_accumulation=True, parent_cuda_inverse=True,
              full_bf16_precision_guard=True, gpu_correctness='unmeasured', speedup='unmeasured')
(ROOT / '.tools/attention-qkswap-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
