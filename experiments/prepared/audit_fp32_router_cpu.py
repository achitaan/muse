"""Numeric model, write coverage and binding checks for the FP32 router draft."""
import ast
import builtins
import hashlib
import json
import symtable
from pathlib import Path
import numpy as np

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
rng = np.random.default_rng(20261028)
mul = np.float32(0.1767766952966369)
max_error = 0.0
max_weight_error = 0.0
cases = 0

def project_model(x, w):
    # Model FP32 FMA accumulation, 32-lane reduction and eight split sums.
    # The float64 multiply/add model is a numerical check, not a claim of
    # instruction-exact GPU emulation. Official GPU coverage remains required.
    m = x.shape[0]
    xx = x.reshape(m, 8, 32, 32)
    ww = w.reshape(256, 8, 32, 32)
    a = np.zeros((m, 256, 8, 32), dtype=np.float32)
    for j in range(32):
        a = (a.astype(np.float64) + xx[:, None, :, j].astype(np.float64) *
             ww[None, :, :, j].astype(np.float64)).astype(np.float32)
    for d in [16, 8, 4, 2, 1]:
        b = a.copy()
        b[..., :32-d] = a[..., :32-d] + a[..., d:]
        a = b
    p = a[..., 0]
    y = np.zeros((m, 256), dtype=np.float32)
    for s in range(8): y += p[..., s]
    return y

def route_model(logits, bias):
    m = logits.shape[0]
    with np.errstate(over='ignore'):
        e = np.exp(-(logits * mul).astype(np.float64)).astype(np.float32)
    score = np.float32(1) / (np.float32(1) + e)
    sel = (score + bias).reshape(m, 8, 32).copy()
    raw = score.reshape(m, 8, 32)
    chosen = np.empty((m, 8), dtype=np.int32)
    vals = np.empty((m, 8), dtype=np.float32)
    total = np.zeros(m, dtype=np.float32)
    lanes = np.arange(32)
    for r in range(8):
        b = np.full((m, 32), -np.inf, dtype=np.float32)
        ids = np.full((m, 32), 2**31-1, dtype=np.int32)
        v = np.zeros((m, 32), dtype=np.float32)
        for j in range(8):
            eid = lanes + j * 32
            take = (sel[:, j] > b) | ((sel[:, j] == b) & (eid < ids))
            b = np.where(take, sel[:, j], b)
            ids = np.where(take, eid, ids)
            v = np.where(take, raw[:, j], v)
        for d in [16, 8, 4, 2, 1]:
            oi, ob, ov = ids[:, lanes ^ d], b[:, lanes ^ d], v[:, lanes ^ d]
            take = (ob > b) | ((ob == b) & (oi < ids))
            b, ids, v = np.where(take, ob, b), np.where(take, oi, ids), np.where(take, ov, v)
        assert np.all(ids == ids[:, :1])
        chosen[:, r], vals[:, r] = ids[:, 0], v[:, 0]
        total += vals[:, r]
        for t in range(m): sel[t, chosen[t, r] >> 5, chosen[t, r] & 31] = -np.inf
    return chosen, vals * (np.float32(1) / (total + np.float32(1e-15)))[:, None]

for m, outliers in [(1, False), (3, False), (8, False), (8, True)]:
    w = (rng.normal(size=(256, 8192)) / np.sqrt(8192)).astype(np.float32)
    x = rng.normal(size=(m, 8192)).astype(np.float32)
    if outliers: x[:, ::257] *= np.float32(64)
    bias = (rng.normal(size=256) * .05).astype(np.float32)
    got = project_model(x, w)
    ref = x.astype(np.float64) @ w.astype(np.float64).T
    # Standard forward-error bound: at most 45 FP32 rounding stages along a
    # contribution's path (32 FMAs, 5 warp sums, 8 split sums), no underflow in
    # these manufactured normal/outlier cases. Oracle is independent FP64.
    u = 2.0**-24
    gamma = 45*u / (1-45*u)
    bound = gamma * (np.abs(x.astype(np.float64)) @ np.abs(w.astype(np.float64)).T) + 1e-9
    assert np.all(np.abs(got-ref) <= bound)
    max_error = max(max_error, float(np.max(np.abs(got-ref))))
    ids, weights = route_model(got, bias)
    ref_score = (1/(1+np.exp(-ref*float(mul)))).astype(np.float32)
    biased = ref_score + bias
    expected_ids = np.stack([np.lexsort((np.arange(256), -row))[:8] for row in biased])
    assert np.array_equal(ids, expected_ids)
    expected_w = np.take_along_axis(ref_score, expected_ids, 1).astype(np.float64)
    expected_w /= expected_w.sum(1, keepdims=True) + 1e-15
    err = float(np.max(np.abs(weights-expected_w)))
    assert err < 1e-6
    max_weight_error = max(max_weight_error, err)
    cases += 1
ids, weights = route_model(np.zeros((8, 256), np.float32), np.zeros(256, np.float32))
assert np.array_equal(ids, np.tile(np.arange(8), (8, 1)))
assert np.array_equal(weights, np.full((8, 8), .125, np.float32))

# Exhaustive index coverage for the fixed model: all router rows and K values
# are consumed once per token; no padding or approximate key/route selection.
coverage = np.zeros((256, 8192), dtype=np.uint8)
for nb in range(16):
    for warp in range(8):
        for ks in range(8):
            k = ks*1024 + np.arange(1024)
            for row in [nb*16+warp*2, nb*16+warp*2+1]: coverage[row, k] += 1
assert np.all(coverage == 1)
for start in [0, 128, 2**31, 2**32-128]:
    v = start
    for _ in range(20):
        last_flags = []
        for _ in rng.permutation(128):
            last_flags.append((v & 127) == 127)
            v = (v+1) & 0xffffffff
        assert sum(last_flags) == 1 and last_flags[-1]

draft = (root / '.tools/decoder-fp32-router-variant.py').read_text(encoding='utf-8')
tree = ast.parse(draft)
table = symtable.symtable(draft, 'draft', 'exec')
known = set(table.get_identifiers()) | set(dir(builtins))
count = 0
def check_bindings(tab):
    global count
    for name in tab.get_identifiers():
        s = tab.lookup(name)
        if s.is_global() and s.is_referenced():
            assert name in known, (tab.get_name(), name)
            count += 1
    for child in tab.get_children(): check_bindings(child)
for child in table.get_children(): check_bindings(child)
cuda = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
            and any(isinstance(a, ast.Name) and a.id == '_RF_CUDA' for a in n.targets))
assert '__fmaf_rn' in cuda and 'mma.' not in cuda and 'bfloat16' not in cuda
assert 'volatile float* parts' in cuda and '__threadfence();' in cuda
result = {'source_sha256': hashlib.sha256(draft.encode()).hexdigest(), 'seed': 20261028,
          'manufactured_cases': cases+1, 'max_logit_error_vs_fp64': max_error,
          'max_route_weight_error_vs_fp64': max_weight_error,
          'coverage_row_k_pairs': int(coverage.size), 'counter_wrap_verified': True,
          'global_bindings_checked': count, 'gpu_status': 'unmeasured',
          'memory_contract': 'fenced volatile partials, last-block reduction; sequential calls'}
(root / '.tools/decoder-fp32-router-cpu-audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
