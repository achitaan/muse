"""Independent factor representation, rounding and ownership checks; CPU only."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import re
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
raw = (ROOT / 'best/moe_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
tree = ast.parse(raw)
original = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
candidate = (ROOT / '.tools/moe-factor-y.cu').read_text()
ledger = json.loads((ROOT / '.tools/moe-factor-y-cuda-edits.json').read_bytes())
restored = candidate
for edit in reversed(ledger):
    a, new = edit['offset'], edit['new']
    assert restored[a:a + len(new)] == new
    restored = restored[:a] + edit['old'] + restored[a + len(new):]
assert restored == original
old_final = original.split('#if OKC_WGS && OKC_WGSFINSS\n')[1].split('#elif OKC_FINCOL')[0]
new_final = candidate.split('#if OKC_WGS && OKC_WGSFINSS\n')[1].split('#elif OKC_FINCOL')[0]
assert old_final.count('CTA_SYNC();') == new_final.count('CTA_SYNC();') == 3
for expression in [
    'a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w',
    'ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);',
    'tsum += red[w * 8 + tid];',
    'p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f)',
    'acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;',
    'acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;',
    '__floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1])',
]:
    assert expression in old_final and expression in new_final
assert 'const __nv_bfloat16* row' in new_final.split('if (qv[sl] >= 0) {', 1)[1]
assert new_final.count('factor_product(saved[sl], gain, a, b);') == 2
assert candidate.count('__fmul_rn(__uint_as_float(') == 8
assert 'factor_marlin_bits2' in candidate
start = original.index('__device__ __forceinline__ float2 achita_marlin_out2(')
old_helper = original[start:original.index('\n}', start) + 2]
expected_helper = old_helper.replace('float2 achita_marlin_out2', '__nv_bfloat162 factor_marlin_bits2')
for old, new in [('encoded_acc0', 'a'), ('encoded_acc1', 'b'), ('encoded_scale', 'scale')]:
    expected_helper = expected_helper.replace(old, new)
expected_helper = expected_helper.replace('return __bfloat1622float2(__hmul2_rn(acc, __bfloat162bfloat162(gs)));',
                                        'return __hmul2_rn(acc, __bfloat162bfloat162(gs));')
start = candidate.index('__device__ __forceinline__ __nv_bfloat162 factor_marlin_bits2(')
new_helper = candidate[start:candidate.index('\n}', start) + 2]
assert expected_helper == new_helper, 'The actual CUDA helper must retain both BF16 boundaries'
unpack = re.findall(r'__fmul_rn\(__uint_as_float\(p\.([xyzw]) (<< 16|& 0xffff0000u)\), gain\[(\d)\]\)', candidate)
assert unpack == [(field, operation, str(i)) for i, (field, operation) in enumerate(
    (field, op) for field in 'xyzw' for op in ('<< 16', '& 0xffff0000u'))]
gain_unpack = re.findall(r'__uint_as_float\(gb\.([xyzw]) (<< 16|& 0xffff0000u)\)', new_final)
assert gain_unpack == [(field, op) for field in 'xyzw' for op in ('<< 16', '& 0xffff0000u')]
range_body = candidate.split('__device__ __forceinline__ int wgs_stage_at(', 1)[1].split('\n}', 1)[0]
assert range_body.split('{', 1)[1].lstrip().startswith('if (nj == 0) return 0;')
assert range_body.index('if (nj == 0)') < range_body.index('S.jC[lo]')


def promote(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def pack_decode(bits):
    pairs = bits.astype(np.uint32).reshape(*bits.shape[:-1], bits.shape[-1] // 2, 2)
    packed = pairs[..., 0] | (pairs[..., 1] << 16)
    low, high = (packed << 16).view(np.float32), (packed & 0xffff0000).view(np.float32)
    return np.stack([low, high], axis=-1).reshape(bits.shape)


all_bits = np.arange(65536, dtype=np.uint16)
assert np.array_equal(pack_decode(all_bits).view(np.uint32), promote(all_bits).view(np.uint32))
finite = all_bits[(all_bits & 0x7f80) != 0x7f80]
gains = np.array([0, 0x8000, 1, 0x8001, 0x80, 0x8080, 0x2000, 0xa000,
                  0x3c80, 0xbc80, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x3fc0,
                  0xbfc0, 0x4000, 0xc000, 0x4380, 0xc380, 0x6000, 0xe000,
                  0x7f7f, 0xff7f], dtype=np.uint16)
b, gain = promote(finite)[:, None], promote(gains)[None, :]
with np.errstate(over='ignore', under='ignore', invalid='ignore'):
    # Every finite BF16 product is exactly representable in FP64; conversion
    # to FP32 supplies an independent RN oracle, including subnormal/overflow.
    oracle = (b.astype(np.float64) * gain.astype(np.float64)).astype(np.float32)
    reconstructed = np.multiply(b, gain, dtype=np.float32)
assert np.array_equal(oracle.view(np.uint32), reconstructed.view(np.uint32))

# Exhaust the full 64-token phase-B tile ownership, independent of row order.
positions = Counter()
for ub in range(16):
    for wg in range(4):
        for wp in range(4):
            r16 = ub * 16 + wg * 4 + wp
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for nt in range(8):
                    for h in range(2):
                        token = nt * 8 + 2 * t + h
                        for col in (r16 * 16 + g, r16 * 16 + g + 8):
                            positions[token, col] += 1
assert len(positions) == 64 * 4096 and set(positions.values()) == {1}
assert all(0 <= t < 64 and 0 <= col < 4096 for t, col in positions)
assert {tid * 8 + c for tid in range(512) for c in range(8)} == set(range(4096))
emax, hidden = 384 * 8, 4096
compact_last_byte = 2 * emax * hidden - 1
fixup_first_byte = 4 * emax * hidden
assert compact_last_byte < fixup_first_byte

rng = np.random.default_rng(20261029)
fixtures, mutant_round_caught, mutant_swap_caught = [], False, False
for m in (1, 17, 32, 128, 256, 385):
    # These are generated representation fixtures, not official routing or GPU inputs.
    values = rng.normal(size=(m, 8, 4096)).astype(np.float32)
    bits32 = values.view(np.uint32)
    bits = ((bits32 + 0x7fff + ((bits32 >> 16) & 1)) >> 16).astype(np.uint16)
    gain32 = (1 + .3 * rng.normal(size=4096)).astype(np.float32)
    gain_bits32 = gain32.view(np.uint32)
    gain_bits = ((gain_bits32 + 0x7fff + ((gain_bits32 >> 16) & 1)) >> 16).astype(np.uint16)
    active = rng.random((m, 8)) < .2
    active[0] = False  # force an all-nonlocal token in every fixture
    if m > 1:
        active[-1] = True
    bits[~active] = 0x7fc1  # invalid slots deliberately contain BF16 NaNs
    old = np.zeros((m, 8, 4096), dtype=np.float32)
    new = np.zeros_like(old)
    old[active] = np.multiply(promote(bits[active]), promote(gain_bits), dtype=np.float32)
    new[active] = np.multiply(pack_decode(bits[active]), promote(gain_bits), dtype=np.float32)
    assert np.array_equal(old.view(np.uint32), new.view(np.uint32)) and np.isfinite(new).all()
    if active.any():
        rounded = ((old[active].view(np.uint32) + 0x7fff + ((old[active].view(np.uint32) >> 16) & 1)) >> 16).astype(np.uint16)
        mutant_round_caught |= not np.array_equal(promote(rounded).view(np.uint32), old[active].view(np.uint32))
        swapped = bits[active].reshape(-1, 2048, 2)[..., ::-1].copy().reshape(-1, 4096)
        mutant_swap_caught |= not np.array_equal(promote(swapped).view(np.uint32), promote(bits[active]).view(np.uint32))
    norm64 = old.astype(np.float64)
    norm64 /= np.sqrt(np.mean(norm64 * norm64, axis=-1, keepdims=True) + 1e-8)
    weights = rng.random((m, 8)).astype(np.float32)
    output64 = np.sum(norm64 * weights[..., None], axis=1)
    assert np.isfinite(output64).all() and np.array_equal(output64[0], np.zeros(4096))
    fixtures.append(dict(m=m, active_routes=int(active.sum()), invalid_nan_slots_masked=True,
                         factor_products_bitwise_exact=True, all_nonlocal_output_exact_zero=True))
assert mutant_round_caught and mutant_swap_caught
report = dict(parent_source_sha256=hashlib.sha256(raw).hexdigest(),
              factor_cuda_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
              original_cuda_inverse_exact=True, norm_route_expression_and_barriers_unchanged=True,
              empty_job_range_returns_before_job_read_or_divide=True,
              all_bf16_pack_promote_bit_patterns=65536,
              finite_product_comparisons_to_fp64_RN_oracle=int(oracle.size),
              signed_zero_subnormal_overflow_covered=True,
              phase_B_tile_unique_stores=len(positions), finalizer_unique_columns=4096,
              compact_last_byte=compact_last_byte, unchanged_fixup_first_byte=fixup_first_byte,
              fixtures=fixtures, rejected_mutants=['extra_bf16_product_round', 'packed_half_swap'],
              numpy=np.__version__, python=platform.python_version(),
              cpu_only=True, gpu_verified=False,
              limitations='Representation and source/ownership proofs; no CUDA compilation, register allocation, graph replay or GPU timing proof')
(ROOT / '.tools/moe-factor-y-cpu-audit.json').write_bytes((json.dumps(report, indent=2) + '\n').encode())
print(json.dumps(report, indent=2))
