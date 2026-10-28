"""Independent bit-level model of Lee Penkman's eight-operation FP4 placement.

Public reference: lee101/openkernels c565e373d34575c5f7df862b05f8be7619078b0f,
muse_moe_w4a16/submission.py, Apache-2.0. No candidate is changed here.
"""
import json
from pathlib import Path
import numpy as np

mask = np.uint32(0x81c081c0)


def rol(q, n):
    return ((q << np.uint32(n)) | (q >> np.uint32(32 - n))).astype(np.uint32)


def dq8(q):
    return [q & mask, rol(q, 10) & mask, rol(q, 13) & mask,
            ((q & np.uint32(0x001b001b)) * np.uint32(0x1084)) & mask]


def dq7(q):
    return [q & mask, (q * np.uint32(16)) & mask, rol(q, 10) & mask,
            ((q * np.uint32(64)) & np.uint32(0x80c080c0)) |
            (rol(q, 14) & np.uint32(0x01000100))]


oldpos, newpos = {}, {}
for r in range(4):
    for up in range(2):
        for b, ob in enumerate([6, 7, 8, 15]):
            output_bit = ob + 16 * up
            oldpos[r, up, b] = (output_bit + (0, 28, 22, 18 if b == 2 else 26)[r]) % 32
            newpos[r, up, b] = ((output_bit - (0, 10, 13)[r]) % 32 if r < 3
                                else [4, 0, 1, 3][b] + 16 * up)
assert sorted(newpos.values()) == list(range(32))

# The multiply only sees these eight bits. Exhaust every possible input to it,
# proving that overlapping shifts/carries cannot alter any selected output bit.
positions = [0, 1, 3, 4, 16, 17, 19, 20]
q = np.zeros(256, dtype=np.uint32)
for i, p in enumerate(positions): q |= ((np.arange(256, dtype=np.uint32) >> i) & 1) << p
expected = np.zeros_like(q)
for up in range(2):
    for ib, ob in zip([4, 0, 1, 3], [6, 7, 8, 15]):
        expected |= ((q >> (ib + 16 * up)) & 1) << (ob + 16 * up)
assert np.array_equal(dq8(q)[3], expected)

rng = np.random.default_rng(20261028)
codes = rng.integers(0, 16, (131072, 4, 2), dtype=np.uint32)
packed = []
for pos in [oldpos, newpos]:
    w = np.zeros(len(codes), dtype=np.uint32)
    for (r, up, b), p in pos.items(): w |= ((codes[:, r, up] >> b) & 1) << p
    packed.append(w)
for r, (a, b) in enumerate(zip(dq7(packed[0]), dq8(packed[1]))):
    assert np.array_equal(a, b)
    expected = np.zeros(len(codes), dtype=np.uint32)
    for up in range(2):
        for bit, ob in enumerate([6, 7, 8, 15]):
            expected |= ((codes[:, r, up] >> bit) & 1) << (ob + 16 * up)
    assert np.array_equal(b, expected)
record = {'seed': 20261028, 'random_full_words': len(codes),
          'weight_codes_checked': int(codes.size), 'all_word_bit_positions_bijective': True,
          'all_multiply_operand_patterns_checked': 256,
          'old_and_new_bf16_fragment_bits_identical': True,
          'candidate_modified': False, 'gpu_status': 'unmeasured',
          'performance_claim': 'none; source operation count is not measured machine code'}
(next(p for p in Path(__file__).resolve().parents if (p / '.git').exists()) / '.tools/moe-dq8-layout-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
