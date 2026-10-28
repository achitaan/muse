"""Independent FP64 rounding audit for pairing the BF16 epilogue products."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
bits = np.arange(65536, dtype=np.uint16)
bits = bits[(bits & 0x7f80) != 0x7f80]  # all finite BF16 inputs, including signed zeros
acc = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
scale_bits = np.array([0, 0x8000, 1, 0x8001, 0x80, 0x8080, 0x2000, 0xa000,
                       0x3c80, 0xbc80, 0x3f00, 0xbf00, 0x3f80, 0xbf80,
                       0x3fc0, 0xbfc0, 0x4000, 0xc000, 0x4380, 0xc380,
                       0x6000, 0xe000, 0x7f7f, 0xff7f], dtype=np.uint16)
scales = torch.from_numpy(scale_bits.view(np.int16).copy()).view(torch.bfloat16)


def exact_bf16_round(value):
    """Analytical RN-even: normal ulp 2**(E-7), subnormal ulp 2**-133."""
    a = np.abs(value)
    _, exponent = np.frexp(a)
    unit_exp = np.maximum(exponent - 1 - 7, -133)
    rounded = np.ldexp(np.rint(np.ldexp(a, -unit_exp)), unit_exp)
    rounded = np.copysign(rounded, value)
    # Rounding up past the largest finite BF16 produces infinity.
    largest = float(torch.tensor([0x7f7f], dtype=torch.int16).view(torch.bfloat16).float()[0])
    rounded = np.where(np.abs(rounded) > largest, np.copysign(np.inf, rounded), rounded)
    return rounded.astype(np.float32)


with np.errstate(over='ignore', invalid='ignore'):
    exact_products = acc.float().numpy().astype(np.float64)[:, None] * scales.float().numpy().astype(np.float64)[None, :]
    expected = exact_bf16_round(exact_products)
    scalar = (acc.float()[:, None] * scales.float()[None, :]).to(torch.bfloat16).float().numpy()
    vector_model = (acc[:, None] * scales[None, :]).float().numpy()
assert np.array_equal(scalar.view(np.uint32), expected.view(np.uint32))
assert np.array_equal(vector_model.view(np.uint32), expected.view(np.uint32))
source = (root / '.tools/moe-paired-epilogue-variant.py').read_bytes()
record = {'source_sha256': hashlib.sha256(source).hexdigest(),
          'all_finite_bf16_accumulators': len(bits), 'bf16_scales': len(scale_bits),
          'products_bitwise_equal_to_independent_fp64_rn_even': int(expected.size),
          'signed_zeros_subnormals_and_overflow_included': True,
          'limitations': 'CPU algebra/rounding model; no CUDA compilation, GPU correctness or speed evidence',
          'gpu_status': 'unmeasured'}
(root / '.tools/moe-paired-epilogue-cpu-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
