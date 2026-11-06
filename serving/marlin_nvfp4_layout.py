"""Coordinate oracle for stock SGLang's Hopper NVFP4 Marlin layout.

Pinned SGLang: a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f.
Derived from gptq_marlin_repack.cuh (4-bit, empty permutation),
marlin_permute_scales and nvfp4_marlin_process_scales.
These helpers describe one expert. They do not allocate a weight bank.
"""

import math
import struct

SGLANG_COMMIT = "a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f"


def weight_address(k: int, n: int, size_n: int) -> tuple[int, int]:
    """Return the flat int32 word index and bit shift of logical W[n, k]."""
    if size_n <= 0 or size_n % 64 or not 0 <= n < size_n or k < 0:
        raise ValueError("expected a nonnegative coordinate and N divisible by 64")
    k_tile, k_in_tile = divmod(k, 16)
    n_tile, n_in_tile = divmod(n, 64)
    warp, n_in_warp = divmod(n_in_tile, 16)
    n_half, column = divmod(n_in_warp, 8)
    k_half, k_in_half = divmod(k_in_tile, 8)
    row, k_odd = divmod(k_in_half, 2)
    lane = column * 4 + row
    word = (k_tile * (size_n // 64) + n_tile) * 128 + lane * 4 + warp
    shift = 4 * (k_odd * 4 + n_half * 2 + k_half)
    return word, shift


def scale_address(k_group: int, n: int, size_n: int) -> int:
    """Return the flat byte index of the scale for W[n, 16*k_group:...]."""
    if size_n <= 0 or size_n % 64 or not 0 <= n < size_n or k_group < 0:
        raise ValueError("expected a nonnegative coordinate and N divisible by 64")
    n_tile, n_in_tile = divmod(n, 64)
    row, col = divmod(n_in_tile, 8)
    transposed = col * 8 + row
    # The extra [0, 2, 1, 3] permutation is its own inverse.
    swizzled = (transposed & ~3) | ((transposed & 1) << 1) | ((transposed & 2) >> 1)
    return k_group * size_n + n_tile * 64 + swizzled


def encode_scale_byte(raw: int) -> int:
    """Return the Marlin storage byte for a finite nonnegative E4M3FN scale."""
    if not 0 <= raw <= 126:
        raise ValueError("expected finite nonnegative E4M3FN scale bits (0x00..0x7e)")
    exponent, mantissa = divmod(raw, 8)
    value = math.ldexp(mantissa if exponent == 0 else 8 + mantissa,
                       -9 if exponent == 0 else exponent - 10)
    half_bits = int.from_bytes(struct.pack("<e", value * 128), "little")
    return (half_bits >> 7) & 0xFF
