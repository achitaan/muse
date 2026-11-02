"""achitaan: experimental fixed-position BF16 codec, NOT an OKC submission.

Independent implementation of the ZipServ-style idea described in
HPMLL/ZipServ_ASPLOS26 (Apache-2.0), arxiv.org/html/2603.17435.
No source from that repository is embedded here.

Intended integration: FMT=6 in chenwainuo's existing gv kernel in bf16_v2.py.
Same MMA fragment order, register ring, X loads, reduction and launch geometry;
only the weight loads/decode change. Never replace the default dispatch.

An input unit is [4 K16 steps, 32 lanes, 8 BF16 values] (1024 weights).
Each lane owns one 32-weight group. It has three uint32 bitplanes and 32 bytes
of sign+mantissa. Code zero escapes; 1..7 index a matrix-wide 8-byte table.
There is ONE uint32 escape base per 1024-weight unit, not per 32-weight group:
lane-group bases are a warp prefix sum of popcount(~(p0|p1|p2)). An individual
escape is base + popcount(mask & ((1<<position)-1)). Non-escape positions
are fixed, so there is no variable-length serial decode or shared-memory LUT.

Exact storage: 11 + 16*escape_fraction + 32/1024 + (64+32)/weight_count bpw
(the last 32 bits are the offset sentinel). Required <=11.16, so even with
infinite matrices this needs coverage >=99.1953125%, stronger than 95%.

This CPU packer/reference decoder is the initial implementation while the
real-weight statistics CHECK runs. GPU integration must remain gated on that
check, then real-weight bitwise roundtrip and both ABBA passes <=0.988.
"""

import numpy as np


def pack_fragments(bits: np.ndarray) -> dict:
    """Pack uint16 [T, KC, 4, 32, 8] bits; reject any non-fragment shape.

    Integer input intentionally preserves every BF16 encoding, including both
    zeros, subnormals, infinities and NaN payloads. No floating conversion.
    """
    bits = np.asarray(bits)
    if bits.dtype != np.uint16 or bits.ndim != 5 or bits.shape[2:] != (4, 32, 8):
        raise ValueError("ZS expects uint16 [T, KC, 4, 32, 8] fragments")
    if bits.size == 0:
        raise ValueError("ZS cannot pack an empty weight matrix")
    units = bits.shape[0] * bits.shape[1]
    if bits.size > np.iinfo(np.uint32).max:
        raise ValueError("ZS uint32 escape offsets require <=2**32-1 weights")
    h = np.zeros(256, dtype=np.int64)
    raw = bits.reshape(units, 4, 32, 8)
    for start in range(0, units, 2048):
        e = (raw[start:start + 2048] >> 7) & 255
        h += np.bincount(e.reshape(-1), minlength=256)
    table = np.zeros(8, dtype=np.uint8)
    table[1:] = np.argsort(h, kind="stable")[-7:]
    code_for_exp = np.zeros(256, dtype=np.uint8)
    code_for_exp[table[1:]] = np.arange(1, 8, dtype=np.uint8)
    plane = np.empty((units, 3, 32), dtype=np.uint32)
    sm = np.empty((units, 2, 32, 16), dtype=np.uint8)
    offsets = np.empty(units + 1, dtype=np.uint32)
    offsets[0] = 0
    escapes = []
    shifts = np.arange(32, dtype=np.uint32)
    for start in range(0, units, 2048):
        b = raw[start:start + 2048]
        n = len(b)
        # Lane-major ordering is the escape-order contract consumed by the
        # warp scan, followed by a popcount prefix within each lane group.
        groups = b.transpose(0, 2, 1, 3).reshape(n, 32, 32)
        codes = code_for_exp[(groups >> 7) & 255]
        for p in range(3):
            plane[start:start + n, p] = np.bitwise_or.reduce(
                (((codes >> p) & 1).astype(np.uint32) << shifts), axis=-1)
        mantissa_sign = ((b & 127) | ((b >> 8) & 128)).astype(np.uint8)
        sm[start:start + n] = mantissa_sign.reshape(n, 2, 2, 32, 8).transpose(0, 1, 3, 2, 4).reshape(n, 2, 32, 16)
        mask = codes == 0
        counts = mask.sum(axis=(1, 2), dtype=np.uint64)
        offsets[start + 1:start + n + 1] = offsets[start] + np.cumsum(counts, dtype=np.uint64)
        escapes.append(groups[mask])
    escaped = np.concatenate(escapes)
    result = dict(shape=bits.shape, table=table, sign_mantissa=sm,
                  planes=plane, offsets=offsets, escapes=escaped)
    # Source of expected values: the original integer BF16 bit patterns, NOT
    # output produced by a floating-point conversion or by another codec.
    if not np.array_equal(unpack_fragments(result), bits):
        raise AssertionError("ZS BF16 bit-exact roundtrip failed")
    result["bpw"] = sum(result[k].nbytes * 8 for k in
                        ("table", "sign_mantissa", "planes", "offsets", "escapes")) / bits.size
    result["top7"] = 1.0 - escaped.size / bits.size
    return result


def unpack_fragments(pack: dict) -> np.ndarray:
    """Independent fixed-position decoder with rank-addressed full BF16 escapes."""
    shape = tuple(pack["shape"])
    units = shape[0] * shape[1]
    out = np.empty((units, 4, 32, 8), dtype=np.uint16)
    for start in range(0, units, 2048):
        p = pack["planes"][start:start + 2048]
        n = len(p)
        c = np.zeros((n, 32, 32), dtype=np.uint8)
        # Deliberately separate from the encoder's plane-reduction operation.
        for position in range(32):
            c[:, :, position] = (((p[:, 0] >> position) & 1)
                                 | (((p[:, 1] >> position) & 1) << 1)
                                 | (((p[:, 2] >> position) & 1) << 2))
        ms = pack["sign_mantissa"][start:start + n].reshape(n, 2, 32, 2, 8)
        ms = ms.transpose(0, 1, 3, 2, 4).reshape(n, 4, 32, 8)
        ms = ms.transpose(0, 2, 1, 3).reshape(n, 32, 32).astype(np.uint16)
        decoded = (ms & 127) | ((ms & 128) << 8) | (pack["table"][c].astype(np.uint16) << 7)
        mask = c == 0
        # Prefix within a unit includes preceding lanes then preceding values;
        # this is exactly the mathematical rank from warp-scan + popcount.
        rank = np.cumsum(mask.reshape(n, 1024), axis=1, dtype=np.uint32) - 1
        addresses = pack["offsets"][start:start + n, None] + rank
        decoded[mask] = pack["escapes"][addresses.reshape(n, 32, 32)[mask]]
        out[start:start + n] = decoded.reshape(n, 32, 4, 8).transpose(0, 2, 1, 3)
    return out.reshape(shape)


def self_test() -> None:
    # Reference: all 65536 BF16 encodings, not a floating-point approximation.
    all_bits = np.arange(65536, dtype=np.uint16).reshape(8, 8, 4, 32, 8)
    pk = pack_fragments(all_bits)
    assert np.array_equal(unpack_fragments(pk), all_bits)
    assert pk["escapes"].size == 65536 - 7 * 256
    # Reference: a single representable exponent requires no fallback at all.
    constant = np.full((1, 2, 4, 32, 8), 0x3f80, dtype=np.uint16)
    pk = pack_fragments(constant)
    assert pk["escapes"].size == 0
    assert pk["bpw"] == 11 + 32 / 1024 + 96 / constant.size
    # Check escapes in first/last bit, adjacent positions, lane/unit boundaries,
    # negative zero, smallest subnormal, infinity and NaN payload preservation.
    mixed = constant.copy()
    mixed.reshape(-1)[[0, 1, 7, 8, 31, 32, 255, 256, 1023, 1024, 2047]] = np.array(
        [0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0x7fff, 0xffff, 0x1234, 0x9876], dtype=np.uint16)
    assert np.array_equal(unpack_fragments(pack_fragments(mixed)), mixed)
    # Deterministic random bit patterns exercise arbitrary signs and payloads.
    rng = np.random.default_rng(20261102)
    for shape in ((1, 1, 4, 32, 8), (2, 3, 4, 32, 8), (3, 5, 4, 32, 8)):
        values = rng.integers(0, 65536, size=shape, dtype=np.uint16)
        assert np.array_equal(unpack_fragments(pack_fragments(values)), values)
    # Reference: original integers across the 2048-unit packing batch boundary;
    # the second batch must continue, rather than restart, its escape offsets.
    boundary = rng.integers(0, 65536, size=(1, 2049, 4, 32, 8), dtype=np.uint16)
    assert np.array_equal(unpack_fragments(pack_fragments(boundary)), boundary)
    print("ZS codec CPU self-tests passed: exhaustive BF16 bits, zero escapes, boundaries, seeded random")


if __name__ == "__main__":
    self_test()
