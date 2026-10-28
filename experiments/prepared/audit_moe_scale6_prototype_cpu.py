"""Finite CPU proof of candidate scale-byte packing; no kernel is changed."""
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
gen = torch.Generator().manual_seed(20261028)

def scale6_eligible(*banks):
    if not banks:
        return False
    for bank in banks:
        if bank.dtype != torch.float8_e4m3fn or bank.numel() == 0:
            return False
        raw = bank.contiguous().view(torch.uint8)
        if bool(((raw < 64) | (raw > 126)).any()):
            return False
    return True

# Analytical domain: restore the constant 0x40 bit; reject all bytes outside
# [0x40,0x7e], including the 0x7f NaN and every signed or subnormal byte.
for code in range(256):
    bank = torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert scale6_eligible(bank) == (64 <= code <= 126)
valid = torch.arange(64, 127, dtype=torch.uint8).view(torch.float8_e4m3fn)
assert scale6_eligible(valid, valid.clone())
assert not scale6_eligible(valid, torch.tensor([63], dtype=torch.uint8).view(torch.float8_e4m3fn))
assert not scale6_eligible(valid, torch.tensor([127], dtype=torch.uint8).view(torch.float8_e4m3fn))
assert not scale6_eligible(valid.view(torch.uint8))
assert not scale6_eligible(torch.empty(0, dtype=torch.float8_e4m3fn))
assert not scale6_eligible()
# Exhaust all adjacent six-bit field pairs in every position, then random packets.
pairs = torch.cartesian_prod(torch.arange(64), torch.arange(64))
groups = []
for j in range(8):
    g = torch.zeros((4096, 8), dtype=torch.int64)
    g[:, j] = pairs[:, 0]
    g[:, (j + 1) % 8] = pairs[:, 1]
    groups.append(g)
groups.append(torch.randint(0, 64, (131072, 8), generator=gen))
fields = torch.cat(groups)
packets = sum(fields[:, j] << (6 * j) for j in range(8))
packed = torch.stack([(packets >> (8 * j)) & 255 for j in range(6)], -1).to(torch.uint8).flatten()
u8 = packed.reshape(-1, 4).long()
words = sum(u8[:, j] << (8 * j) for j in range(4))
i = torch.arange(len(fields))
offset = (i * 3) // 2
shift = (i & 1) * 16
lo, hi = words[offset], words[offset + 1]
x = ((lo >> shift) | (hi << (32 - shift))) & 0xffffffff
y = hi >> shift
recovered = []
for js in range(4):
    pair = (x, x >> 12, (x >> 24) | (y << 8), y >> 4)[js]
    sep = (pair & 63) | ((pair & 4032) << 10)
    proposed = (sep * 16 + 0x60006000) & 0xffffffff
    original = ((fields[:, 2 * js] + 64) | ((fields[:, 2 * js + 1] + 64) << 16)) * 16 + 0x5c005c00
    # Algebraic reference: restoring the original byte's 0x40 bit adds 0x400
    # to each BF16 half, exactly the difference between the two constants.
    assert torch.equal(proposed, original & 0xffffffff)
    recovered.extend([pair & 63, (pair >> 6) & 63])
assert torch.equal(torch.stack(recovered, -1), fields)
# A WGS group has 128 packets (1024 scale bytes). The last two-word read ends
# exactly at byte 768; all ring, warpgroup and TMA boundaries retain alignment.
index = torch.arange(128)
assert int((((3 * index) // 2 + 1) * 4 + 3).max()) == 767
piece, unit = 8192 + 768, 4 * (8192 + 768)
assert piece % 128 == unit % 1024 == (3 * unit) % 1024 == 0
result = {'seed': 20261028, 'scale_values_roundtripped': fields.numel(),
          'eligibility_byte_values_exhausted': 256, 'mixed_bank_and_dtype_fallbacks': True,
          'adjacent_field_pairs_exhausted': 8 * 64 * 64,
          'scale_pair_bits_match_original_prmt_imad': True,
          'first_and_last_packet_two_word_loads_in_bounds': True,
          'warpgroup_piece_bytes': piece, 'stage_bytes': unit,
          'old_stage_bytes': 36864, 'streamed_weight_byte_reduction': 1 - unit / 36864,
          'eligibility_required': 'every local scale byte in 0x40..0x7e; all others use old format',
          'gpu_status': 'no implementation or GPU measurement yet'}
root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
(root / '.tools/moe-s6-prototype-audit.json').write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps(result))
