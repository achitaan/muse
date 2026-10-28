"""Check lossless packing by inversion, independently of GPU performance."""
import ast
import json
from pathlib import Path
import torch

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
tree = ast.parse((root / '.tools/vjeux-moe-upstream.py').read_text())
names = {'_stored_pos', '_split_nibbles', '_scatter_code_bits', '_pack_codes',
         '_pack_scales', '_fat_order', '_unit_order', '_pack_codes_nat', '_wgs_copy'}
ns = {'torch': torch, 'DQ7': True, '_REG_HJ': [(0, 0), (1, 0), (0, 2), (1, 2)]}
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in names:
        exec(compile(ast.Module(body=[n], type_ignores=[]), '<audited-packing>', 'exec'), ns)
ns['_POS'] = ns['_stored_pos']()
torch.manual_seed(20261028)
torch.set_num_threads(2)
R, K = 256, 4096
codes = torch.randint(0, 256, (R, K // 2), dtype=torch.uint8)
scales = torch.randint(0, 256, (R, K // 16), dtype=torch.uint8)
rows = torch.randperm(R)
nib, _ = ns['_split_nibbles'](codes, rows)
packed = ns['_pack_codes'](codes, rows)
w = (packed.to(torch.int64) & 0xffffffff).reshape(R // 16, 8, 8, 8, 4, 4)
rebuilt = torch.zeros((*w.shape, 2, 4), dtype=torch.int64)
for reg, (h, j0) in enumerate(ns['_REG_HJ']):
    for up in range(2):
        for bit in range(4):
            rebuilt[..., h, j0 + up] |= ((w >> ns['_POS'][(reg, up, bit)]) & 1) << bit
rebuilt = rebuilt.permute(0, 6, 3, 1, 2, 4, 5, 7).reshape(R, K)
assert torch.equal(rebuilt.to(torch.uint8), nib)
ps = ns['_pack_scales'](scales, rows)
rs = ps.reshape(R // 16, 8, 8, 4, 8, 2).permute(0, 5, 2, 1, 4, 3).reshape(R, K // 16)
assert torch.equal(rs, scales[rows])
unit = ns['_unit_order'](packed, ps).reshape(R // 16, 36864)
assert torch.equal(unit[:, :32768].reshape(-1), packed.view(torch.uint8))
assert torch.equal(unit[:, 32768:].reshape(-1), ps)
fat = ns['_fat_order'](packed, ps).reshape(R // 256, 16, 36864)
fc = fat[:, :, :32768].contiguous().view(torch.int32)
fc = fc.reshape(R // 256, 8, 2, 16, 4, 32, 4).permute(0, 3, 1, 2, 4, 5, 6).contiguous().reshape(-1)
fs = fat[:, :, 32768:].reshape(R // 256, 8, 2, 16, 32, 4, 2).permute(0, 3, 1, 4, 2, 5, 6).contiguous().reshape(-1)
assert torch.equal(fc, packed) and torch.equal(fs, ps)

# Natural-K pack and WGS copies must preserve all codes/scales, including sign
# and unusual scale bytes; this checks layout identities, not finite-only data.
pn = ns['_pack_codes_nat'](codes, rows)
wn = (pn.to(torch.int64) & 0xffffffff).reshape(R // 16, 8, 8, 8, 4, 4)
rn = torch.zeros((*wn.shape, 2, 2, 2), dtype=torch.int64)
for reg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
    for up in range(2):
        for bit in range(4):
            rn[..., h, q, up] |= ((wn >> ns['_POS'][(reg, up, bit)]) & 1) << bit
rn = rn.permute(0, 6, 3, 1, 2, 5, 7, 4, 8).reshape(R, K)
assert torch.equal(rn.to(torch.uint8), nib)
wgs = ns['_wgs_copy'](codes, scales, rows).reshape(R // 256, 16, 4, 9216)
wc = wgs[:, :, :, :8192].contiguous().view(torch.int32).reshape(R // 256, 16, 4, 4, 4, 32, 4)
wc = wc.permute(0, 2, 4, 1, 3, 6, 5).contiguous().reshape(R // 16, 256, 32)
wc = wc.reshape(R // 16, 8, 8, 4, 8, 4).permute(0, 1, 2, 4, 5, 3).contiguous().reshape(-1)
assert torch.equal(wc, pn)
ws = wgs[:, :, :, 8192:].reshape(R // 256, 16, 4, 4, 4, 8, 4, 2)
ws = ws.permute(0, 2, 4, 7, 5, 1, 3, 6).contiguous().reshape(R, K // 16)
assert torch.equal(ws, scales[rows])

# Precision formula is derived from pinned SGLang marlin_template.h's write:
# BF16(c0), BF16(global_scale), BF16 multiply. Keep the CUDA scaling encoding
# separate so no rounding happens while encoded by 2^-62.
a = torch.randn(8192) * 32
g = torch.rand(8192) * .2 + .01
expected = (a.bfloat16().float() * g.bfloat16().float()).bfloat16().float()
encoded_a, encoded_g = a * 2.0**-62, g * 2.0**62
got = ((encoded_a * 2.0**62).bfloat16().float() *
       (encoded_g * 2.0**-62).bfloat16().float()).bfloat16().float()
assert torch.equal(expected, got)
result = {'seed': 20261028, 'codes_checked': R * K,
          'scales_checked': R * (K // 16),
          'mma_sync_pack_inverse': True, 'natural_pack_inverse': True,
          'fat_unit_wgs_layout_inverse': True, 'bf16_boundary_formula': True,
          'gpu_correctness': 'not established', 'gpu_speedup': 'not measured'}
(root / '.tools/native-moe-cpu-audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
