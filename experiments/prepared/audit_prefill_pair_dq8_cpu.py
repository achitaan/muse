"""Independent grouped packing inverse and BF16 rounding audit; no GPU run.

Expected FP4 output bits follow E2M1's sign/magnitude placement in BF16.
The product oracle uses exact FP64 products and analytical RN-even BF16 ulps,
following the retained Marlin accumulator/global-scale/product boundaries.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
torch.set_num_threads(2)
PARENT = '93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7'
raw = (ROOT / '.tools/prefill-pair-dq8.py').read_bytes()
parent = (ROOT / '.prefill-measurements' / PARENT / 'submission.py').read_bytes()
assert hashlib.sha256(parent).hexdigest() == PARENT
tree, old_tree = ast.parse(raw), ast.parse(parent)


def constants(t):
    return {x.id: ast.literal_eval(n.value) for n in t.body if isinstance(n, ast.Assign)
            for x in n.targets if isinstance(x, ast.Name) and x.id in ('_pg__CODE_POS', '_pg__CUDA_SRC')}


new, old = constants(tree), constants(old_tree)
audit = json.loads((ROOT / '.tools/prefill-pair-dq8-source-audit.json').read_bytes())
assert audit['source_sha256'] == hashlib.sha256(raw).hexdigest()
restored = new['_pg__CUDA_SRC']
for edit in reversed(audit['cuda_edits']):
    assert restored.count(edit['after']) == 1
    restored = restored.replace(edit['after'], edit['before'])
assert restored == old['_pg__CUDA_SRC']
for t in (tree, old_tree):
    for n in t.body:
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id in new for x in n.targets):
            n.value = ast.Constant('changed constant')
assert ast.dump(tree, include_attributes=False) == ast.dump(old_tree, include_attributes=False)


def scope(pos):
    ns = {'torch': torch, '_pg_KS': 64, '_pg_WBLK': 4608, '_pg__CODE_POS': pos}
    for n in old_tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in ('_pg__codes', '_pg__repack'):
            exec(compile(ast.Module(body=[n], type_ignores=[]), '<actual-grouped-pack>', 'exec'), ns)
    return ns


torch.manual_seed(20261028)
N = 256
codes = torch.randint(0, 256, (1, N, 2048), dtype=torch.uint8)
scales = torch.randint(0, 256, (1, N, 256), dtype=torch.uint8)
ns, os = scope(new['_pg__CODE_POS']), scope(old['_pg__CODE_POS'])
packed, previous = ns['_pg__repack'](codes, scales, False), os['_pg__repack'](codes, scales, False)
assert torch.equal(packed[..., 4096:], previous[..., 4096:])
cb = packed[0, :, :, :4096].reshape(N // 128, 64, 2, 4, 8, 4, 4, 4).to(torch.int64)
word = sum(cb[..., i] << (8 * i) for i in range(4))
decoded = torch.zeros((*word.shape, 8), dtype=torch.int64)
for k, slot in enumerate([3, 7, 1, 5, 2, 6, 0, 4]):
    for bit in range(4):
        decoded[..., k] |= ((word >> new['_pg__CODE_POS'][slot][bit]) & 1) << bit
decoded = decoded.reshape(N // 128, 64, 2, 4, 8, 4, 4, 2, 2, 2)
decoded = decoded.permute(0, 2, 3, 7, 4, 1, 6, 8, 5, 9).reshape(N, 4096)
assert torch.equal(decoded.to(torch.uint8), ns['_pg__codes'](codes[0]))
sf = packed[0, :, :, 4096:].reshape(N // 128, 64, 2, 4, 8, 4, 2)
sf = sf.permute(0, 2, 3, 6, 4, 1, 5).reshape(N, 256)
assert torch.equal(sf, scales[0])

# The actual integer CUDA expressions, applied to every packed word.
def rol(w, n):
    return ((w << n) | (w >> (32 - n))) & 0xffffffff


mask = 0x81c081c0
w = word.numpy().astype(np.uint64)
fragments = (w & mask, rol(w, 10) & mask, rol(w, 13) & mask,
             ((w & 0x001b001b) * 0x1084) & mask)
oldcb = previous[0, :, :, :4096].reshape(*word.shape, 4).to(torch.int64)
oldw = sum(oldcb[..., i] << (8 * i) for i in range(4)).numpy().astype(np.uint64)
old_fragments = (oldw & mask, (oldw * 16) & mask, rol(oldw, 10) & mask,
                 ((oldw * 64) & 0x80c080c0) | (rol(oldw, 14) & 0x01000100))
assert all(np.array_equal(a, b) for a, b in zip(fragments, old_fragments))
for reg, slots in enumerate(((3, 7), (2, 6), (1, 5), (0, 4))):
    expected = np.zeros_like(w)
    for upper, slot in enumerate(slots):
        for bit, ob in enumerate((6, 7, 8, 15)):
            expected |= ((w >> new['_pg__CODE_POS'][slot][bit]) & 1) << (ob + 16 * upper)
    assert np.array_equal(fragments[reg], expected)
# Exhaust every eight-bit masked multiply operand to cover carry interference.
q = np.zeros(256, dtype=np.uint64)
expected = q.copy()
for i, pos in enumerate((0, 1, 3, 4, 16, 17, 19, 20)):
    q |= ((np.arange(256, dtype=np.uint64) >> i) & 1) << pos
for upper in (0, 1):
    for ib, ob in zip((4, 0, 1, 3), (6, 7, 8, 15)):
        expected |= ((q >> (ib + 16 * upper)) & 1) << (ob + 16 * upper)
assert np.array_equal(((q * 0x1084) & mask), expected)

# Existing analytic oracle is independent of the new CUDA epilogue.
oracle_tree = ast.parse((ROOT / 'experiments/prepared/audit_moe_paired_epilogue.py').read_bytes())
oracle = next(n for n in oracle_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'exact_bf16_round')
oracle_ns = {'np': np, 'torch': torch}
exec(compile(ast.Module(body=[oracle], type_ignores=[]), '<analytical-bf16-oracle>', 'exec'), oracle_ns)
bits = np.arange(65536, dtype=np.uint16)
bits = bits[(bits & 0x7f80) != 0x7f80]
acc = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
scale_bits = np.array([0, 0x8000, 1, 0x8001, 0x80, 0x8080, 0x2000, 0xa000,
                       0x3c80, 0xbc80, 0x3f00, 0xbf00, 0x3f80, 0xbf80,
                       0x3fc0, 0xbfc0, 0x4000, 0xc000, 0x4380, 0xc380,
                       0x6000, 0xe000, 0x7f7f, 0xff7f], dtype=np.uint16)
gs = torch.from_numpy(scale_bits.view(np.int16).copy()).view(torch.bfloat16)
with np.errstate(over='ignore', invalid='ignore'):
    expected = oracle_ns['exact_bf16_round'](acc.float().numpy().astype(np.float64)[:, None] * gs.float().numpy().astype(np.float64)[None, :])
    got = (acc[:, None] * gs[None, :]).float().numpy()
    scalar = (acc.float()[:, None] * gs.float()[None, :]).bfloat16().float().numpy()
assert np.array_equal(expected.view(np.uint32), got.view(np.uint32))
assert np.array_equal(expected.view(np.uint32), scalar.view(np.uint32))
# Prefill uses a different reversible power-of-two encoding from standalone MoE.
a = torch.randn(131072) * 1024
g = torch.rand(131072) * .2
reference = (a.bfloat16().float() * g.bfloat16().float()).bfloat16()
encoded_a, encoded_g = a * 2.0**-7, g * 2.0**7
paired = ((encoded_a * 128).bfloat16() * (encoded_g * 2.0**-7).bfloat16())
assert torch.equal(reference.view(torch.int16), paired.view(torch.int16))
report = dict(source_sha256=hashlib.sha256(raw).hexdigest(), seed=20261028,
              inverse_recovers_parent_cuda_and_other_ast=True, codes_inverted=N * 4096,
              scale_bytes_inverted=N * 256, original_fragment_bits_identical=True,
              exhaustive_masked_multiply_inputs=256, bf16_products_fp64_oracle=int(expected.size),
              signed_zeros_subnormals_overflow=True, power_of_two_encoding_products=len(a),
              scheduling_routing_norms_attention_dense_unchanged=True,
              gpu_status='unmeasured; CPU evidence does not prove CUDA behavior or speed')
(ROOT / '.tools/prefill-pair-dq8-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
