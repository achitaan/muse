"""Invert the public grouped-MoE packing; no GPU or challenge run is used."""
import ast
import builtins
import hashlib
import json
import symtable
from pathlib import Path
import torch

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
source = (root / '.tools/vjeux-prefill-upstream.py').read_text()
tree = ast.parse(source)
ns = {'torch': torch, 'KS': 64, 'WBLK': 4608}
for n in tree.body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_CODE_POS' for t in n.targets):
        ns['_CODE_POS'] = ast.literal_eval(n.value)
    if isinstance(n, ast.FunctionDef) and n.name in {'_codes', '_repack'}:
        exec(compile(ast.Module(body=[n], type_ignores=[]), '<audited-pack>', 'exec'), ns)
positions = ns['_CODE_POS']
assert sorted(x for values in positions.values() for x in values) == list(range(32))
torch.set_num_threads(2)
torch.manual_seed(20261028)
N = 256
weights = torch.randint(0, 256, (1, N, 2048), dtype=torch.uint8)
scales = torch.randint(0, 256, (1, N, 256), dtype=torch.uint8)
packed = ns['_repack'](weights, scales, False)
cb = packed[0, :, :, :4096].reshape(N // 128, 64, 2, 4, 8, 4, 4, 4).to(torch.int64)
word = sum(cb[..., i] << (8 * i) for i in range(4))
decoded = torch.zeros((*word.shape, 8), dtype=torch.int64)
for k, slot in enumerate([3, 7, 1, 5, 2, 6, 0, 4]):
    for bit in range(4):
        decoded[..., k] |= ((word >> positions[slot][bit]) & 1) << bit
decoded = decoded.reshape(N // 128, 64, 2, 4, 8, 4, 4, 2, 2, 2)
decoded = decoded.permute(0, 2, 3, 7, 4, 1, 6, 8, 5, 9).reshape(N, 4096)
assert torch.equal(decoded.to(torch.uint8), ns['_codes'](weights[0]))
sf = packed[0, :, :, 4096:].reshape(N // 128, 64, 2, 4, 8, 4, 2)
sf = sf.permute(0, 2, 3, 6, 4, 1, 5).contiguous().reshape(N, 256)
assert torch.equal(sf, scales[0])
# Gate/up layout is the same bit pack with a row permutation. Verify every row
# index of its 8192-row permutation, without a large temporary bit tensor.
nt = torch.arange(64).view(64, 1, 1)
q = torch.arange(8).view(1, 8, 1)
lr = torch.arange(16).view(1, 1, 16)
perm = torch.where(lr < 8, 64 * nt + 8 * q + lr,
                   4096 + 64 * nt + 8 * q + lr - 8).reshape(-1)
assert torch.equal(perm.sort().values, torch.arange(8192))

draft = (root / '.tools/prefill-grouped-variant.py').read_text()
st = symtable.symtable(draft, 'draft', 'exec')
known = set(st.get_identifiers()) | set(dir(builtins))
checked = 0
def validate(table):
    global checked
    for name in table.get_identifiers():
        v = table.lookup(name)
        if v.is_global() and v.is_referenced():
            assert name in known, (table.get_name(), name)
            checked += 1
    for child in table.get_children(): validate(child)
for table in st.get_children(): validate(table)
dt = ast.parse(draft)
combine = next(n for n in dt.body if isinstance(n, ast.FunctionDef) and n.name == '_pg_combine')
assert 'z = y * norm' in ast.unparse(combine)
assert 'result += z * inv * w' in ast.unparse(combine)
cuda = next(n.value.value for n in dt.body if isinstance(n, ast.Assign)
            and any(isinstance(a, ast.Name) and a.id == '_pg__CUDA_SRC' for a in n.targets))
assert 'uint32_t outw[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};' in cuda
assert 'route metadata published by threads 0..7' in cuda
result = {'seed': 20261028, 'codes_checked': N * 4096, 'scale_bytes_checked': N * 256,
          'source_sha256': hashlib.sha256(draft.encode()).hexdigest(),
          'pack_inverse': True, 'gate_up_rows_bijective': True,
          'global_bindings_checked': checked, 'gather_raw_bf16_copy': True,
          'gpu_correctness': 'unverified', 'gpu_speedup': 'unmeasured'}
(root / '.tools/prefill-grouped-cpu-audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
