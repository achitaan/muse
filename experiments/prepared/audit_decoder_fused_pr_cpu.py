"""Check exact compression/layouts and integration bindings, without GPU runs."""
import ast
import builtins
import hashlib
import json
import symtable
import types
from pathlib import Path
import torch

root = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
draft = (root / '.tools/decoder-fused-pr-variant.py').read_text()
tree = ast.parse(draft)
helper_names = {'_fpr_bfp_sk2_tileunits', '_fpr_bfp_sk2_compress', '_fpr_bfp_sk2_stream', '_fpr_bfp_sk2_table'}
ns = {'torch': torch}
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in helper_names:
        exec(compile(ast.Module(body=[n], type_ignores=[]), '<audited-codec>', 'exec'), ns)
torch.manual_seed(20261028)
torch.set_num_threads(2)
N, K = 64, 8192
w = (torch.randn(N, K) * .01).bfloat16()
w[0, 0] = 1e-8
w[1, 511] = 0.0
w[15, 791] = -1e-8
comp = ns['_fpr_bfp_sk2_compress'](w)
assert comp is not None
p = comp['packed']
low = p[..., :512].reshape(N // 8, K // 64, 32, 16).to(torch.int32)
codebytes = p[..., 512:].reshape(N // 8, K // 64, 32, 8)
codes = torch.cat([codebytes[..., :4] & 15, codebytes[..., :4] >> 4,
                   codebytes[..., 4:] & 15, codebytes[..., 4:] >> 4], -1).to(torch.int32)
base = comp['base4'] & 255
bits = (((codes >> 3) << 7) | (base + (codes & 7))) << 8 | low
tu = bits.to(torch.int16).view(torch.bfloat16)
def undo_tileunits(x, rows):
    return x.reshape(rows // 8, K // 64, 8, 4, 4, 2, 2).permute(0, 2, 1, 4, 5, 3, 6).contiguous().reshape(rows, K)
decoded = undo_tileunits(tu, N).float()
decoded[comp['rows'], comp['cols']] += comp['delta']
assert torch.equal(decoded.bfloat16().view(torch.int16), w.view(torch.int16))
assert comp['delta'].numel() >= 3

# The stage stream must cover every tile/unit once, without changing bytes.
stream = ns['_fpr_bfp_sk2_stream'](p, N, 32, 4, 1, 8, 1)
rebuilt = torch.empty_like(p)
seen = torch.zeros(p.shape[:2], dtype=torch.int32)
for st in range(K // 512):
    for cta in range(N // 32):
        for warp in range(8):
            for nt in range(4):
                tile, unit = cta * 4 + nt, st * 8 + warp
                rebuilt[tile, unit] = stream[st, cta, warp, 0, nt]
                seen[tile, unit] += 1
assert torch.equal(rebuilt, p) and bool((seen == 1).all())
tab, pmax = ns['_fpr_bfp_sk2_table'](comp, N, 32)
actual = []
for cta in range(N // 32):
    for entry in tab[cta, 1:1 + int(tab[cta, 0, 0])]:
        key, delta = map(int, entry)
        actual.append((cta * 32 + (key >> 16), key & 65535, delta))
expected = list(zip(comp['rows'].tolist(), comp['cols'].tolist(), comp['delta'].view(torch.int32).tolist()))
assert sorted(actual) == sorted(expected)

# Router hi and lo stay BF16 in the native stream. Verify both copies bytewise.
router = torch.randn(256, K, dtype=torch.float32) * .01
hi = router.bfloat16()
lo = (router - hi.float()).bfloat16()
rt = torch.cat([ns['_fpr_bfp_sk2_tileunits'](a).contiguous().view(torch.uint8).reshape(32, 128, 1024)
                for a in [hi, lo]], -1)
rs = ns['_fpr_bfp_sk2_stream'](rt, 256, 8, 1, 1, 8, 1)
rr = torch.empty_like(rt)
for st in range(16):
    for warp in range(8): rr[:, st * 8 + warp] = rs[st, :, warp, 0, 0]
assert torch.equal(rr, rt)
for a, off in [(hi, 0), (lo, 1024)]:
    got = undo_tileunits(rr[..., off:off + 1024].contiguous().view(torch.bfloat16), 256)
    assert torch.equal(got.view(torch.int16), a.view(torch.int16))

baseline = ast.parse((root / 'best/decoder_verified.py').read_text())
old_fn = next(n for n in baseline.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
new_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
branch = next(n for n in new_fn.body if isinstance(n, ast.If) and ast.unparse(n.test) == "8 < tokens <= 32 and state['fpr'] is not None")
restored_body = []
for n in new_fn.body: restored_body.extend(branch.orelse if n is branch else [n])
assert ast.dump(ast.Module(body=restored_body, type_ignores=[])) == ast.dump(ast.Module(body=old_fn.body, type_ignores=[]))

st = symtable.symtable(draft, 'draft', 'exec')
known = set(st.get_identifiers()) | set(dir(builtins))
global_count = 0
def bindings(table):
    global global_count
    for name in table.get_identifiers():
        sym = table.lookup(name)
        if sym.is_global() and sym.is_referenced():
            assert name in known, (table.get_name(), name)
            global_count += 1
    for child in table.get_children(): bindings(child)
for child in st.get_children(): bindings(child)
const = {n.targets[0].id: n.value.value for n in tree.body if isinstance(n, ast.Assign)
         and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
assert const['_fpr_CPP'].startswith('#include <torch/extension.h>' + chr(10))
assert const['_fpr_CUDA'].startswith('#include <type_traits>' + chr(10))
assert '@@' not in const['_fpr_CUDA']
factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_fpr_get_ext')
calls = []
ext = types.SimpleNamespace(sk2_set_pdl=lambda n: calls.append(('pdl', n)))
def fake_loader(**kw):
    assert kw['cpp_sources'] == const['_fpr_CPP'] and kw['cuda_sources'] == const['_fpr_CUDA']
    assert kw['functions'] == ['sk2f', 'sk2_set_pdl']
    calls.append(('build', 1))
    return ext
cache_ns = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))),
            '_fpr_ext': None, '_fpr_load_inline': fake_loader,
            '_fpr_CPP': const['_fpr_CPP'], '_fpr_CUDA': const['_fpr_CUDA']}
exec(compile(ast.Module(body=[factory], type_ignores=[]), '<factory-regression>', 'exec'), cache_ns)
assert cache_ns['_fpr_get_ext']() is ext and cache_ns['_fpr_get_ext']() is ext
assert calls == [('build', 1), ('pdl', 0)]
result = {'source_sha256': hashlib.sha256(draft.encode()).hexdigest(), 'seed': 20261028,
          'bf16_pooler_values_recovered': N * K, 'escapes_checked': len(expected),
          'router_hi_lo_values_recovered': 2 * 256 * K, 'streams_bijective': True,
          'entire_original_else_path_unchanged': True, 'global_bindings_checked': global_count,
          'factory_reuse_and_cpp_headers': True, 'gpu_status': 'unmeasured'}
(root / '.tools/decoder-fused-pr-cpu-audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
