"""Audit lossless SK2 streams, correction tables and small-token dispatch on CPU."""
import argparse
import ast
import builtins
import hashlib
import json
import symtable
import types
from pathlib import Path

import torch

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='.tools/decoder-small-projections-variant.py')
parser.add_argument('--output', default='.tools/decoder-small-projections-cpu-audit.json')
args = parser.parse_args()
raw = (root / args.source).read_bytes()
draft = raw.decode('utf-8')
tree = ast.parse(draft)
constants = {n.targets[0].id: n.value.value for n in tree.body
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
configs = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == '_SP_CONFIGS' for t in n.targets)))
shared = '_SP_DISPATCH' not in constants
tokr = constants.get('_SP_TOKR', 8)
output_only = tokr == 16
symbol = 'sk2output' if output_only else 'sk2small'
helpers = {'_fpr_bfp_sk2_tileunits', '_fpr_bfp_sk2_compress', '_fpr_bfp_sk2_stream',
           '_fpr_bfp_sk2_table', '_sp_pack', '_sp_module', '_fpr_get_ext'}
ns = {'torch': torch, 'hashlib': hashlib, '_SP_TOKR': tokr}
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name in helpers:
        exec(compile(ast.Module(body=[n], type_ignores=[]), '<audited-helper>', 'exec'), ns)

torch.set_num_threads(2)
generator = torch.Generator().manual_seed(20261028)
cases = []
for config in configs:
    ci, layer, n, k, rows, nt, nwr, wsk, upw, nacc, stages = config
    assert n % rows == 0 and rows == nwr * nt * 8 and k % (wsk * upw * 64) == 0
    small_n = rows * 4
    w = (torch.randn((small_n, k), generator=generator) * .01).bfloat16()
    w[0, 0], w[1, 511], w[rows - 1, k - 1] = 1e-8, 0., -1e-8
    comp = ns['_fpr_bfp_sk2_compress'](w)
    assert comp is not None
    packed = comp['packed']
    low = packed[..., :512].reshape(small_n // 8, k // 64, 32, 16).int()
    cb = packed[..., 512:].reshape(small_n // 8, k // 64, 32, 8)
    codes = torch.cat([cb[..., :4] & 15, cb[..., :4] >> 4,
                       cb[..., 4:] & 15, cb[..., 4:] >> 4], -1).int()
    bits = ((((codes >> 3) << 7) | ((comp['base4'] & 255) + (codes & 7))) << 8) | low
    tu = bits.to(torch.int16).view(torch.bfloat16)
    recovered = tu.reshape(small_n // 8, k // 64, 8, 4, 4, 2, 2).permute(0, 2, 1, 4, 5, 3, 6).reshape(small_n, k).float()
    recovered[comp['rows'], comp['cols']] += comp['delta']
    assert torch.equal(recovered.bfloat16().view(torch.int16), w.view(torch.int16))

    stream = ns['_fpr_bfp_sk2_stream'](packed, small_n, rows, nt, nwr, wsk, upw)
    seen = torch.zeros(packed.shape[:2], dtype=torch.int32)
    for st in range(k // (wsk * upw * 64)):
        for cta in range(small_n // rows):
            for warp in range(nwr * wsk):
                for uu in range(upw):
                    for tile_col in range(nt):
                        tile = cta * (rows // 8) + (warp // wsk) * nt + tile_col
                        unit = st * wsk * upw + (warp % wsk) * upw + uu
                        assert torch.equal(stream[st, cta, warp, uu, tile_col], packed[tile, unit])
                        seen[tile, unit] += 1
    assert bool((seen == 1).all())
    table, pmax = ns['_fpr_bfp_sk2_table'](comp, small_n, rows)
    actual = []
    for cta in range(small_n // rows):
        for entry in table[cta, 1:1 + int(table[cta, 0, 0])]:
            key, delta = map(int, entry)
            actual.append((cta * rows + (key >> 16), key & 65535, delta))
    expected = list(zip(comp['rows'].tolist(), comp['cols'].tolist(), comp['delta'].view(torch.int32).tolist()))
    assert sorted(actual) == sorted(expected) and len(expected) >= 3
    reduced_cfg = (ci, layer, small_n, k, rows, nt, nwr, wsk, upw, nacc, stages)
    plan = ns['_sp_pack'](w, reduced_cfg)
    assert plan is not None and plan['smem'] < 232448
    assert torch.equal(plan['packed'], stream) and torch.equal(plan['table'], table)
    # Each epilogue writes exactly M*N values, with no dependence on padded rows.
    # PK is enabled for TOKR=8; the TMA descriptor uses M rows and masks ldmatrix.
    for m in range(1, tokr + 1):
        local = [(idx // (rows // 4), (idx % (rows // 4)) * 4 + j)
                 for idx in range(m * (rows // 4)) for j in range(4)]
        assert len(set(local)) == m * rows
        assert set(local) == {(token, row) for token in range(m) for row in range(rows)}
    for invalid in ('negative_zero', 'nan', 'inf', 'excess_escapes'):
        bad = w.clone()
        if invalid == 'negative_zero': bad.view(torch.int16)[0, 0] = -32768
        elif invalid == 'nan': bad[0, 0] = float('nan')
        elif invalid == 'inf': bad[0, 0] = float('inf')
        else: bad.flatten()[::16] = 1e-8
        assert ns['_sp_pack'](bad, reduced_cfg) is None, invalid
    assert ns['_sp_pack'](w.float(), reduced_cfg) is None
    assert ns['_sp_pack'](w[:, :-1], reduced_cfg) is None
    # A pathological per-CTA correction table must fall back before launch.
    original_table = ns['_fpr_bfp_sk2_table']
    ns['_fpr_bfp_sk2_table'] = lambda *args: (table, 30000)
    assert ns['_sp_pack'](w, reduced_cfg) is None
    ns['_fpr_bfp_sk2_table'] = original_table
    cases.append({'layer': layer, 'values_recovered': w.numel(), 'corrections': len(actual),
                  'stages': stages, 'sample_pmax': pmax, 'sample_dynamic_shared_bytes': plan['smem']})

cuda = (constants['_fpr_CUDA'] if shared else
        constants['_fpr_CUDA'].split('\nvoid sk2f(', 1)[0] + constants['_SP_DISPATCH'])
assert 'if (TOKR == 8 || M < TOKR) go(std::true_type{})' in cuda
assert '(PK && trow[mt] >= M) ? zrow' in cuda
assert 'PK ? (uint32_t)M : (uint32_t)TOKR' in cuda
assert 'for (int idx = tid; idx < R4 * M; idx += NC)' in cuda
if output_only:
    assert cuda.split('\nvoid sk2output(', 1)[1].count('launch_sk2<1, 16,') == 1
else:
    assert cuda.count('launch_sk2<1, 8,') == 3
assert '@@' not in cuda
cpp = constants['_fpr_CPP'] if shared else constants['_SP_CPP']
assert cpp.startswith('#include <torch/extension.h>\n')
calls = []
extension = types.SimpleNamespace(sk2_set_pdl=lambda value: calls.append(('pdl', value)))
def loader(**kwargs):
    assert kwargs['cuda_sources'] == cuda and kwargs['cpp_sources'] == cpp
    assert kwargs['functions'] == (['sk2f', 'sk2_set_pdl', symbol] if shared else [symbol])
    calls.append(1)
    return extension
ns.update(_sp_ext=None, _SP_CUDA=cuda, _SP_CPP=cpp, _fpr_load_inline=loader,
          _fpr_ext=None, _fpr_CPP=cpp, _fpr_CUDA=cuda,
          torch=types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: (9, 0))))
assert ns['_sp_module']() is extension and ns['_sp_module']() is extension
assert calls == ([1, ('pdl', 0)] if shared else [1])

# Exercise only the new dispatch wrapper with fake tensor metadata.
wrapper = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_achita_dense_project'][-1]
sentinel_old, sentinel_new = object(), object()
dispatch_ns = {'torch': torch, '_sp_incumbent_project': lambda *args: sentinel_old,
               '_sp_run': lambda *args: sentinel_new}
exec(compile(ast.Module(body=[wrapper], type_ignores=[]), '<dispatch>', 'exec'), dispatch_ns)
for ci, layer, n, k, *_ in configs:
    plan = {'n': n, 'k': k, 'packed': types.SimpleNamespace(device='cuda:0')}
    state = {'small_projection_choices': ({(layer, 16): plan} if output_only else {(layer, 1): plan, (layer, 8): plan}),
             'dense': {layer: types.SimpleNamespace(shape=(8192 if layer == 'unpooler' else n, k))}}
    for m in (1, 2, 3, 4, 7, 8, *range(9, 18), 32, 128):
        x = types.SimpleNamespace(shape=(m, k), dtype=torch.bfloat16, device='cuda:0',
                                  is_contiguous=lambda: True, data_ptr=lambda: 256)
        rows_arg = n if layer == 'unpooler' else None
        got = dispatch_ns['_achita_dense_project'](x, layer, state, rows_arg)
        eligible = 8 < m <= 16 if output_only else m <= 8
        assert got is (sentinel_new if eligible else sentinel_old)
        x.dtype = torch.float32
        assert dispatch_ns['_achita_dense_project'](x, layer, state, rows_arg) is sentinel_old
    if layer == 'unpooler':
        x.dtype = torch.bfloat16
        x.shape = (1, k)
        assert dispatch_ns['_achita_dense_project'](x, layer, state) is sentinel_old

symbols = symtable.symtable(draft, 'candidate', 'exec')
known = set(symbols.get_identifiers()) | set(dir(builtins))
def audit_bindings(table):
    for name in table.get_identifiers():
        symbol = table.lookup(name)
        if symbol.is_global() and symbol.is_referenced(): assert name in known, (table.get_name(), name)
    for child in table.get_children(): audit_bindings(child)
audit_bindings(symbols)
base = (root / 'best/decoder_verified.py').read_bytes()
assert hashlib.sha256(base).hexdigest() == '09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67'
if shared:
    # CUDA extension integration is independently audited in the companion script.
    assert constants['_fpr_CUDA'].startswith(next(n.value.value for n in ast.parse(base).body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_fpr_CUDA' for t in n.targets)))
else:
    assert raw.startswith(base)
assert len(raw) < 1_000_000
result = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'seed': 20261028, 'cases': cases,
          'lossless_cpu_recovery': True, 'streams_bijective': True, 'correction_tables_exact': True,
          'fallbacks_and_token_store_geometry': True, 'factory_reuse': True, 'global_bindings': True,
          'baseline_bytes_unchanged': not shared, 'gpu_status': 'unmeasured'}
(root / args.output).write_bytes((json.dumps(result, indent=2) + '\n').encode('utf-8'))
print(json.dumps(result))
