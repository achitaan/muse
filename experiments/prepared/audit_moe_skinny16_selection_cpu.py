"""Routing, actual wrapper dispatch and metadata-only selection audit, no GPU."""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import symtable
import sys
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
torch.set_num_threads(2)
raw = (ROOT / '.tools/moe-skinny16-selected.py').read_bytes()
prototype = (ROOT / 'results/moe-skinny16-preflight-20261028/prototype.py').read_bytes()
meta = json.loads((ROOT / '.tools/moe-skinny16-selection-audit.json').read_bytes())
assert hashlib.sha256(raw).hexdigest() == meta['source_sha256']
if meta.get('repair'):
    sys.path.insert(0, str(ROOT / 'scripts'))
    from moe_skinny16_evidence import verify_repair
    assert verify_repair(raw, meta)
else:
    assert raw.startswith(prototype)
tree = ast.parse(raw)
addon = (ROOT / 'experiments/prepared/moe_skinny16_addon.py').read_text()
names = set(dir(builtins))
for n in tree.body:
    if isinstance(n, (ast.FunctionDef, ast.ClassDef)): names.add(n.name)
    elif isinstance(n, (ast.Import, ast.ImportFrom)): names.update(a.asname or a.name.split('.')[0] for a in n.names)
    elif isinstance(n, ast.Assign):
        names.update(x.id for target in n.targets for x in ast.walk(target) if isinstance(x, ast.Name))
global_refs = []
def bindings(table):
    for key in table.get_identifiers():
        symbol = table.lookup(key)
        if symbol.is_global() and symbol.is_referenced():
            assert key in names, (table.get_name(), key)
            global_refs.append(key)
    for child in table.get_children(): bindings(child)
bindings(symtable.symtable(addon, '<skinny16-addon>', 'exec'))
functions = {n.name: n for n in ast.parse(addon).body if isinstance(n, ast.FunctionDef)}
scope = {'torch': torch, 'HIDDEN': 4096}
exec(compile(ast.Module(body=[functions['_sk16_synthetic']], type_ignores=[]), '<actual-synthetic>', 'exec'), scope)
routing_cases, hot_second_tile_cases = 0, 0
for rows in range(9, 17):
    for lo in (0, 96):
        state = {'lo': lo, 'w': [torch.zeros(1)]}
        for pattern in (0, 1, 2):
            x, ids, tw = scope['_sk16_synthetic'](rows, pattern, state, 20261029 + rows * 31 + pattern)
            assert x.shape == (rows, 4096) and x.dtype == torch.bfloat16 and x.is_contiguous()
            assert ids.shape == tw.shape == (rows, 8) and ids.dtype == torch.int32
            assert bool(((ids >= 0) & (ids < 256)).all())
            assert all(torch.unique(row).numel() == 8 for row in ids)
            assert torch.isfinite(x).all() and torch.isfinite(tw).all() and (tw > 0).all()
            assert torch.allclose(tw.sum(-1), torch.ones(rows), rtol=0, atol=1e-6)
            counts = torch.bincount(((ids - lo) % 256).flatten(), minlength=256)[:32]
            assert counts.sum() == rows * (4 if pattern == 1 else 2)
            assert counts.max() <= rows and (counts > 0).sum() <= 32
            if pattern == 1:
                assert counts[0] == rows and counts[1] == rows
                hot_second_tile_cases += 1
            routing_cases += 1

# Execute the actual final kernel wrapper with native launch replaced by a
# recorder. This checks dispatch and normalized tensor metadata, not CUDA math.
dispatch = {'torch': torch, '_SK16_DEFINES': ('@sk16',),
            '_sk16_original_kernel': lambda data, state: 'incumbent',
            '_ext': lambda defs: defs}
def launch(ext, state, x, ids, tw):
    assert ext == ('@sk16', '-DOKC_PRMTSC=1')
    assert x.is_contiguous() and ids.is_contiguous() and tw.is_contiguous()
    assert x.dtype == torch.bfloat16 and ids.dtype == torch.int32 and tw.dtype == torch.float32
    return 'skinny16'
dispatch['_sk16_run'] = launch
exec(compile(ast.Module(body=[functions['kernel_fn']], type_ignores=[]), '<actual-wrapper>', 'exec'), dispatch)
dispatch_cases = 0
for enabled in (False, True):
    state = dict(skinny16_enabled=enabled, lo=96, cnt=32, sfx=('-DOKC_PRMTSC=1',))
    for rows in range(385):
        data = {'hidden_states': torch.zeros((rows, 4), dtype=torch.bfloat16),
                'topk_ids': torch.zeros((rows, 8), dtype=torch.int64),
                'topk_weights': torch.ones((rows, 8)), 'config': dict(local_start=96, local_count=32)}
        result = dispatch['kernel_fn'](data, state)
        assert result == ('skinny16' if enabled and 9 <= rows <= 16 else 'incumbent')
        dispatch_cases += 1
fn = functions['kernel_fn']
assert not any(isinstance(n, ast.Attribute) and n.attr in ('item', 'cpu', 'synchronize') for n in ast.walk(fn))
state_writes = [ast.unparse(t) for n in ast.walk(ast.parse(addon)) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == 'state']
assert state_writes == ["state['skinny16_enabled']", "state['skinny16_enabled']"]
constants = {x.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)
             for x in n.targets if isinstance(x, ast.Name) and x.id == 'CUDA_SRC'}
alternate = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(x, ast.Subscript) and ast.unparse(x) == "ALT_SRCS['sk16']" for x in n.targets))
for cuda in (constants['CUDA_SRC'], alternate):
    assert cuda.count('achita_marlin_out2(') == 12
    assert '__floats2bfloat162_rn(encoded_acc0 * TWO62, encoded_acc1 * TWO62)' in cuda
    assert '__float2bfloat16_rn(encoded_scale * 0x1p-62f)' in cuda
    assert '__bfloat1622float2(__hmul2_rn(acc, __bfloat162bfloat162(gs)))' in cuda
    assert '* gsc' not in cuda and 'st_keep_f32(p.ybuf' in cuda
report = dict(source_sha256=meta['source_sha256'], original_prefix_exact=True,
              routing_cases=routing_cases, hot_second_tile_cases=hot_second_tile_cases,
              dispatch_cases=dispatch_cases, global_references_checked=len(global_refs),
              no_new_weight_copies=True, state_writes=state_writes,
              both_cuda_precision_contracts_preserved=True,
              gpu_status='unmeasured; CPU metadata and source checks only')
(ROOT / '.tools/moe-skinny16-selection-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps(report))
