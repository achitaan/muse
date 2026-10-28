"""Independent bounds, K coverage, output ownership and FP64 cluster-GEMM audit.

These models do not establish CUDA race freedom, compilation or performance.
GPU prepare checks and full9test/30benchmark validation remain mandatory.
"""
import ast
import builtins
import hashlib
import json
from pathlib import Path
import symtable
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
torch.set_num_threads(2)
torch.manual_seed(20261028)
raw = (ROOT / '.tools/dense-cluster.py').read_bytes()
meta = json.loads((ROOT / '.tools/dense-cluster-source-audit.json').read_bytes())
assert hashlib.sha256(raw).hexdigest() == meta['source_sha256']
parent = (ROOT / '.scored-measurements' / meta['parent_sha256'] / 'submission.py').read_bytes()
assert raw[:len(parent)] == parent and hashlib.sha256(parent).hexdigest() == meta['parent_sha256']
tree = ast.parse(raw)
constants = {x.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             for x in n.targets if isinstance(x, ast.Name) and x.id in ('_LG_CUDA', '_LG_CPP', '_LG_CONFIGS')}
cuda = constants['_LG_CUDA']
assert cuda.count('launch_gq<128, 1, 8, 2>') == 1
assert cuda.count('launch_gqx<128, 1, 8, 3>') == 1
assert 'wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16' in cuda
assert 'const int xrows = min(M, BN)' in cuda
assert cuda.count('A_BYTES + xrows * 128') == 4
assert 'if (tt < M)' in cuda and 'if (tok < M)' in cuda
assert 'const int u0 = (crank * U) / CL' in cuda
assert 'const int u0 = ((int)crank * U) / CL' in cuda
# Full source pieces are pinned by independently reading the original constants.
upstream = ast.parse((ROOT / 'results/dense-pinned-codec-screen-20261028/upstream_dense.py').read_bytes())
uv = {}
for n in upstream.body:
    if isinstance(n, ast.Assign):
        for x in n.targets:
            if isinstance(x, ast.Name):
                try:
                    uv[x.id] = ast.literal_eval(n.value)
                except (ValueError, TypeError):
                    pass
for key in ('MBAR_HELPERS', 'WG_HELPERS', 'MAP_SRC'):
    assert uv[key] in cuda
for name in ('GQ_SRC', 'GQX_SRC'):
    prefix, suffix = uv[name].split('@@' + name[:-4] + '_DISPATCH@@')
    assert prefix in cuda and suffix in cuda

# The original WG sweep selected nothing in the exact published parent.
old = json.loads((ROOT / '.scored-measurements' / meta['parent_sha256'] / 'result.json').read_bytes())['result']
lines = [l for l in old.get('log_tail', '').splitlines() if l.startswith('WG_KEEP')]
assert len(lines) == 3 and 'WG_SELECT' not in old.get('log_tail', '')
base_tree = ast.parse(parent)
alias_pos = next(i for i, n in enumerate(base_tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(x, ast.Name) and x.id == '_achita_wg_base_prepare' for x in n.targets))
assert isinstance(base_tree.body[alias_pos].value, ast.Name) and base_tree.body[alias_pos].value.id == 'prepare'
prior_prepare = [n for n in base_tree.body[:alias_pos] if isinstance(n, ast.FunctionDef) and n.name == 'prepare'][-1]
assert "state['v18'] = prepare_v18_extra(weights)" in ast.unparse(prior_prepare)

addon = (ROOT / 'experiments/prepared/dense_cluster_addon.py').read_text()
symbol_table = symtable.symtable(addon, '<cluster-addon>', 'exec')
all_names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
for n in tree.body:
    if isinstance(n, (ast.Import, ast.ImportFrom)):
        all_names.update(a.asname or a.name.split('.')[0] for a in n.names)
    elif isinstance(n, ast.Assign):
        all_names.update(x.id for target in n.targets for x in ast.walk(target) if isinstance(x, ast.Name))
all_names |= set(dir(builtins))
checked_globals = []
def check_names(st):
    for name in st.get_identifiers():
        s = st.lookup(name)
        if s.is_global() and s.is_referenced():
            assert name in all_names, (st.get_name(), name)
            checked_globals.append(name)
    for child in st.get_children():
        check_names(child)
check_names(symbol_table)

key_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_lg_key')
key_scope = {'torch': torch, '_LG_CONFIGS': constants['_LG_CONFIGS']}
exec(compile(ast.Module(body=[key_fn], type_ignores=[]), '<actual-metadata-dispatch>', 'exec'), key_scope)
class Fake:
    def __init__(self, shape, *, dtype=torch.bfloat16, contiguous=True, aligned=True, cuda=True):
        self.shape, self.ndim, self.dtype, self.is_cuda = shape, len(shape), dtype, cuda
        self.contiguous, self.aligned = contiguous, aligned
    def is_contiguous(self): return self.contiguous
    def data_ptr(self): return 256 if self.aligned else 258

dispatch_cases = 0
for layer, n in (('qkv_gate', 2304), ('pooler', 4096), ('o', 8192), ('unpooler', 8192), ('router', 256)):
    for m in range(1, 1025):
        data = {'x': Fake((m, 8192)), 'config': dict(layer=layer, m=m, n=n, k=8192)}
        got = key_scope['_lg_key'](data)
        expected = layer if (layer == 'qkv_gate' and 33 <= m <= 128) or (layer == 'pooler' and 65 <= m <= 128) else None
        assert got == expected
        dispatch_cases += 1
for layer, (_, _, n, k, _, _) in constants['_LG_CONFIGS'].items():
    for bad in (dict(dtype=torch.float32), dict(contiguous=False), dict(aligned=False), dict(cuda=False)):
        data = {'x': Fake((128, k), **bad), 'config': dict(layer=layer, m=128, n=n, k=k)}
        assert key_scope['_lg_key'](data) is None
        dispatch_cases += 1
    for field, value in (('m', 127), ('n', n + 1), ('k', k - 1)):
        cfg = dict(layer=layer, m=128, n=n, k=k); cfg[field] = value
        assert key_scope['_lg_key']({'x': Fake((128, k)), 'config': cfg}) is None
        dispatch_cases += 1

resources = {}
ownership_cases = 0
for kind, cl in (('gq', 2), ('gqx', 3)):
    bn, bm, stages, nth = 128, 64, 8, 128
    stage_bytes = (bm + bn) * 128
    extra = 64 * (bn // 2 + 4) * 4 if cl == 2 else (bn // 8) * nth * 16
    smem = 1024 + stages * stage_bytes + 16 * stages + 16 + extra
    assert smem < 232448  # H200 opt-in shared/block limit; verify actual compilation/resources on GPU.
    peer_start = stages * stage_bytes + 16 * stages + 16
    assert peer_start % 16 == 0
    assert 1023 + peer_start + extra < smem
    u = 8192 // 64
    intervals = [(r * u // cl * 64, (r + 1) * u // cl * 64) for r in range(cl)]
    assert intervals[0][0] == 0 and intervals[-1][1] == 8192
    assert all(intervals[r][1] == intervals[r + 1][0] for r in range(cl - 1))
    coverage = torch.zeros(8192, dtype=torch.int16)
    for lo, hi in intervals:
        for block in range(lo, hi, 64):
            for step in range(4): coverage[block + 16 * step:block + 16 * (step + 1)] += 1
    assert bool((coverage == 1).all())
    if cl == 2:
        assert 2 * 32 * (bn // 2) * 4 == 64 * (bn // 2) * 4
        last_peer_end = (63 * (bn // 2 + 4) + bn // 2) * 4
        assert last_peer_end <= extra
    else:
        assert (16 * 128 - 1) * 16 + 16 == extra
        largest_owned = (16 + cl - 1) // cl * 8 * 64 * 2
        assert largest_owned < stages * stage_bytes
    for m in range(1, 129):
        # Independently enumerate native accumulator->staging->global ownership.
        count = np.zeros((m, 64), dtype=np.int16)
        for rank in range(cl):
            if kind == 'gq':
                keep_lo = 2 * rank
                for warp in range(keep_lo, keep_lo + 2):
                    for lane in range(32):
                        g, t4 = lane >> 2, lane & 3
                        row0 = (warp - keep_lo) * 16 + g + rank * 32
                        for j in range(16):
                            token = 8 * j + 2 * t4
                            for dt, dr in ((0, 0), (0, 8), (1, 0), (1, 8)):
                                if token + dt < m: count[token + dt, row0 + dr] += 1
            else:
                jlo, jhi = rank * 16 // cl, (rank + 1) * 16 // cl
                for tid in range(128):
                    warp, lane = tid >> 5, tid & 31
                    g, t4 = lane >> 2, lane & 3
                    row0 = warp * 16 + g
                    for j in range(jlo, jhi):
                        token = 8 * j + 2 * t4
                        for dt, dr in ((0, 0), (0, 8), (1, 0), (1, 8)):
                            if token + dt < m: count[token + dt, row0 + dr] += 1
        assert np.all(count == 1), (kind, m)
        ownership_cases += 1
    for n in (2304, 4096):
        assert n % 64 == 0
        rows = [r for rb in range(n // 64) for r in range(64 * rb, 64 * (rb + 1))]
        assert rows == list(range(n))
    resources[kind] = dict(cluster=cl, dynamic_shared_bytes=smem,
                           k_intervals=intervals, payloads_fit=True,
                           output_ownership_exact=True, n_blocks=(4096 if cl == 2 else 2304) // 64 * cl)

errors = []
for cl in (2, 3):
    for m in (33, 65, 127, 128):
        x = torch.randn((m, 8192)).bfloat16().float()
        w = (torch.randn((64, 8192)) * .02).bfloat16().float()
        x[::8] = 0
        x[::8, 137] = 2
        pieces = [x[:, r * 128 // cl * 64:(r + 1) * 128 // cl * 64] @
                  w[:, r * 128 // cl * 64:(r + 1) * 128 // cl * 64].T for r in range(cl)]
        ref = x.double() @ w.double().T
        # Each owner starts with its own partial; then adds others in rank order.
        for owner in range(cl):
            total = pieces[owner].clone()
            for r in range(cl):
                if r != owner: total += pieces[r]
            got = total.bfloat16().double()
            error = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref))
            impulse = float(torch.linalg.vector_norm(got[::8] - ref[::8]) / torch.linalg.vector_norm(ref[::8]))
            assert error < .004 and impulse < .001
            errors.append(dict(cluster=cl, m=m, owner=owner, relative_l2=error, impulse_l2=impulse))
report = dict(source_sha256=meta['source_sha256'], seed=20261028,
              original_parent_prefix_exact=True, cuda_pieces_pinned=True,
              old_wg_choices_proven_empty_from_published_log=True,
              global_bindings_checked=len(checked_globals), dispatch_cases=dispatch_cases,
              resources=resources, output_ownership_cases=ownership_cases,
              numerical_cases=errors, no_weight_requantization=True,
              limitations='CPU models, not CUDA memory/race/compilation/performance verification',
              gpu_status='unmeasured')
(ROOT / '.tools/dense-cluster-cpu-audit.json').write_bytes(json.dumps(report, indent=2).encode())
print(json.dumps({k: v for k, v in report.items() if k != 'numerical_cases'}))
