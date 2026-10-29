"""Exercise only host diagnostic helpers; never call its prepare or GPU factory."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import runpy
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
addon_path = ROOT / 'experiments/prepared/moe_phase_diagnostic_addon.py'
addon_raw = addon_path.read_bytes()
tree = ast.parse(addon_raw)
allowed = {'base64', 'bisect', 'hashlib', 'json', 'random', 'statistics', 'struct', 'time', 'zlib'}
for node in tree.body:
    if isinstance(node, ast.Import):
        assert all(n.name in allowed for n in node.names)
    elif isinstance(node, ast.ImportFrom):
        assert node.module == 'torch.utils.cpp_extension' and [n.name for n in node.names] == ['load_inline']
    elif isinstance(node, ast.Assign):
        assert isinstance(node.value, (ast.Name, ast.Constant))
    elif isinstance(node, ast.FunctionDef):
        assert not node.decorator_list and not node.args.defaults and not node.args.kw_defaults
    else:
        assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
# Module scope defines helpers and aliases only. This does not execute the
# submission prefix, compile CUDA, call prepare or allocate any GPU tensor.
ns = runpy.run_path(str(addon_path), init_globals={'prepare': None, 'kernel_fn': None, 'torch': torch})
trace = []
base = 10 ** 12
for i in range(132):
    trace.append([base + 100 + i, base + 110 + i, base + 200 + 2 * i,
                  base + 300 + 3 * i, base + 1000 + i, base + 1100 + 4 * i,
                  base + 130 + i, base + 120 + i])
packed = ns['_phase_pack_trace'](trace, list(range(132)))
raw = ns['_phase_zlib'].decompress(ns['_phase_b64'].b64decode(packed['deltas_u32le_zlib_b64'], validate=True))
# Independent little-endian reconstruction against the known fixture.
recovered = [[packed['column_origins_ns'][c] + int.from_bytes(raw[(i * 8 + c) * 4:(i * 8 + c + 1) * 4], 'little')
              for c in range(8)] for i in range(132)]
assert recovered == trace
summary = ns['_phase_trace_summary'](trace, {'phases': [{}, {'stage_counts': [1] * 132}]})
assert summary['trace_span_ns'] == 1524
assert summary['final_tail_after_latest_B_ns'] == 931
assert summary['CTAs_finishing_nonempty_B_before_latest_A'] == 54
assert summary['interval_min_median_max_ns']['phase_A'] == [70, 135.5, 201]
assert summary['interval_min_median_max_ns']['phase_B_including_readiness'] == [100, 165.5, 231]
assert summary['interval_min_median_max_ns']['post_B_to_grid_join'] == [438, 569.0, 700]
assert summary['interval_min_median_max_ns']['finalization'] == [100, 296.5, 493]
rejected = []
bad = copy.deepcopy(trace); bad[0][2] = base
try:
    ns['_phase_trace_summary'](bad, {'phases': [{}, {'stage_counts': [1] * 132}]})
except AssertionError:
    rejected.append('nonmonotonic_phase_clock')
bad = copy.deepcopy(trace); bad[0][2] += 2 ** 33
try:
    ns['_phase_pack_trace'](bad, list(range(132)))
except AssertionError:
    rejected.append('timestamp_delta_overflow')
assert len(rejected) == 2
ids = torch.arange(8, dtype=torch.int32).expand(32, 8).contiguous()
metadata = ns['_phase_metadata']({'topk_ids': ids}, 132)
assert metadata['counts'] == [32] * 8 + [0] * 24
assert metadata['local_routes'] == 256 and metadata['active_experts'] == 8
assert metadata['jobs'] == 8 and metadata['padded_rows'] == 256 and metadata['zero_local_tokens'] == 0
assert [sum(p['stage_counts']) for p in metadata['phases']] == [4096, 2048]
empty = ns['_phase_metadata']({'topk_ids': ids + 32}, 132)
assert empty['local_routes'] == empty['jobs'] == 0 and empty['zero_local_tokens'] == 32
assert all(sum(p['stage_counts']) == 0 for p in empty['phases'])
candidate = (ROOT / '.tools/moe-phase-diagnostic.py').read_bytes()
report = json.loads((ROOT / '.tools/moe-phase-diagnostic-source-audit.json').read_bytes())
assert hashlib.sha256(candidate).hexdigest() == report['source_sha256']
parent = (ROOT / 'best/moe_verified.py').read_bytes()
assert candidate.startswith(parent)
assert hashlib.sha256(parent).hexdigest() == report['parent_sha256']
original = ast.parse(parent)
cuda = next(ast.literal_eval(n.value) for n in original.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'CUDA_SRC' for t in n.targets))
query = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == '_PHASE_QUERY' for t in n.targets))
instrumented = cuda.replace('#define OKC_DEBUG 0', '#define OKC_DEBUG 1', 1) + query
assert instrumented[:-len(query)].replace('#define OKC_DEBUG 1', '#define OKC_DEBUG 0', 1) == cuda
audit = dict(source_sha256=report['source_sha256'], parent_sha256=report['parent_sha256'],
             host_trace_codec_roundtrip=True, analytical_trace_fixture_exact=True,
             invalid_trace_rejections=rejected, metadata_tests=['eight_local_experts', 'all_nonlocal'],
             diagnostic_cuda_inverse_exact=True, original_prefix_exact=True,
             original_kernel_fn_binding_unchanged=True, cpu_only=True, gpu_verified=False,
             limitations='Synthetic host fixtures and source proofs; no native compilation or timing evidence')
(ROOT / '.tools/moe-phase-diagnostic-cpu-audit.json').write_bytes((json.dumps(audit, indent=2) + '\n').encode())
print(json.dumps(audit, indent=2))
