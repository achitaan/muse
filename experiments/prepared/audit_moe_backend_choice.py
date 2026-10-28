"""Check synthetic routing and native scratch bounds without GPU measurement."""
import ast
import hashlib
import json
from pathlib import Path
import torch

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
tree = ast.parse((root / 'experiments/prepared/moe_backend_choice_addon.py').read_text(encoding='utf-8'))
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_bkt_synthetic')
scope = {'torch': torch, 'HIDDEN': 4096}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<synthetic-route-audit>', 'exec'), scope)
cases = 0
max_jobs = 0
for rows in range(9, 33):
    for start in (0, 96):
        for i, kind in enumerate(('diffuse', 'hot', 'mixed')):
            x, ids, weights = scope['_bkt_synthetic'](rows, kind, {'lo': start, 'w': [torch.zeros(1)]},
                                                     20261028 + rows * 100 + i)
            assert x.shape == (rows, 4096) and x.dtype == torch.bfloat16 and x.is_contiguous()
            assert ids.shape == weights.shape == (rows, 8) and ids.dtype == torch.int32
            assert bool(((ids >= 0) & (ids < 256)).all())
            assert all(torch.unique(row).numel() == 8 for row in ids)
            assert bool((weights > 0).all())
            assert torch.allclose(weights.sum(-1), torch.ones(rows), atol=1e-6, rtol=0)
            rotated = (ids - start) % 256
            counts = torch.bincount(rotated.flatten(), minlength=256)[:32]
            assert int(counts.max()) <= rows  # each token visits an expert at most once
            assert int(counts.sum()) == rows * (4 if kind == 'hot' else 2)
            # A mixed fallback job covers up to 64 rows: for M<=32 there can
            # be at most one job per expert, hence <=32 < native JMAX=88.
            jobs = int((counts > 0).sum())
            assert jobs <= 32 and rows * 8 <= 3072
            max_jobs = max(max_jobs, jobs)
            cases += 1
selector = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_bkt_select')
assert isinstance(selector.body[-1], ast.Return) and ast.unparse(selector.body[-1].value) == 'choices'
assert all(not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Subscript)
           and isinstance(t.value, ast.Name) and t.value.id == 'state' for t in n.targets)) for n in ast.walk(selector))
source = (root / '.tools/moe-backend-choice-variant.py').read_bytes()
record = {'source_sha256': hashlib.sha256(source).hexdigest(),
          'synthetic_cases': cases, 'eligible_rows': '9..32', 'local_starts': [0, 96],
          'route_ids_unique_per_token': True, 'weights_positive_normalized': True,
          'maximum_fallback_jobs_observed': max_jobs, 'fallback_jobs_upper_bound': 32,
          'native_jmax': 88, 'scratch_route_capacity': 3072,
          'selector_returns_backend_metadata_only': True,
          'gpu_correctness_and_performance': 'unmeasured'}
(root / '.tools/moe-backend-choice-cpu-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
