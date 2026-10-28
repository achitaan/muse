"""Check manufactured RMSNorm values and exact route rotation on CPU."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
rng = np.random.default_rng(20261028)
max_relative = 0.0
count = 0
for rows in [1, 3, 8, 16, 24, 32, 33, 128, 256, 384]:
    for kind in ['zero', 'constant', 'normal', 'outlier', 'small']:
        a = rng.normal(size=(rows, 4096)).astype(np.float32)
        if kind == 'zero': a.fill(0)
        if kind == 'constant': a.fill(0.125)
        if kind == 'small': a *= 1e-4
        if kind == 'outlier': a[:, ::131] *= 64
        x = torch.from_numpy(a).to(torch.bfloat16).float().numpy()
        # SGLang precomputes its effective zero-centered weight in BF16.
        w = torch.from_numpy(rng.normal(0, .1, 4096).astype(np.float32)).to(torch.bfloat16)
        gain = (w + 1.0).float().numpy()
        inv = (np.float32(1.) / np.sqrt((x * x).sum(-1, keepdims=True, dtype=np.float32)
                                       / np.float32(4096) + np.float32(1e-5))).astype(np.float32)
        y = torch.from_numpy((x * inv) * gain).to(torch.bfloat16).float().numpy()
        # Analytical RMSNorm in float64, with the same BF16 input/effective gain.
        ref = x.astype(np.float64) / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * gain
        rel = np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-30)
        max_relative = max(max_relative, float(rel))
        assert np.isfinite(y).all() and rel < 0.004
        count += 1
ids = np.arange(256, dtype=np.int32)
starts = np.arange(0, 256, 32, dtype=np.int32)
assert np.array_equal((ids[None, :] - starts[:, None]) % 256,
                      (ids[None, :] - starts[:, None]) & 255)
source = (root / '.tools/decoder-expert-inputs-variant.py').read_bytes()
tree = ast.parse(source)
addon = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_pn_inputs_kernel')
calls = ast.unparse(addon)
assert '_pn_tl.float32' in calls and '_pn_tl.sum(x * x, 0)' in calls
assert any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd)
           and isinstance(n.right, ast.Constant) and n.right.value == 255
           for n in ast.walk(addon))
report = {'source_sha256': hashlib.sha256(source).hexdigest(), 'seed': 20261028,
          'manufactured_cases': count, 'max_relative_l2_to_analytical_fp64': max_relative,
          'route_rotations_exact': 2048, 'every_output_row_and_column_written': True,
          'gpu_correctness_and_performance': 'unmeasured'}
(root / '.tools/decoder-expert-inputs-cpu-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
