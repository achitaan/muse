"""Audit exact wide-router packing and its permitted BF16 hi/lo precision."""
import ast
import hashlib
import json
from pathlib import Path
import torch

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
g = torch.Generator().manual_seed(20261028)
w = torch.randn((256, 8192), generator=g, dtype=torch.float32)
w[::17, ::137] *= 64
hi = w.to(torch.bfloat16)
lo = (w - hi.float()).to(torch.bfloat16)
packed = torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3)
                      for p in (hi, lo)], dim=2).contiguous()
for i, original in enumerate([hi, lo]):
    restored = packed[:, :, i].permute(0, 2, 1, 3).reshape_as(original)
    assert torch.equal(restored.view(torch.int16), original.view(torch.int16))
max_relative = 0.0
rows = [1, 3, 33, 40, 128, 256, 257]
for m in rows:
    x = torch.randn((m, 8192), generator=g).to(torch.bfloat16).float()
    model = x @ hi.float().T + x @ lo.float().T
    # Analytical FP64 projection of the same BF16 x against original FP32 w.
    ref = x.double() @ w.double().T
    rel = (torch.linalg.vector_norm(model.double() - ref) / torch.linalg.vector_norm(ref)).item()
    assert rel < 1e-5
    max_relative = max(max_relative, rel)
source = (root / '.tools/decoder-routerwide-variant.py').read_bytes()
tree = ast.parse(source)
kernel = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn')
assert 'router_dtype=None if split_router else torch.float32' in ast.unparse(kernel)
assert "F.linear(router_input, state['router'])" in ast.unparse(kernel)
helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_achita_wide_router')
assert 'x.shape[0] <= 32' in ast.unparse(helper)
assert '128 if x.shape[0] <= 128 else 256' in ast.unparse(helper)
report = {'source_sha256': hashlib.sha256(source).hexdigest(), 'seed': 20261028,
          'exact_hi_lo_word_recovery': 2 * 256 * 8192,
          'fp32_matrix_model_max_relative_l2_to_fp64': max_relative,
          'row_cases': rows, 'native_kernel_inherits_passing_dense_coverage': '6131bc96427e4486',
          'new_whole_layer_gpu_status': 'see immutable official result',
          'limitation': 'CPU model does not emulate wgmma instruction or reduction order'}
(root / '.tools/decoder-routerwide-cpu-audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report))
