"""Reuse existing generic CUDA with three curated public SK2 instantiations."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
sha = '09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67'
raw = (root / 'results/resumed-decoder-phase6-20261028' / sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == sha
base = raw.decode('utf-8')
upstream = ast.parse((root / '.tools/vjeux-decoder-upstream.py').read_text(encoding='utf-8'))
values = {}
for node in upstream.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                try: values[target.id] = ast.literal_eval(node.value)
                except (ValueError, TypeError): pass
geometry_fn = next(n for n in upstream.body if isinstance(n, ast.FunctionDef) and n.name == 'bfp__sk2_geom')
scope = {'bfp_SK2_K': values['bfp_SK2_K']}
exec(compile(ast.Module(body=[geometry_fn], type_ignores=[]), '<audited-geometry>', 'exec'), scope)
names = ['pooler8h_r72', 'qkv8_r176', 'slice8_r192']
configs = []
dispatch = []
for ci, name in enumerate(names):
    cfg = next(c for c in values['bfp_SK2_CONFIGS'] if c[0] == name)
    _, layer, n, rows, nt, nwr, wsk, upw, nacc, cl = cfg
    hl, stages = scope['bfp__sk2_geom'](cfg, 8)
    assert not hl and cl == 1 and rows == nwr * nt * 8
    k = values['bfp_SK2_K'][layer]
    configs.append((ci, layer, n, k, rows, nt, nwr, wsk, upw, nacc, stages))
    dispatch.append(f'''if (ci == {ci}) {{
      TORCH_CHECK(x.dim() == 2 && x.size(1) == {k} && out.numel() >= M * {n});
      launch_sk2<1, 8, {nt}, {nwr}, {wsk}, {upw}, {stages}, true, false, {nacc}, 1, __nv_bfloat16>(
          Ws, x, out, ptab, pmax, base4, (int)M, {n}, {k}, 0, nullptr); return;
    }}''')
cpp = '''#include <torch/extension.h>
void sk2small(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab,
              int64_t pmax, int64_t base4, int64_t M, int64_t ci);
'''
cuda_tail = '''
void sk2small(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab,
              int64_t pmax, int64_t base4, int64_t M, int64_t ci) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && M >= 1 && M <= 8);
''' + '\n'.join(dispatch) + '\n  TORCH_CHECK(false, "sk2small: unsupported configuration");\n}\n'
setup = (f'_SP_CONFIGS = {configs!r}\n_SP_CPP = {cpp!r}\n_SP_DISPATCH = {cuda_tail!r}\n'
         "_SP_CUDA = _fpr_CUDA[:_fpr_CUDA.index('\\nvoid sk2f(')] + _SP_DISPATCH\n_sp_ext = None\n")
addon = (root / 'experiments/prepared/small_projection_addon.py').read_text(encoding='utf-8')
tree = ast.parse(setup + addon)
# Compact ordinary Python indentation only; all CUDA literals remain visible.
for n in ast.walk(tree):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module)) and n.body:
        if isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
            n.body.pop(0)
ast.fix_missing_locations(tree)
text = ast.unparse(tree)
lines = []
for line in text.splitlines():
    indent = len(line) - len(line.lstrip(' '))
    assert indent % 4 == 0
    lines.append(' ' * (indent // 4) + line[indent:])
addon_text = '\n'.join(lines) + '\n'
assert ast.dump(ast.parse(addon_text)) == ast.dump(tree)
combined = base + '\n# Curated small BF16 projection paths from pinned Vjeux Apache-2.0 source; full license retained above.\n' + addon_text
compile(combined, '<small-projections>', 'exec')
assert len(combined.encode('utf-8')) < 1_000_000
# Original CUDA constants, precision paths and existing factory are untouched.
assert combined[:len(base)] == base
target = root / '.tools/decoder-small-projections-variant.py'
target.write_bytes(combined.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(combined.encode('utf-8')).hexdigest(),
          'base_sha256': sha, 'packaged_bytes': len(combined.encode('utf-8')),
          'curated_configs': configs, 'all_previous_source_bytes_unchanged': True,
          'new_cuda': 'reuse generic SK2 template plus three fixed dispatch instantiations',
          'cached_state': 'static lossless weights plus configuration choices only',
          'gpu_status': 'unmeasured'}
(root / '.tools/decoder-small-projections-source-audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
