"""Test one curated lossless output projection for 9..16 decoder tokens."""
import ast
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
prior_sha = 'ced7a4b9fa1aef6a51549706acf5eccaa44d012b27f83f4ca25e124d5f42b610'
raw = (root / 'results/resumed-decoder-phase8-20261028' / prior_sha / 'submission.py').read_bytes()
assert hashlib.sha256(raw).hexdigest() == prior_sha
prior = raw.decode('utf-8')
base_raw = (root / 'best/decoder_verified.py').read_bytes()
base_sha = '09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67'
assert hashlib.sha256(base_raw).hexdigest() == base_sha
base_tree = ast.parse(base_raw)
constants = {n.targets[0].id: n.value.value for n in base_tree.body
    if isinstance(n, ast.Assign) and len(n.targets) == 1
    and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
upstream = ast.parse((root / '.tools/vjeux-decoder-upstream.py').read_bytes())
tables = {}
for node in upstream.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in ('bfp_SK2_CONFIGS', 'bfp_SK2_K'):
                tables[target.id] = ast.literal_eval(node.value)
geometry = next(n for n in upstream.body if isinstance(n, ast.FunctionDef) and n.name == 'bfp__sk2_geom')
exec(compile(ast.Module(body=[geometry], type_ignores=[]), '<pinned-geometry>', 'exec'), tables)
cfg = next(c for c in tables['bfp_SK2_CONFIGS'] if c[0] == 'o32h_r100')
_, layer, n, rows, nt, nwr, wsk, upw, nacc, cl = cfg
hl, stages = tables['bfp__sk2_geom'](cfg, 16)
k = tables['bfp_SK2_K'][layer]
assert (layer, n, k, rows, nt, nwr, wsk, upw, nacc, cl, hl, stages) == ('o', 8192, 1024, 32, 2, 2, 2, 2, 1, 1, False, 4)
config = (0, layer, n, k, rows, nt, nwr, wsk, upw, nacc, stages)
cpp = constants['_fpr_CPP'] + '''
void sk2output(torch::Tensor Ws, torch::Tensor x, torch::Tensor out, torch::Tensor ptab,
               int64_t pmax, int64_t base4, int64_t M, int64_t ci);
'''
cuda = constants['_fpr_CUDA'] + f'''
void sk2output(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab,
               int64_t pmax, int64_t base4, int64_t M, int64_t ci) {{
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16
              && M > 8 && M <= 16 && ci == 0);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == {k} && out.numel() >= M * {n});
  launch_sk2<1, 16, {nt}, {nwr}, {wsk}, {upw}, {stages}, true, false, {nacc}, 1, __nv_bfloat16>(
      Ws, x, out, ptab, pmax, base4, (int)M, {n}, {k}, 0, nullptr);
}}
'''
prefix = prior.split('\n# Share the existing licensed SK2 extension;', 1)[0]
prefix_tree = ast.parse(prefix)
patches = []
for node in prefix_tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ('_fpr_CPP', '_fpr_CUDA'):
            patches.append((node, f'{name} = {(cpp if name == "_fpr_CPP" else cuda)!r}'))
    elif isinstance(node, ast.FunctionDef) and node.name == '_fpr_get_ext':
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.startswith('okc_fpr_sp_'):
                child.value = 'okc_fpr_op_' + hashlib.sha256(cuda.encode()).hexdigest()[:16] + '_'
            if isinstance(child, ast.keyword) and child.arg == 'functions':
                assert ast.literal_eval(child.value) == ['sk2f', 'sk2_set_pdl', 'sk2small']
                child.value.elts[-1] = ast.Constant('sk2output')
        patches.append((node, ast.unparse(node)))
lines = prefix.splitlines(keepends=True)
for node, replacement in sorted(patches, key=lambda p: p[0].lineno, reverse=True):
    lines[node.lineno - 1:node.end_lineno] = [replacement + '\n']
prefix = ''.join(lines)

# Reuse the checked comparison design; no rejected projection instantiations.
addon = ast.parse(prior).body[len(base_tree.body):]
addon = [node for node in addon if not isinstance(node, ast.Assign)
         or not any(isinstance(t, ast.Name) and t.id == '_SP_CONFIGS' for t in node.targets)]
for node in addon:
    if not isinstance(node, ast.FunctionDef):
        continue
    if node.name == '_sp_pack':
        # XBY uses the 16-row TMA geometry, not the old 8-row geometry.
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'sst' for t in child.targets):
                replacement = ast.parse('sst = ((wsk * upw * _SP_TOKR * 128 + nwr * wsk * upw * nt * 768) + 1023) // 1024 * 1024').body[0]
                child.value = replacement.value
    elif node.name == '_sp_run':
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr == 'sk2small': child.attr = 'sk2output'
    elif node.name == '_sp_select':
        for child in ast.walk(node):
            if isinstance(child, ast.For) and isinstance(child.target, ast.Name) and child.target.id == 'rows':
                assert ast.literal_eval(child.iter) == (1, 8)
                child.iter = ast.Tuple(elts=[ast.Constant(16)], ctx=ast.Load())
    elif node.name == '_achita_dense_project':
        replacement = ast.parse('''
def _achita_dense_project(x, layer, state, rows=None):
    m = x.shape[0]
    plan = state.get('small_projection_choices', {}).get((layer, 16)) if 8 < m <= 16 else None
    if plan is not None:
        target = state['dense'][layer].shape[0] if rows is None else rows
        if (target == plan['n'] and x.shape[1] == plan['k'] and x.dtype == torch.bfloat16
                and x.is_contiguous() and x.device == plan['packed'].device and x.data_ptr() % 16 == 0):
            return _sp_run(x, plan)
    return _sp_incumbent_project(x, layer, state, rows)
''').body[0]
        node.args, node.body = replacement.args, replacement.body
addon_tree = ast.Module(body=addon, type_ignores=[])
ast.fix_missing_locations(addon_tree)
compact = []
for line in ast.unparse(addon_tree).splitlines():
    indent = len(line) - len(line.lstrip(' '))
    assert indent % 4 == 0
    compact.append(' ' * (indent // 4) + line[indent:])
text = '\n'.join(compact) + '\n'
assert ast.dump(ast.parse(text)) == ast.dump(addon_tree)
combined = prefix + '\n# One curated output projection in the existing licensed SK2 extension.\n'
combined += f'_SP_CONFIGS = {[config]!r}\n_SP_TOKR = 16\n' + text
compile(combined, '<decoder-output-projection>', 'exec')
target = root / '.tools/decoder-output-projection-variant.py'
target.write_bytes(combined.encode('utf-8'))
assert target.stat().st_size < 1_000_000
record = {'source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'base_sha256': base_sha,
          'packaged_bytes': target.stat().st_size, 'public_config': cfg, 'instantiation': config,
          'tma_token_rows': 16, 'dispatch_domain': '9..16 tokens; output projection only',
          'comparison': 'two synthetic patterns at 16 tokens; incumbent vs one fixed SK2 config',
          'rejected_small_projection_instantiations': 0, 'extra_extension_builds': 0,
          'gpu_status': 'unmeasured'}
(root / '.tools/decoder-output-projection-source-audit.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
