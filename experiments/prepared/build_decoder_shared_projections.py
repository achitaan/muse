"""Share the existing SK2 extension and disable unreachable router setup."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
base_sha = '09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67'
trial_sha = '92fe19e2e94211b479baf0b8b5ab2bd2a705cd8c50ce09e5060479614d6a68aa'
base_raw = (root / 'best/decoder_verified.py').read_bytes()
assert hashlib.sha256(base_raw).hexdigest() == base_sha
base = base_raw.decode('utf-8')
old_trial = (root / 'results/resumed-decoder-phase7-20261028' / trial_sha / 'submission.py').read_bytes()
assert hashlib.sha256(old_trial).hexdigest() == trial_sha
trial_tree = ast.parse(old_trial)
constants = {n.targets[0].id: n.value.value for n in trial_tree.body
             if isinstance(n, ast.Assign) and len(n.targets) == 1
             and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
configs = ast.literal_eval(next(n.value for n in trial_tree.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == '_SP_CONFIGS' for t in n.targets)))
old_tree = ast.parse(base)
new_cpp = constants['_fpr_CPP'] + constants['_SP_CPP'].split('\n', 1)[1]
new_cuda = constants['_fpr_CUDA'] + constants['_SP_DISPATCH']
patches = []
counts = Counter()
for node in old_tree.body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        name = node.targets[0].id
        if name in ('_fpr_CPP', '_fpr_CUDA'):
            patches.append((node, f'{name} = {(new_cpp if name == "_fpr_CPP" else new_cuda)!r}'))
    if not isinstance(node, ast.FunctionDef):
        continue
    if node.name == '_fpr_get_ext':
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and child.value == 'okc_fpr_f68f77c0d29cc56b_':
                child.value = 'okc_fpr_sp_' + hashlib.sha256(new_cuda.encode()).hexdigest()[:16] + '_'
                counts['factory_name'] += 1
            if isinstance(child, ast.keyword) and child.arg == 'functions':
                assert ast.literal_eval(child.value) == ['sk2f', 'sk2_set_pdl']
                child.value.elts.append(ast.Constant('sk2small'))
                counts['factory_exports'] += 1
        patches.append((node, ast.unparse(node)))
    elif node.name == '_achita_dense_prepare':
        job_list = next((n for n in node.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'jobs' for t in n.targets)), None)
        if job_list:
            for job in job_list.value.elts:
                if ast.literal_eval(job.elts[0]) in ('router', 'router2'):
                    assert isinstance(job.elts[2], ast.Name) and job.elts[2].id == 'weights'
                    job.elts[2] = ast.Dict(keys=[], values=[])
                    counts['unused_router_jobs'] += 1
            patches.append((node, ast.unparse(node)))
    elif node.name == 'prepare_bundle':
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == 'rt_prepare_routertail_candidates':
                assert len(child.args) == 1 and ast.unparse(child.args[0]) == 'weights'
                child.args[0] = ast.Dict(keys=[], values=[])
                counts['unused_routertail_job'] += 1
        patches.append((node, ast.unparse(node)))
assert counts == {'factory_name': 1, 'factory_exports': 1, 'unused_router_jobs': 2, 'unused_routertail_job': 1}

lines = base.splitlines(keepends=True)
for node, replacement in sorted(patches, key=lambda p: p[0].lineno, reverse=True):
    assert node.col_offset == 0
    lines[node.lineno - 1:node.end_lineno] = [replacement + '\n']
modified = ''.join(lines)

addon_tree = ast.parse((root / 'experiments/prepared/small_projection_addon.py').read_text(encoding='utf-8'))
for node in addon_tree.body:
    if isinstance(node, ast.FunctionDef) and node.name == '_sp_module':
        node.body = [ast.Return(ast.Call(func=ast.Name(id='_fpr_get_ext', ctx=ast.Load()), args=[], keywords=[]))]
    if isinstance(node, ast.FunctionDef) and node.name == '_sp_select':
        for child in ast.walk(node):
            # The extension is already built. Preserve 20 seconds after comparison.
            if isinstance(child, ast.Compare) and ast.unparse(child.left) == '_sp_time.monotonic() - started':
                for value in child.comparators:
                    assert isinstance(value, ast.Constant) and value.value in (230, 235, 275)
                    value.value = {230: 260, 235: 265, 275: 280}[value.value]
    if isinstance(node, ast.FunctionDef) and node.name == 'prepare':
        node.body.insert(2, ast.parse("print('SMALL_PROJ_PREPARE_BASE_S', _sp_time.monotonic() - started, flush=True)").body[0])
for node in ast.walk(addon_tree):
    if isinstance(node, (ast.FunctionDef, ast.Module)) and node.body:
        if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body.pop(0)
ast.fix_missing_locations(addon_tree)
compact = []
for line in ast.unparse(addon_tree).splitlines():
    indent = len(line) - len(line.lstrip(' '))
    assert indent % 4 == 0
    compact.append(' ' * (indent // 4) + line[indent:])
addon = '\n'.join(compact) + '\n'
assert ast.dump(ast.parse(addon)) == ast.dump(addon_tree)
combined = modified + '\n# Share the existing licensed SK2 extension; three fixed public BF16 projections.\n'
combined += f'_SP_CONFIGS = {configs!r}\n' + addon
compile(combined, '<shared-small-projections>', 'exec')
assert len(combined.encode('utf-8')) < 1_000_000
new_tree = ast.parse(combined)
new_constants = {n.targets[0].id: n.value.value for n in new_tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
old_constants = {n.targets[0].id: n.value.value for n in ast.parse(base).body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1
                 and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.Constant)}
assert {k: v for k, v in old_constants.items() if k not in ('_fpr_CPP', '_fpr_CUDA')} == {
    k: v for k, v in new_constants.items() if k in old_constants and k not in ('_fpr_CPP', '_fpr_CUDA')}
assert new_constants['_fpr_CUDA'] == old_constants['_fpr_CUDA'] + constants['_SP_DISPATCH']
target = root / '.tools/decoder-shared-projections-variant.py'
target.write_bytes(combined.encode('utf-8'))
record = {'source_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'base_sha256': base_sha, 'packaged_bytes': target.stat().st_size,
          'curated_configs': configs, 'changes': dict(counts),
          'existing_cuda_bytes': 'unchanged; append three sk2small instantiations to existing extension',
          'other_literal_constants': 'unchanged', 'router_runtime_and_norm_math': 'unchanged',
          'extra_extension_builds': 0, 'selection_cutoffs_seconds': [260, 265, 280],
          'gpu_status': 'unmeasured'}
(root / '.tools/decoder-shared-projections-source-audit.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
print(json.dumps(record))
