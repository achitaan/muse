"""Audit precise setup edits and prove retired router branches unreachable."""
import argparse
import ast
import collections
import copy
import __future__
import hashlib
import json
from pathlib import Path
import types

root = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='.tools/decoder-shared-projections-variant.py')
parser.add_argument('--output', default='.tools/decoder-shared-prepare-cpu-audit.json')
args = parser.parse_args()
base = ast.parse((root / 'best/decoder_verified.py').read_bytes())
raw = (root / args.source).read_bytes()
tree = ast.parse(raw)
symbol = 'sk2output' if any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_SP_TOKR' for t in n.targets) for n in tree.body) else 'sk2small'
restored = copy.deepcopy(tree.body[:len(base.body)])
edits = collections.Counter()
for old, node in zip(base.body, restored):
    if ast.dump(old) == ast.dump(node):
        continue
    if isinstance(node, ast.Assign):
        name = node.targets[0].id
        assert name in ('_fpr_CPP', '_fpr_CUDA')
        # Append-only extension integration must retain every old source byte.
        assert node.value.value.startswith(old.value.value)
        node.value = copy.deepcopy(old.value)
        edits[name] += 1
    elif isinstance(node, ast.FunctionDef) and node.name == '_fpr_get_ext':
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value.startswith(('okc_fpr_sp_', 'okc_fpr_op_')):
                child.value = 'okc_fpr_f68f77c0d29cc56b_'
                edits['factory_name'] += 1
            elif isinstance(child, ast.keyword) and child.arg == 'functions':
                assert ast.literal_eval(child.value) == ['sk2f', 'sk2_set_pdl', symbol]
                child.value.elts.pop()
                edits['factory_exports'] += 1
    elif isinstance(node, ast.FunctionDef) and node.name == '_achita_dense_prepare':
        jobs = next(n for n in node.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'jobs' for t in n.targets))
        for job in jobs.value.elts:
            if ast.literal_eval(job.elts[0]) in ('router', 'router2'):
                assert ast.literal_eval(job.elts[2]) == {}
                job.elts[2] = ast.Name(id='weights', ctx=ast.Load())
                edits['router_jobs'] += 1
    elif isinstance(node, ast.FunctionDef) and node.name == 'prepare_bundle':
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == 'rt_prepare_routertail_candidates':
                assert ast.literal_eval(child.args[0]) == {}
                child.args[0] = ast.Name(id='weights', ctx=ast.Load())
                edits['routertail_job'] += 1
    else:
        raise AssertionError(ast.dump(node)[:200])
    # Reversing only the enumerated changes must recover the original full AST.
    assert ast.dump(node) == ast.dump(old)
assert edits == {'_fpr_CPP': 1, '_fpr_CUDA': 1, 'factory_name': 1, 'factory_exports': 1,
                 'router_jobs': 2, 'routertail_job': 1}
assert ast.dump(ast.Module(body=restored, type_ignores=[])) == ast.dump(base)

# All original entrypoint calls name BF16 projection families, never router.
families = {'qkv_gate', 'o', 'pooler', 'unpooler'}
calls = [n for n in ast.walk(base) if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Name) and n.func.id == '_achita_dense_project']
assert calls and all(isinstance(n.args[1], ast.Constant) and n.args[1].value in families for n in calls)
configs = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                               and any(isinstance(t, ast.Name) and t.id == '_SP_CONFIGS' for t in n.targets)))
assert {c[1] for c in configs} <= families

# Dispatch-only test: force every eligible dense backend to fall through while
# making reads/calls to the three retired router components fail immediately.
# The numerical oracle is the separate codec audit and the official full check.
def forbidden(*args, **kwargs):
    raise AssertionError('Retired router backend reached')
class Poison(dict):
    __getitem__ = get = forbidden
fake_weight = types.SimpleNamespace(device='cuda:0')
fake_torch = types.SimpleNamespace(bfloat16='bf16', empty=lambda *args, **kwargs: object())
table_nodes = [n for n in base.body if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id in ('OLAYER_PRIMARY_CONFIGS', 'OLAYER_NEARBY_CONFIGS', 'OLAYER_CONFIGS') for t in n.targets)]
table_namespace = {}
exec(compile(ast.Module(body=table_nodes, type_ignores=[]), '<pure-config-tables>', 'exec'), table_namespace)
namespace = {'torch': fake_torch, 'OLAYER_CONFIGS': table_namespace['OLAYER_CONFIGS']}
kernel_names = {'candidate_v8', 'candidate_base', 'candidate_v9', 'candidate_v10',
                'candidate_v11', 'candidate_v12', 'candidate_v13', 'candidate_v14',
                'candidate_v15', '_achita_dense_kernel', 'run_bundle', 'run_bundle2'}
aliases = {'_bundle_base_kernel', '_b2_base_kernel', '_v18_base_kernel', '_achita_wg_base_kernel'}
nodes = [n for n in tree.body[:len(base.body)]
         if isinstance(n, ast.FunctionDef) and n.name in kernel_names
         or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in aliases for t in n.targets)]
for node in nodes:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
            namespace.setdefault(child.func.id, lambda *args, **kwargs: None)
namespace.update(run_router=forbidden, run_router2=forbidden,
                 rt_run_routertail_candidate=forbidden,
                 candidate_sglang=lambda *args, **kwargs: 'fallback')
for node in nodes:
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<dispatch-only>', 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
state = collections.defaultdict(dict)
state.update(router=Poison(), router2=Poison(), achita_wg_choices={},
             bundle={'dd': {}, 'rw': {}, 'rt': Poison(), 'mt': {}, 'qkv': None},
             bundle2={'qkv': {}, 'pooler': {}, 'o': {'weight': fake_weight}, 'diet': {}})
cases = 0
for m in (*range(1, 301), 512, 1024):
    for layer in sorted(families):
        k = 1024 if layer == 'o' else 4096 if layer == 'unpooler' else 8192
        n = 2304 if layer == 'qkv_gate' else 4096 if layer == 'pooler' else 1024 if layer == 'unpooler' and m <= 8 else 8192
        x = types.SimpleNamespace(shape=(m, k), ndim=2, dtype='bf16', is_cuda=True,
                                  device='cuda:0', is_contiguous=lambda: True, data_ptr=lambda: 256)
        data = {'x': x, 'config': {'m': m, 'n': n, 'k': k, 'layer': layer}}
        assert namespace['_achita_dense_kernel'](data, state) == 'fallback'
        cases += 1

# Actual retired prepare functions return before any CUDA work when weightless.
disabled_ns = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('prepare_router', 'prepare_router2', 'rt_prepare_routertail_candidates'):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<disabled-prepare>', 'exec',
                     flags=__future__.annotations.compiler_flag), disabled_ns)
        assert disabled_ns[node.name]({}) == {'enabled': False}

result = {'source_sha256': hashlib.sha256(raw).hexdigest(), 'enumerated_edits': dict(edits),
          'reversing_only_enumerated_edits_recovers_entire_baseline_ast': True,
          'all_original_cuda_bytes_preserved': True, 'dense_projection_call_families': sorted(families),
          'dispatch_fallthrough_cases': cases, 'retired_router_accesses': 0,
          'actual_disabled_prepare_functions': 3, 'numerical_status': 'see codec audit; GPU unmeasured'}
(root / args.output).write_bytes((json.dumps(result, indent=2) + '\n').encode())
print(json.dumps(result))
