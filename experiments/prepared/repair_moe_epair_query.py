"""Repair only nine resource-query namespace references; launch nothing."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
FAILED = 'e8e3086d34a30de0283e37f2f40d3b1d9742313203fc624c1720bc6a6badced8'


def assignment(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def query_binding(tree):
    cuda = ast.literal_eval(assignment(tree, '_EPAIR_CUDA').value)
    query = ast.literal_eval(assignment(tree, '_EPAIR_QUERY').value)
    assert 'namespace okc_epair {' in cuda
    refs = re.findall(r'\b(\w+)::(moe_fused|NTL)\b', query)
    assert len(refs) == 9 and all(n == 'okc_epair' for n, _ in refs), 'Resource namespace mismatch'
    assert [m for _, m in refs].count('moe_fused') == 6
    assert [m for _, m in refs].count('NTL') == 3
    assert 'std::vector<int64_t> okc_epair_info()' in query
    module = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_epair_module')
    loader = next(n for n in ast.walk(module) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == '_epair_load')
    exports = ast.literal_eval(next(k.value for k in loader.keywords if k.arg == 'functions'))
    assert exports == ['okc_run', 'okc_set_pdl', 'okc_epair_info']
    cpp = ast.unparse(next(k.value for k in loader.keywords if k.arg == 'cpp_sources'))
    assert 'std::vector<int64_t> okc_epair_info();' in cpp
    prepare = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare'][-1]
    calls = [n for n in ast.walk(prepare) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'module']
    assert [n.func.attr for n in calls] == ['okc_epair_info']
    return cuda, query


def main():
    failed = (ROOT / 'results/moe-epair-store-preflight-20261029/submission.py').read_bytes()
    assert hashlib.sha256(failed).hexdigest() == FAILED
    assert failed.count(b'okc_epair_y::') == 9
    repaired = failed.replace(b'okc_epair_y::', b'okc_epair::')
    old, new = ast.parse(failed), ast.parse(repaired)
    expected = copy.deepcopy(old)
    before = assignment(expected, '_EPAIR_QUERY')
    original_query = ast.literal_eval(before.value)
    assert original_query.count('okc_epair_y::') == 9
    before.value = ast.Constant(value=original_query.replace('okc_epair_y::', 'okc_epair::'))
    assert ast.dump(expected) == ast.dump(new), 'Any other AST change is out of scope'
    cuda, query = query_binding(new)
    assert ast.literal_eval(assignment(old, '_EPAIR_CUDA').value) == cuda
    # Bind against the resource query that actually compiled in the earlier
    # factor experiment; only its namespace and exported function are renamed.
    known_tree = ast.parse((ROOT / 'results/moe-factor-y-20261029/submission.py').read_bytes())
    known = ast.literal_eval(assignment(known_tree, '_FACTOR_QUERY').value)
    assert query == known.replace('okc_factor_y::', 'okc_epair::').replace('okc_factor_info', 'okc_epair_info')
    mutants = [('original_wrong_namespace', old)]
    wrong = copy.deepcopy(new)
    assignment(wrong, '_EPAIR_QUERY').value = ast.Constant(value=query.replace('okc_epair::NTL', 'bad::NTL', 1))
    mutants.append(('wrong_geometry_namespace', wrong))
    wrong = copy.deepcopy(new)
    loader = next(n for n in ast.walk(next(n for n in wrong.body if isinstance(n, ast.FunctionDef)
                                          and n.name == '_epair_module'))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_epair_load')
    next(k for k in loader.keywords if k.arg == 'functions').value = ast.List(
        elts=[ast.Constant(value=v) for v in ['okc_run', 'okc_set_pdl']], ctx=ast.Load())
    mutants.append(('missing_export', wrong))
    wrong = copy.deepcopy(new)
    prep = [n for n in wrong.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare'][-1]
    method = next(n for n in ast.walk(prep) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                  and n.func.attr == 'okc_epair_info')
    method.func.attr = 'wrong_info'
    mutants.append(('wrong_host_method', wrong))
    rejected = []
    for name, mutant in mutants:
        try:
            query_binding(mutant)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Query-binding mutant escaped: ' + name)
    report = dict(failed_source_sha256=FAILED, source_sha256=hashlib.sha256(repaired).hexdigest(),
                  source_bytes=len(repaired), query_namespace_references_corrected=9,
                  only_query_constant_AST_changed=True, every_runtime_function_AST_unchanged=True,
                  cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(), cuda_algorithm_bytes_unchanged=True,
                  same_resource_query_as_previously_compiled_factor_module_after_identifier_binding=True,
                  exports_declarations_and_prepare_method_agree=True, rejected_mutants=rejected,
                  numerical_resource_selection_and_performance_gates_unchanged=True,
                  previous_CPU_layout_and_host_audits_reused_without_relabelling=True,
                  gpu_verified=False, compile_verified=False, new_gpu_jobs=0,
                  next_check_not_before_utc='2026-10-29T06:27:37.228843Z')
    folder = ROOT / 'results/moe-epair-store-repair-20261029'
    folder.mkdir(exist_ok=True)
    for name, data in [('submission.py', repaired), ('query-repair-audit.json', (json.dumps(report, indent=2) + '\n').encode())]:
        p = folder / name
        if p.exists():
            assert p.read_bytes() == data, 'Frozen repair collision: ' + name
        else:
            p.write_bytes(data)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
