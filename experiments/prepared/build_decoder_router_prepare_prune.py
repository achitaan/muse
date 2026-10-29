"""Remove two unreachable router GPU preparations; preserve public Decode dispatch."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = 'bc528949d73aa09ae9a330d030cadd5654f9ff7d20d48e8ff26bbeed96ae1c43'


def main():
    source = ROOT / 'results/decoder-source-package-20261029' / PARENT / 'submission.py'
    raw = source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    old, lines, edits = ast.parse(raw), raw.splitlines(keepends=True), []
    for fn in old.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name != '_achita_dense_prepare':
            continue
        for node in fn.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'jobs' for t in node.targets):
                for job in node.value.elts:
                    if ast.literal_eval(job.elts[0]) in ('router', 'router2'):
                        argument = job.elts[2]
                        assert isinstance(argument, ast.Name) and argument.id == 'weights'
                        a = sum(map(len, lines[:argument.lineno - 1])) + argument.col_offset
                        b = sum(map(len, lines[:argument.end_lineno - 1])) + argument.end_col_offset
                        edits.append((a, b, b'{}'))
    assert len(edits) == 2
    candidate = raw
    for a, b, replacement in sorted(edits, reverse=True):
        candidate = candidate[:a] + replacement + candidate[b:]
    restored = copy.deepcopy(ast.parse(candidate))
    reversed_count = 0
    for fn in restored.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name != '_achita_dense_prepare':
            continue
        for node in fn.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'jobs' for t in node.targets):
                for job in node.value.elts:
                    if ast.literal_eval(job.elts[0]) in ('router', 'router2'):
                        assert ast.literal_eval(job.elts[2]) == {}
                        job.elts[2] = ast.Name(id='weights', ctx=ast.Load())
                        reversed_count += 1
    assert reversed_count == 2 and ast.dump(restored) == ast.dump(old)
    families = {'qkv_gate', 'o', 'pooler', 'unpooler'}
    calls = [n for n in ast.walk(old) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == '_achita_dense_project']
    assert calls and all(isinstance(n.args[1], ast.Constant) and n.args[1].value in families for n in calls)
    guarded_reads = []
    for name in ('candidate_v8', 'candidate_base'):
        fn = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        route_if = next(n for n in fn.body if isinstance(n, ast.If))
        terms = route_if.test.values if isinstance(route_if.test, ast.BoolOp) else [route_if.test]
        assert any(ast.unparse(term) == "layer == 'router'" for term in terms)

        def reads(node):
            return [n for n in ast.walk(node) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                    and n.value.id == 'state' and isinstance(n.slice, ast.Constant)
                    and n.slice.value in ('router', 'router2')]

        guarded = [n for body in route_if.body for n in reads(body)]
        assert {id(n) for n in guarded} == {id(n) for n in reads(fn)}
        guarded_reads.append(dict(function=name, retired_state_reads=len(guarded)))
    for name in ('prepare_router', 'prepare_router2'):
        fn = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.unparse(fn.body[0]) == "w = weights.get('router')"
        branch = fn.body[1]
        assert isinstance(branch, ast.If) and isinstance(branch.test, ast.BoolOp) and isinstance(branch.test.op, ast.Or)
        assert ast.unparse(branch.test.values[0]) == 'w is None'
        assert isinstance(branch.body[0], ast.Return) and ast.literal_eval(branch.body[0].value) == {'enabled': False}
    digest = hashlib.sha256(candidate).hexdigest()
    assert digest == 'df68deb60a132159d630928d19d93bcb9513de86b98a8c9517cfadd5e2e2f731'
    report = dict(source_sha256=digest, parent_package_sha256=PARENT, bytes=len(candidate),
                  changed_prepare_arguments=2, all_other_ast_exact=True,
                  public_projection_families=sorted(families), public_projection_calls=len(calls),
                  retired_state_reads_guarded=guarded_reads, weightless_factories_return_before_cuda=True,
                  original_fp32_router_and_wide_router_retained=True, gpu_verified=False,
                  preparation_savings_s=None, timed_speedup_claim=False)
    directory = ROOT / 'results/decoder-router-prepare-prune-20261029'
    directory.mkdir(exist_ok=True)
    for name, data in [('submission.py', candidate), ('cpu-source-audit.json', (json.dumps(report, indent=2) + '\n').encode())]:
        target = directory / name
        if target.exists():
            assert target.read_bytes() == data, ('Frozen collision', name)
        else:
            target.write_bytes(data)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
