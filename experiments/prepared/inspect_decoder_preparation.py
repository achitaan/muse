"""Map pinned Decode preparation using AST and cached evidence only; no GPU imports."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').is_file())
PARENT = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracked-evidence-only', action='store_true',
                        help='Use frozen parent records, as on a fresh checkout; never contact the service')
    args = parser.parse_args()
    raw = (ROOT / 'best/decoder_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    tree, bindings = ast.parse(raw), {}
    # Reproduce top-level function/alias binding, without evaluating any code.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            bindings[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if isinstance(node.value, ast.Name) and node.value.id in bindings:
                        bindings[target.id] = bindings[node.value.id]
                    else:
                        bindings.pop(target.id, None)
    stages, seen = [], set()

    def visit(node):
        if node.lineno in seen:
            return
        seen.add(node.lineno)
        calls = sorted((n for n in ast.walk(node) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name)), key=lambda n: (n.lineno, n.col_offset))
        for call in calls:
            dependency = bindings.get(call.func.id)
            if dependency is not None and 'prepare' in dependency.name:
                visit(dependency)
        stages.append(dict(function=node.name, parent_line=node.lineno,
                           function_ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest()))

    visit(bindings['prepare'])
    owners = [(node, sub) for node in tree.body if isinstance(node, ast.FunctionDef)
              for sub in ast.walk(node) if isinstance(sub, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'jobs' for t in sub.targets)
              and isinstance(sub.value, ast.List)]
    assert len(owners) == 1
    owner, jobs_node = owners[0]
    jobs = [dict(name=ast.literal_eval(job.elts[0]), function=ast.unparse(job.elts[1]),
                 weight_expression=ast.unparse(job.elts[2]), options=ast.literal_eval(job.elts[3]))
            for job in jobs_node.value.elts]
    pools = [n for n in ast.walk(owner) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == '_PreparePool']
    assert len(pools) == 1
    workers = next(ast.literal_eval(k.value) for k in pools[0].keywords if k.arg == 'max_workers')
    assert len(jobs) == 21 and workers == 4
    # These entries share a function, but their inputs/options differ. They
    # cannot be collapsed merely because the function name repeats.
    pair = [j for j in jobs if j['function'] == 'tt_v10_m32x_prepare_m32x']
    assert len(pair) == 2 and pair[0]['options'] != pair[1]['options']
    prepare = bindings['_achita_dense_prepare']
    empty_choice = any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                       and not n.value.keys and any(isinstance(t, ast.Subscript)
                       and isinstance(t.slice, ast.Constant) and t.slice.value == 'achita_wg_choices'
                       for t in n.targets) for n in ast.walk(prepare))
    assert empty_choice, 'Reassess: the final WG sweep is no longer statically disabled'
    package = json.loads((ROOT / '.tools/decoder-source-package-source-audit.json').read_bytes())
    directory = ROOT / 'results/decoder-source-package-20261029' / package['source_sha256']
    assert directory.is_dir()
    local_cache = ROOT / '.decode-measurements' / PARENT
    use_frozen = args.tracked_evidence_only or not (local_cache / 'result.json').is_file()
    cached = directory / 'parent-result.json' if use_frozen else local_cache / 'result.json'
    launch_file = directory / 'parent-launch.json' if use_frozen else local_cache / 'launch.json'
    launch = json.loads(launch_file.read_bytes())
    assert launch['source_sha256'] == PARENT and launch['job_id'] == '9560c915e85f4f50'
    result = json.loads(cached.read_bytes())
    assert result['id'] == '9560c915e85f4f50' and result['status'] == 'done'
    assert result['result']['passed'] is True
    records = [r for r in result['result']['records'] if r.get('event') == 'setup']
    assert len(records) == 1 and records[0]['status'] == 'pass'
    if not use_frozen:
        assert hashlib.sha256((local_cache / 'submission.py').read_bytes()).hexdigest() == PARENT
    for name, source in [('parent-result.json', cached), ('parent-launch.json', launch_file)]:
        target = directory / name
        if target.exists():
            assert target.read_bytes() == source.read_bytes(), 'Parent evidence differs from frozen record'
        else:
            target.write_bytes(source.read_bytes())
    dense_audit = json.loads((ROOT / 'results/dense-cluster-preflight-20261029/source-audit.json').read_bytes())
    dense_parent_bytes = (ROOT / 'best/verified.py').stat().st_size
    dense_increment = dense_audit['bytes'] - dense_parent_bytes
    report = dict(
        parent_sha256=PARENT, packaged_source_sha256=package['source_sha256'],
        method='static AST and exact source-bound cached result; no submission code executed',
        preparation_functions_dependency_first=stages, parallel_jobs=jobs,
        parallel_jobs_parent_line=jobs_node.lineno, parallel_workers=workers,
        wg_preparation_sweep_already_disabled=True,
        same_function_different_inputs_not_deduplicated=pair,
        historical_setup=dict(job=result['id'], source_sha256=PARENT,
                              prepare_s=records[0]['prepare_s'],
                              cached_result=(directory / 'parent-result.json').relative_to(ROOT).as_posix(),
                              result_sha256=hashlib.sha256(cached.read_bytes()).hexdigest(),
                              launch_sha256=hashlib.sha256(launch_file.read_bytes()).hexdigest()),
        prepare_limit_s=300, historical_remaining_s=300 - records[0]['prepare_s'],
        new_prepare_measurement=None, per_stage_gpu_times_available=False,
        preparation_optimized=False,
        source_capacity_screen=dict(packaged_bytes=package['bytes'],
                                    standalone_dense_increment_bytes=dense_increment,
                                    raw_append_total_bytes=package['bytes'] + dense_increment,
                                    remaining_bytes_before_adaptation=1_000_000 - package['bytes'] - dense_increment,
                                    integration_built=False),
        conclusions=[
            'Source representation now has capacity for the raw D1 increment; no adapter has been built.',
            'The 300-second whole-layer preparation budget remains unresolved.',
            'Removing the already disabled WG sweep yields no additional savings.',
            'Repeated factory names do not prove duplicate builds or equivalent weight layouts.',
            'D1 must use one whole-layer budget and preserve the existing Decode state and dispatch.',
        ], gpu_jobs_launched=0,
    )
    # Retain the first local-cache map. This portable review adds raw parent
    # records and canonical paths without changing any measured evidence.
    target = directory / 'preparation-review.json'
    data = (json.dumps(report, indent=2) + '\n').encode()
    if target.exists():
        assert target.read_bytes() == data, 'Refusing to overwrite immutable preparation evidence'
    else:
        target.write_bytes(data)
    print(json.dumps(dict(source_sha256=package['source_sha256'], stages=len(stages),
                          parallel_jobs=len(jobs), workers=workers,
                          historical_prepare_s=records[0]['prepare_s'],
                          historical_remaining_s=report['historical_remaining_s'],
                          source_capacity_screen=report['source_capacity_screen'],
                          preparation_optimized=False, gpu_jobs_launched=0), indent=2))


if __name__ == '__main__':
    main()
