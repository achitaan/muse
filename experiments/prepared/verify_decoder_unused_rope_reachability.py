"""Static root-state escape audit for the preparation-only RoPE removal."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
FOLDER = ROOT / 'results/decoder-unused-rope-prune-20261029'
SHA = '307dd791187807cd4895346cec314a91ff8a2af49add7d7ac5c7a27f04007072'


def main():
    raw = (FOLDER / 'submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == SHA
    tree = ast.parse(raw)
    # Only these local functions receive the whole root state during a timed
    # call. All other callees receive data or an explicitly extracted substate.
    argument_positions = {'kernel_fn': 1, '_achita_dense_project': 2,
                          '_pn_expert_inputs': 3, '_experts': 3,
                          '_sfn_boundary': 2, '_achita_wide_router': 1}
    final_bindings = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            final_bindings[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in argument_positions:
                    final_bindings[target.id] = node
    summary = []
    for name, position in argument_positions.items():
        fn = final_bindings[name]
        assert isinstance(fn, ast.FunctionDef)
        assert fn.args.args[position].arg == 'state'
        parents = {child: parent for parent in ast.walk(fn) for child in ast.iter_child_nodes(parent)}
        keys, transfers = set(), []
        for node in ast.walk(fn):
            if not isinstance(node, ast.Name) or node.id != 'state' or not isinstance(node.ctx, ast.Load):
                continue
            parent = parents[node]
            if isinstance(parent, ast.Subscript) and parent.value is node:
                assert isinstance(parent.slice, ast.Constant) and isinstance(parent.slice.value, str)
                assert parent.slice.value != 'cos_sin'
                keys.add(parent.slice.value)
            else:
                assert isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name)
                callee = parent.func.id
                assert callee in argument_positions and callee != 'kernel_fn'
                assert parent.args[argument_positions[callee]] is node
                transfers.append(callee)
        summary.append(dict(function=name, root_keys=sorted(keys), whole_state_transfers=transfers))
    # No root-state aliases, dynamic indexing or whole-state external escapes
    # are accepted by the source walk above. Substates retain every old value.
    report = dict(source_sha256=SHA, final_runtime_bindings_audited=True,
                  no_dynamic_root_state_indexing=True, no_whole_root_state_escape=True,
                  no_root_state_aliases=True, cos_sin_unread_in_all_root_state_consumers=True,
                  functions=summary, cpu_only=True, gpu_verified=False)
    data = (json.dumps(report, indent=2) + '\n').encode()
    target = FOLDER / 'reachability-audit.json'
    if target.exists():
        assert target.read_bytes() == data, 'Frozen audit collision'
    else:
        target.write_bytes(data)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
