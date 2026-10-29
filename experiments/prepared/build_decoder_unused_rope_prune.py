"""Bypass an unread preparation-only RoPE table; retain every runtime function."""
import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = 'df68deb60a132159d630928d19d93bcb9513de86b98a8c9517cfadd5e2e2f731'


def main():
    path = ROOT / 'results/decoder-router-prepare-prune-20261029/submission.py'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT
    tree, lines = ast.parse(raw), raw.splitlines(keepends=True)
    uses = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == 'cos_sin']
    assert len(uses) == 1, 'Any other exact-key reference requires a new reachability review'
    base = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    state = next(n for n in base.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'state' for t in n.targets))
    assert isinstance(state.value, ast.Dict)
    index = next(i for i, k in enumerate(state.value.keys) if k is uses[0])
    value = state.value.values[index]
    assert ast.unparse(value) == 'torch.cat((freqs.cos(), freqs.sin()), dim=-1)'
    removed = [n for n in base.body if
               isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id in ('inv_freq', 'freqs') or
               isinstance(n, ast.Delete) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
               and n.targets[0].id == 'freqs']
    assert len(removed) == 3
    definitions = [n.targets[0].id for n in removed if isinstance(n, ast.Assign)]
    assert definitions == ['inv_freq', 'freqs']
    assert [n.id for n in ast.walk(base) if isinstance(n, ast.Name) and n.id in ('freqs', 'inv_freq')].count('freqs') == 4
    # Exact bounds from the declared table dimensions and FP32 construction.
    declarations = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign)
                    and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)}
    rope_positions = declarations['ROPE_POSITIONS']
    assert ast.unparse(rope_positions) == '1 << 21'
    assert 'torch.arange(0, 64, 2' in ast.unparse(removed[0])
    assert 'dtype=torch.float' in ast.unparse(removed[0]) and 'dtype=torch.float' in ast.unparse(removed[1])

    edits = []
    for n in removed:
        assert n.lineno == n.end_lineno
        a = sum(map(len, lines[:n.lineno - 1]))
        b = a + len(lines[n.lineno - 1])
        edits.append((a, b, b''))
    a = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
    b = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
    edits.append((a, b, b'None'))
    candidate = raw
    for a, b, replacement in sorted(edits, reverse=True):
        candidate = candidate[:a] + replacement + candidate[b:]
    revised = ast.parse(candidate)
    expected = copy.deepcopy(tree)
    target = next(n for n in expected.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    removed_lines = {n.lineno for n in removed}
    target.body = [n for n in target.body if n.lineno not in removed_lines]
    target_state = next(n for n in target.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'state' for t in n.targets))
    target_state.value.values[index] = ast.Constant(value=None)
    assert ast.dump(revised) == ast.dump(expected)
    # Runtime entry and every helper/preparation wrapper remain exact. The
    # single changed base prepare keeps the same state keys and all norms.
    assert all(ast.dump(a) == ast.dump(b) for a, b in zip(tree.body, revised.body) if a is not base)
    changed = next(n for n in revised.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
    assert not any(isinstance(n, ast.Name) and n.id in ('inv_freq', 'freqs') for n in ast.walk(changed))
    digest = hashlib.sha256(candidate).hexdigest()
    report = dict(source_sha256=digest, parent_sha256=PARENT, bytes=len(candidate),
                  exact_cos_sin_key_occurrences=1, unused_state_key_retained_as_none=True,
                  removed_base_prepare_statements=3, all_other_module_ast_exact=True,
                  all_runtime_functions_exact=True, all_native_attention_and_fallback_source_exact=True,
                  table_positions=1 << 21, table_columns=64, element_bytes=4,
                  persistent_table_bytes=(1 << 21) * 64 * 4,
                  table_expression_tensor_payload_peak_bytes=(1 << 21) * 32 * 4 * 5,
                  peak_scope='freqs + separate cos/sin + concatenated output; excludes allocator/workspace and other state',
                  peak_is_representation_arithmetic_not_measurement=True,
                  preparation_savings_s=None, gpu_verified=False, timed_speedup_claim=False)
    folder = ROOT / 'results/decoder-unused-rope-prune-20261029'
    folder.mkdir(exist_ok=True)
    for name, data in [('submission.py', candidate), ('cpu-source-audit.json', (json.dumps(report, indent=2) + '\n').encode())]:
        target = folder / name
        if target.exists():
            assert target.read_bytes() == data, ('Frozen collision', name)
        else:
            target.write_bytes(data)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
