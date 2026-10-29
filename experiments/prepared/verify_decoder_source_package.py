"""Statically verify the package against D9; no submission code is executed."""
import argparse
import ast
import base64
import copy
import hashlib
import json
from pathlib import Path
import zlib

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').is_file())
PARENT = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
TARGET = '_native_moe_CUDA_SRC'
OUTPUT = ROOT / '.tools/decoder-source-package.py'
REPORT = ROOT / '.tools/decoder-source-package-source-audit.json'
FROZEN = ROOT / 'results/decoder-source-package-20261029'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def assignment(tree, name):
    matches = [n for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(matches) == 1, ('Ambiguous assignment', name)
    return matches[0]


def is_loader_import(node):
    return isinstance(node, ast.Import) and [(x.name, x.asname) for x in node.names] == [
        ('base64', '_zsrc_b64'), ('zlib', '_zsrc_zlib')]


def verify_program(parent, candidate):
    """Use the pinned checked program as the authority, not the builder report."""
    old, new = ast.parse(parent), ast.parse(candidate)
    original, packed = assignment(old, TARGET), assignment(new, TARGET)
    call = packed.value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == '_zsrc'
    assert len(call.args) == 1 and not call.keywords
    payload = ast.literal_eval(call.args[0])
    assert isinstance(payload, str)
    expected = ast.literal_eval(original.value).encode('utf-8')
    compressed = base64.b64decode(payload, validate=True)
    decoder = zlib.decompressobj()
    decoded = decoder.decompress(compressed, len(expected) + 1)
    assert decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    assert decoded == expected, 'Decoded CUDA changed'
    old_helper = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == '_zsrc')
    helpers = [n for n in new.body if isinstance(n, ast.FunctionDef) and n.name == '_zsrc']
    imports = [n for n in new.body if is_loader_import(n)]
    assert len(helpers) == len(imports) == 1
    assert ast.dump(helpers[0]) == ast.dump(old_helper)
    assert new.body.index(imports[0]) < new.body.index(helpers[0]) < new.body.index(packed)
    # The helper's only module-scope bindings are its definition and its import
    # aliases. Relocation cannot replace another earlier binding of these names.
    names = {'_zsrc', '_zsrc_b64', '_zsrc_zlib'}
    for node in old.body:
        if node is old_helper or is_loader_import(node):
            continue
        if isinstance(node, ast.Assign):
            assert not any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    remove = {id(helpers[0]), id(imports[0])}
    new.body = [n for n in new.body if id(n) not in remove]
    packed.value = copy.deepcopy(original.value)
    for index, node in enumerate(old.body):
        if node is old_helper or is_loader_import(node):
            new.body.insert(index, copy.deepcopy(node))
    assert ast.dump(new) == ast.dump(old), 'Program changed after literal expansion'
    assert candidate.splitlines()[0] == parent.splitlines()[0]
    assert len(candidate) < 1_000_000
    return dict(decoded_cuda_sha256=sha(expected), decoded_cuda_bytes=len(expected),
                exact_expanded_module_ast=True, existing_loader_ast_exact=True,
                all_runtime_functions_and_fallbacks_unchanged=True,
                all_prepare_functions_unchanged=True, all_expanded_cuda_unchanged=True)


def mutation_probes(parent, candidate):
    # Expected behavior comes from exact equivalence to checked D9, rather than
    # a new expected numerical output or a weakened track tolerance.
    tree = ast.parse(candidate)
    mutations = []
    changed = copy.deepcopy(tree)
    call = assignment(changed, TARGET).value
    cuda = zlib.decompress(base64.b64decode(call.args[0].value))
    call.args[0].value = base64.b64encode(zlib.compress(cuda + b'\n', 9)).decode()
    mutations.append(('decoded_cuda_byte', ast.unparse(changed).encode()))
    changed = copy.deepcopy(tree)
    loader = next(n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name == '_zsrc')
    loader.body = [ast.Return(value=ast.Constant(value='wrong'))]
    mutations.append(('loader_body', ast.unparse(changed).encode()))
    changed = copy.deepcopy(tree)
    assignment(changed, 'ROUTER_FP32_MAX_TOKENS').value = ast.Constant(value=9)
    mutations.append(('precision_dispatch_boundary', ast.unparse(changed).encode()))
    rejected = []
    for name, raw in mutations:
        try:
            verify_program(parent, raw)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError(('Mutation escaped the audit', name))
    return rejected


def freeze_file(path, raw):
    if path.exists():
        assert path.read_bytes() == raw, ('Refusing to replace immutable evidence', str(path))
    else:
        path.write_bytes(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true', help='Save immutable CPU evidence; launches nothing')
    args = parser.parse_args()
    parent, candidate = (ROOT / 'best/decoder_verified.py').read_bytes(), OUTPUT.read_bytes()
    report = json.loads(REPORT.read_bytes())
    assert sha(parent) == PARENT == report['parent_sha256']
    assert sha(candidate) == report['source_sha256']
    checks = verify_program(parent, candidate)
    restored = candidate
    for edit in sorted(report['patches'], key=lambda p: p['candidate_start'], reverse=True):
        a, b = edit['candidate_start'], edit['candidate_end']
        old = parent[edit['parent_start']:edit['parent_end']]
        assert sha(restored[a:b]) == edit['replacement_sha256']
        assert sha(old) == edit['original_sha256']
        restored = restored[:a] + old + restored[b:]
    assert restored == parent, 'Inverse must recover every parent byte, including licenses'
    audit = dict(source_sha256=sha(candidate), parent_sha256=PARENT,
                 inverse_parent_bytes_exact=True, mutation_rejections=mutation_probes(parent, candidate),
                 bytes=len(candidate), source_headroom_bytes=1_000_000 - len(candidate),
                 submission_code_executed=False, gpu_jobs_launched=0, evaluated=False, **checks)
    audit_raw = (json.dumps(audit, indent=2) + '\n').encode()
    (ROOT / '.tools/decoder-source-package-cpu-audit.json').write_bytes(audit_raw)
    if args.freeze:
        directory = FROZEN / sha(candidate)
        directory.mkdir(parents=True, exist_ok=True)
        freeze_file(directory / 'submission.py', candidate)
        freeze_file(directory / 'source-audit.json', REPORT.read_bytes())
        freeze_file(directory / 'cpu-audit.json', audit_raw)
        audit['frozen_directory'] = str(directory.relative_to(ROOT))
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
