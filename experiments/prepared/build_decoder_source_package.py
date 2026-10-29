"""Package one pinned CUDA literal losslessly; never import the GPU submission."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').is_file())
PARENT = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
TARGET = '_native_moe_CUDA_SRC'
OUTPUT = ROOT / '.tools/decoder-source-package.py'
REPORT = ROOT / '.tools/decoder-source-package-source-audit.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def span(node, lines, *, whole_lines=False):
    start = sum(map(len, lines[:node.lineno - 1]))
    end = sum(map(len, lines[:node.end_lineno - 1]))
    if whole_lines:
        return start, end + len(lines[node.end_lineno - 1])
    return start + node.col_offset, end + node.end_col_offset


def build():
    raw = (ROOT / 'best/decoder_verified.py').read_bytes()
    if sha(raw) != PARENT:
        raise ValueError('Verified Decode changed: reassess the exact parent before building')
    tree, lines = ast.parse(raw), raw.splitlines(keepends=True)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == TARGET for t in n.targets))
    cuda = ast.literal_eval(assignment.value)
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_zsrc')
    imports = next(n for n in tree.body if isinstance(n, ast.Import)
                   and [(x.name, x.asname) for x in n.names]
                   == [('base64', '_zsrc_b64'), ('zlib', '_zsrc_zlib')])
    import_start, _ = span(imports, lines, whole_lines=True)
    _, helper_end = span(helper, lines, whole_lines=True)
    bootstrap = raw[import_start:helper_end]
    # Moving the already present import/helper ahead of this literal introduces
    # no dependency and keeps every CUDA byte, license and runtime function.
    assert len(ast.parse(bootstrap).body) == 2
    literal_start, literal_end = span(assignment.value, lines)
    insert_at, _ = span(assignment, lines, whole_lines=True)
    payload = base64.b64encode(zlib.compress(cuda.encode('utf-8'), 9)).decode('ascii')
    expression = ('_zsrc(' + repr(payload) + ')').encode('ascii')
    edits = [
        (insert_at, insert_at, bootstrap + b'\n', 'move_loader_before_first_use'),
        (literal_start, literal_end, expression, 'encode_one_cuda_literal'),
        (import_start, helper_end, b'', 'remove_old_loader_location'),
    ]
    result, cursor, patches = bytearray(), 0, []
    for start, end, replacement, kind in sorted(edits):
        assert cursor <= start <= end
        result.extend(raw[cursor:start])
        candidate_start = len(result)
        result.extend(replacement)
        patches.append(dict(kind=kind, parent_start=start, parent_end=end,
                            candidate_start=candidate_start, candidate_end=len(result),
                            original_sha256=sha(raw[start:end]), replacement_sha256=sha(replacement)))
        cursor = end
    result.extend(raw[cursor:])
    candidate = bytes(result)
    compile(candidate, '<decoder-source-package>', 'exec')  # Python compile only, no execution/imports.
    assert candidate.splitlines()[0] == raw.splitlines()[0]
    assert len(candidate) < 1_000_000
    report = dict(
        parent_sha256=PARENT, source_sha256=sha(candidate), parent_bytes=len(raw),
        bytes=len(candidate), saved_bytes=len(raw) - len(candidate),
        source_cap_bytes=1_000_000, source_headroom_bytes=1_000_000 - len(candidate),
        encoded_symbol=TARGET, decoded_cuda_bytes=len(cuda.encode('utf-8')),
        decoded_cuda_sha256=sha(cuda.encode('utf-8')), encoding='base64(zlib(level=9))',
        zlib_runtime=zlib.ZLIB_RUNTIME_VERSION, patches=patches,
        scope='source representation only; existing loader moved before its first new use',
        gpu_verified=False, evaluated=False, active_candidate_changed=False,
        preparation_time_improvement='unmeasured; no compilation or preparation work removed',
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_bytes(candidate)
    REPORT.write_bytes((json.dumps(report, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({k: v for k, v in report.items() if k != 'patches'}, indent=2))


if __name__ == '__main__':
    build()
