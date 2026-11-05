"""Build standalone decode_1h; incumbent helper CUDA is losslessly deduplicated."""
import ast
import difflib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DONOR = Path('/Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse-fp16/kb/exp/decode_bundle.py')


def extract(source, names):
    nodes = [n for n in ast.parse(source).body
             if (ast.unparse(n.targets[0]) if isinstance(n, ast.Assign)
                 else getattr(n, 'name', None)) in names]
    assert len(nodes) == len(names)
    return '\n\n'.join(ast.get_source_segment(source, n) for n in nodes)


def build():
    base = (HERE.parent / 'v1/decode_v1.py').read_text()
    nodes = ast.parse(base).body
    main = next(ast.literal_eval(n.value) for n in nodes if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == 'MOE_CUDA_SRC')
    node = next(n for n in nodes if isinstance(n, ast.Assign)
                and ast.unparse(n.targets[0]) == "_MA_ALT_SRCS['moe_helper0']")
    helper = ast.literal_eval(node.value)
    a, b = main.splitlines(keepends=True), helper.splitlines(keepends=True)
    edits = [(i, j, ''.join(b[k:l])) for op, i, j, k, l in
             difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != 'equal']
    restored = list(a)
    for i, j, text in reversed(edits):
        restored[i:j] = text.splitlines(keepends=True)
    assert ''.join(restored) == helper
    replacement = '''# Lossless readable deduplication; incumbent CUDA bytes remain identical.
def _d1h_cuda_lines(source, edits):
    lines = source.splitlines(keepends=True)
    for lo, hi, text in reversed(edits):
        lines[lo:hi] = text.splitlines(keepends=True)
    return ''.join(lines)

_MA_ALT_SRCS['moe_helper0'] = _d1h_cuda_lines(MOE_CUDA_SRC, (
'''
    for i, j, text in edits:
        assert "'''" not in text
        replacement += f"    ({i}, {j}, r'''{text}'''),\n"
    replacement += '))'
    base = base.replace(ast.get_source_segment(base, node), replacement, 1)
    donor = DONOR.read_text()
    names = {'AH_CUDA', 'AH_CPP', '_AHLaunchRejected', '_bd_ah_workspace',
             '_bd_ah_assert_launch', '_bd_ah_launch', '_bd_inputs', '_bd_compatible', '_bd_key'}
    addition = extract(donor, names).replace('_bd_', '_d1h_').replace('_BD_AH_WS', '_D1H_WS').replace('_BD_MODULES', '_D1H_MODULES').replace('_BD_SHAPES', '_D1H_SHAPES')
    source = base + '\n\n# One-head attention: byte-identical to the independently checked attention-board kernel.\n' + addition
    source += '\n\n' + (HERE / 'decode_1h_arm.py').read_text()
    ast.parse(source)
    assert len(source.encode()) < 1_000_000, len(source.encode())
    (HERE / 'decode_1h.py').write_text(source)
    print(f'decode_1h.py: {len(source.encode())} bytes; all incumbent CUDA restored exactly')


if __name__ == '__main__':
    build()
