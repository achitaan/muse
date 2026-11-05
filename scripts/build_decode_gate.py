"""Readable clean-base output-gate overlap experiment. CPU build only by default."""
import ast
from pathlib import Path
from build_decode_gate_diag import compact, build as build_diag
from gate_build_flags import pin_inline_builds


def build(root):
    base = compact((root / 'kb/v1/decode_v1.py').read_text())
    # Use editable sources, never a stale generated diagnostic file.
    diag = build_diag(root)
    tree = ast.parse(diag)
    helpers = []
    for name in ('_dg_inputs', '_dg_sk_source', '_dg_capture', '_dg_cache', '_dg_reset', '_dg_pair'):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        source = ast.get_source_segment(diag, node)
        if name == '_dg_sk_source':
            source = source.replace('def _dg_sk_source(', 'def _gg_original_sk_source(', 1)
        if name == '_dg_pair':
            source = source.replace('_dg_stats.', '_gg_stats.').replace('_dg_time.', '_gg_time.')
        helpers.append(source)
    impl = (root / 'kb/exp/gate_overlap_impl.py').read_text()
    out = pin_inline_builds(base + '\n\n' + '\n\n'.join(helpers) + '\n\n' + impl)
    ast.parse(out)
    assert len(out.encode()) < 1_000_000, len(out.encode())
    return out


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    source = build(root)
    path = root / 'kb/exp/decode_gate.py'
    path.write_text(source)
    print(f'{path.relative_to(root)}: {len(source.encode())} bytes; not submitted')
