"""Build a readable, standalone gate ceiling diagnostic from CLEAN decode_v1.

Only Python comments/blank lines OUTSIDE string literals are compacted. AST
equality proves that no embedded kernel source or executable base code changes.
"""
import ast
import io
from pathlib import Path
import tokenize
from gate_build_flags import pin_inline_builds


def compact(source):
    tokens = [t for t in tokenize.generate_tokens(io.StringIO(source).readline)
              if t.type != tokenize.COMMENT or t.start[0] <= 13]
    protected = set()
    for t in tokens:
        if t.type == tokenize.STRING and t.end[0] > t.start[0]:
            protected.update(range(t.start[0], t.end[0] + 1))
    text = tokenize.untokenize(tokens)
    result = ''.join(line if n in protected else line.rstrip() + '\n'
                     for n, line in enumerate(text.splitlines(keepends=True), 1)
                     if n in protected or line.strip())
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(result))
    return result


def build(root):
    base = compact((root / 'kb/v1/decode_v1.py').read_text())
    # Copy ONLY the public-contract synthetic-input helper, not the encoded
    # chain module or any of its kernels. This function is ordinary source.
    source = (root / 'kb/exp/decode_chain.py').read_text()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_ch_inputs')
    inputs = ast.get_source_segment(source, node).replace('def _ch_inputs(', 'def _dg_inputs(', 1)
    # One-request scored shapes use fixed ctx=80000, rather than agentic spread.
    inputs = inputs.replace('seq = torch.randint(74000, 86001, (B,), generator=gen, device=dev)',
                            'seq = (torch.full((1,), 80000, dtype=torch.int64, device=dev) if B == 1 else\n'
                            '               torch.randint(74000, 86001, (B,), generator=gen, device=dev))')
    # The native B=1 speculative prologue uses page-table WIDTH, not seq_lens.
    # Match the scored 80000 context in both places; leaving width=86000 would
    # time a speculative miss on every replay, unlike the actual scored shapes.
    inputs = inputs.replace('H, per = st["qkv"].shape[1], 86000 + L',
                            'H, per = st["qkv"].shape[1], (80000 if B == 1 else 86000) + L')
    impl = (root / 'kb/exp/gate_diag_impl.py').read_text()
    out = pin_inline_builds(base + '\n\n' + inputs + '\n\n' + impl)
    assert len(out.encode()) < 1_000_000, len(out.encode())
    ast.parse(out)
    return out


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    out = build(root)
    target = root / 'kb/exp/decode_gate_diag.py'
    target.write_text(out)
    print(f'{target.relative_to(root)}: {len(out.encode())} bytes')
