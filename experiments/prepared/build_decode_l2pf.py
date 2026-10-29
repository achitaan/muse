"""Build the L2-prefetch Decode candidate from the retained D9 bytes plus one add-on."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = ROOT / 'best/decoder_verified.py'
PARENT_SHA = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
ADDON = ROOT / 'experiments/prepared/decode_l2pf_addon.py'
OUT = ROOT / 'candidates/decode_l2pf.py'
LIMIT = 1_000_000  # official upload cap (422 above it)


def kernel_body(text):
    tree = ast.parse(text)
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'kernel_fn'][-1]
    return text.splitlines()[node.lineno - 1:node.end_lineno]


def main():
    parent = PARENT.read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA, 'retained D9 bytes changed'
    addon = ADDON.read_bytes().replace(b'\r\n', b'\n')
    raw = parent + (b'' if parent.endswith(b'\n') else b'\n') + addon
    text = raw.decode('utf-8')
    ast.parse(text)
    # The new kernel_fn must equal the retained one except the two prefetch lines
    # and the dropped unused `dense` local.
    old = [l for l in kernel_body(parent.decode('utf-8')) if l != " dense = state['dense']"]
    new = [l for l in kernel_body(text) if l not in (' pf = _pfbeg(state, cfg, tokens)', ' if pf is not None:', '  pf[0].wait_stream(pf[1])')]
    assert old == new, 'kernel_fn drifted from the retained body'
    assert len(raw) <= LIMIT, f'{len(raw)} bytes exceeds the {LIMIT}-byte upload cap'
    OUT.write_bytes(raw)
    print(json.dumps({'parent_sha256': PARENT_SHA, 'addon_sha256': hashlib.sha256(addon).hexdigest(),
                      'candidate': str(OUT.relative_to(ROOT)), 'candidate_sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw), 'headroom_bytes': LIMIT - len(raw), 'kernel_fn_identical_except_prefetch': True},
                     indent=2))


if __name__ == '__main__':
    main()
