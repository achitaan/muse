"""Build a Decode candidate = retained D9 + attention split overrides proven in the Attention track.

usage: python experiments/prepared/build_decode_attn_tuned.py '<json list of [B, L, sliding, pairs, {override}]>'
Only the single `_native_attn__TUNED = {...}` line of D9 changes; everything else is byte-identical.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = ROOT / 'best/decoder_verified.py'
PARENT_SHA = 'd9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'
OUT = ROOT / 'candidates/decode_attn_tuned.py'
OLD = (b"_native_attn__TUNED = {(32, 8, True, 1): dict(n_splits=4, nc=1, coop=1), "
       b"(1, 1, False, 1): dict(nc=8, coop=1, n_splits=252)}")


def main():
    extra = json.loads(sys.argv[1])
    raw = PARENT.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PARENT_SHA
    assert raw.count(OLD) == 1
    entries = ["(32, 8, True, 1): dict(n_splits=4, nc=1, coop=1)", "(1, 1, False, 1): dict(nc=8, coop=1, n_splits=252)"]
    for (b, l, sliding, pairs, override) in extra:
        assert (b, l, sliding, pairs) not in ((32, 8, True, 1), (1, 1, False, 1))
        args = ', '.join(f'{k}={int(v)}' for (k, v) in override.items())
        entries.append(f'({int(b)}, {int(l)}, {bool(sliding)}, {int(pairs)}): dict({args})')
    new = ('_native_attn__TUNED = {' + ', '.join(entries) + '}').encode()
    out = raw.replace(OLD, new)
    ast.parse(out.decode('utf-8'))
    assert len(out) <= 1_000_000
    OUT.write_bytes(out)
    print(json.dumps({'parent_sha256': PARENT_SHA, 'candidate_sha256': hashlib.sha256(out).hexdigest(),
                      'bytes': len(out), 'tuned_line': new.decode()}, indent=2))


if __name__ == '__main__':
    main()
