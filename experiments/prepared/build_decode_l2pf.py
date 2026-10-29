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


def main():
    parent = PARENT.read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA, 'retained D9 bytes changed'
    addon = ADDON.read_bytes().replace(b'\r\n', b'\n')
    raw = parent + (b'' if parent.endswith(b'\n') else b'\n') + addon
    ast.parse(raw.decode('utf-8'))
    OUT.write_bytes(raw)
    print(json.dumps({'parent_sha256': PARENT_SHA, 'addon_sha256': hashlib.sha256(addon).hexdigest(),
                      'candidate': str(OUT.relative_to(ROOT)), 'candidate_sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw)}, indent=2))


if __name__ == '__main__':
    main()
