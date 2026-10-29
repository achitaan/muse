"""Build the Prefill FA3-selector candidate: retained bytes plus one appended add-on."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = ROOT / 'best/prefill_verified.py'
PARENT_SHA = '93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7'
ADDON = ROOT / 'experiments/prepared/prefill_fa_select_addon.py'
OUT = ROOT / 'candidates/prefill_fa_select.py'


def main():
    parent = PARENT.read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA
    addon = ADDON.read_bytes().replace(b'\r\n', b'\n')
    raw = parent + (b'' if parent.endswith(b'\n') else b'\n') + addon
    tree = ast.parse(raw.decode('utf-8'))
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    assert names.count('kernel_fn') >= 1 and names[-1] == 'prepare'
    assert len(raw) <= 1_000_000
    OUT.write_bytes(raw)
    print(json.dumps({'parent_sha256': PARENT_SHA, 'addon_sha256': hashlib.sha256(addon).hexdigest(),
                      'candidate_sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}, indent=2))


if __name__ == '__main__':
    main()
