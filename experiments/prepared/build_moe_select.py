"""Build the MoE build-selector candidate: retained bytes plus one appended add-on."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'STATE.json').exists())
PARENT = ROOT / 'best/moe_verified.py'
PARENT_SHA = '331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
ADDON = ROOT / 'experiments/prepared/moe_select_addon.py'
OUT = ROOT / 'candidates/moe_select.py'
NEEDED = ('HIDDEN', 'CHUNK_TOKENS', 'TINY_MAX_M', 'TINY2_MAX_M', 'MID_MAX_M',
          'TINY_DEFINES', 'TINY2_DEFINES', 'MID_DEFINES', '_main_defines', '_ext', 'prepare', 'kernel_fn')
RETAINED_CHAIN = '''    if TINY_DEFINES and M <= TINY_MAX_M:
        ext = _ext(tuple(TINY_DEFINES) + sfx)
    elif TINY2_DEFINES and M <= TINY2_MAX_M and (sfx or "-DOKC_WGS=1" not in tuple(TINY2_DEFINES)):
        ext = _ext(tuple(TINY2_DEFINES) + sfx)  # (the WGS build needs the PRMT scale decode: else MID/MAIN)
    elif MID_DEFINES and M <= MID_MAX_M:
        ext = _ext(tuple(MID_DEFINES) + sfx)
    else:
        ext = _ext(tuple(_main_defines(sfx)) + sfx)'''


def main():
    parent = PARENT.read_bytes()
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA
    ptext = parent.decode('utf-8')
    assert ptext.count(RETAINED_CHAIN) == 1, 'retained dispatch chain differs from the encoded default'
    tree = ast.parse(ptext)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    defined |= {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    missing = [n for n in NEEDED if n not in defined]
    assert not missing, missing
    addon = ADDON.read_bytes().replace(b'\r\n', b'\n')
    raw = parent + (b'' if parent.endswith(b'\n') else b'\n') + addon
    t2 = ast.parse(raw.decode('utf-8'))
    fns = [n.name for n in t2.body if isinstance(n, ast.FunctionDef)]
    assert fns[-1] == 'prepare' and fns[-2] == 'kernel_fn'
    assert len(raw) <= 1_000_000
    OUT.write_bytes(raw)
    print(json.dumps({'parent_sha256': PARENT_SHA, 'addon_sha256': hashlib.sha256(addon).hexdigest(),
                      'candidate_sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}, indent=2))


if __name__ == '__main__':
    main()
