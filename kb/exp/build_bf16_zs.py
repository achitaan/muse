"""achitaan: materialize a readable standalone submission, never encode source."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:80], text.count(old))
    return text.replace(old, new, 1)


def build():
    base = (HERE / "bf16_v2.py").read_text(encoding="utf-8")
    base = replace_once(base,
        "'dtstart': _dtstart, 'dtplans': _dtplans, 'dtarmcall': _dtarmcall, 'flushext': _flushext}",
        "'dtstart': _dtstart, 'dtplans': _dtplans, 'dtarmcall': _dtarmcall, 'flushext': _flushext,\n"
        "            'zs_gv_source': GVCU}  # achitaan: source export only; base instances unchanged")
    base = replace_once(base,
        "return {'prepare': prepare, 'kernel_fn': kernel_fn, 'warm': _get_ext}",
        "return {'prepare': prepare, 'kernel_fn': kernel_fn, 'warm': _get_ext,\n"
        "            'zs_cg_source': RGEMV_SRC}  # achitaan: source export only; base instances unchanged")
    codec = (HERE / "zs_codec.py").read_text(encoding="utf-8").split("def self_test() -> None:", 1)[0]
    arms = (HERE / "zs_pooler.py").read_text(encoding="utf-8")
    source = base + "\n\n# achitaan: standalone lossless weight codec helpers.\n" + codec + "\n" + arms
    ast.parse(source)
    data = source.encode("utf-8")
    assert len(data) < 900000 and b"\r" not in data
    assert not any(t.encode() in data for t in ("base64", "b64decode", "b85", "zlib.decompress", "lzma"))
    (HERE / "bf16_zs.py").write_bytes(data)
    print("bf16_zs.py AST/LF/source checks passed:", len(data), "bytes")


if __name__ == "__main__":
    build()
