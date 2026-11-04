"""Development-only assembler; the emitted candidate is self-contained."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
base = (HERE.parent / "moe_headroom/moe_headroom.py").read_bytes()
cuda = (HERE / "restream.cu").read_text()
host = (HERE / "host.py").read_text()
assert "'''" not in cuda
(HERE / "moe_restream.py").write_bytes(base + b"\n\n# TQ64 experimental variant, separately compiled and gated.\n_RS_CUDA = r'''" + cuda.encode() + b"'''\n\n" + host.encode())
