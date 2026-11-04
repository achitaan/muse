"""Rebuild the standalone candidate from the read-only VMS5 reference.

Usage: python3 scripts/build_moe_vmsgate.py /path/to/moe_vms5.py
The candidate's handwritten gate after GATE_MARKER is preserved. No other
checkout is modified. Run the CPU preflight after regenerating.
"""
import ast
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "candidates/moe_vmsgate.py"
GATE_MARKER = "# === MOE_VMSGATE: distribution diagnostic and conservative static fallback ==="


def build(reference: str, gate: str) -> str:
    tree = ast.parse(reference)
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("VMS_SRC", "VMS_CPP"):
                values[node.targets[0].id] = ast.literal_eval(node.value)
    prefix = reference[:reference.index("# =====================================================================================================================\n# achitaan: VMS / VML")]
    prefix = prefix[prefix.index("# ps1:"):]
    prefix = "# okc: problem=muse_moe_w4a16\n# vmsgate: VMS5 incumbent + control-only VMS/VML + 32-draw null-controlled diagnostic.\n" + prefix
    # Match host and device dialects; no kernel arithmetic or incumbent selector changes.
    prefix = prefix.replace('extra_cuda_cflags=["-O3",', 'extra_cflags=["-O3", "-std=c++20"],\n            extra_cuda_cflags=["-O3", "-std=c++20",')
    src = values["VMS_SRC"]
    start = src.index("constexpr int NVAR = 8;")
    stop = src.index("__host__ __device__ constexpr bool var_fits", start)
    src = src[:start] + "constexpr int NVAR = 2;\nconstexpr Var VAR[NVAR] = {{0, 3, 1, 0}, {0, 4, 1, 0}};\n" + src[stop:]
    start = src.index("template <bool PB>\nKern pick(")
    stop = src.index("static void ensure_smem", start)
    src = src[:start] + """template <bool PB>
Kern pick(int diag, int v) {
  constexpr int CTL = MAXT <= 16 ? 1 : 0;
  if (diag != 0 || v != CTL) return nullptr;
  return kern_of<PB, 0, CTL>();
}
""" + src[stop:]
    # Keep only the established control instantiated. Dead templated experimental
    # paths remain readable, but no stream-only / math-only kernel is compiled.
    src = src.replace("// achitaan: VMS / VML (iteration 5)", "// achitaan: VMS / VML (iteration 5, control-only build)")
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in values.items()}
    generated = "\n# Control source inherited from moe_vms5.py; arithmetic unchanged.\n"
    generated += "# Original embedded-source SHA256: " + str(hashes) + "\n"
    # A literal rather than repr keeps the CUDA source human-readable.
    generated += "VMS_SRC = r'''" + src + "'''\n\nVMS_CPP = r'''" + values["VMS_CPP"] + "'''\n\n"
    result = prefix + generated + GATE_MARKER + gate.split(GATE_MARKER, 1)[1]
    ast.parse(result)
    return result


if __name__ == "__main__":
    TARGET.write_text(build(Path(sys.argv[1]).read_text(), TARGET.read_text()))
    print(f"generated {TARGET}: {TARGET.stat().st_size} bytes")
