"""Reproduce the single-file experiment from the pinned incumbent (no imports)."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "board_best/moe_m3_scored_2.3797.py"
DEST = ROOT / "candidates/moe_verify_sync.py"


def once(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new)


def build():
    original = BASE.read_text()
    tree = ast.parse(original)
    # Preserve the complete incumbent, including its existing selector and
    # all non-verify dispatch. The new experiment does not tune those builds.
    source = original
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(x, ast.Name) and x.id == "CUDA_SRC" for x in n.targets))
    cuda = ast.literal_eval(assignment.value)
    cuda = once(cuda, "#define OKC_WGSNX 3", "#ifndef OKC_VERIFY_SYNC\n#define OKC_VERIFY_SYNC 0\n#endif\n#define OKC_WGSNX 3")
    core = (ROOT / "candidates/moe_verify_sync_core.cuh").read_text()
    cuda = once(cuda, "template <bool PB>\n__device__ __forceinline__ void wgs_phase(",
                core + "\ntemplate <bool PB>\n__device__ __forceinline__ void wgs_phase(")
    cuda = once(cuda, "      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;",
                "#if OKC_VERIFY_SYNC\n"
                "      case 1: wgs_seg_sync8<PB>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n"
                "#else\n"
                "      case 1: wgs_seg<PB, 1>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n"
                "#endif")
    # Kept correctness fix from ledger row 73 / job 2cba38a5, in both arms.
    # Without it the incumbent cannot replay an all-remote route buffer.
    cuda = once(cuda, "  wgs_prefix(S, nj);\n  wgs_range(S, nj, 0,",
                "  if (nj == 0) {\n"
                "    S.wgr[0] = S.wgr[1] = S.wgr[2] = S.wgr[3] = 0;\n"
                "    return;\n"
                "  }\n"
                "  wgs_prefix(S, nj);\n  wgs_range(S, nj, 0,")
    cuda += """
// Untimed resource guard. Both arms query the actual BIG kernel used at M>=128.
std::vector<int64_t> okc_verify_resources() {
  cudaFuncAttributes a{};
  cudaError_t err = cudaFuncGetAttributes(&a, okc::moe_fused<true>);
  TORCH_CHECK(err == cudaSuccess, "cudaFuncGetAttributes: ", cudaGetErrorString(err));
  return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)okc_smem_bytes()};
}
"""
    old_assignment = original.splitlines()[assignment.lineno - 1]
    assert "'''" not in cuda
    source = once(source, old_assignment, "CUDA_SRC = r'''" + cuda + "'''")
    source = once(source, "void okc_set_pdl(int64_t on);", "void okc_set_pdl(int64_t on);\nstd::vector<int64_t> okc_verify_resources();")
    source = once(source, 'functions=["okc_run", "okc_set_pdl"]',
                  'functions=["okc_run", "okc_set_pdl", "okc_verify_resources"]')
    source = once(source, "okc_moe_w4a16_v216x_pair1dq8_", "okc_moe_w4a16_verify_sync8_v1_")
    source = once(source, '    ext = _ext(_MOE_SEL.get(M) or _moe_default(M, sfx))',
                  '    ext = _ext(_MOE_SEL.get(M) or _moe_default(M, sfx))\n'
                  '    if M in (128, 256):\n'
                  '        ext = state.get("verify_sync", {}).get(M, ext)\n'
                  )
    source = once(source, "import torch\n", "import torch\nimport math\n")
    source = once(source, "# Local changes: remove diagnostics;", "# Local changes (Nov 04): gated verify-only N=8 mma.sync; J==0 guard;\n# retain incumbent NVFP4 weights/math and all other shapes.\n# Original local changes: remove diagnostics;")
    source += "\n\n" + (ROOT / "scripts/moe_verify_sync_gate.py").read_text()
    source = "\n".join(line.rstrip() for line in source.splitlines()) + "\n"
    ast.parse(source)
    return source


if __name__ == "__main__":
    source = build()
    DEST.write_text(source)
    print(f"{DEST}: {len(source.encode())} bytes")
