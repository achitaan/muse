"""Reproducibly derive the experimental WGS port from the retained source.

This produces an experimental artifact, never a production dispatch decision.
"""
import ast
import hashlib
from pathlib import Path

BASE_SHA = "f72cdf2b5438c70369b0fb823e171f4022b22b97f0dff66c6170dfe3aaf845fb"


def build():
    root = Path(__file__).resolve().parents[1]
    source = root / "board_best/moe_m3_scored_2.3797.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == BASE_SHA
    tree = ast.parse(source.read_text())
    src = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))

    def replace(old, new):
        nonlocal src
        assert src.count(old) == 1, (old[:80], src.count(old))
        src = src.replace(old, new)

    replace("#include <cuda_bf16.h>", "#include <cuda_bf16.h>\n#include <cuda.h>")
    replace("struct Params {", """struct Params {
  const CUtensorMap* maps;
  __nv_bfloat16* gemm1_debug;
  int* local_out;
  int phases;
""")
    replace("// ---------------------------------------------------------------- routing prologue (every CTA, identical result)",
            '#include "native_marlin_tiles.cuh"\n\n'
            '// ---------------------------------------------------------------- routing prologue (every CTA, identical result)')
    assert src.count("for (int pb = 0; pb < 2; ++pb) {") == 2
    src = src.replace("for (int pb = 0; pb < 2; ++pb) {", "for (int pb = 0; pb < p.phases; ++pb) {")
    # x producer has its own phase loop with a different declaration.
    start = src.index("__device__ void wgs_xproducer")
    end = src.index("struct WgsPend", start)
    section = src[start:end].replace("pb < 2", "pb < p.phases")
    src = src[:start] + section + src[end:]
    start = src.index("      const unsigned char* src = base +", src.index("void wgs_wproducer"))
    end = src.index("      if (++ws", start)
    src = src[:start] + """      const int row = ub * (pb ? 256 : 128);
      const int kg = S.jE[j] * 256 + st * 16;
      unsigned char* dst = ring + ws * UNIT_BYTES;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\\n"
          :: "r"(smem_u32(bar)), "r"(UNIT_BYTES) : "memory");
      native_tma(dst, p.maps + pb * 2, row, kg, bar);
      native_tma(dst + 32768, p.maps + pb * 2 + 1, row, kg, bar);
      if (!pb) {
        native_tma(dst + 16384, p.maps, row + 4096, kg, bar);
        native_tma(dst + 34816, p.maps + 1, row + 4096, kg, bar);
      }
""" + src[end:]
    replace("const unsigned char* pc0 = ring + wg * WGS_PIECE;", "const unsigned char* pc0 = ring;")
    start = src.index("    // codes and scales one 4-step group", src.index("void wgs_seg"))
    end = src.index("        wg_fence();", start)
    src = src[:start] + """#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
#pragma unroll
      for (int h = 0; h < 4 / GS; ++h) {
        const int ci = lg * (4 / GS) + h;
#pragma unroll
        for (int jj = 0; jj < GS; ++jj) {
          native_regs<PB>(pc, lg * 4 + h * GS + jj, wg, wp, lane, af[ci & 1][jj]);
        }
        if (ci == 16 / GS - 1 && lane == 0) mbar_arrive1(S.bars + BW_E + ws);
""" + src[end:]
    start = src.index("    for (int nt = 0; nt < NTT; ++nt)\n      asm volatile(\"red.global.add.v4.f32", src.index("void wgs_seg"))
    end = src.index("    // kind 3:", start)
    src = src[:start] + """    for (int nt = 0; nt < NTT; ++nt) {
#pragma unroll
      for (int component = 0; component < 4; ++component)
        native_add_preserve(fb + nt * 512 + component, d[nt][component]);
    }
""" + src[end:]
    # Preserve Marlin's *encoded* accumulator BF16 boundary, including subnormals.
    replace("__floats2bfloat162_rn(encoded_acc0 * TWO62, encoded_acc1 * TWO62)",
            "__floats2bfloat162_rn(encoded_acc0, encoded_acc1)")
    replace("__float2bfloat16_rn(encoded_scale * 0x1p-62f)", "__float2bfloat16_rn(encoded_scale)")
    # Only the active WGS epilogue changes; inactive board variants remain historical.
    start = src.index("void wgs_epi_tile")
    end = src.index("template <bool PB>", start + 30)
    section = src[start:end].replace("p.gsA[e] * TWO62", "p.gsA[e]").replace("p.gsB[e] * TWO62", "p.gsB[e]")
    section = section.replace("const float gt = achita_pair_9.x, up = achita_pair_9.y;", """const float gt = achita_pair_9.x, up = achita_pair_9.y;
        if (p.gemm1_debug) {
          const size_t slot = S.ent[jst + tk];
          p.gemm1_debug[slot * 8192 + col] = __float2bfloat16_rn(gt);
          p.gemm1_debug[slot * 8192 + 4096 + col] = __float2bfloat16_rn(up);
          p.inter[slot * 4096 + col] = __float2bfloat16_rn(SILU_DIV(gt, 1.f + __expf(-gt)) * up);
        }""")
    section = section.replace("const float w0 = __bfloat162float(__ldg(p.normw + n0)), w1 = __bfloat162float(__ldg(p.normw + n0 + 8));", "")
    section = section.replace("achita_pair_wgs.x * w0", "achita_pair_wgs.x").replace("achita_pair_wgs.y * w1", "achita_pair_wgs.y")
    src = src[:start] + section + src[end:]
    replace("  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  g0 =", "  if (!nj) { g0 = g1 = 0; return; }\n  const uint32_t tot = pb ? S.wtotB : S.wtot;\n  g0 =")
    replace("    wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);", "    if (p.phases == 2) wgs_phase<true>(p, S, ring, S.wgr[2], S.wgr[3], seq);")
    start = src.index("  // final: out[t] =")
    end = src.index("  DBG(5);", start)
    src = src[:start] + """  // Every route slot, including remote/padding, is initialized on every replay.
  if (p.phases == 2) {
    for (int slot = blockIdx.x * NT + tid; slot < p.M * 8; slot += gridDim.x * NT)
      p.local_out[slot] = (int)S.eid[slot];
    for (int i = blockIdx.x * NT + tid; i < p.M * 8 * HD / 2; i += gridDim.x * NT) {
      const int slot = i / (HD / 2), col = 2 * (i % (HD / 2));
      const int q = S.pos[slot];
      const float a = q >= 0 ? p.ybuf[(size_t)q * HD + col] : 0.f;
      const float b = q >= 0 ? p.ybuf[(size_t)q * HD + col + 1] : 0.f;
      reinterpret_cast<__nv_bfloat162*>(p.out)[i] = __floats2bfloat162_rn(a, b);
    }
  }
""" + src[end:]
    src = src[:src.index("static int g_okc_pdl")]
    src += (Path(__file__).parent / "native_marlin_host.inc").read_text()
    # Keep all added source lines inside the contest's length restriction.
    lines = []
    for line in src.splitlines():
        while len(line) > 390:
            cut = line.rfind(", ", 0, 380)
            if cut < 0:
                raise ValueError("line requires manual wrapping: " + line[:100])
            lines.append(line[:cut + 1])
            line = "                 " + line[cut + 2:]
        lines.append(line.rstrip())
    output = Path(__file__).parent / "native_marlin_wgs.cu"
    notice = source.read_text().split('"""', 1)[0]
    notice = "\n".join(("//" + line[1:] if line.startswith("#") else line).rstrip()
                       for line in notice.splitlines())
    output.write_text(notice + "\n// Serving port: native Marlin TMA tiles, explicit route outputs,\n"
                      "// stock encoded accumulation, subnormal-preserving stream-K fixups.\n"
                      + "\n".join(lines) + "\n", newline="\n")
    print(output, hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
