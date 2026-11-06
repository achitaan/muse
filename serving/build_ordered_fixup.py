"""One bounded follow-up; derive separate sources without altering frozen v3."""
import hashlib
from pathlib import Path


def build():
    root = Path(__file__).resolve().parent
    src = (root / "native_marlin_wgs.cu").read_text()
    assert hashlib.sha256((root / "native_marlin_wgs.cu").read_bytes()).hexdigest() == "3eb38b8d658e3becd0d314c2ce4ad717786b9859c7570cee4392458411340736"
    out = root / "experiments/ordered_fixup"
    out.mkdir(parents=True, exist_ok=True)

    def replace(old, new):
        nonlocal src
        assert src.count(old) == 1, (old[:70], src.count(old))
        src = src.replace(old, new)

    # Double the old partial bank: each CTA owns two endpoints in each phase.
    # Counter indexing remains by the first contributor, exactly as in v3.
    src = src.replace("(size_t)2 * WGS_SLOTS * 4 * WGS_FIXW", "(size_t)2 * WGS_SLOTS * 2 * 4 * WGS_FIXW")
    src = src.replace("(size_t)pb * WGS_SLOTS * 4 * WGS_FIXW", "(size_t)pb * WGS_SLOTS * 2 * 4 * WGS_FIXW")
    src = src.replace("(size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW", "(size_t)2 * okc::WGS_SLOTS * 2 * 4 * okc::WGS_FIXW")
    replace("template <bool PB>\n__device__ __forceinline__ void wgs_flush", """// Forward declarations for deterministic contributor enumeration.
__device__ __forceinline__ void wgs_range(const Smem&, int, int, int, int, int&, int&);
__device__ __forceinline__ int wgs_cta_of(const Smem&, int, int, int);
template <bool PB>
__device__ __forceinline__ void wgs_flush""")
    start = src.index("  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot")
    end = src.index("  if (wt == 0) *fc = 0;", start)
    src = src[:start] + """  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  const int first = wgs_cta_of(S, PB, pd.u << 4, gridDim.x);
  const int last = wgs_cta_of(S, PB, (pd.u << 4) + 15, gridDim.x);
  for (int nt = 0; nt < ntt; ++nt) {
    float d4[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c = first; c <= last; ++c) {
      int a, b;
      wgs_range(S, S.njF + S.njS, PB, c, gridDim.x, a, b);
      if (a >= b || a >= ((pd.u + 1) << 4) || b <= (pd.u << 4)) continue;
      const int endpoint = (a >> 4) == pd.u ? 0 : 1;
      const float* partial = wgs_fix(p, PB)
          + (size_t)((c * 2 + endpoint) * 4 + wg) * WGS_FIXW + wt * 4 + nt * 512;
      const float4 v = __ldcg(reinterpret_cast<const float4*>(partial));
      // Explicit RN additions retain subnormals, in increasing stage-owner order.
      d4[0] = __fadd_rn(d4[0], v.x); d4[1] = __fadd_rn(d4[1], v.y);
      d4[2] = __fadd_rn(d4[2], v.z); d4[3] = __fadd_rn(d4[3], v.w);
    }
    wgs_epi_tile<PB>(p, S, j, r16, nt, d4);
  }
""" + src[end:]
    start = src.index("  if (slot >= 0) {  // a share of a unit split")
    end = src.index("    // kind 3:", start)
    src = src[:start] + """  if (slot >= 0) {
    // Only boundary units of this CTA's contiguous range can be partial.
    // The two locations have unique writers and are not reused in this phase.
    const int endpoint = ((PB ? S.wgr[2] : S.wgr[0]) >> 4) == u ? 0 : 1;
    float* fb = wgs_fix(p, PB)
        + (size_t)(((int)blockIdx.x * 2 + endpoint) * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(d[nt][0], d[nt][1], d[nt][2], d[nt][3]);
""" + src[end:]
    assert "native_add_preserve(fb" not in src
    (out / "native_marlin_wgs.cu").write_text(src, newline="\n")
    for name in ("native_marlin.py", "native_marlin_tiles.cuh"):
        (out / name).write_bytes((root / name).read_bytes())
    print(hashlib.sha256((out / "native_marlin_wgs.cu").read_bytes()).hexdigest())


if __name__ == "__main__":
    build()
