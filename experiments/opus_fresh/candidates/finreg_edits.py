"""Exact CUDA text edits for the MoE WGS phase-B in-register finisher candidate (FINREG).

Only code inside `#if OKC_WGS` changes, so the TINY/TINY2/MID device code is untouched.
"""

OLD_FLUSH_DF = """  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // designated finisher (kind 3): the share processed last in time by construction (phase A: the one holding the
  // unit's last stage, run early by the reorder; phase B: the one holding stage 0, at its CTA's end). The others just
  // count with a fire-and-forget release (no atomic round trip, no drain wait); the finisher waits for their counts
  if (pd.kind == 2) {
    if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");
    pd.kind = 0;
    return;
  }
  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }
  asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
  pd.kind = 0;
  } else {"""

NEW_FLUSH_DF = """  constexpr bool DF = (OKC_WGSDFIN >> (PB ? 1 : 0)) & 1;  // designated finisher in this phase
  if constexpr (DF) {
  // FINREG: only non-finisher shares (kind 2) reach here; the designated finisher completes its unit in wgs_seg
  // with its own partial still in registers. Fire-and-forget release of this share's stage count.
  if (wt == 0) asm volatile("red.release.gpu.global.add.s32 [%0], %1;\\n" ::"l"(fc), "r"(pd.nst) : "memory");
  pd.kind = 0;
  return;
  } else {"""

OLD_SEG_TAIL = """  if (slot >= 0) {  // a share of a unit split between CTAs: add it now, count it later (deferred)
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                   "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
    // kind 3: this share finishes the unit (designated finisher, OKC_WGSDFIN): phase A the share holding stage 15,
    // phase B the one holding stage 0
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2, u, slot, s1 - s0};
    return;
  }"""

NEW_SEG_TAIL = """  if (slot >= 0) {  // a share of a unit split between CTAs
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
    if (((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) {
      // FINREG designated finisher: wait for the other shares' counts (each released after its red.add), then add
      // their summed partial to this share's registers and fall through to the unit epilogue below.
      int* fc = wgs_cnt(p, PB) + slot * 4 + wg;
      if (wt == 0) { while (ld_acquire(fc) < 16 - (s1 - s0)) __nanosleep(32); }
      asm volatile("bar.sync %0, 128;\\n" ::"r"(3 + wg) : "memory");
      asm volatile("fence.acq_rel.gpu;\\n" ::: "memory");
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt) {
        const float4 v = __ldcg(reinterpret_cast<const float4*>(fb + nt * 512));
        *reinterpret_cast<float4*>(fb + nt * 512) = make_float4(0.f, 0.f, 0.f, 0.f);
        d[nt][0] += v.x; d[nt][1] += v.y; d[nt][2] += v.z; d[nt][3] += v.w;
      }
      if (wt == 0) *fc = 0;
    } else {
#pragma unroll
      for (int nt = 0; nt < NTT; ++nt)
        asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512), "f"(d[nt][0]), "f"(d[nt][1]),
                     "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
      pd = WgsPend{2, u, slot, s1 - s0};
      return;
    }
  }"""

EDITS = ((OLD_FLUSH_DF, NEW_FLUSH_DF), (OLD_SEG_TAIL, NEW_SEG_TAIL))


def apply(cuda: str) -> str:
    out = cuda
    for old, new in EDITS:
        assert out.count(old) == 1, "edit anchor must occur exactly once"
        out = out.replace(old, new)
    return out


def invert(cuda: str) -> str:
    out = cuda
    for old, new in EDITS:
        assert out.count(new) == 1
        out = out.replace(new, old)
    return out
