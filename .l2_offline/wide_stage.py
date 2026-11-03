"""Wide-only live-range reduction, preserving narrow stage source verbatim."""
def reduce_wide_live_ranges(src):
    a=src.index('void wgs_seg(');end=src.index('\n}',a)+2
    s=src[a:end]
    s=s.replace('  const uint32_t k16c =', '  if constexpr (WGS_L2 && NTT > 8) { if (pd.kind) wgs_flush<PB>(p, S, pd); }\n  const uint32_t k16c =')
    s=s.replace('if (FIRST && ci ==','if (FIRST && NTT <= 8 && ci ==')
    beg=s.index('      const uint4 cwg = cwn;');stop=s.index('#pragma unroll\n      for (int h =',beg)
    old=s[beg:stop]
    s=s[:beg]+'''      uint4 cwg; uint2 scg;
      if constexpr (WGS_L2 && NTT > 8) {
        // N72..96 cannot carry next-group packed words alongside the wider
        // accumulator at96 registers. Load just the current four k16 steps.
        cwg = *reinterpret_cast<const uint4*>(pc + ((lg * 4 + wp) * 32 + lane) * 16);
        scg = *reinterpret_cast<const uint2*>(pc + 8192 + ((lg * 4 + wp) * 8 + g) * 8);
        if (lg == 3) {
          const uint32_t loaded = cwg.x ^ cwg.y ^ cwg.z ^ cwg.w ^ scg.x ^ scg.y;
          const uint32_t mask = 0xffffffffu ^ (loaded & p.zero);
          asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");
          if (lane == 0) mbar_arrive1(S.bars + BW_E + ws);
        }
      } else {
'''+old.replace('const uint4 cwg = cwn;','cwg = cwn;').replace('const uint2 scg = scn;','scg = scn;')+'''      }
'''+s[stop:]
    return src[:a]+s+src[end:]
