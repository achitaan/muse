    // Same SwiGLU/BF16 values, redistributed across lane bit2 for paired stores.
    const int ntt = (cnt + 7) >> 3;
    const float gsc = p.gsA[e] * TWO62;
    uint32_t vb[2] = {0u, 0u};
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tk = nt * 8 + 2 * t + h;
      if (tk < cnt) {
        const float2 achita_pair_9 = achita_marlin_out2(d4[h], d4[2 + h], gsc);
        const float gt = achita_pair_9.x, up = achita_pair_9.y;
        const float v = SILU_DIV(gt, 1.f + __expf(-gt)) * up;
        vb[h] = (uint32_t)__bfloat16_as_ushort(__float2bfloat16(probe_q(v)));
      }
    }
    // All lanes participate, including short-tail lanes whose vb is zero.
    // g-even sends h1; g-odd sends h0. XOR4 exchanges adjacent columns.
    const bool odd = (g & 1) != 0;
    const uint32_t cross = __shfl_xor_sync(0xffffffffu, odd ? vb[0] : vb[1], 4);
    const uint32_t packed = odd ? (cross | (vb[1] << 16)) : (vb[0] | (cross << 16));
    const int tk = nt * 8 + 2 * t + (int)odd;
    const int col = r16 * 8 + (g & ~1);
    if (tk < cnt) {
      __nv_bfloat16* dst = p.xB + (size_t)S.jrow[j] * HD + (col & 7)
          + (size_t)((((col >> 8) * ntt + (tk >> 3)) * 32 + ((col & 255) >> 3)) * 64 + (tk & 7) * 8);
      epair_st_keep_u32(reinterpret_cast<uint32_t*>(dst), packed);
    }
