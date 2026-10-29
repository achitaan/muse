#if OKC_WGS && OKC_WGSFINSS
  {
    static_assert(HD == NT * 8, "8 columns per thread");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    const uint4 gb = __ldg(reinterpret_cast<const uint4*>(p.normw) + tid);
    const float gain[8] = {__uint_as_float(gb.x << 16), __uint_as_float(gb.x & 0xffff0000u),
                           __uint_as_float(gb.y << 16), __uint_as_float(gb.y & 0xffff0000u),
                           __uint_as_float(gb.z << 16), __uint_as_float(gb.z & 0xffff0000u),
                           __uint_as_float(gb.w << 16), __uint_as_float(gb.w & 0xffff0000u)};
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
      int qv[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) qv[sl] = S.pos[tk * 8 + sl];
      uint4 saved[8];
      float ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        ps[sl] = 0.f;
        saved[sl] = make_uint4(0u, 0u, 0u, 0u);
        if (qv[sl] >= 0) {
          const __nv_bfloat16* row = reinterpret_cast<const __nv_bfloat16*>(p.ybuf) + (size_t)qv[sl] * HD;
          saved[sl] = __ldcg(reinterpret_cast<const uint4*>(row) + tid);
          float4 a, b;
          factor_product(saved[sl], gain, a, b);
          ps[sl] = a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w + b.x * b.x + b.y * b.y + b.z * b.z + b.w * b.w;
        }
      }
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float tsum = 0.f;
        for (int w = 0; w < NW; ++w) tsum += red[w * 8 + tid];
        red[NW * 8 + tid] = tsum;
      }
      CTA_SYNC();
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl)
        if (qv[sl] >= 0) {
          const float r = p.topk_w[tk * 8 + sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          float4 a, b;
          factor_product(saved[sl], gain, a, b);
          acc[0] += r * a.x; acc[1] += r * a.y; acc[2] += r * a.z; acc[3] += r * a.w;
          acc[4] += r * b.x; acc[5] += r * b.y; acc[6] += r * b.z; acc[7] += r * b.w;
        }
      uint4 o;
      uint32_t* ow = reinterpret_cast<uint32_t*>(&o);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        __nv_bfloat162 v = __floats2bfloat162_rn(acc[2 * i], acc[2 * i + 1]);
        ow[i] = *reinterpret_cast<uint32_t*>(&v);
      }
      *reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 8) = o;
      CTA_SYNC();
    }
  }
