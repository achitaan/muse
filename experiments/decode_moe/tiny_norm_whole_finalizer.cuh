#if OKC_TINY && OKC_WS
  // One CTA owns a complete token row. All ybuf producers and the helper
  // have joined before this point. The shared weight ring is no longer live.
  {
    static_assert(NT == 256 && HD == NT * 16 && NW == 8, "tiny whole-row geometry");
    float* red = reinterpret_cast<float*>(xbuf);
    const int lane = tid & 31, warp = WARP_IDX(tid);
    for (int tk = blockIdx.x; tk < p.M; tk += gridDim.x) {
      int qv[8];
      float wv[8], ps[8];
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        const bool first = hasp && tk == tkp;
        qv[sl] = first ? qvp[sl] : (int)S.pos[tk * 8 + sl];
        wv[sl] = qv[sl] >= 0 ? (first ? wvp[sl] : p.topk_w[tk * 8 + sl]) : 0.f;
        ps[sl] = 0.f;
        if (qv[sl] >= 0) {
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 4;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float4 v = __ldcg(src + k);
            ps[sl] += v.x * v.x;
            ps[sl] += v.y * v.y;
            ps[sl] += v.z * v.z;
            ps[sl] += v.w * v.w;
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) ps[sl] += __shfl_xor_sync(0xffffffffu, ps[sl], o);
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int sl = 0; sl < 8; ++sl) red[warp * 8 + sl] = ps[sl];
      }
      CTA_SYNC();
      if (tid < 8) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < NW; ++w) sum += red[w * 8 + tid];
        red[NW * 8 + tid] = sum;
        // One token-owning CTA and one lane own each valid route statistic.
        // Preserve the completed-generation diagnostic/scratch contract.
#pragma unroll
        for (int sl = 0; sl < 8; ++sl)
          if (tid == sl && qv[sl] >= 0) ss_cur()[qv[sl]] = sum;
      }
      CTA_SYNC();
      float acc[16];
#pragma unroll
      for (int k = 0; k < 16; ++k) acc[k] = 0.f;
#pragma unroll
      for (int sl = 0; sl < 8; ++sl) {
        if (qv[sl] >= 0) {
          const float r = wv[sl] * rsqrtf(red[NW * 8 + sl] * (1.f / HD) + 1e-8f);
          const float4* src = reinterpret_cast<const float4*>(p.ybuf + (size_t)qv[sl] * HD) + tid * 4;
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            const float4 v = __ldcg(src + k);
            acc[4*k] += r * v.x; acc[4*k+1] += r * v.y;
            acc[4*k+2] += r * v.z; acc[4*k+3] += r * v.w;
          }
        }
      }
      uint4 packed[2];
      uint32_t* words = reinterpret_cast<uint32_t*>(packed);
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(acc[2*k], acc[2*k+1]);
        words[k] = *reinterpret_cast<const uint32_t*>(&v);
      }
      uint4* dst = reinterpret_cast<uint4*>(p.out + (size_t)tk * HD + tid * 16);
      dst[0] = packed[0]; dst[1] = packed[1];
      CTA_SYNC();  // this CTA may own another token when gridDim.x < M
    }
  }
