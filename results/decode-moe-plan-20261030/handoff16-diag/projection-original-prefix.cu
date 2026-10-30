
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define M32D __device__ __forceinline__

M32D int4 m16trace_59fcb5b72994_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint4 m16trace_59fcb5b72994_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

M32D uint32_t m16trace_59fcb5b72994_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

M32D uint32_t m16trace_59fcb5b72994_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

M32D void m16trace_59fcb5b72994_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
M32D void m16trace_59fcb5b72994_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  const uint32_t cm = C & 0x77777777u;
  const uint32_t hlo = m16trace_59fcb5b72994_prmt(T0, T1, cm);
  const uint32_t hhi = m16trace_59fcb5b72994_prmt(T0, T1, cm >> 16);
  a[0] = m16trace_59fcb5b72994_prmt(L0, hlo, 0x5140u) | ((C << 12) & 0x80008000u);
  a[1] = m16trace_59fcb5b72994_prmt(L0, hlo, 0x7362u) | ((C << 8) & 0x80008000u);
  a[2] = m16trace_59fcb5b72994_prmt(L1, hhi, 0x5140u) | ((C << 4) & 0x80008000u);
  a[3] = m16trace_59fcb5b72994_prmt(L1, hhi, 0x7362u) | (C & 0x80008000u);
}

__device__ __noinline__ void m16trace_59fcb5b72994_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
M32D void m16trace_59fcb5b72994_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) m16trace_59fcb5b72994_never(p);
}

M32D uint32_t m16trace_59fcb5b72994_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

M32D uint32_t m16trace_59fcb5b72994_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

M32D uint32_t m16trace_59fcb5b72994_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

M32D void m16trace_59fcb5b72994_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

M32D void m16trace_59fcb5b72994_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

M32D void m16trace_59fcb5b72994_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
M32D void m16trace_59fcb5b72994_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
M32D void m16trace_59fcb5b72994_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

M32D void m16trace_59fcb5b72994_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

M32D bool m16trace_59fcb5b72994_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

M32D uint64_t m16trace_59fcb5b72994_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
M32D void m16trace_59fcb5b72994_wait_bar(uint32_t bar) {
  if (m16trace_59fcb5b72994_mbar_try(bar)) return;
  const uint64_t t0 = m16trace_59fcb5b72994_gtime();
  while (!m16trace_59fcb5b72994_mbar_try(bar))
    if (m16trace_59fcb5b72994_gtime() - t0 > 20000000ull) __trap();
}

M32D void m16trace_59fcb5b72994_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

M32D float4 m16trace_59fcb5b72994_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

M32D void m16trace_59fcb5b72994_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
M32D void m16trace_59fcb5b72994_store(float* p, float v) { *p = v; }

template <int FMT> struct M32Fmt;
template <> struct M32Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct M32Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct M32Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct M32Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
M32D void m16trace_59fcb5b72994_decode(const int4 (&w)[M32Fmt<FMT>::NL], uint32_t (&A)[M32Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < M32Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < M32Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      m16trace_59fcb5b72994_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      m16trace_59fcb5b72994_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      m16trace_59fcb5b72994_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      m16trace_59fcb5b72994_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
M32D void m16trace_59fcb5b72994_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
M32D void m16trace_59fcb5b72994_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) m16trace_59fcb5b72994_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
__device__ __noinline__ void pdlo16_wait(uint32_t checksum) {
  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");
}

template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
__global__ void __launch_bounds__(NTMAX) pdlo16_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = M32Fmt<FMT>::NL, NA = M32Fmt<FMT>::NA, MTT = MT * TG;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3 || RED == 4) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int xbytes = (XM == 2 || XM == 3 || XM == 5) ? MTT * 8 * (K / (CS ? CS : gridDim.y) + 8) * 2 : 0;
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem);
  const int wbytes = XM == 4 ? CWR * CWK * RT * NA * 2048 : 0;
  int4* sw = reinterpret_cast<int4*>(smem + xbytes);
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, pw = tid >> 5, warp = pw / TG, tg = pw % TG, lane = tid & 31;
    const int nt = CWR ? CWR * CWK * TG * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MTT, (m + 1) >> 1);
    const uint32_t bar = m16trace_59fcb5b72994_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = m16trace_59fcb5b72994_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = (RED == 2 || RED == 4) ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        m16trace_59fcb5b72994_mbar_init(bar, 1);
        m16trace_59fcb5b72994_mbar_expect(bar, parts * RT * (RED == 4 ? MTT * 4 : TV) * 128u);
        m16trace_59fcb5b72994_fence_init();
      }
      m16trace_59fcb5b72994_cl_arrive();
    }

    if constexpr (XM == 2 || XM == 3 || XM == 5) {
      const int ks = K / S;
      for (int v = tid; v < MTT * 8 * ks / 8; v += nt) {
        const int t = v / (ks / 8), c = v % (ks / 8);
        const void* src = x + (size_t)min(t, m - 1) * K + blockIdx.y * ks + c * 8;
        void* dst = sx + t * (ks + 8) + c * 8;
        if constexpr (XM == 3 || XM == 5) {
          const uint32_t sa = m16trace_59fcb5b72994_smem_u32(dst);
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(src) : "memory");
        } else *reinterpret_cast<uint4*>(dst) = m16trace_59fcb5b72994_ldx(src);
      }
      if constexpr (XM == 3 || XM == 5) asm volatile("cp.async.commit_group;" ::: "memory");
      else __syncthreads();
    }
    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = m16trace_59fcb5b72994_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * (XM >= 9 ? 8 : 16);

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if (XM != 4 || tg == 0) w[slot][j][l] = m16trace_59fcb5b72994_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && (XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = m16trace_59fcb5b72994_ldx(xp);
          xv[slot][n][1] = m16trace_59fcb5b72994_ldx(xp + (XM >= 9 ? 32 : 8));
        }
      }
    };
static_assert(FMT==1 && MT==2 && RT==1 && ST==1 && RED==0 && DIAG==0 && XM==7 && TG==1, "fixed O16 preamble");
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    pdlo16_wait(my_e ^ (uint32_t)w[0][0][0].x ^ (uint32_t)w[0][0][0].y ^ (uint32_t)w[0][0][0].z ^ (uint32_t)w[0][0][0].w ^ (uint32_t)w[0][0][1].x ^ (uint32_t)w[0][0][1].y ^ (uint32_t)w[0][0][1].z ^ (uint32_t)w[0][0][1].w ^ (uint32_t)w[0][0][2].x ^ (uint32_t)w[0][0][2].y ^ (uint32_t)w[0][0][2].z ^ (uint32_t)w[0][0][2].w);
    m16trace_59fcb5b72994_order_point(m, outp);
    if constexpr (XM == 3 || XM == 5) {
      asm volatile("cp.async.wait_group 0;" ::: "memory");
      __syncthreads();
    }

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          m16trace_59fcb5b72994_order_point(m, outp);
        } else {
          int4 current[RT][NL];
#pragma unroll
          for(int j=0;j<RT;++j)
#pragma unroll
            for(int z=0;z<NL;++z)current[j][z]=w[s][j][z];
          if(si+ST<KU)load(s,si+ST);
          uint32_t b[MT][8];
          if constexpr (XM == 7 || XM == 8 || XM == 11) {
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              const __nv_bfloat16* xp = x0 + (size_t)min((tg * MT + n) * 8 + (lane >> 2), m - 1) * K + si * 64;
              const uint4 z0=m16trace_59fcb5b72994_ldx(xp), z1=m16trace_59fcb5b72994_ldx(xp+(XM >= 9 ? 32 : 8));
              b[n][0]=z0.x; b[n][1]=z0.y; b[n][2]=z0.z; b[n][3]=z0.w;
              b[n][4]=z1.x; b[n][5]=z1.y; b[n][6]=z1.z; b[n][7]=z1.w;
            }
            m16trace_59fcb5b72994_order_point(m, outp);
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) if (XM != 4 || tg == 0) m16trace_59fcb5b72994_decode<FMT>(current[j], A[j], T0, T1, U0, U1);
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr ((XM == 0 || XM == 6 || XM == 9 || XM == 10)) {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }

          m16trace_59fcb5b72994_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (XM == 4 && tg != 0) ? 0xFFFFFFFEu : (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) m16trace_59fcb5b72994_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
          if constexpr (XM == 4) {
            if (tg == 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q)
                  sw[((warp * RT + j) * NA * 4 + q) * 32 + lane] =
                    make_int4(A[j][q*4], A[j][q*4+1], A[j][q*4+2], A[j][q*4+3]);
            }
            __syncthreads();
            if (tg != 0) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int q = 0; q < NA * 4; ++q) {
                  const int4 v = sw[((warp * RT + j) * NA * 4 + q) * 32 + lane];
                  A[j][q*4]=v.x; A[j][q*4+1]=v.y; A[j][q*4+2]=v.z; A[j][q*4+3]=v.w;
                }
            }
          }
          if constexpr (XM == 6 || XM == 8 || XM == 10) {
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
              for (int n = 0; n < MT; ++n)
#pragma unroll
                for (int j = 0; j < RT; ++j)
                  m16trace_59fcb5b72994_mma(acc[0][j][n], A[j] + q * 4, b[n][2*q], b[n][2*q+1]);
          } else if constexpr (XM == 5) {
#pragma unroll
            for (int q = 0; q < 4; ++q) {
#pragma unroll
              for (int np = 0; np < (MT + 1) / 2; ++np) {
                const int tok = tg * MT * 8 + np * 16 + (lane & 7) + (MT == 1 ? 0 : (lane >> 4) * 8);
                const int col = (wk * KU + si) * 64 + q * 16 + ((lane >> 3) & 1) * 8;
                const uint32_t addr = m16trace_59fcb5b72994_smem_u32(sx + tok * (K / S + 8) + col);
                uint32_t b0,b1,b2,b3;
                if constexpr (MT == 1)
                  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(b0), "=r"(b1) : "r"(addr));
                else
                  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
#pragma unroll
                for (int j = 0; j < RT; ++j) {
                  m16trace_59fcb5b72994_mma(acc[0][j][np * 2], A[j] + q * 4, b0, b1);
                  if constexpr (MT > 1) m16trace_59fcb5b72994_mma(acc[0][j][np * 2 + 1], A[j] + q * 4, b2, b3);
                }
              }
            }
          } else {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n) {
              uint32_t bx[8];
              if constexpr (XM >= 1 && XM <= 4 && !NOX) {
                const int tok = min((tg * MT + n) * 8 + (lane >> 2), m - 1);
                const __nv_bfloat16* xp;
                if constexpr (XM == 2 || XM == 3 || XM == 5)
                  xp = sx + (size_t)tok * (K / S + 8) + (wk * KU + si) * 64 + (lane & 3) * 16;
                else xp = x0 + (size_t)tok * K + si * 64;
                uint4 z0, z1;
                if constexpr (XM == 2 || XM == 3 || XM == 5) {
                  z0 = *reinterpret_cast<const uint4*>(xp);
                  z1 = *reinterpret_cast<const uint4*>(xp + 8);
                } else {
                  z0 = m16trace_59fcb5b72994_ldx(xp); z1 = m16trace_59fcb5b72994_ldx(xp + (XM >= 9 ? 32 : 8));
                }
                bx[0]=z0.x; bx[1]=z0.y; bx[2]=z0.z; bx[3]=z0.w;
                bx[4]=z1.x; bx[5]=z1.y; bx[6]=z1.z; bx[7]=z1.w;
              }
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]) ^ ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]);
                } else {
                  m16trace_59fcb5b72994_mma(acc[0][j][n], A[j] + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                  if constexpr (HILO) m16trace_59fcb5b72994_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q] : b[n][2 * q]), ((XM >= 1 && XM <= 4 && !NOX) ? bx[2 * q + 1] : b[n][2 * q + 1]));
                }
              }
            }
          }
          if constexpr (XM == 4) __syncthreads();
        }
      }
    }

    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          m16trace_59fcb5b72994_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) m16trace_59fcb5b72994_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = (tg * MT + n) * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                m16trace_59fcb5b72994_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MTT, tile = rest / MTT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) m16trace_59fcb5b72994_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) m16trace_59fcb5b72994_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          const int NI = WR * RT * MTT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[(((size_t)warp * RT + j) * MTT + tg * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            m16trace_59fcb5b72994_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MTT * 32;
            m16trace_59fcb5b72994_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) m16trace_59fcb5b72994_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = RED == 4 ? WR * RT * MTT * 32 : WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + xbytes + wbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2 || RED == 4) {
            m16trace_59fcb5b72994_cl_wait();
            if (RED == 4 || t < TV) {
              const uint32_t rbar = m16trace_59fcb5b72994_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = (tg * MT + n) * 4 + t;
                  if (RED == 4 || tn < TV) {
                    const int idx = RED == 4 ? ((wr * RT + j) * MTT + tg * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else m16trace_59fcb5b72994_st_async4(m16trace_59fcb5b72994_mapa(m16trace_59fcb5b72994_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            m16trace_59fcb5b72994_wait_bar(bar);
            if constexpr (RED == 4) m16trace_59fcb5b72994_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_full);
            else m16trace_59fcb5b72994_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (RED == 4 || t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if ((tg * MT + n) * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + (tg * MT + n) * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            m16trace_59fcb5b72994_cl_wait();
            const uint32_t rbar = m16trace_59fcb5b72994_mapa(bar, 0);
            m16trace_59fcb5b72994_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else m16trace_59fcb5b72994_st_async4(m16trace_59fcb5b72994_mapa(m16trace_59fcb5b72994_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            m16trace_59fcb5b72994_wait_bar(bar);
            m16trace_59fcb5b72994_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int m16trace_59fcb5b72994_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int XM, int TG>
static int m16trace_59fcb5b72994_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = pdlo16_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, XM, TG>;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * TG * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * TG * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT * TG, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * TG * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * TG * 512;
    else if (RED == 2 || RED == 4) sm = (size_t)S * NW * RT * (RED == 4 ? MT * TG * 4 : TV) * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  if (XM == 2 || XM == 3 || XM == 5) sm += (size_t)MT * TG * 8 * (K / S + 8) * 2;
  if (XM == 4) sm += (size_t)WR * WK * RT * M32Fmt<FMT>::NA * 2048;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = m16trace_59fcb5b72994_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[2];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    lc.numAttrs = 1;
  }
  if (query == 0) {
    const int pdli = lc.numAttrs;
    at[pdli].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[pdli].val.programmaticStreamSerializationAllowed = 1;
    lc.attrs = at; lc.numAttrs = pdli + 1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  const cudaError_t pdli_launch_rc = cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  if (pdli_launch_rc != cudaSuccess) return -static_cast<int>(pdli_launch_rc);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int m16trace_59fcb5b72994_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
case 0: return m16trace_59fcb5b72994_launch_t<1,2,1,1,0,0,256,0,8192,1024,2,4,4,1,7,1>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);
// Private O16 twin supports only exact C12 plans; callers keep the original BF16 fallback.
    default: return -99;
  }
}


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {

  asm volatile("{ .reg .pred ready, expired; .reg .b64 start, now, elapsed;\n"
    "mbarrier.try_wait.parity.shared::cta.b64 ready,[%0],%1; @ready bra DONE;\n"
    "mov.u64 start, %%globaltimer;\n"
    "WAIT: mbarrier.try_wait.parity.shared::cta.b64 ready,[%0],%1; @ready bra DONE;\n"
    "mov.u64 now, %%globaltimer; sub.u64 elapsed,now,start;\n"
    "setp.gt.u64 expired,elapsed,20000000; @expired trap; bra WAIT;\n"
    "DONE: }" :: "r"(b), "r"(ph) : "memory");

}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
OD void ol_mma32(float* a,uint64_t w,uint64_t x) {
          asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;" : "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),"+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]) : "l"(w),"l"(x));
        }

__device__ __noinline__ void pdlo32_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void pdlo32_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  static_assert(BN==32 && BK==128 && ST==8 && PIPE==1 && PACK==1 && HINT==1 && TW==1, "fixed full-K O32 ring");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto loadw=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
  };
  auto loadx=[&](int i) {
    const int s=i%ST;
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  auto load=[&](int i) { loadw(i); loadx(i); };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) loadw(i);
    pdlo32_wait();
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) loadx(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        if constexpr(BN==32) ol_mma32(a,aw,ax);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    int row=(tid%128/32)*16+(tid%32)/4;
    int col=2*(tid%4);
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      #pragma unroll
      for(int h=0;h<2;h++) {
        int token=mt*T+(tid/128)*BN+col+8*j;
        if(token<m) out[(long)token*8192+nt*R+row+8*h]=__float2bfloat16_rn(a[4*j+2*h]);
        if(token+1<m) out[(long)(token+1)*8192+nt*R+row+8*h]=__float2bfloat16_rn(a[4*j+2*h+1]);
      }
    }
  }

}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=pdlo32_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  cudaLaunchConfig_t lc = {};
  lc.gridDim=dim3(8192/R,(m+T-1)/T);
  lc.blockDim=dim3(128*TW+((PIPE&2)?128:0));
  lc.dynamicSmemBytes=SMEM; lc.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr[1];
  attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed=1;
  lc.attrs=attr; lc.numAttrs=1;
  const cudaError_t launch_rc=cudaLaunchKernelEx(&lc,fn,tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  if(launch_rc!=cudaSuccess) return int(launch_rc);
  return int(cudaGetLastError());
}
int pdlo32_info(int query) {
  if(query!=1) return -90;
  int blocks=0;
  const cudaError_t rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,pdlo32_projection<32,128,8,1,1,1,1>,128,196736);
  return rc==cudaSuccess ? blocks : -static_cast<int>(rc);
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {case 0: return ol_launch_t<32,128,8,1,1,1,1>(x,w,out,m,stream,query); default:return -99;}
}


#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <algorithm>
#include <type_traits>
#include <cooperative_groups.h>

#define CPSSD __device__ __forceinline__

CPSSD int4 dd_cap2_b_cpssm_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint4 dd_cap2_b_cpssm_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

CPSSD uint32_t dd_cap2_b_cpssm_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

CPSSD uint32_t dd_cap2_b_cpssm_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

CPSSD void dd_cap2_b_cpssm_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
CPSSD void dd_cap2_b_cpssm_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x5140u);
  a[1] = dd_cap2_b_cpssm_prmt(L0, low_high, 0x7362u);
  a[2] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x5140u);
  a[3] = dd_cap2_b_cpssm_prmt(L1, high_high, 0x7362u);
}

__device__ __noinline__ void dd_cap2_b_cpssm_never(void* p) { asm volatile("trap;"); }

__device__ __noinline__ void pdlu16_wait(uint32_t checksum) {
  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");
}

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
CPSSD void dd_cap2_b_cpssm_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) dd_cap2_b_cpssm_never(p);
}

CPSSD uint32_t dd_cap2_b_cpssm_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

CPSSD uint32_t dd_cap2_b_cpssm_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

CPSSD uint32_t dd_cap2_b_cpssm_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

CPSSD void dd_cap2_b_cpssm_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

CPSSD void dd_cap2_b_cpssm_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

CPSSD void dd_cap2_b_cpssm_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
CPSSD void dd_cap2_b_cpssm_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
CPSSD void dd_cap2_b_cpssm_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

CPSSD void dd_cap2_b_cpssm_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

CPSSD bool dd_cap2_b_cpssm_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

CPSSD uint64_t dd_cap2_b_cpssm_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
CPSSD void dd_cap2_b_cpssm_wait_bar(uint32_t bar) {
  if (dd_cap2_b_cpssm_mbar_try(bar)) return;
  const uint64_t t0 = dd_cap2_b_cpssm_gtime();
  while (!dd_cap2_b_cpssm_mbar_try(bar))
    if (dd_cap2_b_cpssm_gtime() - t0 > 20000000ull) __trap();
}

CPSSD void dd_cap2_b_cpssm_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

CPSSD float4 dd_cap2_b_cpssm_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

CPSSD void dd_cap2_b_cpssm_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
CPSSD void dd_cap2_b_cpssm_store(float* p, float v) { *p = v; }

template <int FMT> struct CpsSmFmt;
template <> struct CpsSmFmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct CpsSmFmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct CpsSmFmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct CpsSmFmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
CPSSD void dd_cap2_b_cpssm_decode(const int4 (&w)[CpsSmFmt<FMT>::NL], uint32_t (&A)[CpsSmFmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < CpsSmFmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < CpsSmFmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      dd_cap2_b_cpssm_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      dd_cap2_b_cpssm_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      dd_cap2_b_cpssm_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      dd_cap2_b_cpssm_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
CPSSD void dd_cap2_b_cpssm_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
CPSSD void dd_cap2_b_cpssm_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) dd_cap2_b_cpssm_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
__global__ void __launch_bounds__(NTMAX) pdlu16_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = CpsSmFmt<FMT>::NL, NA = CpsSmFmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
  const int rawbytes = CP * WR * WK * RT * NL * 512;
  if constexpr (EMPTY) {
    return;
  } else {
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, nt = CWR ? CWR * CWK * 32 : blockDim.x;
    const int NW = WR * WK, wr = warp % WR, wk = warp / WR;
    const int S = CS ? CS : gridDim.y;
    const int GX = CN ? CN / (WR * RT * 16) : gridDim.x;
    const int G = GX * S * NW;
    const int gw = (blockIdx.y * GX + blockIdx.x) * NW + warp;
    const int rt0 = (blockIdx.x * WR + wr) * RT;
    const int c0 = (blockIdx.y * WK + wk) * KU;
    const int TV = min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = dd_cap2_b_cpssm_smem_u32(&s_bar);
    uint32_t rank = 0;
    if constexpr (PUSH) {
      rank = dd_cap2_b_cpssm_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        dd_cap2_b_cpssm_mbar_init(bar, 1);
        dd_cap2_b_cpssm_mbar_expect(bar, parts * RT * TV * 128u);
        dd_cap2_b_cpssm_fence_init();
      }
      dd_cap2_b_cpssm_cl_arrive();
    }

    uint32_t my_e = 0xFFFFFFFFu;
    if constexpr (C12 && !RO) my_e = dd_cap2_b_cpssm_lde(esc + (size_t)gw * 32 + lane);

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4* rawring = reinterpret_cast<int4*>(smem);
    auto stage_weights = [&](int slot, int si) {
      if constexpr (CP > 0) {
        if (si < KU) {
#pragma unroll
          for (int j=0; j<RT; ++j)
#pragma unroll
            for (int l=0; l<NL; ++l) {
              const uint32_t dst=dd_cap2_b_cpssm_smem_u32(rawring + (((slot*NW+warp)*RT+j)*NL+l)*32+lane);
              const int4* src=wq+(size_t)si*sstride+(j*NL+l)*32;
              asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst), "l"(src) : "memory");
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
      }
    };
#pragma unroll
    static_assert(FMT==1 && MT==2 && RT==1 && ST==2 && RED==0 && DIAG==0 && CP==2, "fixed CP2 U16 preamble");
    for (int q=0; q<CP; ++q) stage_weights(q,q);
    pdlu16_wait(my_e);
    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = dd_cap2_b_cpssm_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = dd_cap2_b_cpssm_ldx(xp);
          xv[slot][n][1] = dd_cap2_b_cpssm_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    dd_cap2_b_cpssm_order_point(m, outp);

    float acc[NA][RT][MT][4];
#pragma unroll
    for (int h = 0; h < NA; ++h)
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) acc[h][j][n][c] = 0.f;
    uint32_t fold = 0;
    uint32_t nxt = 0xFFFFFFFFu;
    int ep = 0;
    if constexpr (C12 && !RO) nxt = __shfl_sync(0xffffffffu, my_e, 0);

#pragma unroll 1
    for (int i = 0; i < KU; i += ST) {
#pragma unroll
      for (int s = 0; s < ST; ++s) {
        const int si = i + s;
        if (ST == 3 && si >= KU) break;
        if constexpr (RO) {
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int l = 0; l < NL; ++l)
              fold ^= (uint32_t)(w[s][j][l].x ^ w[s][j][l].y ^ w[s][j][l].z ^ w[s][j][l].w);
          if (si + ST < KU) load(s, si + ST);
          dd_cap2_b_cpssm_order_point(m, outp);
        } else {
          if constexpr (CP > 0) {
            asm volatile("cp.async.wait_group %0;" :: "n"(CP-1) : "memory");
#pragma unroll
            for (int j=0; j<RT; ++j)
#pragma unroll
              for (int l=0; l<NL; ++l) {
                const uint32_t src=dd_cap2_b_cpssm_smem_u32(rawring+((((si%CP)*NW+warp)*RT+j)*NL+l)*32+lane);
                int4& v=w[s][j][l];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "r"(src) : "memory");
              }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) dd_cap2_b_cpssm_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if constexpr (CP > 0) stage_weights(si%CP,si+CP);
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          if (si + ST < KU) load(s, si + ST);
          dd_cap2_b_cpssm_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) dd_cap2_b_cpssm_patch<NA * 16>(A[j], nxt);
                  ++ep;
                  nxt = ep < 32 ? __shfl_sync(0xffffffffu, my_e, ep) : 0xFFFFFFFFu;
                } while ((nxt >> 18) == useq);
              }
            }
          }
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
#pragma unroll
              for (int q = 0; q < 4; ++q) {
                if constexpr (NOMMA) {
#pragma unroll
                  for (int h = 0; h < NA; ++h)
                    fold ^= A[j][16 * h + 4 * q] ^ A[j][16 * h + 4 * q + 1] ^ A[j][16 * h + 4 * q + 2] ^
                            A[j][16 * h + 4 * q + 3];
                  fold ^= b[n][2 * q] ^ b[n][2 * q + 1];
                } else {
                  dd_cap2_b_cpssm_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) dd_cap2_b_cpssm_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
        }
      }
    }

    if constexpr (CP > 0) asm volatile("cp.async.wait_group 0;" ::: "memory");
    if constexpr (RO) {
      if constexpr (RO_CODE) {
        if (warp == 0) {
          uint32_t a = fold, b = fold ^ 1u, c = fold ^ 2u, d = fold ^ 3u;
          const uint32_t k = (uint32_t)m | 1u;
#pragma unroll
          for (int q = 0; q < 64; ++q)
            asm volatile("mad.lo.u32 %0, %0, %4, %5;\n mad.lo.u32 %1, %1, %4, %5;\n mad.lo.u32 %2, %2, %4, %5;\n"
                         " mad.lo.u32 %3, %3, %4, %5;" : "+r"(a), "+r"(b), "+r"(c), "+r"(d) : "r"(k), "r"(q));
          fold = a ^ b ^ c ^ d;
        }
      }
      if constexpr (RO_STORE) {
        if ((lane & 3) == 0)
          dd_cap2_b_cpssm_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) dd_cap2_b_cpssm_store(out, 1.f);
      }
      return;
    } else {
      float r[RT][MT][4];
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int n = 0; n < MT; ++n)
#pragma unroll
          for (int c = 0; c < 4; ++c) r[j][n][c] = HILO ? acc[0][j][n][c] + acc[NA - 1][j][n][c] : acc[0][j][n][c];
      if constexpr (NOMMA) {
        if (fold == 0x9e3779b9u) r[0][0][0] = 1.f;
      }

      const int RB = WR * RT * 16;
      const int row0 = blockIdx.x * RB;
      if (NORED || (RED == 0 && WK == 1)) {
#pragma unroll
        for (int j = 0; j < RT; ++j)
#pragma unroll
          for (int n = 0; n < MT; ++n)
#pragma unroll
            for (int c = 0; c < 4; ++c) {
              const int tok = n * 8 + 2 * (lane & 3) + (c & 1);
              const int row = (rt0 + j) * 16 + (lane >> 2) + 8 * (c >> 1);
              if (NOSTORE ? __float_as_uint(r[j][n][c]) == 0x9e3779b9u : tok < m)
                dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, r[j][n][c]);
            }
        return;
      }
      if constexpr (!NORED) {
        // Full slots: idx = (tile * MT + n) * 32 + lane. Compact slots (MT 1): idx = (tile * TV + t) * 8 + g.
        auto emit_full = [&](int idx, float4 v) {
          const int ln = idx & 31, rest = idx >> 5, n = rest % MT, tile = rest / MT;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = n * 8 + 2 * (ln & 3) + (c & 1);
            const int row = row0 + tile * 16 + (ln >> 2) + 8 * (c >> 1);
            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) dd_cap2_b_cpssm_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem + rawbytes);
          const int NI = WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              part[((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, emit_full);
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            dd_cap2_b_cpssm_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) dd_cap2_b_cpssm_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem + rawbytes);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            dd_cap2_b_cpssm_cl_wait();
            if (t < TV) {
              const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((wr * RT + j) * TV + tn) * 8 + g;
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            dd_cap2_b_cpssm_wait_bar(bar);
            dd_cap2_b_cpssm_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((wr * RT + j) * TV + n * 4 + t) * 8 + g] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            dd_cap2_b_cpssm_cl_wait();
            const uint32_t rbar = dd_cap2_b_cpssm_mapa(bar, 0);
            dd_cap2_b_cpssm_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else dd_cap2_b_cpssm_st_async4(dd_cap2_b_cpssm_mapa(dd_cap2_b_cpssm_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            dd_cap2_b_cpssm_wait_bar(bar);
            dd_cap2_b_cpssm_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
}

static int dd_cap2_b_cpssm_smem_optin() {
  static int v = -1;
  if (v < 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

// query 0 launches; 1 returns max active clusters (cluster launch) or blocks per SM; 2 returns regs*1000+local bytes.
template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int CP>
static int dd_cap2_b_cpssm_launch_t(int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
                       int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  auto kern = pdlu16_gemv<FMT, MT, RT, ST, RED, DIAG, NTMAX, CM, CN, CK, CWR, CWK, CKU, CS, CP>;
  const int policy = query / 100;
  query %= 100;
  if (query == 2) {
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, kern) != cudaSuccess) return -10;
    return fa.numRegs * 1000 + (int)std::min<size_t>(fa.localSizeBytes, 999);
  }
  if ((CM && m != CM) || (CN && N != CN) || (CK && K != CK)) return -12;
  if ((CWR && WR != CWR) || (CWK && WK != CWK) || (CKU && KU != CKU) || (CS && S != CS)) return -12;
  const int NW = WR * WK, nt = NW * 32, RB = WR * RT * 16;
  if (WR < 1 || WK < 1 || nt > NTMAX) return -1;
  if (N % RB) return -2;
  if ((int64_t)S * WK * KU * 64 != K) return -3;
  if ((ST != 3 && KU % ST) || KU < ST) return -4;
  if (m < 1 || m > MT * 8) return -5;
  const bool cl = RED != 0;
  if (!cl && S != 1) return -9;
  if (cl && (S < 2 || S > 16)) return -9;
  const int TV = std::min(4 * MT, (m + 1) / 2);
  size_t sm = 0;
  if (!(DIAG & 28)) {
    if (RED == 0) sm = WK > 1 ? (size_t)NW * RT * MT * 512 : 0;
    else if (RED == 1) sm = (size_t)(NW + WR) * RT * MT * 512;
    else if (RED == 2) sm = (size_t)S * NW * RT * TV * 128;
    else sm = (size_t)(S + WK) * WR * RT * TV * 128;
  }
  sm += (size_t)CP * NW * RT * CpsSmFmt<FMT>::NL * 512;
  static int max_dyn = -1;
  if (max_dyn < 0) {
    cudaFuncAttributes fa;
    cudaFuncGetAttributes(&fa, kern);
    max_dyn = dd_cap2_b_cpssm_smem_optin() - (int)fa.sharedSizeBytes;
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dyn);
    if (cl) cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  }
  if ((int64_t)sm > max_dyn) return -6;
  cudaLaunchConfig_t lc = {};
  cudaLaunchAttribute at[3];
  lc.gridDim = dim3(N / RB, S, 1);
  lc.blockDim = dim3(nt, 1, 1);
  lc.dynamicSmemBytes = sm;
  lc.stream = (cudaStream_t)stream;
  if (cl) {
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 1;
    at[0].val.clusterDim.y = S;
    at[0].val.clusterDim.z = 1;
    lc.attrs = at;
    at[1].id = cudaLaunchAttributeClusterSchedulingPolicyPreference;
    at[1].val.clusterSchedulingPolicyPreference = static_cast<cudaClusterSchedulingPolicy>(policy);
    lc.numAttrs = 2;
  }
  if (query == 3) { int v=0; cudaError_t e=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v,kern,nt,sm); return e==cudaSuccess?v:-11; }
  if (query == 4) return (int)sm;
  if (query == 0) {
    const int dep_attr=lc.numAttrs;
    at[dep_attr].id=cudaLaunchAttributeProgrammaticStreamSerialization;
    at[dep_attr].val.programmaticStreamSerializationAllowed=1;
    lc.attrs=at;lc.numAttrs=dep_attr+1;
  }
  if (query == 1) {
    int v = 0;
    cudaError_t e = cl ? cudaOccupancyMaxActiveClusters(&v, kern, &lc)
                       : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&v, kern, nt, sm);
    cudaGetLastError();
    return e == cudaSuccess ? v : -11;
  }
  const cudaError_t pdlu_rc=cudaLaunchKernelEx(&lc, kern, reinterpret_cast<const int4*>(wp), reinterpret_cast<const __nv_bfloat16*>(x),
                     reinterpret_cast<void*>(out), reinterpret_cast<const uint32_t*>(esc), m, N, K, WR, WK, KU,
                     (uint32_t)(t01 & 0xffffffff), (uint32_t)((uint64_t)t01 >> 32), (uint32_t)(u01 & 0xffffffff),
                     (uint32_t)((uint64_t)u01 >> 32));
  if(pdlu_rc!=cudaSuccess)return -static_cast<int>(pdlu_rc);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}

int dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query) {
  switch (inst) {
case 0: return dd_cap2_b_cpssm_launch_t<1,2,1,2,0,0,256,0,8192,4096,2,4,16,1,2>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);
    default: return -99;
  }
}

// dd_cap2_b_


#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define D __device__ __forceinline__
D uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }
D void waitbar(uint32_t b,int p) {
  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");
}
D void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
D void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
D uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
D void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");
}
D uint4 ldw(const uint4* p,uint64_t policy) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));
  return v;
}
D uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {
  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;
}
D void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high = (C & 0x87878787u) + T0;
  const uint32_t high_high = (rotated & 0x87878787u) + T0;
  a[0] = prmt(L0, low_high, 0x5140u);
  a[1] = prmt(L0, low_high, 0x7362u);
  a[2] = prmt(L1, high_high, 0x5140u);
  a[3] = prmt(L1, high_high, 0x7362u);
}
D void patch(uint32_t* a,uint32_t e) {
  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);
  #pragma unroll
  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;
}

D void mma16(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9,%10,%11}, %24, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%12,%13,%14,%15}, %25, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%16,%17,%18,%19}, %26, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, {%20,%21,%22,%23}, %27, 1, 1, 1, 0; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]),"l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}
D void mma32(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %32, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%20,%21,%22,%23}, %33, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%24,%25,%26,%27}, %34, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%28,%29,%30,%31}, %35, 1, 1, 1, 0; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]),"l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}
D void mma64(float* d,uint32_t* a,uint64_t b) {asm volatile("wgmma.fence.sync.aligned; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %48, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%36,%37,%38,%39}, %49, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%40,%41,%42,%43}, %50, 1, 1, 1, 0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%44,%45,%46,%47}, %51, 1, 1, 1, 0; wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;" : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]),"l"(b+0),"l"(b+2),"l"(b+4),"l"(b+6) : "memory");}

__device__ __noinline__ void pdlu32_wait(uint32_t checksum) {
  asm volatile("{ .reg .b32 used; mov.b32 used, %0; griddepcontrol.wait; }" :: "r"(checksum) : "memory");
}

template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>
__global__ __launch_bounds__(128*WK,CAP) void pdlu32_gemm(
    const __grid_constant__ CUtensorMap tx,const uint4* __restrict__ w,
    const uint32_t* __restrict__ esc,__nv_bfloat16* out,int m,int n,int k,int split,uint32_t t0,uint32_t t1) {
  static_assert(BN==32 && R==64 && WK==2 && PF==2 && Q==2 && C12 && CAP==2, "fixed U32 cluster twin");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  extern __shared__ __align__(1024) uint8_t sm[];
  constexpr int RW=R/16,NT=128*WK,RP=R+4,SB=BN*128,WS=(Q*(SB+8)+1023)/1024*1024,NL=C12?3:4;
  int tid=threadIdx.x,wg=tid/128,warp=tid%128/32,lane=tid%32;
  uint32_t base0=addr(sm),base=base0+wg*WS,full=base+Q*SB;
  int rank=blockIdx.x%split,nt=blockIdx.x/split,mt=blockIdx.y;
  int kb0=(k/64)*(rank*WK+wg)/(split*WK),nkb=(k/64)*(rank*WK+wg+1)/(split*WK)-kb0;
  uint64_t pw,px;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0,1.0;":"=l"(pw));
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0,1.0;":"=l"(px));
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  auto loadx=[&](int i) {
    if(tid%128==0 && i<nkb) {
      int s=i%Q;
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
      tma(base+s*SB,&tx,(kb0+i)*64,mt*BN,full+s*8,px);
    }
  };
  // PDL: initial activation TMA follows the independent weight preamble.
  uint32_t e0=0xffffffffu,e1=0xffffffffu,nxt=0xffffffffu;int ep=0;
  if(C12 && warp<RW) {
    const uint32_t* el=esc+((nt*split*WK+rank*WK+wg)*RW+warp)*64;
    e0=el[lane];e1=el[lane+32];nxt=__shfl_sync(0xffffffffu,e0,0);
  }
  uint4 raw[PF][NL]={};
  auto load=[&](int slot,int i) {
    if(i<nkb && warp<RW) {
      const uint4* src=w+((long)(nt*(k/64)+kb0+i)*RW+warp)*NL*32+lane;
      #pragma unroll
      for(int l=0;l<NL;l++)raw[slot][l]=ldw(src+l*32,pw);
    }
  };
  #pragma unroll
  for(int p=0;p<PF;p++)load(p,p);
  pdlu32_wait(e0 ^ e1 ^ raw[0][0].x ^ raw[0][0].y ^ raw[0][0].z ^ raw[0][0].w ^ raw[0][1].x ^ raw[0][1].y ^ raw[0][1].z ^ raw[0][1].w ^ raw[0][2].x ^ raw[0][2].y ^ raw[0][2].z ^ raw[0][2].w ^ raw[1][0].x ^ raw[1][0].y ^ raw[1][0].z ^ raw[1][0].w ^ raw[1][1].x ^ raw[1][1].y ^ raw[1][1].z ^ raw[1][1].w ^ raw[1][2].x ^ raw[1][2].y ^ raw[1][2].z ^ raw[1][2].w);
  #pragma unroll
  for(int s=0;s<Q;s++)loadx(s);
  float acc[BN/2];
  #pragma unroll
  for(int j=0;j<BN/2;j++)asm volatile("mov.b32 %0,0;":"=f"(acc[j]));
  #pragma unroll 1
  for(int ib=0;ib<nkb;ib+=PF) {
    #pragma unroll
    for(int p=0;p<PF;p++) {
      int i=ib+p;if(i>=nkb)break;
      uint32_t a[16];
      if constexpr(C12) {
        dec(raw[p][0].x,raw[p][0].y,raw[p][2].x,t0,t1,a);
        dec(raw[p][0].z,raw[p][0].w,raw[p][2].y,t0,t1,a+4);
        dec(raw[p][1].x,raw[p][1].y,raw[p][2].z,t0,t1,a+8);
        dec(raw[p][1].z,raw[p][1].w,raw[p][2].w,t0,t1,a+12);
      } else {
        #pragma unroll
        for(int l=0;l<4;l++) {
          a[4*l]=raw[p][l].x;a[4*l+1]=raw[p][l].y;a[4*l+2]=raw[p][l].z;a[4*l+3]=raw[p][l].w;
        }
      }
      load(p,i+PF);
      if constexpr(C12) {
        while((nxt>>18)==uint32_t(i)) {
          if(lane==int((nxt>>13)&31u))patch(a,nxt);
          ++ep;
          uint32_t ev=ep<32?e0:e1;
          nxt=__shfl_sync(0xffffffffu,ev,ep&31);
          if(ep>=64)nxt=0xffffffffu;
        }
      }
      int s=i%Q;
      waitbar(full+s*8,(i/Q)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");
      if constexpr(BN==16)mma16(acc,a,desc(base+s*SB));
      if constexpr(BN==32)mma32(acc,a,desc(base+s*SB));
      if constexpr(BN==64)mma64(acc,a,desc(base+s*SB));
      #pragma unroll
      for(int j=0;j<BN/2;j++)asm volatile("":"+f"(acc[j])::"memory");
      loadx(i+Q);
    }
  }
  __syncthreads();
  if(tid%128==0) {
    for(int s=0;s<Q;s++)asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(full+s*8) : "memory");
  }
  __syncthreads();
  int row=warp*16+lane/4,col=2*(lane%4);
  #pragma unroll
  for(int j=0;j<BN/8;j++) {
    #pragma unroll
    for(int h=0;h<2;h++) {
      if(warp>=RW)continue;
      ((float*)sm)[(wg*BN+col+8*j)*RP+row+8*h]=acc[4*j+2*h];
      ((float*)sm)[(wg*BN+col+8*j+1)*RP+row+8*h]=acc[4*j+2*h+1];
    }
  }
  __syncthreads();
  if(split>1)cluster_sync();
  for(int q=tid+rank*NT;q<BN*(R/4);q+=NT*split) {
    int token=q/(R/4),row=(q%(R/4))*4;
    if(mt*BN+token>=m)continue;
    float4 sum=make_float4(0,0,0,0);
    for(int pp=0;pp<split*WK;pp++) {
      int s=pp/WK;
      uint32_t remote=base0+((pp%WK*BN+token)*RP+row)*4;
      if(split>1)asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(remote):"r"(remote),"r"(s));
      float4 v;
      asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3},[%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"r"(remote):"memory");
      sum.x+=v.x;sum.y+=v.y;sum.z+=v.z;sum.w+=v.w;
    }
    __nv_bfloat162 lo=__floats2bfloat162_rn(sum.x,sum.y),hi=__floats2bfloat162_rn(sum.z,sum.w);
    *(uint2*)(out+(long)(mt*BN+token)*n+nt*R+row)=make_uint2(*(uint32_t*)&lo,*(uint32_t*)&hi);
  }
  __syncthreads();
  if(split>1)cluster_sync();
}


using EncodeFn=decltype(&cuTensorMapEncodeTiled);
int map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {
  static EncodeFn encode=nullptr;
  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}
  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}


template<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>
int launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {
  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;
  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;
  auto kern=pdlu32_gemm<BN,R,WK,PF,Q,C12,CAP>;
  static bool ready=false;
  if(!ready) {
    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;
    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;
    ready=true;
  }
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;
  cudaLaunchAttribute attr[2]={};
  int na=0;
  if(split>1){attr[na].id=cudaLaunchAttributeClusterDimension;attr[na].val.clusterDim={unsigned(split),1,1};++na;}
  attr[na].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[na].val.programmaticStreamSerializationAllowed=1;++na;
  cfg.attrs=attr;cfg.numAttrs=na;
  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));
}
int launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {
  switch(id) {case 0: return launch_t<32,64,2,2,2,true,2>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);}
  return -1;
}
int pdlu32_occupancy(){
  auto kern=pdlu32_gemm<32,64,2,2,2,true,2>;
  cudaError_t rc=cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,18432);
  if(rc!=cudaSuccess)return -static_cast<int>(rc);
  rc=cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1);
  if(rc!=cudaSuccess)return -static_cast<int>(rc);
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(256,1,1);cfg.blockDim=dim3(256,1,1);cfg.dynamicSmemBytes=18432;
  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={2,1,1};
  cfg.attrs=&attr;cfg.numAttrs=1;
  int clusters=0;rc=cudaOccupancyMaxActiveClusters(&clusters,kern,&cfg);
  return rc==cudaSuccess?clusters:-static_cast<int>(rc);
}
int info(int id){cudaFuncAttributes fa={};switch(id){case 0:cudaFuncGetAttributes(&fa,pdlu32_gemm<32,64,2,2,2,true,2>);break;}return fa.numRegs*10000+fa.localSizeBytes;}
