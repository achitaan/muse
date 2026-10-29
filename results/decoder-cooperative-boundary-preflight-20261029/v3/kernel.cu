
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include <type_traits>
#include <cooperative_groups.h>

#define SMALLM2D __device__ __forceinline__

SMALLM2D int4 smallm2_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMALLM2D uint4 smallm2_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SMALLM2D uint32_t smallm2_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SMALLM2D uint32_t smallm2_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SMALLM2D void smallm2_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SMALLM2D void smallm2_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high=(C & 0x87878787u)+T0;
  const uint32_t high_high=(rotated & 0x87878787u)+T0;
  a[0]=smallm2_prmt(L0,low_high,0x5140u);
  a[1]=smallm2_prmt(L0,low_high,0x7362u);
  a[2]=smallm2_prmt(L1,high_high,0x5140u);
  a[3]=smallm2_prmt(L1,high_high,0x7362u);
}

__device__ __noinline__ void smallm2_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SMALLM2D void smallm2_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) smallm2_never(p);
}

SMALLM2D uint32_t smallm2_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SMALLM2D uint32_t smallm2_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SMALLM2D uint32_t smallm2_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SMALLM2D void smallm2_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SMALLM2D void smallm2_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SMALLM2D void smallm2_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SMALLM2D void smallm2_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SMALLM2D void smallm2_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SMALLM2D void smallm2_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SMALLM2D bool smallm2_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SMALLM2D uint64_t smallm2_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a device fault instead of consuming unfinished data.
SMALLM2D void smallm2_wait_bar(uint32_t bar) {
  if (smallm2_mbar_try(bar)) return;
  const uint64_t t0 = smallm2_gtime();
  while (!smallm2_mbar_try(bar))
    if (smallm2_gtime() - t0 > 20000000ull) __trap();
}

SMALLM2D void smallm2_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SMALLM2D float4 smallm2_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SMALLM2D void smallm2_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SMALLM2D void smallm2_store(float* p, float v) { *p = v; }

template <int FMT> struct Smallm2Fmt;
template <> struct Smallm2Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct Smallm2Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct Smallm2Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct Smallm2Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SMALLM2D void smallm2_decode(const int4 (&w)[Smallm2Fmt<FMT>::NL], uint32_t (&A)[Smallm2Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < Smallm2Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < Smallm2Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      smallm2_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      smallm2_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      smallm2_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      smallm2_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SMALLM2D void smallm2_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SMALLM2D void smallm2_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) smallm2_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
#ifndef OKC_COOP_BOUNDARY_HELPERS
#define OKC_COOP_BOUNDARY_HELPERS

struct CoopBoundary {
  const float *stream, *alpha, *beta;
  const __nv_bfloat16 *pre_weight;
  float *new_stream, *router;
  __nv_bfloat16 *output;
  float post_eps, pre_eps;
};
__device__ __forceinline__ float coop_sum(float value, float *scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int d=16; d; d>>=1) value += __shfl_down_sync(0xffffffff, value, d);
  if (!lane) scratch[warp] = value;
  __syncthreads();
  if (!warp) {
    value = lane < blockDim.x/32 ? scratch[lane] : 0.f;
    for (int d=16; d; d>>=1) value += __shfl_down_sync(0xffffffff, value, d);
    if (!lane) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}
// Three streaming passes avoid keeping a full FP32 residual row and all gate
// vectors live in the GEMV register allocation. They deliberately add reads.
__device__ __forceinline__ void coop_boundary(
    const __nv_bfloat16 *branch, int m, CoopBoundary p, float *scratch) {
  cooperative_groups::this_grid().sync();
  const int row = blockIdx.x;
  if (row >= m) return;
  constexpr int H = 8192;
  const size_t base = (size_t)row * H;
  float sum = 0.f;
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float b = __bfloat162float(branch[base+h]);
    sum = fmaf(b,b,sum);
  }
  const float post = rsqrtf(coop_sum(sum,scratch)/H+p.post_eps);
  float pre_sum = 0.f;
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float b = __bfloat162float(branch[base+h]);
    const float normed = __bfloat162float(__float2bfloat16_rn(b*post));
    const float a = __fmul_rn(p.alpha[h],p.stream[base+h]);
    const float c = __fmul_rn(p.beta[h],normed);
    const float value = __fadd_rn(a,c);
    p.new_stream[base+h] = value;
    pre_sum = fmaf(value,value,pre_sum);
  }
  // Separate scratch prevents a WAR race on the first returned sum.
  const float pre = rsqrtf(coop_sum(pre_sum,scratch+32)/H+p.pre_eps);
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float value = __fmul_rn(__fmul_rn(p.new_stream[base+h],pre),
                                 __bfloat162float(p.pre_weight[h]));
    p.output[base+h] = __float2bfloat16_rn(value);
    // Exact selected CUDA source retains FP32 here with pre_norm_bf16=false.
    // Native/Triton router alternatives must not be silently substituted.
    p.router[base+h] = value;
  }
}

#endif

template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
__global__ void __launch_bounds__(NTMAX) cb_smallm2_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1, CoopBoundary boundary) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = Smallm2Fmt<FMT>::NL, NA = Smallm2Fmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
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
    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = smallm2_smem_u32(&s_bar);
    uint32_t rank = 0, my_e = 0xFFFFFFFFu;
    if constexpr (!(OPT & 1)) {
    if constexpr (PUSH) {
      rank = smallm2_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smallm2_mbar_init(bar, 1);
        smallm2_mbar_expect(bar, parts * RT * TV * 128u);
        smallm2_fence_init();
      }
      smallm2_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);

    }

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = smallm2_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && !(OPT & 2)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = smallm2_ldx(xp);
          xv[slot][n][1] = smallm2_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
    if constexpr (OPT & 1) {
    if constexpr (PUSH) {
      rank = smallm2_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        smallm2_mbar_init(bar, 1);
        smallm2_mbar_expect(bar, parts * RT * TV * 128u);
        smallm2_fence_init();
      }
      smallm2_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = smallm2_lde(esc + (size_t)gw * 32 + lane);

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
          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr (OPT & 2) {
              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;
              uint4 u = smallm2_ldx(xp), v = smallm2_ldx(xp + 8);
              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;
              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) smallm2_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) smallm2_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) smallm2_patch<NA * 16>(A[j], nxt);
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
                  smallm2_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) smallm2_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
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
          smallm2_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) smallm2_store(out, 1.f);
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
                smallm2_store(out + (size_t)tok * N + row, r[j][n][c]);
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
            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          if constexpr (OPT & 8) { emit_full(idx, v); return; }
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) smallm2_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)
              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            smallm2_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) smallm2_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            smallm2_cl_wait();
            if (t < TV) {
              const uint32_t rbar = smallm2_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            smallm2_wait_bar(bar);
            smallm2_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            smallm2_cl_wait();
            const uint32_t rbar = smallm2_mapa(bar, 0);
            smallm2_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else smallm2_st_async4(smallm2_mapa(smallm2_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            smallm2_wait_bar(bar);
            smallm2_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
  coop_boundary(reinterpret_cast<const __nv_bfloat16*>(outp), m, boundary, reinterpret_cast<float*>(smem));
}



#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include <type_traits>
#include <cooperative_groups.h>

#define SM3D __device__ __forceinline__

SM3D int4 v18s_ldw(const int4* p) {
  int4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.s32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint4 v18s_ldx(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

SM3D uint32_t v18s_lde(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

SM3D uint32_t v18s_prmt(uint32_t a, uint32_t b, uint32_t s) {
  uint32_t d;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s));
  return d;
}

SM3D void v18s_mma(float (&c)[4], const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 8 weights of one k16 step: low bytes in L0 (values 0-3) and L1 (values 4-7), 4-bit codes in C.
SM3D void v18s_dec(uint32_t L0, uint32_t L1, uint32_t C, uint32_t T0, uint32_t T1, uint32_t* a) {
  uint32_t rotated;
  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));
  const uint32_t low_high=(C & 0x87878787u)+T0;
  const uint32_t high_high=(rotated & 0x87878787u)+T0;
  a[0]=v18s_prmt(L0,low_high,0x5140u);
  a[1]=v18s_prmt(L0,low_high,0x7362u);
  a[2]=v18s_prmt(L1,high_high,0x5140u);
  a[3]=v18s_prmt(L1,high_high,0x7362u);
}

__device__ __noinline__ void v18s_never(void* p) { asm volatile("trap;"); }

// Never taken, but ptxas cannot prove it: ends the basic block so loads issue where written.
SM3D void v18s_order_point(int m, void* p) {
  uint32_t t;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
  if ((t & (uint32_t)m) == 0x7fffffffu) v18s_never(p);
}

SM3D uint32_t v18s_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

SM3D uint32_t v18s_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

SM3D uint32_t v18s_mapa(uint32_t a, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(rank));
  return d;
}

SM3D void v18s_mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

SM3D void v18s_mbar_expect(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(tx) : "memory");
}

SM3D void v18s_fence_init() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }
SM3D void v18s_cl_arrive() { asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory"); }
SM3D void v18s_cl_wait() { asm volatile("barrier.cluster.wait.aligned;" ::: "memory"); }

SM3D void v18s_st_async4(uint32_t ra, float4 v, uint32_t rbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1, %2, %3, %4}, [%5];"
               ::"r"(ra), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w), "r"(rbar) : "memory");
}

SM3D bool v18s_mbar_try(uint32_t bar) {
  uint32_t ok;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], 0;\n"
               " selp.u32 %0, 1, 0, p;\n}" : "=r"(ok) : "r"(bar) : "memory");
  return ok != 0;
}

SM3D uint64_t v18s_gtime() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

// Bounded: a broken transaction count shows up as a wrong result instead of a hang.
SM3D void v18s_wait_bar(uint32_t bar) {
  if (v18s_mbar_try(bar)) return;
  const uint64_t t0 = v18s_gtime();
  while (!v18s_mbar_try(bar))
    if (v18s_gtime() - t0 > 20000000ull) __trap();
}

SM3D void v18s_add4(float4& a, const float4& b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w; }

SM3D float4 v18s_shfl_xor4(const float4& v, int o) {
  return make_float4(__shfl_xor_sync(0xffffffffu, v.x, o), __shfl_xor_sync(0xffffffffu, v.y, o),
                     __shfl_xor_sync(0xffffffffu, v.z, o), __shfl_xor_sync(0xffffffffu, v.w, o));
}

SM3D void v18s_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }
SM3D void v18s_store(float* p, float v) { *p = v; }

template <int FMT> struct Sm3Fmt;
template <> struct Sm3Fmt<0> { static constexpr int NL = 4, NA = 1; };
template <> struct Sm3Fmt<1> { static constexpr int NL = 3, NA = 1; };
template <> struct Sm3Fmt<2> { static constexpr int NL = 8, NA = 2; };
template <> struct Sm3Fmt<3> { static constexpr int NL = 6, NA = 2; };

template <int FMT>
SM3D void v18s_decode(const int4 (&w)[Sm3Fmt<FMT>::NL], uint32_t (&A)[Sm3Fmt<FMT>::NA * 16], uint32_t T0, uint32_t T1,
                   uint32_t U0, uint32_t U1) {
  if constexpr (FMT == 0 || FMT == 2) {
#pragma unroll
    for (int l = 0; l < Sm3Fmt<FMT>::NL; ++l) {
      A[4 * l + 0] = (uint32_t)w[l].x;
      A[4 * l + 1] = (uint32_t)w[l].y;
      A[4 * l + 2] = (uint32_t)w[l].z;
      A[4 * l + 3] = (uint32_t)w[l].w;
    }
  } else {
#pragma unroll
    for (int h = 0; h < Sm3Fmt<FMT>::NA; ++h) {
      const int4* u = w + 3 * h;
      const uint32_t t0 = h ? U0 : T0, t1 = h ? U1 : T1;
      v18s_dec(u[0].x, u[0].y, u[2].x, t0, t1, A + 16 * h + 0);
      v18s_dec(u[0].z, u[0].w, u[2].y, t0, t1, A + 16 * h + 4);
      v18s_dec(u[1].x, u[1].y, u[2].z, t0, t1, A + 16 * h + 8);
      v18s_dec(u[1].z, u[1].w, u[2].w, t0, t1, A + 16 * h + 12);
    }
  }
}

// Escape entry: seq << 18 | lane << 13 | register << 8 | half << 7 | 7-bit XOR of the high byte's exponent bits.
template <int NR>
SM3D void v18s_patch(uint32_t (&A)[NR], uint32_t e) {
  const uint32_t reg = (e >> 8) & 31u;
  const uint32_t xr = (e & 0x7Fu) << (8 + ((e >> 7) & 1u) * 16);
#pragma unroll
  for (int r = 0; r < NR; ++r)
    if (reg == (uint32_t)r) A[r] ^= xr;
}

// Adjacent lanes read adjacent float4 slots, with each lane summing that slot's split-K partials.
template <class Emit>
SM3D void v18s_reduce(const float4* src, int NI, int P, int PSTR, int tid, int nt, Emit emit) {
  for (int idx = tid; idx < NI; idx += nt) {
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int p = 0; p < P; ++p) v18s_add4(v, src[(size_t)p * PSTR + idx]);
    emit(idx, v);
  }
}

// MT n8 token tiles, RT 16-row tiles per warp, ST units in flight per row tile. DIAG (timing only): 1 no MMA,
// 2 no x loads, 4 no reduction, 8 loads only, 16 empty kernel, 32 (with 8) plain output stores, 64 (with 8) a 4 KB
// straight-line code block at the end, 128 (with 4) no output stores.
#ifndef OKC_COOP_BOUNDARY_HELPERS
#define OKC_COOP_BOUNDARY_HELPERS

struct CoopBoundary {
  const float *stream, *alpha, *beta;
  const __nv_bfloat16 *pre_weight;
  float *new_stream, *router;
  __nv_bfloat16 *output;
  float post_eps, pre_eps;
};
__device__ __forceinline__ float coop_sum(float value, float *scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int d=16; d; d>>=1) value += __shfl_down_sync(0xffffffff, value, d);
  if (!lane) scratch[warp] = value;
  __syncthreads();
  if (!warp) {
    value = lane < blockDim.x/32 ? scratch[lane] : 0.f;
    for (int d=16; d; d>>=1) value += __shfl_down_sync(0xffffffff, value, d);
    if (!lane) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}
// Three streaming passes avoid keeping a full FP32 residual row and all gate
// vectors live in the GEMV register allocation. They deliberately add reads.
__device__ __forceinline__ void coop_boundary(
    const __nv_bfloat16 *branch, int m, CoopBoundary p, float *scratch) {
  cooperative_groups::this_grid().sync();
  const int row = blockIdx.x;
  if (row >= m) return;
  constexpr int H = 8192;
  const size_t base = (size_t)row * H;
  float sum = 0.f;
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float b = __bfloat162float(branch[base+h]);
    sum = fmaf(b,b,sum);
  }
  const float post = rsqrtf(coop_sum(sum,scratch)/H+p.post_eps);
  float pre_sum = 0.f;
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float b = __bfloat162float(branch[base+h]);
    const float normed = __bfloat162float(__float2bfloat16_rn(b*post));
    const float a = __fmul_rn(p.alpha[h],p.stream[base+h]);
    const float c = __fmul_rn(p.beta[h],normed);
    const float value = __fadd_rn(a,c);
    p.new_stream[base+h] = value;
    pre_sum = fmaf(value,value,pre_sum);
  }
  // Separate scratch prevents a WAR race on the first returned sum.
  const float pre = rsqrtf(coop_sum(pre_sum,scratch+32)/H+p.pre_eps);
  for (int h=threadIdx.x; h<H; h+=blockDim.x) {
    const float value = __fmul_rn(__fmul_rn(p.new_stream[base+h],pre),
                                 __bfloat162float(p.pre_weight[h]));
    p.output[base+h] = __float2bfloat16_rn(value);
    // Exact selected CUDA source retains FP32 here with pre_norm_bf16=false.
    // Native/Triton router alternatives must not be silently substituted.
    p.router[base+h] = value;
  }
}

#endif

template <int FMT, int MT, int RT, int ST, int RED, int DIAG, int NTMAX, int CM, int CN, int CK, int CWR, int CWK, int CKU, int CS, int OPT>
__global__ void __launch_bounds__(NTMAX) cb_v18s_gemv(
    const int4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, void* __restrict__ outp,
    const uint32_t* __restrict__ esc, int m_arg, int N_arg, int K_arg, int WR_arg, int WK_arg, int KU_arg, uint32_t T0, uint32_t T1,
    uint32_t U0, uint32_t U1, CoopBoundary boundary) {
  const int m = CM ? CM : m_arg;
  const int N = CN ? CN : N_arg, K = CK ? CK : K_arg;
  const int WR = CWR ? CWR : WR_arg, WK = CWK ? CWK : WK_arg, KU = CKU ? CKU : KU_arg;
  constexpr int NL = Sm3Fmt<FMT>::NL, NA = Sm3Fmt<FMT>::NA;
  constexpr bool C12 = FMT == 1 || FMT == 3;
  constexpr bool HILO = FMT >= 2;
  constexpr bool NOMMA = DIAG & 1, NOX = DIAG & 2, NORED = DIAG & 4, RO = DIAG & 8, EMPTY = DIAG & 16;
  constexpr bool RO_STORE = DIAG & 32, RO_CODE = DIAG & 64, NOSTORE = DIAG & 128;
  constexpr bool PUSH = (RED == 2 || RED == 3) && !NORED && !RO && !EMPTY;
  using OutT = typename std::conditional<HILO, float, __nv_bfloat16>::type;
  OutT* out = reinterpret_cast<OutT*>(outp);
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ __align__(8) uint64_t s_bar;
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
    const int TV = (OPT & 8) ? 4 * MT : min(4 * MT, (m + 1) >> 1);
    const uint32_t bar = v18s_smem_u32(&s_bar);
    uint32_t rank = 0, my_e = 0xFFFFFFFFu;
    if constexpr (!(OPT & 1)) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

    }

    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;
    const size_t sstride = (size_t)G * (RT * NL * 32);
    const __nv_bfloat16* x0 = x + (size_t)c0 * 64 + (lane & 3) * 16;

    int4 w[ST][RT][NL];
    uint4 xv[ST][MT][2];
    auto load = [&](int slot, int s) {
#pragma unroll
      for (int j = 0; j < RT; ++j)
#pragma unroll
        for (int l = 0; l < NL; ++l) w[slot][j][l] = v18s_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);
      if constexpr (!NOX && !RO && !(OPT & 2)) {
#pragma unroll
        for (int n = 0; n < MT; ++n) {
          const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + s * 64;
          xv[slot][n][0] = v18s_ldx(xp);
          xv[slot][n][1] = v18s_ldx(xp + 8);
        }
      }
    };
#pragma unroll
    for (int s = 0; s < ST; ++s) load(s, s);
    if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
    if constexpr (OPT & 1) {
    if constexpr (PUSH) {
      rank = v18s_rank();
      if (rank == 0 && tid == 0) {
        const uint32_t parts = RED == 2 ? (uint32_t)((S - 1) * NW) : (uint32_t)((S - 1) * WR);
        v18s_mbar_init(bar, 1);
        v18s_mbar_expect(bar, parts * RT * TV * 128u);
        v18s_fence_init();
      }
      v18s_cl_arrive();
    }

    if constexpr (C12 && !RO) my_e = v18s_lde(esc + (size_t)gw * 32 + lane);

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
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
        } else {
          uint32_t b[MT][8];
#pragma unroll
          for (int n = 0; n < MT; ++n) {
            if constexpr (NOX) {
#pragma unroll
              for (int e = 0; e < 8; ++e) b[n][e] = 0x3f803f80u;
            } else if constexpr (OPT & 2) {
              const __nv_bfloat16* xp = x0 + (size_t)min(n * 8 + (lane >> 2), m - 1) * K + si * 64;
              uint4 u = v18s_ldx(xp), v = v18s_ldx(xp + 8);
              b[n][0]=u.x; b[n][1]=u.y; b[n][2]=u.z; b[n][3]=u.w;
              b[n][4]=v.x; b[n][5]=v.y; b[n][6]=v.z; b[n][7]=v.w;
            } else {
              b[n][0] = xv[s][n][0].x; b[n][1] = xv[s][n][0].y; b[n][2] = xv[s][n][0].z; b[n][3] = xv[s][n][0].w;
              b[n][4] = xv[s][n][1].x; b[n][5] = xv[s][n][1].y; b[n][6] = xv[s][n][1].z; b[n][7] = xv[s][n][1].w;
            }
          }
          uint32_t A[RT][NA * 16];
#pragma unroll
          for (int j = 0; j < RT; ++j) v18s_decode<FMT>(w[s][j], A[j], T0, T1, U0, U1);
          if (si + ST < KU) load(s, si + ST);
          if constexpr (!(OPT & 4)) v18s_order_point(m, outp);
          if constexpr (C12) {
#pragma unroll
            for (int j = 0; j < RT; ++j) {
              const uint32_t useq = (uint32_t)(si * RT + j);
              if ((nxt >> 18) == useq) {
                do {
                  if (lane == (int)((nxt >> 13) & 31u)) v18s_patch<NA * 16>(A[j], nxt);
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
                  v18s_mma(acc[0][j][n], A[j] + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                  if constexpr (HILO) v18s_mma(acc[NA - 1][j][n], A[j] + 16 + 4 * q, b[n][2 * q], b[n][2 * q + 1]);
                }
              }
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
          v18s_store(out + rt0 * 16 + (lane >> 2), __uint_as_float(fold & 0x3fffffffu));
      } else {
        if (fold == 0x9e3779b9u) v18s_store(out, 1.f);
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
                v18s_store(out + (size_t)tok * N + row, r[j][n][c]);
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
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        auto emit_compact = [&](int idx, float4 v) {
          if constexpr (OPT & 8) { emit_full(idx, v); return; }
          const int g = idx & 7, rest = idx >> 3, t = rest % TV, tile = rest / TV;
          const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            const int tok = 2 * t + (c & 1);
            const int row = row0 + tile * 16 + g + 8 * (c >> 1);
            if (tok < m) v18s_store(out + (size_t)tok * N + row, vv[c]);
          }
        };
        if constexpr (RED == 0 || RED == 1) {
          float4* part = reinterpret_cast<float4*>(smem);
          const int NI = (OPT & 16) ? WR * RT * TV * 8 : WR * RT * MT * 32;
#pragma unroll
          for (int j = 0; j < RT; ++j)
#pragma unroll
            for (int n = 0; n < MT; ++n)
              if (!(OPT & 16) || n * 4 + (lane & 3) < TV)
              part[(OPT & 16) ? (size_t)wk * NI + ((wr * RT + j) * TV + n * 4 + (lane & 3)) * 8 + (lane >> 2) : ((size_t)warp * RT * MT + j * MT + n) * 32 + lane] =
                  make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
          __syncthreads();
          if constexpr (RED == 0) {
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { if constexpr (OPT & 16) emit_compact(idx, v); else emit_full(idx, v); });
          } else {
            float4* csum = part + (size_t)NW * RT * MT * 32;
            v18s_reduce(part, NI, WK, NI, tid, nt, [&](int idx, float4 v) { csum[idx] = v; });
            cooperative_groups::cluster_group cl = cooperative_groups::this_cluster();
            cl.sync();
            const int rk = (int)cl.block_rank();
            const int per = (NI + S - 1) / S;
            const int e1 = min(NI, (rk + 1) * per);
            for (int idx = rk * per + tid; idx < e1; idx += nt) {
              float4 v = *cl.map_shared_rank(csum + idx, 0);
              for (int q = 1; q < S; ++q) v18s_add4(v, *cl.map_shared_rank(csum + idx, q));
              emit_full(idx, v);
            }
            cl.sync();
          }
        } else {
          const int NIc = WR * RT * TV * 8;
          const int t = lane & 3, g = lane >> 2;
          float4* recv = reinterpret_cast<float4*>(smem);
          // The leader's own partials use plain shared stores; only remote CTAs push with st.async.
          if constexpr (RED == 2) {
            v18s_cl_wait();
            if (t < TV) {
              const uint32_t rbar = v18s_mapa(bar, 0);
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n) {
                  const int tn = n * 4 + t;
                  if (tn < TV) {
                    const int idx = ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + tn) * 8 + g);
                    float4* dst = recv + (size_t)(rank * WK + wk) * NIc + idx;
                    const float4 v = make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
                    if (rank == 0) *dst = v;
                    else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
                  }
                }
            }
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S * WK, NIc, tid, nt, emit_compact);
          } else {
            float4* part = recv + (size_t)S * NIc;
            if (t < TV) {
#pragma unroll
              for (int j = 0; j < RT; ++j)
#pragma unroll
                for (int n = 0; n < MT; ++n)
                  if (n * 4 + t < TV)
                    part[(size_t)wk * NIc + ((OPT & 8) ? ((wr * RT + j) * MT + n) * 32 + lane : ((wr * RT + j) * TV + n * 4 + t) * 8 + g)] =
                        make_float4(r[j][n][0], r[j][n][1], r[j][n][2], r[j][n][3]);
            }
            __syncthreads();
            v18s_cl_wait();
            const uint32_t rbar = v18s_mapa(bar, 0);
            v18s_reduce(part, NIc, WK, NIc, tid, nt, [&](int idx, float4 v) {
              float4* dst = recv + (size_t)rank * NIc + idx;
              if (rank == 0) *dst = v;
              else v18s_st_async4(v18s_mapa(v18s_smem_u32(dst), 0), v, rbar);
            });
            if (rank != 0) return;
            __syncthreads();
            v18s_wait_bar(bar);
            v18s_reduce(recv, NIc, S, NIc, tid, nt, emit_compact);
          }
        }
      }
    }
  }
  coop_boundary(reinterpret_cast<const __nv_bfloat16*>(outp), m, boundary, reinterpret_cast<float*>(smem));
}


// Local cooperative O1/O8 launch admission; inherited device code is attributed
// in THIRD_PARTY_NOTICES.md and the integrated submission's full notices.
template <int FAMILY, class Kernel>
static int64_t cb_launch_t(Kernel kernel, int fmt, int query, int m,
    int64_t wp_arg, int64_t x_arg, int64_t branch_arg, int64_t esc_arg,
    int64_t stream_arg, int64_t alpha_arg, int64_t beta_arg, int64_t gain_arg,
    int64_t new_stream_arg, int64_t output_arg, int64_t router_arg,
    int64_t t01, int64_t u01, int64_t cuda_stream_arg, float post_eps, float pre_eps) {
  constexpr int NT = FAMILY == 1 ? 1024 : 256;
  constexpr int GRID = FAMILY == 1 ? 128 : 512;
  constexpr int WR = FAMILY == 1 ? 4 : 1;
  constexpr size_t SHARED = FAMILY == 1 ? 16384 : 4096;
  if (query) {
    int device = 0, sm_count = 0, supported = 0, blocks = 0;
    cudaFuncAttributes attributes{};
    if (cudaGetDevice(&device) != cudaSuccess) return -101;
    if (cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,device) != cudaSuccess) return -102;
    if (cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device) != cudaSuccess) return -103;
    if (cudaFuncGetAttributes(&attributes,kernel) != cudaSuccess) return -104;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,NT,SHARED) != cudaSuccess) return -105;
    switch(query) {
      case 1: return supported && GRID <= sm_count*blocks;
      case 2: return attributes.numRegs;
      case 3: return attributes.localSizeBytes;
      case 4: return attributes.sharedSizeBytes;
      case 5: return blocks;
      case 6: return sm_count;
      case 7: return supported;
      case 8: return SHARED;
      default: return -106;
    }
  }
  // Resource admission was already done in prepare, outside graph capture.
  // Reject unsupported shapes before a launch, preserving original fallbacks.
  if (m < 1 || m > 8 || (FAMILY == 1 && m != 1)) return -107;
  const int4* wp = reinterpret_cast<const int4*>(wp_arg);
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(x_arg);
  void* branch = reinterpret_cast<void*>(branch_arg);
  const uint32_t* esc = reinterpret_cast<const uint32_t*>(esc_arg);
  CoopBoundary params{
    reinterpret_cast<const float*>(stream_arg),
    reinterpret_cast<const float*>(alpha_arg),
    reinterpret_cast<const float*>(beta_arg),
    reinterpret_cast<const __nv_bfloat16*>(gain_arg),
    reinterpret_cast<float*>(new_stream_arg),
    reinterpret_cast<float*>(router_arg),
    reinterpret_cast<__nv_bfloat16*>(output_arg),post_eps,pre_eps};
  cudaLaunchAttribute cooperative{};
  cooperative.id = cudaLaunchAttributeCooperative;
  cooperative.val.cooperative = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(GRID,1,1);
  config.blockDim = dim3(NT,1,1);
  config.dynamicSmemBytes = SHARED;
  config.stream = reinterpret_cast<cudaStream_t>(cuda_stream_arg);
  config.attrs = &cooperative;
  config.numAttrs = 1;
  cudaError_t status = cudaLaunchKernelEx(&config,kernel,wp,x,branch,esc,
      m,8192,1024,WR,8,2,
      static_cast<uint32_t>(t01),static_cast<uint32_t>(static_cast<uint64_t>(t01)>>32),
      static_cast<uint32_t>(u01),static_cast<uint32_t>(static_cast<uint64_t>(u01)>>32),params);
  return status == cudaSuccess ? 0 : -1000-static_cast<int>(status);
}

int64_t cb_launch(int family, int fmt, int query, int m,
    int64_t wp, int64_t x, int64_t branch, int64_t esc,
    int64_t stream, int64_t alpha, int64_t beta, int64_t gain,
    int64_t new_stream, int64_t output, int64_t router,
    int64_t t01, int64_t u01, int64_t cuda_stream, float post_eps, float pre_eps) {
#define CB_ARGS fmt,query,m,wp,x,branch,esc,stream,alpha,beta,gain,new_stream,output,router,t01,u01,cuda_stream,post_eps,pre_eps
  if (family == 1) {
    if (fmt == 1) return cb_launch_t<1>(cb_smallm2_gemv<1,1,1,2,0,0,1024,0,8192,1024,4,8,2,1,7>,CB_ARGS);
    if (fmt == 0) return cb_launch_t<1>(cb_smallm2_gemv<0,1,1,2,0,0,1024,0,8192,1024,4,8,2,1,7>,CB_ARGS);
  }
  if (family == 8) {
    if (fmt == 1) return cb_launch_t<8>(cb_v18s_gemv<1,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6>,CB_ARGS);
    if (fmt == 0) return cb_launch_t<8>(cb_v18s_gemv<0,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6>,CB_ARGS);
  }
#undef CB_ARGS
  return -108;
}
