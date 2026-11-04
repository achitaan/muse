// SYNTAX ONLY: CUDA/ATen stubbed and PTX removed. DO NOT EXECUTE.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <type_traits>
#define __device__
#define __global__
#define __forceinline__ inline
#define __grid_constant__
#define __shared__
#define __align__(...)
#define __launch_bounds__(...)
#define C10_CUDA_CHECK(...) do { (void)(__VA_ARGS__); } while(false)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() do {} while(false)
#define TORCH_CHECK(cond, ...) do { if(!(cond)) throw 1; } while(false)
using std::min;
struct float2 { float x,y; }; struct float4 { float x,y,z,w; };
struct uint2 { uint32_t x,y; }; struct uint4 { uint32_t x,y,z,w; }; struct int2 { int x,y; }; struct int4 { int x,y,z,w; };
inline float2 make_float2(float a,float b){return {a,b};}
inline float4 make_float4(float a,float b,float c,float d){return {a,b,c,d};}
inline uint2 make_uint2(uint32_t a,uint32_t b){return {a,b};}
inline int2 make_int2(int a,int b){return {a,b};}
struct __nv_bfloat16 { uint16_t v; }; struct __nv_bfloat162 { uint32_t v; };
inline float __bfloat162float(__nv_bfloat16){return 0;}
inline float2 __bfloat1622float2(__nv_bfloat162){return {};}
inline __nv_bfloat16 __float2bfloat16_rn(float){return {};}
inline __nv_bfloat162 __floats2bfloat162_rn(float,float){return {};}
inline float __uint_as_float(uint32_t){return 0;}
inline uint32_t __float_as_uint(float){return 0;}
inline uint32_t __byte_perm(uint32_t,uint32_t,uint32_t){return 0;}
inline uint32_t __funnelshift_r(uint32_t,uint32_t,unsigned){return 0;}
inline uintptr_t __cvta_generic_to_shared(const void* p){return uintptr_t(p);}
template<class T> T __ldg(const T* p){return *p;}
template<class T> T __shfl_sync(unsigned,T v,int){return v;}
template<class T> T __shfl_xor_sync(unsigned,T v,int){return v;}
inline void __syncthreads(){} inline void __syncwarp(){} inline void __threadfence(){}
struct dim3 { unsigned x=0,y=1,z=1; dim3(unsigned a=0,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
extern dim3 threadIdx,blockIdx,gridDim;
struct CUtensorMap { unsigned char opaque[128]; };
using CUresult=int; using cuuint64_t=uint64_t; using cuuint32_t=uint32_t;
using PFN_cuTensorMapEncodeTiled_v12000=CUresult(*)(...);
using cudaDriverEntryPointQueryResult=int; using cudaStream_t=void*;
constexpr int CUDA_SUCCESS=0,cudaEnableDefault=0;
constexpr int CU_TENSOR_MAP_DATA_TYPE_BFLOAT16=0,CU_TENSOR_MAP_INTERLEAVE_NONE=0,CU_TENSOR_MAP_SWIZZLE_128B=0,
 CU_TENSOR_MAP_L2_PROMOTION_L2_256B=0,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=0,cudaLaunchAttributeClusterDimension=0,
 cudaFuncAttributeNonPortableClusterSizeAllowed=0,cudaDevAttrMaxSharedMemoryPerBlockOptin=0,cudaDevAttrMultiProcessorCount=0;
struct cudaFuncAttributes { int numRegs=0; size_t localSizeBytes=0,sharedSizeBytes=0; int maxThreadsPerBlock=0; };
struct cudaLaunchAttribute { int id=0; struct { dim3 clusterDim; } val; };
struct cudaLaunchConfig_t { dim3 gridDim,blockDim; size_t dynamicSmemBytes=0; cudaStream_t stream=nullptr;
 cudaLaunchAttribute* attrs=nullptr; unsigned numAttrs=0; };
inline int cudaGetDriverEntryPoint(const char*,void**,int,cudaDriverEntryPointQueryResult*){return 0;}
inline int cudaGetDevice(int*){return 0;} inline int cudaDeviceGetAttribute(int*,int,int){return 0;}
template<class F> int cudaFuncSetAttribute(F,int,int){return 0;}
template<class F> int cudaFuncGetAttributes(cudaFuncAttributes*,F){return 0;}
template<class F> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*,F,int,size_t){return 0;}
template<class F,class... A> int cudaLaunchKernelEx(cudaLaunchConfig_t*,F,A...){return 0;}
namespace at {
constexpr int kBFloat16=0,kFloat=1,kInt=2;
struct Tensor {
 int64_t size(int) const{return 0;} int64_t numel() const{return 0;}
 int scalar_type() const{return 0;} bool is_contiguous() const{return true;}
 void* data_ptr() const{return nullptr;} template<class T> T* data_ptr() const{return nullptr;}
};
namespace cuda { inline cudaStream_t getCurrentCUDAStream(){return nullptr;} }
}
#include <stdint.h>


__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  do {} while(false);
}

__device__ __forceinline__ void mma_bf16(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  do {} while(false);
}

__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d;
  do {} while(false);
  return d;
}

__device__ __forceinline__ uint32_t lop3_sel(uint32_t a, uint32_t b, uint32_t c) {
  // c ? b : a, bitwise
  uint32_t d;
  do {} while(false);
  return d;
}

// Decode 8 weights: low bytes of weights 0..3 in lw0, 4..7 in lw1, codes in nw
// (low nibble of byte b -> weight b, high nibble -> weight 4+b; code = s<<3 | o).
// Output: 4 bf16x2 registers r[0..3] = (w0,w1) (w2,w3) (w4,w5) (w6,w7).
__device__ __forceinline__ void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = (nw & 0x87878787u) + base4;
  uint32_t hb = (__funnelshift_r(nw, nw, 4) & 0x87878787u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}

template <typename T> __device__ __forceinline__ T cvt_out(float v);
template <> __device__ __forceinline__ float cvt_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 cvt_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

// Weights, uncompressed (COMP=0): [num_rg][S][32 lanes][16 B] A fragments, S = Kp / 16.
// Weights, compressed (COMP=1): [num_rg][S/4 units][3 parts][32 lanes][16 B]: parts 0,1 = low
//   bytes of the lane's 32 weights of 4 k-steps, part 2 = their 32 codes.
// x: [M][Kx] bf16.  Weight column k uses x column (k % Kx) (router hi|lo concat).
// CTA (rb, sp): row groups rb*RG .. +RG, k range [sp*Kc, (sp+1)*Kc).
// Warp w: row group rb*RG + w / WS, k sub-range (w % WS).
// D: prefetch depth in loads of 16 B per lane (k-steps if !COMP, unit parts if COMP).

// ------------------------------------------------------------------ mbarrier / bulk copy helpers
__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  do {} while(false);
}
__device__ __forceinline__ bool mbar_test(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  do {} while(false);
  return ok != 0;
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  while (!mbar_test(bar, parity)) {}
}
// escape correction from the stored original BF16 weight t: its 12-bit decode keeps the sign and low byte and clamps the
// exponent into [base, base + 7]; the exact difference (asserted in prepare) is formed here in FP32
__device__ __forceinline__ float esc_delta(uint32_t t, uint32_t bq) {
  const uint32_t hi = (t >> 8) & 0xFFu, e71 = hi & 0x7Fu;
  const uint32_t dhi = (hi & 0x80u) | (e71 < bq ? bq : bq + 7u);
  return __uint_as_float(t << 16) - __uint_as_float(((dhi << 8) | (t & 0xFFu)) << 16);
}
// wait with a suspend-time hint: the warp sleeps in hardware until the phase completes instead of re-polling
__device__ __forceinline__ void mbar_wait_sleep(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    do {} while(false);
  } while (!ok);
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  do {} while(false);
}
__device__ __forceinline__ void named_bar(int id, int n) {
  do {} while(false);
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  do {} while(false);
}
// L2 policies (v132: used by the 2-D TMA boxes of the m >= 33 routes only - gq/gqx/gemm90/wide): weights evict_first,
// x evict_last
__device__ __forceinline__ uint64_t l2_pol_ef() {
  uint64_t p; do {} while(false); return p;
}
__device__ __forceinline__ uint64_t l2_pol_el() {
  uint64_t p; do {} while(false); return p;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  do {} while(false);
}
// wait on a local mbarrier whose phase another CTA of the cluster completes (acquire at cluster scope)
__device__ __forceinline__ void mbar_wait_cluster(uint32_t bar, uint32_t parity) {
  uint32_t ok = 0;
  do {
    do {} while(false);
  } while (!ok);
}




#include <vector>
__device__ __forceinline__ uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}
template <int ROWS, int NTH, int K, bool ILP>
__global__ void __launch_bounds__(NTH + 32) r2_pool_kernel(const uint8_t* __restrict__ L, const uint8_t* __restrict__ C,
    const __nv_bfloat16* __restrict__ x, const int2* __restrict__ etab, int pmax,
    __nv_bfloat16* __restrict__ out, uint32_t base4) {
  constexpr int PER = K / (16 * NTH);
  constexpr int NWARP = NTH / 32;
  static_assert(PER >= 1, "cgemv: K >= 16 NTH");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int r0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][NWARP + 1];
  if (warp == NWARP) {   // escape warp: lane i takes entries 1 + i, 33 + i, ... (entry 0 = count); x[k] * delta per row
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = make_int2(0, 0);
    if (lane < pmax) mine = __ldg(&tb[lane]);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) ec[r] = 0.f;
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = (i <= cnt) ? __ldg(&tb[i]) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
        const float c = __bfloat162float(x[en.x & 0xFFFF]) * esc_delta((uint32_t)en.y, base4 & 0xFFu);
        const int rl = en.x >> 16;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) ec[r] += (rl == r) ? c : 0.f;
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      float v = ec[r];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
      if (lane == 0) red[r][NWARP] = v;
    }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER]; uint4 xv[PER][2];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t e0 = (size_t)(r0 + r) * K + (size_t)j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + e0));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + e0 / 2));
    }
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    const __nv_bfloat16* xp = x + j * 16 * NTH + 16 * tid;
    xv[j][0] = __ldg(reinterpret_cast<const uint4*>(xp));
    xv[j][1] = __ldg(reinterpret_cast<const uint4*>(xp + 8));
  }
  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    float xf[16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t* xw = reinterpret_cast<const uint32_t*>(&xv[j][h]);
#pragma unroll
      for (int e = 0; e < 4; ++e) { xf[8 * h + 2 * e] = __uint_as_float(xw[e] << 16); xf[8 * h + 2 * e + 1] = __uint_as_float(xw[e] & 0xFFFF0000u); }
    }
    if constexpr (ILP) {

#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint32_t packed[ROWS][4];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
        const uint32_t c = g ? cw[r][j].y : cw[r][j].x;
        const uint32_t ha = (c & 0x87878787u) + base4;
        const uint32_t hb = (__funnelshift_r(c, c, 4) & 0x87878787u) + base4;
        const uint32_t a = lws[g * 2], b = lws[g * 2 + 1];
        packed[r][0] = __byte_perm(a, ha, 0x5140);
        packed[r][1] = __byte_perm(a, ha, 0x7362);
        packed[r][2] = __byte_perm(b, hb, 0x5140);
        packed[r][3] = __byte_perm(b, hb, 0x7362);
      }
      // Alternate the two independent row chains. Within each row, the
      // sequence j -> g -> element is EXACTLY the original scalar GEMV.
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          const uint32_t p = packed[r][e / 2];
          const float w = __uint_as_float((e & 1) ? (p & 0xffff0000u) : (p << 16));
          acc[r] = fmaf(w, xf[g * 8 + e], acc[r]);
        }
    }

    } else {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        // Signs are prepositioned: byte MSBs for values 0..3, ROR4's MSBs for 4..7.
        // base <= 120, so offset+base+sign <= 255: no cross-byte carries.
        const uint32_t ha = (cws[g] & 0x87878787u) + base4;
        const uint32_t hb = (__funnelshift_r(cws[g], cws[g], 4) & 0x87878787u) + base4;
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = acc[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) red[r][warp] = v;
  }
  __syncthreads();
  if (tid < ROWS) {
    float v = 0.f;
#pragma unroll
    for (int q = 0; q <= NWARP; ++q) v += red[tid][q];
    out[r0 + tid] = __float2bfloat16_rn(v);
  }
}


void r2_pool(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor tab, int64_t pm, at::Tensor out, int64_t b4, int64_t ilp) {
  TORCH_CHECK(x.size(0)==1 && L.size(1)==8192 && out.size(1)%2==0 && out.size(1)<=L.size(0), "rot2 pool geometry");
  auto lp=L.data_ptr<uint8_t>(); auto cp=C.data_ptr<uint8_t>();
  auto xp=reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()); auto tp=reinterpret_cast<const int2*>(tab.data_ptr());
  auto op=reinterpret_cast<__nv_bfloat16*>(out.data_ptr()); auto st=at::cuda::getCurrentCUDAStream();
  if(ilp) r2_pool_kernel<2,512,8192,true>(lp,cp,xp,tp,pm,op,b4);
  else r2_pool_kernel<2,512,8192,false>(lp,cp,xp,tp,pm,op,b4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
template<bool ILP> std::vector<int64_t> r2_pool_info_t() {
  cudaFuncAttributes a; int blocks=0; auto fn=r2_pool_kernel<2,512,8192,ILP>;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,fn));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,544,0));
  return {(int64_t)a.numRegs,(int64_t)a.localSizeBytes,(int64_t)blocks};
}
std::vector<int64_t> r2_pool_info(int64_t ilp) { return ilp ? r2_pool_info_t<true>() : r2_pool_info_t<false>(); }

// Scalar row-major cgemv for 2..8 tokens. A one-unit activation register window
// limits liveness at K=4096. Each output uses the M=1 kernel's exact j/g/e order.
template <int NTH, int K, bool ROT>
__global__ void __launch_bounds__(NTH + 32) r2_multi_kernel(
    const uint8_t* L, const uint8_t* C, const __nv_bfloat16* x, const int2* etab,
    int pmax, __nv_bfloat16* out, uint32_t base4, int M, int N) {
  constexpr int ROWS = 2, TOKS = 8, PER = K / (16 * NTH), NW = NTH / 32;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, row0 = blockIdx.x * ROWS;
  __shared__ float red[ROWS][TOKS][NW + 1];
  if (warp == NW) {
    const int2* tb = etab + (size_t)blockIdx.x * pmax;
    int2 mine = lane < pmax ? __ldg(tb + lane) : make_int2(0, 0);
    const int cnt = __shfl_sync(0xffffffffu, mine.x, 0);
    float ec[ROWS][TOKS] = {};
    for (int i0 = 0; i0 <= cnt; i0 += 32) {
      const int i = i0 + lane;
      int2 en = mine;
      if (i0 > 0) en = i <= cnt ? __ldg(tb + i) : make_int2(0, 0);
      if (i >= 1 && i <= cnt) {
#pragma unroll
        for (int t = 0; t < TOKS; ++t) {
          if (t >= M) break;
          const float c = __bfloat162float(x[(size_t)t * K + (en.x & 0xffff)]) * esc_delta((uint32_t)en.y, base4 & 255u);
#pragma unroll
          for (int r = 0; r < ROWS; ++r) ec[r][t] += (en.x >> 16) == r ? c : 0.f;
        }
      }
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r)
#pragma unroll
      for (int t = 0; t < TOKS; ++t) {
        if (t >= M) break;
        float v = ec[r][t];
#pragma unroll
        for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
        if (lane == 0) red[r][t][NW] = v;
      }
    __syncthreads();
    return;
  }
  uint4 lw[ROWS][PER]; uint2 cw[ROWS][PER];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int j = 0; j < PER; ++j) {
      const size_t pos = (size_t)(row0 + r) * K + j * 16 * NTH + 16 * tid;
      lw[r][j] = __ldg(reinterpret_cast<const uint4*>(L + pos));
      cw[r][j] = __ldg(reinterpret_cast<const uint2*>(C + pos / 2));
    }
  float acc[ROWS][TOKS] = {};
#pragma unroll
  for (int j = 0; j < PER; ++j) {
    uint4 xv[TOKS][2];
#pragma unroll
    for (int t = 0; t < TOKS; ++t) {
      if (t >= M) break;
      const __nv_bfloat16* p = x + (size_t)t * K + j * 16 * NTH + 16 * tid;
      xv[t][0] = __ldg(reinterpret_cast<const uint4*>(p));
      xv[t][1] = __ldg(reinterpret_cast<const uint4*>(p + 8));
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint32_t packed[ROWS][4];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
        const uint32_t c = g ? cw[r][j].y : cw[r][j].x;
        if constexpr (ROT) decode8(lws[g * 2], lws[g * 2 + 1], c, base4, packed[r]);
        else {
          const uint32_t ha = cg_hib(c & 0x0f0f0f0fu, base4), hb = cg_hib((c >> 4) & 0x0f0f0f0fu, base4);
          packed[r][0] = __byte_perm(lws[g * 2], ha, 0x5140);
          packed[r][1] = __byte_perm(lws[g * 2], ha, 0x7362);
          packed[r][2] = __byte_perm(lws[g * 2 + 1], hb, 0x5140);
          packed[r][3] = __byte_perm(lws[g * 2 + 1], hb, 0x7362);
        }
      }
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int t = 0; t < TOKS; ++t) {
          if (t >= M) break;
          const uint32_t xx = reinterpret_cast<const uint32_t*>(&xv[t][g])[e / 2];
          const float xf = __uint_as_float((e & 1) ? (xx & 0xffff0000u) : (xx << 16));
#pragma unroll
          for (int r = 0; r < ROWS; ++r) {
            const uint32_t p = packed[r][e / 2];
            const float w = __uint_as_float((e & 1) ? (p & 0xffff0000u) : (p << 16));
            acc[r][t] = fmaf(w, xf, acc[r][t]);
          }
        }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int t = 0; t < TOKS; ++t) {
      if (t >= M) break;
      float v = acc[r][t];
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
      if (lane == 0) red[r][t][warp] = v;
    }
  __syncthreads();
  if (tid < ROWS * TOKS) {
    const int r = tid / TOKS, t = tid % TOKS;
    if (t < M) {
      float v = 0.f;
#pragma unroll
      for (int q = 0; q <= NW; ++q) v += red[r][t][q];
      out[(size_t)t * N + row0 + r] = __float2bfloat16_rn(v);
    }
  }
}
template <int NTH, int K, bool ROT>
void r2_multi_launch(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor tab, int pm, at::Tensor out, uint32_t b4) {
  r2_multi_kernel<NTH,K,ROT>(
      L.data_ptr<uint8_t>(), C.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const int2*>(tab.data_ptr()), pm, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), b4, x.size(0), out.size(1));
}
void r2_multi(at::Tensor L, at::Tensor C, at::Tensor x, at::Tensor tab, int64_t pm, at::Tensor out, int64_t b4, int64_t rot) {
  TORCH_CHECK(x.size(0)>=1 && x.size(0)<=8 && out.size(1)%2==0 && out.size(1)<=L.size(0), "rot2 multi geometry");
  if (L.size(1)==1024) {
    if(rot) r2_multi_launch<64,1024,true>(L,C,x,tab,pm,out,b4); else r2_multi_launch<64,1024,false>(L,C,x,tab,pm,out,b4);
  } else if (L.size(1)==4096) {
    if(rot) r2_multi_launch<128,4096,true>(L,C,x,tab,pm,out,b4); else r2_multi_launch<128,4096,false>(L,C,x,tab,pm,out,b4);
  } else TORCH_CHECK(false, "rot2 multi K");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#include <cstring>
__device__ __forceinline__ void tma_load_2d_pol(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, uint64_t pol) {
  do {} while(false);
}
__device__ __forceinline__ void tma_load_2d_ef(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_ef());
}
__device__ __forceinline__ void tma_load_2d_el(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1) {
  tma_load_2d_pol(dst, map, bar, c0, c1, l2_pol_el());
}
__device__ __forceinline__ void tma_load_3d(uint32_t dst, const CUtensorMap* map, uint32_t bar, int c0, int c1, int c2) {
  do {} while(false);
}
static PFN_cuTensorMapEncodeTiled_v12000 get_encode() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* p = nullptr;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &q);
    TORCH_CHECK(p != nullptr, "cuTensorMapEncodeTiled not found");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}

// x [M, K] bf16 as 3-D {64 (k in unit), M (token), K/64 (unit)}: one TMA brings `units` tiles of [rows x 128 B]
static CUtensorMap make_map_x3d(const void* ptr, uint64_t m, uint64_t k, uint32_t box_rows, uint32_t units) {
  CUtensorMap mp;
  cuuint64_t dims[3] = {64, m, k / 64};
  cuuint64_t strides[2] = {k * 2, 128};
  cuuint32_t box[3] = {64, box_rows, units};
  cuuint32_t estr[3] = {1, 1, 1};
  CUresult r = get_encode()(&mp, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 3, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled (3d) failed: ", (int)r);
  return mp;
}

static CUtensorMap make_map_2d(const void* ptr, uint64_t rows, uint64_t cols, uint32_t box_rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {cols, rows};
  cuuint64_t strides[1] = {cols * 2};
  cuuint32_t box[2] = {64, box_rows};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = get_encode()(&m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}



#define SK2_WAIT(bar, par) mbar_wait_sleep((bar), (par))
#define SK2_DIAG 0
// ============================================================================ sk2 (m <= 32): swapped operands
// A = x tile (16 tokens x 16 k; ldmatrix from the TMA-loaded, 128B-swizzled tile), B = weights (8 rows x 16 k),
// decoded to registers from the 12-bit stream (BF16 hi + lo for the router).  Each CTA owns R rows over the whole
// K: no cross-CTA reduction.  Warps split the CTA's rows (NWR x NT n8 tiles) and, inside every stage, K (WSK).
// A stage = UPS 64-k units: x tiles [UPS][TOKR rows x 128 B] (TMA, rows >= M zero-filled) + this CTA's weights.
template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT,
          bool PK>
__global__ void __launch_bounds__((NWR * WSK + 1 + (COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1)) * 32, 1)
r2_sk2_kernel(const __grid_constant__ CUtensorMap xmap, const uint8_t* __restrict__ Ws,
           const __nv_bfloat16* __restrict__ x, OutT* __restrict__ out,
           const int2* __restrict__ ptab, int pmax, uint32_t base4, int M, int N, int K, int dg,
           unsigned long long* __restrict__ dbg) {
  const int dgv = SK2_DIAG ? dg : 0;   // diagnostics (dg) compiled out of production builds
  constexpr int NW = NWR * WSK;
  constexpr int NC = NW * 32;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XT = TOKR * 128;
  constexpr int XBY = UPS * XT;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  // partial sums [warp][token][RP] (rows fastest, padded): 64-bit stores in, 128-bit loads out
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);   // >= R, = 8 (mod 32): conflict-free 64-bit stores
  constexpr int CP = R + 1;                         // corr[token][CP]: odd stride, conflict-free per-token updates
  extern __shared__ __align__(1024) unsigned char smem_raw[];
  const uint32_t sraw = (uint32_t)__cvta_generic_to_shared(smem_raw);
  const uint32_t sbase = (sraw + 1023u) & ~1023u;
  unsigned char* gbase = smem_raw + (sbase - sraw);
  const uint32_t bars = sbase + NSTAGE * SST;
  const uint32_t full0 = bars, empty0 = bars + 8 * NSTAGE, pbar = bars + 16 * NSTAGE, cbar = pbar + 8, qbar = pbar + 16;
  // corrections, one copy per escape warp: [NESC][TOKS tokens][CP]; escape warps take turns on the stages
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  static_assert(NESC <= NSTAGE, "sk2: escape warps need NESC <= NSTAGE");
  float* corr = reinterpret_cast<float*>(gbase + NSTAGE * SST + 16 * NSTAGE + 32);
  constexpr int CSZ = (TOKS * CP + 3) & ~3;
  float* peer = corr + NESC * CSZ;                       // [CL-1][WSK][TOKS][RP]: ranks > 0 per-warp sums
  int2* ptab_s = reinterpret_cast<int2*>(peer + (CL > 1 ? (CL - 1) * WSK * TOKS * RP : 0));   // [pmax]: escapes
  const uint32_t zrow = (uint32_t)__cvta_generic_to_shared(ptab_s + pmax);   // 128 B of zeros (x rows >= M)
  // m <= 8: the x box has exactly M rows per 64-k unit (packed, M*128 B per unit; no zero rows moved)
  // PK (M < TOKR, or TOKR == 8): packed x, exactly M rows per 64-k unit (no out-of-bounds TMA rows; rows >= M
  // read zrow); else the full TOKR-row tile with compile-time addressing
  const int Mb = PK ? M : TOKR;
  const uint32_t XU = (uint32_t)Mb * 128u;
  const uint32_t XBYr = UPS * XU;
  float* red = reinterpret_cast<float*>(gbase);   // reuses the ring once every stage is consumed

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // CL = 2: a cluster of two CTAs shares a row block; each streams one half of K, rank 1 hands its sums to rank 0
  const int cta = blockIdx.x / CL, crank = CL > 1 ? (int)(blockIdx.x % CL) : 0;
  const int tst = K / (UPS * 64);                      // stages over all of K
  const int gs0 = (crank * tst) / CL;                  // this CTA's first stage (uneven split for CL = 3)
  const int nst = ((crank + 1) * tst) / CL - gs0;      // and its stage count
  // stage-major weight stream: [stage][row block][...] (at any moment all CTAs read one contiguous region)
  const uint8_t* wsrc = Ws + (size_t)cta * WBY;
  const size_t wstride = (size_t)(gridDim.x / CL) * WBY;

  if (tid == 0) {
    // stage 0 first of all (the first bytes' latency is the ramp of the whole kernel): the weight copy needs only
    // its barrier; the tensor map prefetch goes out before it so the x box follows right behind. In a real cluster
    // (CL > 1) a cluster-scope release issued AFTER a copy is in flight waits for it (~0.7 us): there every barrier
    // is initialized and fenced before the first copy.
    do {} while(false);
    mbar_init(full0, 1);
    if constexpr (CL > 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      if (crank == 0) mbar_expect_tx(qbar, (CL - 1) * NC * MT * NT * 16);
    }
    do {} while(false);
    mbar_expect_tx(full0, ((dgv & 4) ? 0 : XBYr) + WBY);
    bulk_g2s(sbase + XBY, wsrc + (size_t)gs0 * wstride, WBY, full0);
    if (!(dgv & 4)) tma_load_3d(sbase, &xmap, full0, 0, 0, gs0 * UPS);
    if (COMP) {   // this CTA's escape table
      if constexpr (CL == 1) {
        mbar_init(pbar, 1);
        do {} while(false);
      }
      mbar_expect_tx(pbar, pmax * 8);
      bulk_g2s((uint32_t)__cvta_generic_to_shared(ptab_s), ptab + (size_t)cta * pmax, pmax * 8, pbar);
    }
    if constexpr (CL == 1) {
      mbar_init(empty0, NW + (COMP ? NESC : 0));
      for (int i = 1; i < NSTAGE; ++i) { mbar_init(full0 + 8 * i, 1); mbar_init(empty0 + 8 * i, NW + (COMP ? NESC : 0)); }
      if (!COMP) mbar_init(pbar, 1);
      mbar_init(cbar, 32 * NESC);
      mbar_init(qbar, 1);
      do {} while(false);
    }
  }
  if (tid >= 32 && tid < 64) reinterpret_cast<uint32_t*>(ptab_s + pmax)[tid - 32] = 0u;
  // local barriers are all this CTA needs to start streaming; the cluster barrier (rank 0's qbar initialized before
  // ranks > 0 send into it) is only arrived at here and waited on right before the remote stores
  __syncthreads();
  if constexpr (CL > 1) do {} while(false);   // inits fenced above

  if (warp == NW) {
    // producer warp: lane 0 refills the ring
    if (lane == 0) {
      for (int st = 1; st < nst; ++st) {
        const int slot = st % NSTAGE;
        if (st >= NSTAGE) SK2_WAIT(empty0 + 8 * slot, ((st / NSTAGE) - 1) & 1);
        const uint32_t s0 = sbase + slot * SST;
        mbar_expect_tx(full0 + 8 * slot, ((dgv & 4) ? 0 : XBYr) + WBY);
        bulk_g2s(s0 + XBY, wsrc + (size_t)(gs0 + st) * wstride, WBY, full0 + 8 * slot);
        if (!(dgv & 4)) tma_load_3d(s0, &xmap, full0 + 8 * slot, 0, 0, (gs0 + st) * UPS);
      }
    }
    __syncwarp();
    if constexpr (CL > 1) do {} while(false);
    return;
  }
  if (warp >= NW + 1) {
    // NESC escape warps take turns on the stages (each has NESC stage-times per stage): exact corrections for the
    // escapes, read from the x tiles of the ring as each stage lands (the table is in k order); lane t owns token t,
    // so corr[t][row] accumulates with no atomics; each warp has its own copy
    const int ew = warp - (NW + 1);
    if (COMP) {
      float* cr = corr + ew * CSZ;
      for (int i = lane; i < TOKS * CP; i += 32) cr[i] = 0.f;
      mbar_wait_sleep(pbar, 0);
      const int cnt = ptab_s[0].x;
      constexpr int KST = UPS * 64;
      const uint32_t bq = base4 & 0xFFu;
      int pe = 1;
      for (int st = 0; st < nst; ++st) {
        const int kbeg = (gs0 + st) * KST, kend = kbeg + KST;
        const int slot = st % NSTAGE;
        // Every escape warp observes every stage, in order, works only on its own (st % NESC == ew), and hands EVERY
        // stage back (the empty barriers count NW + NESC arrivals). The producer therefore never refills a slot
        // before every escape warp has seen the stage it held, so no warp can find its slot two phases ahead (a
        // parity wait that aliases and wedges the ring: astra review 2B, reproduced with a delayed escape warp).
        SK2_WAIT(full0 + 8 * slot, (st / NSTAGE) & 1);
        if ((st % NESC) != ew) {   // another warp's stage: skip its escapes, hand the slot back
          while (pe <= cnt && (ptab_s[pe].x & 0xFFFF) < kend) ++pe;
          __syncwarp();
          if (lane == 0) mbar_arrive(empty0 + 8 * slot);
          continue;
        }
        const unsigned char* xs = gbase + slot * SST;
        while (pe <= cnt) {
          const int2 e = ptab_s[pe];
          const int kk = e.x & 0xFFFF;
          if (kk >= kend) break;
          if (lane < M && kk >= kbeg) {
            const int kl = kk - kbeg, u = kl >> 6, kc = kl & 63;
            const __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(
                xs + u * XU + lane * 128 + (((kc >> 3) ^ ((u * Mb + lane) & 7)) << 4) + (kc & 7) * 2);
            cr[lane * CP + (e.x >> 16)] += esc_delta((uint32_t)e.y, bq) * __bfloat162float(v);
          }
          ++pe;
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
    }
    mbar_arrive(cbar);
    __syncwarp();
    if constexpr (CL > 1) do {} while(false);
    return;
  }

  // ---------------- consumers
  const int wr = warp / WSK, wk = warp % WSK;
  const int g = lane >> 2, t4 = lane & 3;
  const int q = lane >> 3;
  const int chi = q >> 1;
  int trow[MT];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) trow[mt] = TOKR >= 16 ? mt * 16 + (lane & 7) + 8 * (q & 1) : (lane & 7);
  float acc[NACC][MT][NT][4];
#pragma unroll
  for (int c = 0; c < NACC; ++c)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) acc[c][mt][nt][0] = acc[c][mt][nt][1] = acc[c][mt][nt][2] = acc[c][mt][nt][3] = 0.f;

  for (int st = 0; st < nst; ++st) {
    const int slot = st % NSTAGE;
    mbar_wait(full0 + 8 * slot, (st / NSTAGE) & 1);
    const uint32_t s0 = sbase + slot * SST;
#pragma unroll
    for (int uu = 0; uu < UPW; ++uu) {
      const int ui = wk * UPW + uu;
      const uint32_t xt = s0 + ui * XU;
      const uint32_t wb = s0 + XBY + (warp * UPW + uu) * NT * TU;
      // 1) every shared-memory load of this unit first (weights of all tiles, x fragments of all 4 k-steps)
      uint4 lw[NT];
      uint2 cw[NT];
      uint32_t b[NT][8];
      uint32_t bl[HL ? NT : 1][8];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        if constexpr (COMP) {
          do {} while(false);
          do {} while(false);
        } else {
          const uint32_t a0 = wb + nt * TU + lane * 32;
          do {} while(false);
          do {} while(false);
          if constexpr (HL) {
            do {} while(false);
            do {} while(false);
          }
        }
      }
      uint32_t a[4][MT][4];
      if (dgv & 16) {
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int mt = 0; mt < MT; ++mt)
            ldsm_x4((PK && trow[mt] >= M) ? zrow
                        : xt + trow[mt] * 128 + (((2 * j + chi) ^ ((ui * Mb + trow[mt]) & 7)) << 4),
                    a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3]);
      }
      // the stage's shared memory is in registers now: hand the slot back before decoding and multiplying
      if (UPW == 1 && uu == 0) {
        __syncwarp();
        if (lane == 0) mbar_arrive(empty0 + 8 * slot);
      }
      // 2) decode
      if constexpr (COMP) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          decode8(lw[nt].x, lw[nt].y, cw[nt].x, base4, &b[nt][0]);
          decode8(lw[nt].z, lw[nt].w, cw[nt].y, base4, &b[nt][4]);
        }
      }
      // 3) tensor-core work: independent accumulators per k-step
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            constexpr int HM = HL ? 2 : 1;
            mma_bf16(acc[((uu * 4 + j) * HM) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], b[nt][2 * j], b[nt][2 * j + 1]);
            if constexpr (HL)
              mma_bf16(acc[((uu * 4 + j) * HM + 1) % NACC][mt][nt], a[j][mt][0], a[j][mt][1], a[j][mt][2], a[j][mt][3], bl[nt][2 * j], bl[nt][2 * j + 1]);
          }
    }
    if (UPW > 1) {
      __syncwarp();
      if (lane == 0) mbar_arrive(empty0 + 8 * slot);
    }
  }

  // ---------------- epilogue: K-warps reduce through shared memory (the ring is free now)
  if constexpr (COMP) mbar_wait(cbar, 0);   // the escape warp reads the ring's x tiles until it is done
  if (CL > 1 && crank > 0) {
    do {} while(false);   // rank 0's qbar is initialized
    // ranks > 0: every accumulator goes straight into rank 0's peer buffer (slot crank - 1) by st.async
    // (completes rank 0's qbar transaction count; no fence, no wait), escape corrections folded into k-warp 0's
    const uint32_t peer_l = (uint32_t)__cvta_generic_to_shared(peer + (crank - 1) * WSK * TOKS * RP);
    uint32_t peer_r, qbar_r;
    do {} while(false);
    do {} while(false);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int row = (wr * NT + nt) * 8 + 2 * t4;
        const int tok = mt * 16 + g;
        float v[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          v[e] = acc[0][mt][nt][e];
#pragma unroll
          for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
        }
        if (COMP && wk == 0) {
          v[0] += corr[tok * CP + row];
          v[1] += corr[tok * CP + row + 1];
          if (tok + 8 < 32) {
            v[2] += corr[(tok + 8) * CP + row];
            v[3] += corr[(tok + 8) * CP + row + 1];
          }
        }
        const uint32_t a0 = peer_r + (uint32_t)(((wk * TOKS + tok) * RP + row) * 4);
        do {} while(false);
        do {} while(false);
      }
    return;
  }
  named_bar(1, NC);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int row = (wr * NT + nt) * 8 + 2 * t4;
      const int tok = mt * 16 + g;
      float* r0 = red + (wk * TOKS + tok) * RP + row;
      float v[4];
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        v[e] = acc[0][mt][nt][e];
#pragma unroll
        for (int c = 1; c < NACC; ++c) v[e] += acc[c][mt][nt][e];
      }
      *reinterpret_cast<float2*>(r0) = make_float2(v[0], v[1]);
      *reinterpret_cast<float2*>(r0 + 8 * RP) = make_float2(v[2], v[3]);
    }
  named_bar(1, NC);
  if constexpr (CL > 1) {
    do {} while(false);
    mbar_wait_cluster(qbar, 0);
  }
  constexpr int R4 = R / 4;
  for (int idx = tid; idx < R4 * M; idx += NC) {
    const int tt = idx / R4, row = (idx - tt * R4) * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if constexpr (COMP) {
#pragma unroll
      for (int c = 0; c < NESC; ++c) {
        v.x += corr[c * CSZ + tt * CP + row];
        v.y += corr[c * CSZ + tt * CP + row + 1];
        v.z += corr[c * CSZ + tt * CP + row + 2];
        v.w += corr[c * CSZ + tt * CP + row + 3];
      }
    }
#pragma unroll
    for (int w = 0; w < WSK; ++w) {
      const float4 p = *reinterpret_cast<const float4*>(red + (w * TOKS + tt) * RP + row);
      v.x += p.x; v.y += p.y; v.z += p.z; v.w += p.w;
#pragma unroll
      for (int c = 0; c < CL - 1; ++c) {
        const float4 q = *reinterpret_cast<const float4*>(peer + ((c * WSK + w) * TOKS + tt) * RP + row);
        v.x += q.x; v.y += q.y; v.z += q.z; v.w += q.w;
      }
    }
    OutT* o = out + (size_t)tt * N + cta * R + row;
    if (COMP && ptab_s[0].y != 0)   // escape-balanced row order (v135): 4-row groups moved whole, so they stay contiguous
      o = out + (size_t)tt * N + reinterpret_cast<const int*>(ptab_s + ptab_s[0].y)[row];
    if constexpr (sizeof(OutT) == 4) {
      *reinterpret_cast<float4*>(o) = v;
    } else {
      __nv_bfloat162 lo = __floats2bfloat162_rn(v.x, v.y), hi = __floats2bfloat162_rn(v.z, v.w);
      uint2 pk;
      pk.x = *reinterpret_cast<uint32_t*>(&lo);
      pk.y = *reinterpret_cast<uint32_t*>(&hi);
      *reinterpret_cast<uint2*>(o) = pk;
    }
  }
}

template <int MT, int TOKR, int NT, int NWR, int WSK, int UPW, int NSTAGE, bool COMP, bool HL, int NACC, int CL, typename OutT>
static void launch_r2_sk2(const at::Tensor& Ws, const at::Tensor& x, at::Tensor& out, const at::Tensor& ptab,
                       int64_t pmax, int64_t base4, int M, int N, int K, int64_t dg, unsigned long long* dbg) {
  constexpr int NW = NWR * WSK;
  constexpr int R = NWR * NT * 8;
  constexpr int UPS = WSK * UPW;
  constexpr int TU = COMP ? 768 : (HL ? 2048 : 1024);
  constexpr int WBY = NW * UPW * NT * TU;
  constexpr int XBY = UPS * TOKR * 128;
  constexpr int SST = ((XBY + WBY) + 1023) & ~1023;
  constexpr int TOKS = 16 * MT;
  constexpr int RP = R + ((40 - (R % 32)) % 32);
  constexpr int CP = R + 1;
  constexpr int NESC = COMP ? (NSTAGE >= 4 ? 4 : NSTAGE) : 1;
  const size_t smem = 1024 + (size_t)NSTAGE * SST + 16 * NSTAGE + 32 + (size_t)NESC * ((TOKS * CP + 3) & ~3) * 4 +
                      (CL > 1 ? (size_t)(CL - 1) * WSK * TOKS * RP * 4 : 0) + (size_t)pmax * 8 + 128;
  auto go = [&](auto pkc) {
    constexpr bool PK = decltype(pkc)::value;
    auto kern = r2_sk2_kernel<MT, TOKR, NT, NWR, WSK, UPW, NSTAGE, COMP, HL, NACC, CL, OutT, PK>;
    static bool attr_done = false;
    if (!attr_done) {
      int dev = 0, optin = 0;
      cudaGetDevice(&dev);
      cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
      cudaFuncAttributes fa;
      cudaFuncGetAttributes(&fa, kern);
      cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes);
      attr_done = true;
    }
    CUtensorMap xm = make_map_x3d(x.data_ptr(), (uint64_t)M, (uint64_t)K, PK ? (uint32_t)M : (uint32_t)TOKR, UPS);
    const uint8_t* wsp = reinterpret_cast<const uint8_t*>(Ws.data_ptr());
    const __nv_bfloat16* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    OutT* op = reinterpret_cast<OutT*>(out.data_ptr());
    const int2* tp = reinterpret_cast<const int2*>(ptab.data_ptr());
    if constexpr (CL > 1) {
      cudaLaunchConfig_t lc = {};
      lc.gridDim = dim3(N / R * CL);
      lc.blockDim = dim3((NW + 1 + NESC) * 32);
      lc.dynamicSmemBytes = smem;
      lc.stream = at::cuda::getCurrentCUDAStream();
      cudaLaunchAttribute at_[1];
      at_[0].id = cudaLaunchAttributeClusterDimension;
      at_[0].val.clusterDim.x = CL;
      at_[0].val.clusterDim.y = 1;
      at_[0].val.clusterDim.z = 1;
      lc.attrs = at_;
      lc.numAttrs = 1;
      C10_CUDA_CHECK(cudaLaunchKernelEx(&lc, kern, xm, wsp, xp, op, tp, (int)pmax, (uint32_t)base4, M, N, K, (int)dg, dbg));
    } else {
      kern(xm, wsp, xp, op, tp, (int)pmax,
                                                                               (uint32_t)base4, M, N, K, (int)dg, dbg);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
  };
  if (TOKR == 8 || M < TOKR) go(std::true_type{}); else go(std::false_type{});
}

void r2_sk2(at::Tensor Ws, at::Tensor x, at::Tensor out, at::Tensor ptab, int64_t pmax, int64_t base4,
         int64_t M, int64_t N, int64_t K, int64_t cfg, int64_t tokr, int64_t dg, at::Tensor dbgt) {
  unsigned long long* dbg = dbgt.numel() >= 8192 + 3 * 1024 ? reinterpret_cast<unsigned long long*>(dbgt.data_ptr()) : nullptr;
  if(cfg==0 && tokr==32){launch_r2_sk2<2,32,2,2,2,2,3,true,false,1,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==1 && tokr==8){launch_r2_sk2<1,8,4,1,4,1,4,true,false,2,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==2 && tokr==32){launch_r2_sk2<2,32,2,2,2,2,5,true,false,1,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==3 && tokr==8){launch_r2_sk2<1,8,1,2,2,4,3,true,false,1,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==4 && tokr==32){launch_r2_sk2<2,32,3,1,4,2,4,true,false,1,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==6 && tokr==8){launch_r2_sk2<1,8,1,1,8,2,4,true,false,2,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==7 && tokr==32){launch_r2_sk2<2,32,4,2,2,2,3,true,false,1,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==8 && tokr==16){launch_r2_sk2<1,16,2,2,2,4,4,true,false,1,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==9 && tokr==16){launch_r2_sk2<1,16,2,2,2,2,3,true,false,1,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==10 && tokr==8){launch_r2_sk2<1,8,3,1,4,2,5,true,false,2,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==11 && tokr==16){launch_r2_sk2<1,16,3,1,4,2,5,true,false,2,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
if(cfg==12 && tokr==16){launch_r2_sk2<1,16,2,2,1,4,3,true,false,2,1,__nv_bfloat16>(Ws,x,out,ptab,pmax,base4,M,N,K,dg,dbg);return;}
  TORCH_CHECK(false, "sk2: no config");
}
