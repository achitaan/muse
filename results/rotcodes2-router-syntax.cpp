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
#include <algorithm>
#define RD __device__ __forceinline__
RD uint32_t r2_routertail_smem(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
RD void r2_routertail_init(uint32_t b, int n) { do {} while(false); }
RD void r2_routertail_expect(uint32_t b, int n) { do {} while(false); }
RD void r2_routertail_wait(uint32_t b, int ph) {
  do {} while(false);
}
RD void r2_routertail_sync() { do {} while(false); }
RD uint32_t r2_routertail_map(uint32_t a, int r) { uint32_t b; do {} while(false);return b; }
RD void r2_routertail_push(uint32_t d, float4 v, uint32_t b) {
  do {} while(false);
}
RD float4 r2_routertail_pull(uint32_t d) {
  float4 v; do {} while(false);return v;
}
RD void r2_routertail_add(float4& a,float4 b) { a.x+=b.x;a.y+=b.y;a.z+=b.z;a.w+=b.w; }
RD uint64_t r2_routertail_desc(uint32_t a) { return (uint64_t)((a&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62); }
RD void r2_routertail_tma(uint32_t dst, const void* tm, int k,int r,uint32_t bar) {
  do {} while(false);
}

// ---- c12: the same kernel on lossless 12-bit planes (hi and lo BF16 of the FP32 router): per (16 rows x 64 k) warp tile and
// plane the lane reads 3 x 16 B (2 low-byte words groups + 1 nibble group) instead of 4 x 16 B; exact escape corrections in an
// extra warp. Plane layout [ku][S][16][WK][h][part 0..2][lane] int4; nibble = sign | (exp7 - base) (3 bits), 4 per byte pair.
template<bool ROT> RD void r2_rt12_decode8(uint32_t l0, uint32_t l1, uint32_t nw, uint32_t b4, uint32_t* r) {
  uint32_t ha, hb;
  if constexpr (ROT) {
    ha = (nw & 0x87878787u) + b4;
    hb = (__funnelshift_r(nw, nw, 4) & 0x87878787u) + b4;
  } else {
    ha = ((nw & 0x07070707u) + b4) | ((nw << 4) & 0x80808080u);
    hb = (((nw >> 4) & 0x07070707u) + b4) | (nw & 0x80808080u);
  }
  r[0] = __byte_perm(l0, ha, 0x5140); r[1] = __byte_perm(l0, ha, 0x7362);
  r[2] = __byte_perm(l1, hb, 0x5140); r[3] = __byte_perm(l1, hb, 0x7362);
}
template<int MT,int S,int WK,bool ROT>
__global__ __launch_bounds__(WK*32) void r2_routertail_c12_kernel(const int4* wp,const __nv_bfloat16* x,float* out,int m,const unsigned* etab,uint32_t bqh,uint32_t bql) {
  constexpr int KU=128/S/WK, NI=MT*32, P=4, OWN=NI/P, PARTS=S*1;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,rank=blockIdx.x%S,row=blockIdx.x/S*16;
  const int gw=(rank*16+row/16)*WK+warp;
  extern __shared__ __align__(16) unsigned char sm[];
  float4* recv=(float4*)sm;
  float4* part=recv+PARTS*OWN;
  const uint32_t rb=r2_routertail_smem(sm+(PARTS*OWN+WK*NI)*16);
  int4 w[KU][2][3];uint4 xx[KU][MT][2];
  const unsigned eword=lane<16?__ldg(etab+(size_t)(blockIdx.x*WK+warp)*16+lane):0u;   // [cta][warp][16]: word 0 = count, then entries
#pragma unroll
  for(int k=0;k<KU;k++) {
#pragma unroll
    for(int j=0;j<6;j++) {
      const int4* p=wp+((k*(16*S*WK)+gw)*6+j)*32+lane;
      do {} while(false);
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      const __nv_bfloat16* p=x+min(n*8+(lane>>2),m-1)*8192+((rank*WK+warp)*KU+k)*64+(lane&3)*16;
#pragma unroll
      for(int h=0;h<2;h++) do {} while(false);
    }
  }
  if(rank<P && tid==0) {r2_routertail_init(rb,1);do {} while(false);}
  __syncthreads();
  if(rank<P && tid==0) {r2_routertail_expect(rb,PARTS*OWN*16);do {} while(false);}
  do {} while(false);
  float c[2][MT][4]={};
  const int ecnt=__shfl_sync(0xffffffffu,(int)eword,0);
  const uint32_t b4h=bqh*0x01010101u,b4l=bql*0x01010101u;
#pragma unroll
  for(int k=0;k<KU;k++) {
    uint32_t a[2][4][4];
#pragma unroll
    for(int h=0;h<2;h++) {
      const uint32_t b4=h?b4l:b4h;
      r2_rt12_decode8<ROT>(w[k][h][0].x,w[k][h][0].y,w[k][h][2].x,b4,a[h][0]);
      r2_rt12_decode8<ROT>(w[k][h][0].z,w[k][h][0].w,w[k][h][2].y,b4,a[h][1]);
      r2_rt12_decode8<ROT>(w[k][h][1].x,w[k][h][1].y,w[k][h][2].z,b4,a[h][2]);
      r2_rt12_decode8<ROT>(w[k][h][1].z,w[k][h][1].w,w[k][h][2].w,b4,a[h][3]);
    }
    // exact escapes: overwrite the mis-decoded element with the true BF16 (entry: lane 0-4, q 5-6, reg 7-8, half 9, h 10, k 11, bits 16-31)
    for(int e=0;e<ecnt;e++) {
      const unsigned en=__shfl_sync(0xffffffffu,(int)eword,e+1);
      if(((en>>11)&1)!=(unsigned)k && KU>1) continue;
      if((int)(en&31)!=lane) continue;
      const int q=(en>>5)&3,rg=(en>>7)&3,hf=(en>>9)&1,hh=(en>>10)&1;
      const uint32_t sh=hf*16,msk=0xFFFFu<<sh,val=(en>>16)<<sh;
#pragma unroll
      for(int hq=0;hq<2;hq++)
#pragma unroll
        for(int qq=0;qq<4;qq++)
#pragma unroll
          for(int rr=0;rr<4;rr++)
            if(hq==hh && qq==q && rr==rg) a[hq][qq][rr]=(a[hq][qq][rr]&~msk)|val;
    }
#pragma unroll
    for(int n=0;n<MT;n++) {
      uint32_t b[8]={xx[k][n][0].x,xx[k][n][0].y,xx[k][n][0].z,xx[k][n][0].w,xx[k][n][1].x,xx[k][n][1].y,xx[k][n][1].z,xx[k][n][1].w};
#pragma unroll
      for(int q=0;q<4;q++) {
#pragma unroll
        for(int h=0;h<2;h++) {
          do {} while(false);
        }
      }
    }
  }
#pragma unroll
  for(int n=0;n<MT;n++) part[warp*NI+n*32+lane]=make_float4(c[0][n][0]+c[1][n][0],c[0][n][1]+c[1][n][1],c[0][n][2]+c[1][n][2],c[0][n][3]+c[1][n][3]);
  __syncthreads();
  do {} while(false);
  if(tid<NI) {
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<WK;k++) r2_routertail_add(sum,part[k*NI+tid]);
    const int dst=tid/OWN,local=tid%OWN;
    r2_routertail_push(r2_routertail_map(r2_routertail_smem(recv+rank*OWN+local),dst),sum,r2_routertail_map(rb,dst));
  }

  if(rank<P && tid<OWN) {
    r2_routertail_wait(rb,0);
    float4 sum=make_float4(0,0,0,0);
#pragma unroll
    for(int k=0;k<PARTS;k++) r2_routertail_add(sum,recv[k*OWN+tid]);
    const int v=rank*OWN+tid,ln=v%32,tok=(v/32)*8+(ln&3)*2,wr=row+(ln>>2);
    if(tok<m){out[tok*256+wr]=sum.x;out[tok*256+wr+8]=sum.z;}
    if(tok+1<m){out[(tok+1)*256+wr]=sum.y;out[(tok+1)*256+wr+8]=sum.w;}
  }
}
template<int MT,int S,int WK,bool ROT>
int r2_routertail_c12_t(int64_t w,int64_t x,int64_t out,int m,int64_t etab,int bqh,int bql,int64_t stream) {
  auto fn=r2_routertail_c12_kernel<MT,S,WK,ROT>;
  constexpr int bytes=(S*1/4+WK)*MT*512+16;
  static bool configured=false;
  if(!configured){cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);cudaFuncSetAttribute(fn,cudaFuncAttributeNonPortableClusterSizeAllowed,1);configured=true;}
  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(16*S);cfg.blockDim=dim3(WK*32);cfg.dynamicSmemBytes=bytes;cfg.stream=(cudaStream_t)stream;
  cudaLaunchAttribute at;at.id=cudaLaunchAttributeClusterDimension;at.val.clusterDim={S,1,1};cfg.attrs=&at;cfg.numAttrs=1;
  return (int)cudaLaunchKernelEx(&cfg,fn,(const int4*)w,(const __nv_bfloat16*)x,(float*)out,m,(const unsigned*)etab,(uint32_t)bqh,(uint32_t)bql);
}
template<bool ROT> int r2_routertail_c12_dispatch(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t etab,int bqh,int bql,int64_t stream) {
#define RSC(S,W) if(s==S && wk==W) {if(m<=8)return r2_routertail_c12_t<1,S,W,ROT>(w,x,out,m,etab,bqh,bql,stream);if(m<=16)return r2_routertail_c12_t<2,S,W,ROT>(w,x,out,m,etab,bqh,bql,stream);if(m<=32)return r2_routertail_c12_t<4,S,W,ROT>(w,x,out,m,etab,bqh,bql,stream);}
RSC(8,8) RSC(16,8) RSC(16,4)
return -99;
}

int r2_routertail_c12(int64_t w,int64_t x,int64_t out,int m,int s,int wk,int64_t etab,int bh,int bl,int64_t stream,int rot) {
  if(rot) return r2_routertail_c12_dispatch<true>(w,x,out,m,s,wk,etab,bh,bl,stream);
  return r2_routertail_c12_dispatch<false>(w,x,out,m,s,wk,etab,bh,bl,stream);
}
