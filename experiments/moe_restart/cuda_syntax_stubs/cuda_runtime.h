// Declaration-only shim for clang's CUDA host-side semantic checks on macOS.
// It does not emulate CUDA, compile PTX, or replace real headers in a check.
#pragma once
#include <stdint.h>
#include <stddef.h>
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __forceinline__ inline __attribute__((always_inline))
#define __align__(n) __attribute__((aligned(n)))
#define __launch_bounds__(n, m) __attribute__((launch_bounds(n, m)))
// NVCC's CUDA>=11.7 grid-constant parameter contract is not modeled by Clang
// without the SDK. Const/reference semantics are still checked below.
#define __grid_constant__
struct __align__(8) uint2 { unsigned x,y; };
struct __align__(16) uint4 { unsigned x,y,z,w; };
struct __align__(8) float2 { float x,y; };
struct __align__(16) float4 { float x,y,z,w; };
struct dim3 {
  unsigned x,y,z;
  __host__ __device__ constexpr dim3(unsigned a=1,unsigned b=1,unsigned c=1) : x(a),y(b),z(c) {}
};
extern __device__ const dim3 threadIdx, blockIdx, blockDim, gridDim;
__device__ inline uint2 make_uint2(unsigned a,unsigned b) { return {a,b}; }
__device__ inline uint4 make_uint4(unsigned a,unsigned b,unsigned c,unsigned d) { return {a,b,c,d}; }
__device__ inline float4 make_float4(float a,float b,float c,float d) { return {a,b,c,d}; }
__device__ size_t __cvta_generic_to_shared(const void*);
__device__ void __threadfence();
__device__ void __syncthreads();
__device__ void __nanosleep(unsigned);
__device__ unsigned long long clock64();
__device__ unsigned __ballot_sync(unsigned,bool);
__device__ int __popc(unsigned);
__device__ unsigned __shfl_sync(unsigned,unsigned,int);
__device__ float __shfl_xor_sync(unsigned,float,int);
__device__ unsigned __funnelshift_l(unsigned,unsigned,unsigned);
__device__ unsigned __byte_perm(unsigned,unsigned,unsigned);
__device__ int atomicAdd(int*,int);
__device__ float __fdividef(float,float);
__device__ float __expf(float);
__device__ float rsqrtf(float);
template<class T> __device__ T __ldg(const T*);
template<class T> __device__ T __ldcg(const T*);

// Host launcher declarations, deliberately not implementations. These extend
// frontend coverage to the real host wrapper; they do not check CUDA's ABI.
using cudaStream_t = void*;
using cudaError_t = int;
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize };
struct cudaDeviceProp { int multiProcessorCount; };
struct cudaFuncAttributes {
  size_t sharedSizeBytes, localSizeBytes;
  int numRegs, maxDynamicSharedSizeBytes;
};
template<class F> __host__ cudaError_t cudaFuncSetAttribute(F,cudaFuncAttribute,int);
template<class F> __host__ cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*,F);
template<class F> __host__ cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*,F,int,size_t);
extern "C" __host__ cudaError_t cudaConfigureCall(dim3,dim3,size_t=0,cudaStream_t=nullptr);
