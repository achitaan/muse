#pragma once
#include "cuda_runtime.h"
struct __nv_bfloat16 { unsigned short bits; };
struct __align__(4) __nv_bfloat162 { __nv_bfloat16 lo, hi; };
__device__ __nv_bfloat162 __hmul2(__nv_bfloat162,__nv_bfloat162);
__device__ __nv_bfloat162 __hmul2_rn(__nv_bfloat162,__nv_bfloat162);
__device__ __nv_bfloat162 __low2bfloat162(__nv_bfloat162);
__device__ __nv_bfloat162 __high2bfloat162(__nv_bfloat162);
__device__ __nv_bfloat162 __floats2bfloat162_rn(float,float);
__device__ __nv_bfloat16 __float2bfloat16_rn(float);
__device__ __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16);
__device__ float __bfloat162float(__nv_bfloat16);
__device__ float2 __bfloat1622float2(__nv_bfloat162);
