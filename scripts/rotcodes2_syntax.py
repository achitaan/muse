"""Ordinary C++20 syntax only: CUDA/ATen stubbed, PTX/launch syntax removed.

Never execute the generated files. This does not validate CUDA, PTX or ABI.
"""
from pathlib import Path
import re
import shutil
import subprocess

STUBS = r'''
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
'''


def strip_asm(source):
    count = 0
    pattern = re.compile(r'\basm\s*(?:volatile\s*)?\(')
    while hit := pattern.search(source):
        i, depth, quote, escaped = hit.end(), 1, None, False
        while depth:
            ch = source[i]
            if quote:
                if escaped: escaped = False
                elif ch == '\\': escaped = True
                elif ch == quote: quote = None
            elif ch in ('"', "'"): quote = ch
            elif ch == '(': depth += 1
            elif ch == ')': depth -= 1
            i += 1
        while source[i].isspace(): i += 1
        assert source[i] == ';'
        source = source[:hit.start()] + 'do {} while(false);' + source[i+1:]
        count += 1
    return source, count


def check(source, output: Path):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        return dict(result='SKIPPED: no CPU C++ compiler')
    body = re.sub(r'^#include [<"](?:ATen/|c10/|cuda)[^\n]+\n', '', source, flags=re.M)
    body, count = strip_asm(body)
    body = re.sub(r'<<<.*?>>>', '', body, flags=re.S)
    output.write_text('// SYNTAX ONLY: CUDA/ATen stubbed and PTX removed. DO NOT EXECUTE.\n' + STUBS + body)
    command = [compiler,'-std=c++20','-fsyntax-only','-Wno-unknown-pragmas',str(output)]
    result = subprocess.run(command,capture_output=True,text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return dict(result='PASS: C++20 syntax ONLY, CUDA/ATen stubbed, PTX removed', removed_asm=count, command=command)
