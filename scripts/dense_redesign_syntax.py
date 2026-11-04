"""C++20 syntax preflight with CUDA/ATen declarations stubbed and PTX removed.

This catches ordinary C++ template/scope/signature mistakes only. It is NOT
CUDA, PTX, Torch ABI, or device-code compilation. No generated code is run.
"""
from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess


STUBS = r"""
#include <algorithm>
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
#define C10_CUDA_CHECK(...) do { (void)(__VA_ARGS__); } while (false)
#define TORCH_CHECK(cond, ...) do { if (!(cond)) throw 1; } while (false)
using std::min;
struct float4 { float x,y,z,w; };
struct uint2 { uint32_t x,y; };
inline float4 make_float4(float a,float b,float c,float d) { return {a,b,c,d}; }
inline uint2 make_uint2(uint32_t a,uint32_t b) { return {a,b}; }
struct __nv_bfloat16 { uint16_t bits; };
struct __nv_bfloat162 { uint32_t bits; };
inline __nv_bfloat162 __floats2bfloat162_rn(float,float) { return {}; }
inline uintptr_t __cvta_generic_to_shared(const void* p) { return uintptr_t(p); }
inline void __syncthreads() {}
inline void __threadfence() {}
inline float4 __ldcg(const float4* p) { return *p; }
inline void __stcg(float4* p,float4 v) { *p=v; }
inline int atomicExch(int* p,int v) { int old=*p; *p=v; return old; }
struct dim3 { unsigned x=0,y=1,z=1; dim3(unsigned n=0):x(n){} };
extern dim3 threadIdx, blockIdx;
struct CUtensorMap { unsigned char opaque[128]; };
using CUresult=int;
using cuuint64_t=uint64_t;
using cuuint32_t=uint32_t;
using PFN_cuTensorMapEncodeTiled_v12000=CUresult(*)(...);
using cudaDriverEntryPointQueryResult=int;
constexpr int CUDA_SUCCESS=0, cudaEnableDefault=0;
constexpr int CU_TENSOR_MAP_DATA_TYPE_BFLOAT16=0, CU_TENSOR_MAP_INTERLEAVE_NONE=0,
  CU_TENSOR_MAP_SWIZZLE_128B=0, CU_TENSOR_MAP_L2_PROMOTION_L2_256B=0, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE=0;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=0, cudaLaunchAttributeClusterDimension=0;
struct cudaFuncAttributes { int numRegs=0; size_t localSizeBytes=0,sharedSizeBytes=0; int maxThreadsPerBlock=0; };
struct cudaLaunchAttribute { int id=0; struct { dim3 clusterDim; } val; };
struct cudaLaunchConfig_t { dim3 gridDim,blockDim; size_t dynamicSmemBytes=0; void* stream=nullptr;
  cudaLaunchAttribute* attrs=nullptr; unsigned numAttrs=0; };
inline int cudaGetDriverEntryPoint(const char*,void**,int,cudaDriverEntryPointQueryResult*) { return 0; }
template<class F> int cudaFuncSetAttribute(F,int,int) { return 0; }
template<class F> int cudaFuncGetAttributes(cudaFuncAttributes*,F) { return 0; }
template<class F,class... A> int cudaLaunchKernelEx(cudaLaunchConfig_t*,F,A...) { return 0; }
namespace at {
constexpr int kBFloat16=0,kFloat=1,kInt=2;
struct Tensor {
  int64_t size(int) const { return 0; }
  int64_t numel() const { return 0; }
  int scalar_type() const { return 0; }
  bool is_contiguous() const { return true; }
  void* data_ptr() const { return nullptr; }
  template<class T> T* data_ptr() const { return nullptr; }
};
namespace cuda { inline void* getCurrentCUDAStream() { return nullptr; } }
}
"""


def strip_asm(source):
    pattern = re.compile(r'\basm\s*(?:volatile\s*)?\(')
    count = 0
    while match := pattern.search(source):
        i, level, quote, escaped = match.end(), 1, None, False
        while level:
            ch = source[i]
            if quote:
                if escaped:
                    escaped = False
                elif ch == '\\':
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
            elif ch == '(':
                level += 1
            elif ch == ')':
                level -= 1
            i += 1
        while source[i].isspace():
            i += 1
        assert source[i] == ';'
        source = source[:match.start()] + 'do {} while (false);' + source[i + 1:]
        count += 1
    return source, count


def check(source: str, output: Path):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        return dict(result='SKIPPED: no CPU C++ compiler')
    # Only external CUDA/ATen headers are omitted; standard library stays real.
    body = re.sub(r'^#include [<"](?:ATen/|c10/|cuda)[^\n]+\n', '', source, flags=re.M)
    body, count = strip_asm(body)
    output.write_text('// C++ SYNTAX ONLY: CUDA/ATen stubs; all PTX removed. Never execute.\n' + STUBS + body)
    command = [compiler, '-std=c++20', '-fsyntax-only', '-Wno-unknown-pragmas', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return dict(result='PASS: C++20 templates/host-wrapper syntax, CUDA/ATen stubbed and PTX removed',
                removed_asm_statements=count, command=command)
