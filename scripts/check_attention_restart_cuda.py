"""Clang CUDA C++ syntax checks with explicit local API stubs, NOT CUDA compilation.

This is useful on macOS but cannot validate PTX instructions, real CUDA headers,
resource usage or execution. The report states these limitations explicitly.
"""
import ast
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STUB=r'''
#include <stddef.h>
#include <stdint.h>
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(a,b) __attribute__((launch_bounds(a,b)))
#define __align__(a) __attribute__((aligned(a)))
#define __grid_constant__
#include <vector>
#ifndef INFINITY
#define INFINITY __builtin_inff()
#endif
#define C10_CUDA_CHECK(x) (void)(x)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() ((void)0)
#define TORCH_CHECK(x,...) ((void)(x))
typedef void* cudaStream_t;
struct dim3 { unsigned x,y,z; __host__ __device__ constexpr dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
extern __device__ const dim3 threadIdx,blockIdx;
struct __attribute__((aligned(8))) uint2 { uint32_t x,y; };
struct __attribute__((aligned(16))) uint4 { uint32_t x,y,z,w; };
struct __attribute__((aligned(8))) float2 { float x,y; };
struct __attribute__((aligned(16))) float4 { float x,y,z,w; };
__device__ uint2 make_uint2(uint32_t,uint32_t);
__device__ uint4 make_uint4(uint32_t,uint32_t,uint32_t,uint32_t);
__device__ float2 make_float2(float,float);
__device__ float4 make_float4(float,float,float,float);
struct __nv_bfloat16 { unsigned short v; };
struct __nv_bfloat162 { unsigned v; };
struct __half2 { unsigned v; };
struct __half { unsigned short v; };
__device__ __half2 __floats2half2_rn(float,float);
__device__ float2 __half22float2(__half2);
__device__ float __bfloat162float(__nv_bfloat16);
__device__ float2 __bfloat1622float2(__nv_bfloat162);
__device__ __nv_bfloat16 __float2bfloat16_rn(float);
__device__ __nv_bfloat162 __floats2bfloat162_rn(float,float);
__device__ unsigned long long __cvta_generic_to_shared(const void*);
__device__ float __shfl_xor_sync(unsigned,float,int,int=32);
__device__ int __shfl_sync(unsigned,int,int,int=32);
__device__ float fmaxf(float,float);
__device__ float fmaf(float,float,float);
__device__ float rintf(float);
__device__ float rsqrtf(float);
__device__ float __expf(float);
__device__ float __fdividef(float,float);
__device__ void __sincosf(float,float*,float*);
__device__ void __syncthreads();
__device__ void __threadfence();
template<class T> __device__ T __ldcg(const T*);
template<class T> __device__ T __ldg(const T*);
template<class T> __device__ T max(T a,T b){return a>b?a:b;}
template<class T> __device__ T min(T a,T b){return a<b?a:b;}
namespace cuda {
enum { thread_scope_device, memory_order_acq_rel, memory_order_release };
template<class T,int scope> struct atomic_ref {
  __device__ atomic_ref(T&);
  __device__ T fetch_add(T,int);
  __device__ void store(T,int);
}; }
namespace torch { constexpr int kFloat16=1,kFloat32=2; struct Tensor {
  template<class T> T* data_ptr() const;
  void* data_ptr() const;
  int size(int) const;
  int device() const;
  int scalar_type() const;
  int64_t numel() const;
}; }
namespace c10 { namespace cuda { struct CUDAGuard { CUDAGuard(int); }; } }
namespace at { namespace cuda { cudaStream_t getCurrentCUDAStream(); } }
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize=1;
template<class F> int cudaFuncSetAttribute(F,int,int);
struct cudaFuncAttributes { int numRegs,maxThreadsPerBlock; size_t localSizeBytes; };
template<class F> int cudaFuncGetAttributes(cudaFuncAttributes*,F);
template<class F> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*,F,int,size_t);
__host__ int cudaConfigureCall(dim3,dim3,size_t=0,cudaStream_t=nullptr);
__host__ int cudaSetupArgument(const void*,size_t,size_t);
__host__ int cudaLaunch(const void*);
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--v2',action='store_true')
    parser.add_argument('--v3',action='store_true')
    args=parser.parse_args()
    stem='attention_restart_v3' if args.v3 else 'attention_restart_v2' if args.v2 else 'attention_restart'
    tree=ast.parse((ROOT/'candidates'/f'{stem}.py').read_text())
    src=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
             and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='RR_CUDA')
    # Only external includes are replaced. The complete new core, including
    # all template bodies and host launch instantiations, is left unchanged.
    text=STUB+'\n'+''.join(line for line in src.splitlines(keepends=True) if not line.startswith('#include'))
    target=ROOT/'.tools'/f'{stem}_syntax.cu'
    target.write_text(text)
    result={}
    for side in ('host','device'):
        cmd=['clang++','-x','cuda','--cuda-%s-only'%side,'--cuda-gpu-arch=sm_90',
             '-nocudainc','-nocudalib','-std=c++20','-fsyntax-only',str(target)]
        run=subprocess.run(cmd,capture_output=True,text=True)
        result[side]=dict(command=cmd,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr)
        print(side,run.returncode,run.stderr)
    result['scope']='CUDA C++ syntax with API stubs only; not NVRTC/ptxas or real-header compilation'
    output='attn_restart_v3_cuda_syntax.json' if args.v3 else 'attn_restart_v2_cuda_syntax.json' if args.v2 else 'attn_restart_cuda_syntax.json'
    (ROOT/'results'/output).write_text(json.dumps(result,indent=2)+'\n')
    assert all(result[s]['returncode']==0 for s in ('host','device'))


if __name__=='__main__': main()
