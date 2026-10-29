// Decode residual/norm-only prototype. Original projections remain untouched.
// Numerical boundaries follow retained D9 and its admitted CoopBoundary math.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace okc_cluster_norm {
constexpr int H=8192, PART=2048, NT=256, ELEMENTS=PART/NT;
struct Params {
  const __nv_bfloat16 *branch, *post_weight, *pre_weight;
  const float *stream, *alpha, *beta;
  __nv_bfloat16 *output;
  float *updated, *router;
  int rows, branch_stride, stream_stride;
  float post_eps, pre_eps;
};
struct Shared {
  // Distinct banks prevent a fast CTA's next reduction overwriting a peer's
  // still-live first sum. Both banks survive until the final cluster join.
  float partial[2][NT/32];
  float inverse[2];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int delta=16;delta;delta>>=1) v=__fadd_rn(v,__shfl_down_sync(0xffffffffu,v,delta));
  return v;
}

template<int PHASE>
__device__ __forceinline__ float row_inverse(float v,Shared& s,
    cooperative_groups::cluster_group cluster,float eps) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  v=warp_sum(v);
  if (lane==0) s.partial[PHASE][warp]=v;
  __syncthreads();
  if (warp==0) {
    v=lane<NT/32?s.partial[PHASE][lane]:0.f;
    v=warp_sum(v);
    if (lane==0) s.partial[PHASE][0]=v;
  }
  cluster.sync();
  if (threadIdx.x==0) {
    float sum=0.f;
#pragma unroll
    for (int peer=0;peer<4;++peer) {
      float* remote=cluster.map_shared_rank(&s.partial[PHASE][0],peer);
      sum=__fadd_rn(sum,*remote);
    }
    s.inverse[PHASE]=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/H),eps));
  }
  __syncthreads();
  return s.inverse[PHASE];
}

// KIND0: attention + FP32 router input; KIND1: attention without router output;
// KIND2: FFN with the task's T<=8 implicit zero tail; KIND3: full FFN.
// These are semantic modes, not independently tuned geometry variants.
template<int KIND>
__global__ void __launch_bounds__(NT) boundary(const __grid_constant__ Params p) {
  static_assert(KIND>=0 && KIND<=3,"boundary kind");
  constexpr bool POST_SCALE=KIND>=2,SLICE=KIND==2,ROUTER=KIND==0;
  __shared__ Shared s;
  auto cluster=cooperative_groups::this_cluster();
  const int row=blockIdx.x/4,rank=cluster.block_rank();
  const int col0=rank*PART+threadIdx.x;
  const size_t stream_base=(size_t)row*p.stream_stride;
  const size_t output_base=(size_t)row*H;
  float values[ELEMENTS];
  float ss=0.f;
#pragma unroll
  for (int i=0;i<ELEMENTS;++i) {
    const int col=col0+i*NT;
    float b=0.f;
    if (!SLICE || col<1024) b=__bfloat162float(p.branch[(size_t)row*p.branch_stride+col]);
    if constexpr (POST_SCALE) {
      const float w=__bfloat162float(p.post_weight[col]);
      b=__bfloat162float(__float2bfloat16_rn(__fmul_rn(b,w)));
    }
    values[i]=b;
    ss=fmaf(b,b,ss);
  }
  const float post=row_inverse<0>(ss,s,cluster,p.post_eps);
  float pre_ss=0.f;
#pragma unroll
  for (int i=0;i<ELEMENTS;++i) {
    const int col=col0+i*NT;
    const float normed=__bfloat162float(__float2bfloat16_rn(__fmul_rn(values[i],post)));
    const float residual=__fmul_rn(p.alpha[col],p.stream[stream_base+col]);
    const float branch=__fmul_rn(p.beta[col],normed);
    const float value=__fadd_rn(residual,branch);
    values[i]=value;
    p.updated[output_base+col]=value;
    pre_ss=fmaf(value,value,pre_ss);
  }
  const float pre=row_inverse<1>(pre_ss,s,cluster,p.pre_eps);
#pragma unroll
  for (int i=0;i<ELEMENTS;++i) {
    const int col=col0+i*NT;
    const float value=__fmul_rn(__fmul_rn(values[i],pre),__bfloat162float(p.pre_weight[col]));
    p.output[output_base+col]=__float2bfloat16_rn(value);
    if constexpr (ROUTER) p.router[output_base+col]=value;
  }
  // DSM memory must remain alive until every peer finishes its second sum.
  cluster.sync();
}
} // namespace okc_cluster_norm

// Appended after decoder_cluster_norm.cu by the source-bound candidate builder.
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

template<int KIND>
std::vector<int64_t> cn_info_t() {
  auto kernel=okc_cluster_norm::boundary<KIND>;
  cudaFuncAttributes a{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
  int device=0,sms=0,supported=0,clusters=0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,device));
  cudaLaunchAttribute attr{};
  attr.id=cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim={4,1,1};
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(128,1,1);config.blockDim=dim3(256,1,1);
  config.attrs=&attr;config.numAttrs=1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&clusters,kernel,&config));
  return {KIND,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,clusters,sms,supported,4,256};
}

std::vector<int64_t> cn_info(int kind) {
  switch(kind) {
    case 0:return cn_info_t<0>();case 1:return cn_info_t<1>();
    case 2:return cn_info_t<2>();case 3:return cn_info_t<3>();
    default:TORCH_CHECK(false,"invalid cluster norm kind");
  }
}

template<int KIND>
void cn_launch_t(int rows,int branch_stride,int stream_stride,
    int64_t branch,int64_t post,int64_t pre,int64_t stream,int64_t alpha,int64_t beta,
    int64_t output,int64_t updated,int64_t router,double post_eps,double pre_eps,int64_t cuda_stream) {
  TORCH_CHECK(rows>=1 && rows<=32 && stream_stride>=8192,"cluster norm shape");
  TORCH_CHECK(branch_stride>=(KIND==2?1024:8192),"cluster norm branch stride");
  if constexpr (KIND==2) TORCH_CHECK(rows<=8,"implicit tail is only legal for T<=8");
  TORCH_CHECK(branch && pre && stream && alpha && beta && output && updated,"null cluster norm input");
  if constexpr (KIND>=2) TORCH_CHECK(post,"missing post scale");
  if constexpr (KIND==0) TORCH_CHECK(router,"missing FP32 router destination");
  okc_cluster_norm::Params p{
    reinterpret_cast<const __nv_bfloat16*>(branch),reinterpret_cast<const __nv_bfloat16*>(post),
    reinterpret_cast<const __nv_bfloat16*>(pre),reinterpret_cast<const float*>(stream),
    reinterpret_cast<const float*>(alpha),reinterpret_cast<const float*>(beta),
    reinterpret_cast<__nv_bfloat16*>(output),reinterpret_cast<float*>(updated),reinterpret_cast<float*>(router),
    rows,branch_stride,stream_stride,(float)post_eps,(float)pre_eps};
  cudaLaunchAttribute attr{};
  attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={4,1,1};
  cudaLaunchConfig_t config{};
  config.gridDim=dim3(rows*4,1,1);config.blockDim=dim3(256,1,1);
  config.stream=reinterpret_cast<cudaStream_t>(cuda_stream);config.attrs=&attr;config.numAttrs=1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config,okc_cluster_norm::boundary<KIND>,p));
}

void cn_launch(int kind,int rows,int branch_stride,int stream_stride,
    int64_t branch,int64_t post,int64_t pre,int64_t stream,int64_t alpha,int64_t beta,
    int64_t output,int64_t updated,int64_t router,double post_eps,double pre_eps,int64_t cuda_stream) {
#define CN_ARGS rows,branch_stride,stream_stride,branch,post,pre,stream,alpha,beta,output,updated,router,post_eps,pre_eps,cuda_stream
  switch(kind) {
    case 0:cn_launch_t<0>(CN_ARGS);return;case 1:cn_launch_t<1>(CN_ARGS);return;
    case 2:cn_launch_t<2>(CN_ARGS);return;case 3:cn_launch_t<3>(CN_ARGS);return;
    default:TORCH_CHECK(false,"invalid cluster norm kind");
  }
#undef CN_ARGS
}
