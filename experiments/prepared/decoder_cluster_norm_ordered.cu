// Bounded repair: preserve the pinned scalar kernel's logical reduction tree.
//256 logical threads, vec8/four packs and all8warp partials; four physical CTAs.
// Required BF16 materializations precede the actual second FP32 norm statistic.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace okc_cluster_norm_ordered {
constexpr int H=8192, NT=64, LOGICAL_THREADS=256, PACKS=4, VEC=8;
struct Params {
  const __nv_bfloat16 *branch, *post_weight, *pre_weight;
  const float *stream, *alpha, *beta;
  __nv_bfloat16 *output;
  float *updated, *router;
  int rows, branch_stride, stream_stride;
  float post_eps, pre_eps;
};
struct Shared {
  // Each CTA publishes its two original warp partials, without collapsing
  // them to a CTA sum. Separate banks survive until the final lifetime join.
  float partial[2][NT/32];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int mask=16;mask;mask>>=1) v=__fadd_rn(v,__shfl_xor_sync(0xffffffffu,v,mask));
  return v;
}

template<int PHASE>
__device__ __forceinline__ float row_inverse(float v,Shared& s,
    cooperative_groups::cluster_group cluster,float eps) {
  const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  v=warp_sum(v);
  if (lane==0) s.partial[PHASE][warp]=v;
  __syncthreads();
  cluster.sync();
  float partial=0.f;
  if (lane<8) {
    const int peer=lane/2,peer_warp=lane%2;
    float* remote=cluster.map_shared_rank(&s.partial[PHASE][peer_warp],peer);
    partial=*remote;
  }
  // This is the original global warp0 tree over eight warp partials. Every
  // physical warp performs it; no sum-of-CTA-sums changes the association.
  const float sum=warp_sum(partial);
  return rsqrtf(sum*(1.f/H)+eps);
}

//0attention/routerFP32,1attention/no router,2FFN slicedT<=8,3fullFFN.
template<int KIND>
__global__ void __launch_bounds__(NT) boundary(const __grid_constant__ Params p) {
  static_assert(KIND>=0 && KIND<=3,"boundary kind");
  constexpr bool POST_SCALE=KIND>=2,SLICE=KIND==2,ROUTER=KIND==0;
  __shared__ Shared s;
  auto cluster=cooperative_groups::this_cluster();
  const int row=blockIdx.x/4,logical_thread=cluster.block_rank()*NT+threadIdx.x;
  const size_t stream_base=(size_t)row*p.stream_stride,output_base=(size_t)row*H;
  float values[PACKS][VEC];
  float ss=0.f;
#pragma unroll
  for (int pack=0;pack<PACKS;++pack) {
    const int vector_index=logical_thread+pack*LOGICAL_THREADS;
#pragma unroll
    for (int v=0;v<VEC;++v) {
      const int col=vector_index*VEC+v;
      float b=0.f;
      if (!SLICE || col<1024) b=__bfloat162float(p.branch[(size_t)row*p.branch_stride+col]);
      if constexpr (POST_SCALE) {
        const float w=__bfloat162float(p.post_weight[col]);
        b=__bfloat162float(__float2bfloat16_rn(__fmul_rn(b,w)));
      }
      values[pack][v]=b;
      ss=fmaf(b,b,ss);
    }
  }
  const float post=row_inverse<0>(ss,s,cluster,p.post_eps);
  float pre_ss=0.f;
#pragma unroll
  for (int pack=0;pack<PACKS;++pack) {
    const int vector_index=logical_thread+pack*LOGICAL_THREADS;
#pragma unroll
    for (int v=0;v<VEC;++v) {
      const int col=vector_index*VEC+v;
      const float normed=__bfloat162float(__float2bfloat16_rn(__fmul_rn(values[pack][v],post)));
      const float residual=__fmul_rn(p.alpha[col],p.stream[stream_base+col]);
      const float branch=__fmul_rn(p.beta[col],normed);
      const float value=__fadd_rn(residual,branch);
      values[pack][v]=value;
      p.updated[output_base+col]=value;
      pre_ss=fmaf(value,value,pre_ss);
    }
  }
  const float pre=row_inverse<1>(pre_ss,s,cluster,p.pre_eps);
#pragma unroll
  for (int pack=0;pack<PACKS;++pack) {
    const int vector_index=logical_thread+pack*LOGICAL_THREADS;
#pragma unroll
    for (int v=0;v<VEC;++v) {
      const int col=vector_index*VEC+v;
      const float value=__fmul_rn(__fmul_rn(values[pack][v],pre),__bfloat162float(p.pre_weight[col]));
      p.output[output_base+col]=__float2bfloat16_rn(value);
      if constexpr (ROUTER) p.router[output_base+col]=value;
    }
  }
  cluster.sync(); // no peer may exit while DSM statistics remain in use
}
} // namespace okc_cluster_norm_ordered
