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
