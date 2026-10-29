
#include <vector>
template<bool Q>
std::vector<int64_t> lg_info_t() {
  auto kernel=Q ? gqx_kernel<128,1,8,3> : gq_kernel<128,1,8,2>;
  constexpr int cl=Q?3:2,grid=Q?108:128,bytes=Q?230544:215184;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));
  cudaFuncAttributes a{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a,kernel));
  int device=0,sms=0,cap=0,blocks=0,supported=0;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,device));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&supported,cudaDevAttrClusterLaunch,device));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,160,bytes));
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={cl,1,1};
  cudaLaunchConfig_t config{};config.gridDim=dim3(grid,1,1);config.blockDim=dim3(160,1,1);
  config.dynamicSmemBytes=bytes;config.attrs=&attr;config.numAttrs=1;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveClusters(&cap,kernel,&config));
  return {Q?0:1,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,sms,cap,bytes,160,cl,grid,(int64_t)a.maxDynamicSharedSizeBytes,blocks,supported};
}
std::vector<int64_t> lg_info(int kind) {
  if (kind==0) return lg_info_t<true>();
  if (kind==1) return lg_info_t<false>();
  TORCH_CHECK(false,"invalid fixed Dense family");
}
