// Native residency queries are mandatory before the cooperative tail is used.
std::vector<int64_t> okc_tail_resources() {
  int dev = 0;
  cudaError_t ed = cudaGetDevice(&dev);
  cudaDeviceProp prop = {};
  cudaError_t ep = ed == cudaSuccess ? cudaGetDeviceProperties(&prop, dev) : ed;
  const size_t sm = okc_smem_bytes();
  std::vector<int64_t> result;
  auto one = [&](auto fn, int big) {
    cudaFuncAttributes attr = {};
    int blocks = 0;
    cudaError_t es = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
    cudaError_t ea = cudaFuncGetAttributes(&attr, fn);
    cudaError_t eo = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, okc_tail::NTL, sm);
    result.insert(result.end(), {big, (int64_t)ed, (int64_t)ep, (int64_t)es,
                  (int64_t)ea, (int64_t)eo, attr.numRegs, (int64_t)attr.localSizeBytes,
                  (int64_t)attr.sharedSizeBytes, (int64_t)sm, prop.multiProcessorCount,
                  blocks, prop.cooperativeLaunch, okc_tail::NTL});
  };
  one(okc_tail::moe_fused<false>, 0);
  one(okc_tail::moe_fused<true>, 1);
  return result;
}
