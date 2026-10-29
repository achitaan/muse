// Local cooperative O1/O8 launch admission; inherited device code is attributed
// in THIRD_PARTY_NOTICES.md and the integrated submission's full notices.
template <int FAMILY, class Kernel>
static int64_t cb_launch_t(Kernel kernel, int fmt, int query, int m,
    int64_t wp_arg, int64_t x_arg, int64_t branch_arg, int64_t esc_arg,
    int64_t stream_arg, int64_t alpha_arg, int64_t beta_arg, int64_t gain_arg,
    int64_t new_stream_arg, int64_t output_arg, int64_t router_arg,
    int64_t t01, int64_t u01, int64_t cuda_stream_arg, float post_eps, float pre_eps) {
  constexpr int NT = FAMILY == 1 ? 1024 : 256;
  constexpr int GRID = FAMILY == 1 ? 128 : 512;
  constexpr int WR = FAMILY == 1 ? 4 : 1;
  constexpr size_t SHARED = FAMILY == 1 ? 16384 : 4096;
  if (query) {
    int device = 0, sm_count = 0, supported = 0, blocks = 0;
    cudaFuncAttributes attributes{};
    if (cudaGetDevice(&device) != cudaSuccess) return -101;
    if (cudaDeviceGetAttribute(&sm_count,cudaDevAttrMultiProcessorCount,device) != cudaSuccess) return -102;
    if (cudaDeviceGetAttribute(&supported,cudaDevAttrCooperativeLaunch,device) != cudaSuccess) return -103;
    if (cudaFuncGetAttributes(&attributes,kernel) != cudaSuccess) return -104;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,NT,SHARED) != cudaSuccess) return -105;
    switch(query) {
      case 1: return supported && GRID <= sm_count*blocks;
      case 2: return attributes.numRegs;
      case 3: return attributes.localSizeBytes;
      case 4: return attributes.sharedSizeBytes;
      case 5: return blocks;
      case 6: return sm_count;
      case 7: return supported;
      case 8: return SHARED;
      default: return -106;
    }
  }
  // Resource admission was already done in prepare, outside graph capture.
  // Reject unsupported shapes before a launch, preserving original fallbacks.
  if (m < 1 || m > 8 || (FAMILY == 1 && m != 1)) return -107;
  const int4* wp = reinterpret_cast<const int4*>(wp_arg);
  const __nv_bfloat16* x = reinterpret_cast<const __nv_bfloat16*>(x_arg);
  void* branch = reinterpret_cast<void*>(branch_arg);
  const uint32_t* esc = reinterpret_cast<const uint32_t*>(esc_arg);
  CoopBoundary params{
    reinterpret_cast<const float*>(stream_arg),
    reinterpret_cast<const float*>(alpha_arg),
    reinterpret_cast<const float*>(beta_arg),
    reinterpret_cast<const __nv_bfloat16*>(gain_arg),
    reinterpret_cast<float*>(new_stream_arg),
    reinterpret_cast<float*>(router_arg),
    reinterpret_cast<__nv_bfloat16*>(output_arg),post_eps,pre_eps};
  cudaLaunchAttribute cooperative{};
  cooperative.id = cudaLaunchAttributeCooperative;
  cooperative.val.cooperative = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(GRID,1,1);
  config.blockDim = dim3(NT,1,1);
  config.dynamicSmemBytes = SHARED;
  config.stream = reinterpret_cast<cudaStream_t>(cuda_stream_arg);
  config.attrs = &cooperative;
  config.numAttrs = 1;
  cudaError_t status = cudaLaunchKernelEx(&config,kernel,wp,x,branch,esc,
      m,8192,1024,WR,8,2,
      static_cast<uint32_t>(t01),static_cast<uint32_t>(static_cast<uint64_t>(t01)>>32),
      static_cast<uint32_t>(u01),static_cast<uint32_t>(static_cast<uint64_t>(u01)>>32),params);
  return status == cudaSuccess ? 0 : -1000-static_cast<int>(status);
}

int64_t cb_launch(int family, int fmt, int query, int m,
    int64_t wp, int64_t x, int64_t branch, int64_t esc,
    int64_t stream, int64_t alpha, int64_t beta, int64_t gain,
    int64_t new_stream, int64_t output, int64_t router,
    int64_t t01, int64_t u01, int64_t cuda_stream, float post_eps, float pre_eps) {
#define CB_ARGS fmt,query,m,wp,x,branch,esc,stream,alpha,beta,gain,new_stream,output,router,t01,u01,cuda_stream,post_eps,pre_eps
  if (family == 1) {
    if (fmt == 1) return cb_launch_t<1>(cb_smallm2_gemv<1,1,1,2,0,0,1024,0,8192,1024,4,8,2,1,7>,CB_ARGS);
    if (fmt == 0) return cb_launch_t<1>(cb_smallm2_gemv<0,1,1,2,0,0,1024,0,8192,1024,4,8,2,1,7>,CB_ARGS);
  }
  if (family == 8) {
    if (fmt == 1) return cb_launch_t<8>(cb_v18s_gemv<1,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6>,CB_ARGS);
    if (fmt == 0) return cb_launch_t<8>(cb_v18s_gemv<0,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6>,CB_ARGS);
  }
#undef CB_ARGS
  return -108;
}
