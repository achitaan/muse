#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

static void checked_tensor(const torch::Tensor& t, torch::ScalarType dtype) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == dtype,
              "expected contiguous CUDA tensor of required dtype");
}

torch::Tensor native_maps(std::vector<torch::Tensor> w) {
  TORCH_CHECK(w.size() == 6, "six stock weights/scales required");
  c10::cuda::CUDAGuard guard(w[0].device());
  auto host = torch::empty({4 * (int64_t)sizeof(CUtensorMap)}, torch::dtype(torch::kUInt8));
  for (int pb = 0; pb < 2; ++pb) {
    const int n = pb ? 4096 : 8192, width = pb ? 256 : 128;
    const auto& q = w[pb * 3];
    const auto& s = w[pb * 3 + 1];
    checked_tensor(q, torch::kInt32);
    TORCH_CHECK(s.is_cuda() && s.is_contiguous() && s.element_size() == 1,
                "encoded stock scale bytes required");
    TORCH_CHECK(q.sizes() == torch::IntArrayRef({32, 256, n * 2}), "weight shape");
    TORCH_CHECK(s.sizes() == torch::IntArrayRef({32, 256, n}), "scale shape");
    for (int scale = 0; scale < 2; ++scale) {
      alignas(64) CUtensorMap map;
      uint64_t dims[2] = {(uint64_t)n, 32 * 256};
      uint64_t strides[1] = {(uint64_t)n * (scale ? 1 : 8)};
      uint32_t box[2] = {(uint32_t)width, 16}, elem[2] = {1, 1};
      auto error = cuTensorMapEncodeTiled(
          &map, scale ? CU_TENSOR_MAP_DATA_TYPE_UINT8 : CU_TENSOR_MAP_DATA_TYPE_UINT64,
          2, scale ? s.data_ptr() : q.data_ptr(), dims, strides, box, elem,
          CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
          CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
      TORCH_CHECK(error == CUDA_SUCCESS, "cuTensorMapEncodeTiled error ", (int)error);
      memcpy(host.data_ptr<uint8_t>() + (pb * 2 + scale) * sizeof(map), &map, sizeof(map));
    }
  }
  return host.to(w[0].device());
}

std::vector<torch::Tensor> native_workspace(torch::Tensor exemplar) {
  c10::cuda::CUDAGuard guard(exemplar.device());
  using namespace okc;
  const size_t fs = EMAX + (size_t)EMAX * HD + WGS_FIXF;
  const size_t bs = ((size_t)EMAX + 2 * WGS_XROWS) * HD;
  return {
      torch::zeros({8 + 1024 + EMAX}, exemplar.options().dtype(torch::kInt32)),
      torch::zeros({(int64_t)fs}, exemplar.options().dtype(torch::kFloat32)),
      torch::zeros({(int64_t)bs}, exemplar.options().dtype(torch::kBFloat16))};
}

std::vector<int64_t> native_initialize(torch::Tensor exemplar) {
  c10::cuda::CUDAGuard guard(exemplar.device());
  const size_t smem = okc_smem_bytes();
  C10_CUDA_CHECK(cudaFuncSetAttribute(okc::moe_fused<true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  int blocks = 0;
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, okc::moe_fused<true>, okc::NTL, smem));
  cudaDeviceProp prop;
  int device;
  C10_CUDA_CHECK(cudaGetDevice(&device));
  C10_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  TORCH_CHECK(prop.major == 9 && prop.minor == 0 && blocks >= 1, "resident Hopper grid required");
  TORCH_CHECK(prop.multiProcessorCount < okc::WGS_SLOTS, "fixup slot capacity exceeded");
  return {prop.multiProcessorCount, (int64_t)smem, blocks, okc::NTL};
}

void native_run(std::vector<torch::Tensor> w, std::vector<torch::Tensor> scratch,
                torch::Tensor maps, torch::Tensor x, torch::Tensor ids,
                torch::Tensor out, torch::Tensor local, torch::Tensor debug, int64_t lo,
                int64_t grid, int64_t phases) {
  TORCH_CHECK(w.size() == 6 && scratch.size() == 3, "invalid bound state");
  c10::cuda::CUDAGuard guard(x.device());
  checked_tensor(x, torch::kBFloat16);
  checked_tensor(ids, torch::kInt32);
  checked_tensor(out, torch::kBFloat16);
  checked_tensor(local, torch::kInt32);
  checked_tensor(maps, torch::kUInt8);
  TORCH_CHECK(x.dim() == 2 && x.size(1) == 4096 && x.size(0) <= 384, "input capacity");
  TORCH_CHECK(ids.sizes() == torch::IntArrayRef({x.size(0), 8}), "route shape");
  TORCH_CHECK(local.sizes() == ids.sizes(), "local route shape");
  TORCH_CHECK(out.sizes() == torch::IntArrayRef({x.size(0), 8, 4096}), "output shape");
  TORCH_CHECK(lo >= 0 && lo <= 224 && lo % 32 == 0, "EP8 offset");
  TORCH_CHECK(grid >= 1 && grid < okc::WGS_SLOTS, "resident grid capacity");
  TORCH_CHECK(phases == 1 || phases == 2, "phase count");
  if (!x.size(0)) return;
  okc::Params p{};
  p.maps = reinterpret_cast<const CUtensorMap*>(maps.data_ptr());
  p.gsA = w[2].data_ptr<float>(); p.gsB = w[5].data_ptr<float>();
  p.x = (const __nv_bfloat16*)x.data_ptr(); p.topk_ids = ids.data_ptr<int>();
  p.M = x.size(0); p.lo = lo; p.nloc = 32; p.phases = phases;
  int* ib = scratch[0].data_ptr<int>();
  p.ctr = ib; p.doneA = ib + 8; p.xready = ib + 8 + 512;
  p.ss = scratch[1].data_ptr<float>();
  p.ss1 = reinterpret_cast<float*>(ib + 8 + 1024);
  p.ybuf = p.ss + okc::EMAX;
  p.inter = (__nv_bfloat16*)scratch[2].data_ptr();
  p.xA = p.inter + (size_t)okc::EMAX * okc::HD;
  p.xB = p.xA + (size_t)okc::WGS_XROWS * okc::HD;
  p.doneAs = reinterpret_cast<int*>(p.ybuf + (size_t)okc::EMAX * okc::HD
      + (size_t)2 * okc::WGS_SLOTS * 4 * okc::WGS_FIXW) + 2 * okc::WGS_SLOTS * 4;
  p.out = (__nv_bfloat16*)out.data_ptr();
  p.local_out = local.data_ptr<int>();
  p.gemm1_debug = debug.numel() ? (__nv_bfloat16*)debug.data_ptr() : nullptr;
  TORCH_CHECK(!debug.numel() || debug.numel() == x.size(0) * 8 * 8192, "debug capacity");
  okc::moe_fused<true><<<(int)grid, okc::NTL, okc_smem_bytes(), at::cuda::getCurrentCUDAStream()>>>(p);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <bool PB>
__global__ void native_tile_kernel(const CUtensorMap* maps, float* out, int expert, int ub, int st) {
  __shared__ __align__(128) unsigned char tile[36864];
  __shared__ __align__(8) uint64_t bar;
  const int tid = threadIdx.x;
  if (!tid) {
    okc::mbar_init(&bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(okc::smem_u32(&bar)), "r"(36864) : "memory");
    const int row = ub * (PB ? 256 : 128), kg = expert * 256 + st * 16;
    okc::native_tma(tile, maps + PB * 2, row, kg, &bar);
    okc::native_tma(tile + 32768, maps + PB * 2 + 1, row, kg, &bar);
    if (!PB) {
      okc::native_tma(tile + 16384, maps, row + 4096, kg, &bar);
      okc::native_tma(tile + 34816, maps + 1, row + 4096, kg, &bar);
    }
  }
  __syncthreads();
  okc::mbar_wait(&bar, 0);
  const int wg = tid / 128, wp = (tid / 32) % 4, lane = tid % 32;
  for (int step = 0; step < 16; ++step) {
    uint32_t r[4];
    okc::native_regs<PB>(tile, step, wg, wp, lane, r);
    for (int reg = 0; reg < 4; ++reg) {
      const float2 f = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&r[reg]));
      const int n = wg * 64 + wp * 16 + lane / 4 + (reg % 2) * 8;
      const int k = step * 16 + (reg / 2) * 8 + (lane % 4) * 2;
      out[n * 256 + k] = f.x * 0x1p119f;
      out[n * 256 + k + 1] = f.y * 0x1p119f;
    }
  }
}

torch::Tensor native_tile(torch::Tensor maps, int64_t phase, int64_t expert,
                          int64_t unit, int64_t stage) {
  c10::cuda::CUDAGuard guard(maps.device());
  auto out = torch::empty({256, 256}, maps.options().dtype(torch::kFloat32));
  if (phase) native_tile_kernel<true><<<1, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const CUtensorMap*)maps.data_ptr(), out.data_ptr<float>(), expert, unit, stage);
  else native_tile_kernel<false><<<1, 512, 0, at::cuda::getCurrentCUDAStream()>>>(
      (const CUtensorMap*)maps.data_ptr(), out.data_ptr<float>(), expert, unit, stage);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("maps", &native_maps);
  m.def("workspace", &native_workspace);
  m.def("initialize", &native_initialize);
  m.def("run", &native_run);
  m.def("tile", &native_tile);
}
