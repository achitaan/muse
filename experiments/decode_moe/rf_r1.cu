
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <math.h>
#include <limits.h>

// H200: 16 row tiles * 8 K splits = 128 CTAs, approximately one SM wave.
// Each warp handles two router rows, sharing their weights across all tokens.
template<int TM, int BPC>
__global__ void route_fp32(const float* __restrict__ x, const float* __restrict__ w,
                          const float* __restrict__ bias, volatile float* parts,
                          unsigned int* counter, float* outw, int* outi, int M) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // R1: BPC consecutive (row tile, K split) blocks per CTA, each with the original arithmetic.
#pragma unroll 1
  for (int bi = 0; bi < BPC; ++bi) {
  const int blk = blockIdx.x * BPC + bi;
  const int nb = blk & 15, ks = blk >> 4;
  const int row0 = nb * 16 + warp * 2, row1 = row0 + 1;
  float a[TM], b[TM];
#pragma unroll
  for (int m = 0; m < TM; ++m) a[m] = b[m] = 0.f;
#pragma unroll
  for (int j = 0; j < 32; ++j) {
    const int k = ks * 1024 + j * 32 + lane;
    const float w0 = w[(size_t)row0 * 8192 + k], w1 = w[(size_t)row1 * 8192 + k];
#pragma unroll
    for (int m = 0; m < TM; ++m) {
      if (m < M) {
        const float v = x[(size_t)m * 8192 + k];
        a[m] = __fmaf_rn(v, w0, a[m]);
        b[m] = __fmaf_rn(v, w1, b[m]);
      }
    }
  }
#pragma unroll
  for (int m = 0; m < TM; ++m) {
    if (m < M) {
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        a[m] += __shfl_down_sync(0xffffffffu, a[m], d);
        b[m] += __shfl_down_sync(0xffffffffu, b[m], d);
      }
      if (lane == 0) {
        parts[((size_t)ks * 8 + m) * 256 + row0] = a[m];
        parts[((size_t)ks * 8 + m) * 256 + row1] = b[m];
      }
    }
  }
  }
  // Publish all eight producer warps' partials before this CTA's completion.
  // Fenced stores + volatile partials follow CUDA's last-block reduction pattern:
  // docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
  if (lane == 0) __threadfence();
  __syncthreads();
  __shared__ int last;
  if (tid == 0) {
    unsigned int old = atomicAdd(counter, (unsigned int)BPC);
    last = (((old + (unsigned int)BPC) & 127u) == 0u);
  }
  __syncthreads();
  if (!last || warp >= M) return;
  __threadfence();
  // The last CTA reduces every split, then routes one token per warp. The
  // counter advances 128 per invocation; its low seven bits safely cycle even
  // across uint32 wrap. All consumed partials are overwritten on every call.
  float score[8], sel[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int e = lane + 32 * j;
    float v = 0.f;
#pragma unroll
    for (int s = 0; s < 8; ++s) v += parts[((size_t)s * 8 + warp) * 256 + e];
    const float z = __fmul_rn(v, 0.1767766952966369f);
    score[j] = 1.f / (1.f + expf(-z));
    sel[j] = score[j] + bias[e];
  }
  float chosen_score[8];
  int chosen_id[8];
  float sum = 0.f;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    float best = -INFINITY, raw = 0.f;
    int id = INT_MAX;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int e = lane + j * 32;
      if (sel[j] > best || (sel[j] == best && e < id)) {
        best = sel[j]; raw = score[j]; id = e;
      }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, d);
      const float os = __shfl_xor_sync(0xffffffffu, raw, d);
      const int oi = __shfl_xor_sync(0xffffffffu, id, d);
      if (ob > best || (ob == best && oi < id)) { best = ob; raw = os; id = oi; }
    }
    chosen_score[r] = raw; chosen_id[r] = id; sum += raw;
#pragma unroll
    for (int j = 0; j < 8; ++j) if (lane + j * 32 == id) sel[j] = -INFINITY;
  }
  const float inv = 1.f / (sum + 1e-15f);
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    if (lane == r) {
      outw[warp * 8 + r] = chosen_score[r] * inv;
      outi[warp * 8 + r] = chosen_id[r];
    }
  }
}

void run(at::Tensor x, at::Tensor w, at::Tensor bias, at::Tensor parts,
         at::Tensor counter, at::Tensor outw, at::Tensor outi, int64_t bpc) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat);
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kFloat);
  TORCH_CHECK(x.dim() == 2 && x.size(0) >= 1 && x.size(0) <= 8 && x.size(1) == 8192);
  TORCH_CHECK(w.dim() == 2 && w.size(0) == 256 && w.size(1) == 8192);
  const int M = (int)x.size(0);
  auto stream = at::cuda::getCurrentCUDAStream();
  auto* p = (volatile float*)parts.data_ptr<float>();
  auto* c = reinterpret_cast<unsigned int*>(counter.data_ptr<int>());
#define RF_L(T, B) route_fp32<T, B><<<128 / B, 256, 0, stream>>>(x.data_ptr<float>(), w.data_ptr<float>(), \
      bias.data_ptr<float>(), p, c, outw.data_ptr<float>(), outi.data_ptr<int>(), M)
#define RF_S(T) switch (bpc) { case 2: RF_L(T, 2); break; case 4: RF_L(T, 4); break; case 8: RF_L(T, 8); break; \
      default: RF_L(T, 1); }
  if (M == 1) { RF_S(1) } else { RF_S(8) }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
