
#include <cuda_bf16.h>
#include <stdint.h>
namespace std { using size_t = decltype(sizeof(0)); }
#define SGL_DEVICE __device__ __forceinline__
namespace sglang {
using bf16_t = __nv_bfloat16;
using bf16x2_t = __nv_bfloat162;
using fp32_t = float;
using fp32x2_t = float2;
namespace device {
constexpr int kMaxVecBytes = 16, kWarpThreads = 32;
template<bool Enabled> SGL_DEVICE void PDLWaitPrimary() {
 if constexpr (Enabled) asm volatile("griddepcontrol.wait;" ::: "memory");
}
template<bool Enabled> SGL_DEVICE void PDLTriggerSecondary() {
 if constexpr (Enabled) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}
template<class T, class U> SGL_DEVICE T cast(const U& v);
template<> SGL_DEVICE float2 cast<float2,__nv_bfloat162>(const __nv_bfloat162& v) {
 return __bfloat1622float2(v);
}
template<> SGL_DEVICE __nv_bfloat162 cast<__nv_bfloat162,float2>(const float2& v) {
 return __float22bfloat162_rn(v);
}
namespace math { SGL_DEVICE float rsqrt(float x) { return rsqrtf(x); } }
namespace warp {
template<class T> SGL_DEVICE T reduce_sum(T value) {
 #pragma unroll
 for (int mask=16;mask>0;mask>>=1) value += __shfl_xor_sync(0xffffffff,value,mask,32);
 return value;
}
}
}
}
namespace sglang {

namespace device {

namespace details {

/// \brief Maps byte-width to the corresponding unsigned integer type.
template <std::size_t N>
struct uint_trait {};

template <>
struct uint_trait<1> {
  using type = uint8_t;
};

template <>
struct uint_trait<2> {
  using type = uint16_t;
};

template <>
struct uint_trait<4> {
  using type = uint32_t;
};

template <>
struct uint_trait<8> {
  using type = uint64_t;
};

/// \brief Alias: maps `sizeof(T)` to matching unsigned int type.
template <typename T>
using sized_int = typename uint_trait<sizeof(T)>::type;

}  // namespace details

/// \brief Raw aligned storage for `N` elements of type `T`.
template <typename T, std::size_t N>
struct alignas(sizeof(T) * N) AlignedStorage {
  T data[N];
};

/**
 * \brief Aligned vector for vectorized memory access on GPU.
 *
 * Stores `N` elements of type `T` with natural alignment so that a single
 * `load`/`store` call compiles to a wide memory transaction.
 *
 * \tparam T Element type (e.g. `fp16_t`, `bf16_t`, `float`).
 * \tparam N Number of elements. Must be a power of two and
 *           `sizeof(T) * N <= 32` (256 bits).
 *
 * Example:
 * \code
 *   AlignedVector<fp16_t, 8> vec;  // 16 bytes, 128-bit aligned
 *   vec.load(input_ptr, tid);      // vectorized load
 *   vec[0] = vec[0] + 1;
 *   vec.store(output_ptr, tid);    // vectorized store
 * \endcode
 */
template <typename T, std::size_t N>
struct AlignedVector {
 private:
  static_assert(
      (N > 0 && (N & (N - 1)) == 0) && sizeof(T) * N <= kMaxVecBytes,
      "CUDA vector size exceeds arch limit: max 16 bytes on pre-Blackwell/AMD, "
      "32 bytes on Blackwell or greater");
  using element_t = typename details::sized_int<T>;
  using storage_t = AlignedStorage<element_t, N>;

 public:
  /// \brief Vectorized load from `ptr` at the given element `offset`.
  SGL_DEVICE void load(const void* ptr, int64_t offset = 0) {
    m_storage = reinterpret_cast<const storage_t*>(ptr)[offset];
  }
  /// \brief Vectorized store to `ptr` at the given element `offset`.
  SGL_DEVICE void store(void* ptr, int64_t offset = 0) const {
    reinterpret_cast<storage_t*>(ptr)[offset] = m_storage;
  }
  /// \brief Fill all N elements with the same `value`.
  SGL_DEVICE void fill(T value) {
    const auto store_value = *reinterpret_cast<element_t*>(&value);
#pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      m_storage.data[i] = store_value;
    }
  }

  SGL_DEVICE auto operator[](std::size_t idx) -> T& {
    return reinterpret_cast<T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto operator[](std::size_t idx) const -> T {
    return reinterpret_cast<const T*>(&m_storage)[idx];
  }
  SGL_DEVICE auto data() -> T* {
    return reinterpret_cast<T*>(&m_storage);
  }
  SGL_DEVICE auto data() const -> const T* {
    return reinterpret_cast<const T*>(&m_storage);
  }

 private:
  storage_t m_storage;
};


} }
namespace sglang::device::cta {
template <int kNumThreads, typename T>
SGL_DEVICE T reduce_sum(T value, T* smem) {
  static_assert(kNumThreads % kWarpThreads == 0);
  constexpr auto kNumWarps = kNumThreads / kWarpThreads;
  value = warp::reduce_sum(value);
  if constexpr (kNumWarps == 1) {
    return value;
  } else {
    const auto lane = threadIdx.x % kWarpThreads;
    if (lane == 0) smem[threadIdx.x / kWarpThreads] = value;
    __syncthreads();
    return warp::reduce_sum(lane < kNumWarps ? smem[lane] : T{0});
  }
}
}
namespace sglang {

struct NormResidualParams {
  const void* __restrict__ branch;
  const void* __restrict__ stream;
  void* __restrict__ output;
  void* __restrict__ new_stream;
  void* __restrict__ router_out;
  const void* __restrict__ post_weight;
  const void* __restrict__ pre_weight;
  const void* __restrict__ alpha;
  const void* __restrict__ beta;
  float post_eps;
  float pre_eps;
  int num_rows;
  bool gate_channel_wise;
  bool has_pre_weight;
  bool pre_norm_bf16;
};

namespace device::norm_residual {

// 16B per vectorized bf16 access; the fp32 tensors are covered by 256-bit
// accesses over the same element range (two 128-bit halves pre-Blackwell) so
// both dtypes share one thread map. All row data lives in registers as
// fp32x2_t pairs, so conversions and the bf16 round-trip use the paired
// cvt forms instead of per-element scalars.
inline constexpr int kVec = 8;
inline constexpr int kFVec = kMaxVecBytes >= 32 ? kVec : kVec / 2;
inline constexpr int kFParts = kVec / kFVec;
inline constexpr int kPairs = kVec / 2;
inline constexpr int kFPairs = kFVec / 2;

// Triton's `.to(tl.bfloat16).to(tl.float32)` round-trip; load-bearing for parity.
SGL_DEVICE fp32x2_t round_bf16(fp32x2_t v) {
  return cast<fp32x2_t>(cast<bf16x2_t>(v));
}

}  // namespace device::norm_residual

// One CTA per row; the row stays in registers across both RMS reductions.
// `kPrefetchWeights` additionally issues the gate / pre-norm weight loads before
// the first reduction, so a small-M launch costs one DRAM round trip instead of
// three; at large M the register cost of that outweighs the latency it hides.
template <
    int64_t kHidden,
    int kNumThreads,
    bool kUsePDL,
    bool kPrefetchWeights,
    bool kPostNormIsInputScale,
    bool kHasRouter,
    bool kPostWeight,
    bool kGate>
__global__ void __launch_bounds__(kNumThreads)
    pdln_norm_kernel(const __grid_constant__ NormResidualParams params) {
  using namespace device;
  using namespace device::norm_residual;

  constexpr int kPacks = kHidden / (kNumThreads * kVec);
  constexpr int kNumWarps = kNumThreads / kWarpThreads;
  constexpr int kPreSlots = kPrefetchWeights ? kPacks : 1;
  static_assert(kHidden % (kNumThreads * kVec) == 0);

  using BF16Vec = AlignedVector<bf16x2_t, kPairs>;
  using F32Vec = AlignedVector<fp32x2_t, kFPairs>;

  const int64_t base = static_cast<int64_t>(blockIdx.x) * kHidden;
  const auto branch = static_cast<const bf16_t*>(params.branch) + base;
  const auto stream = static_cast<const fp32_t*>(params.stream) + base;
  const auto out = static_cast<bf16_t*>(params.output) + base;
  const auto new_stream = static_cast<fp32_t*>(params.new_stream) + base;
  const auto router = static_cast<fp32_t*>(params.router_out) + (kHasRouter ? base : 0);
  const auto post_weight = static_cast<const bf16_t*>(params.post_weight);
  const auto pre_weight = static_cast<const bf16_t*>(params.pre_weight);
  const auto alpha_ptr = static_cast<const fp32_t*>(params.alpha);
  const auto beta_ptr = static_cast<const fp32_t*>(params.beta);

  // [0, kNumWarps) and [kNumWarps, 2*kNumWarps) are the two reduction scratch
  // areas; separate areas keep the second reduction free of a WAR barrier.
  __shared__ float smem[kNumWarps == 1 ? 1 : 2 * kNumWarps];

  const bool channel_wise = kGate && params.gate_channel_wise;
  const bool has_pre_weight = params.has_pre_weight;
  const bool pre_bf16 = params.pre_norm_bf16;

  fp32x2_t vals[kPacks][kPairs];
  fp32x2_t post_w[kPostWeight ? kPacks : 1][kPairs];
  F32Vec stream_v[kPacks][kFParts];
  F32Vec alpha_pf[kPreSlots][kFParts], beta_pf[kPreSlots][kFParts];
  BF16Vec pre_w_pf[kPreSlots];

  PDLTriggerSecondary<kUsePDL>();  // independent downstream weight prologue
  PDLWaitPrimary<kUsePDL>();

  float sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec branch_v;
    branch_v.load(branch, idx);
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      stream_v[p][j].load(stream, idx * kFParts + j);
    if constexpr (kPrefetchWeights) {
      if (has_pre_weight) pre_w_pf[p].load(pre_weight, idx);
      if constexpr (kGate) {
        if (channel_wise) {
#pragma unroll
          for (int j = 0; j < kFParts; ++j) {
            alpha_pf[p][j].load(alpha_ptr, idx * kFParts + j);
            beta_pf[p][j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    if constexpr (kPostWeight) {
      BF16Vec weight_v;
      weight_v.load(post_weight, idx);
#pragma unroll
      for (int q = 0; q < kPairs; ++q)
        post_w[p][q] = cast<fp32x2_t>(weight_v[q]);
    }
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      auto v = cast<fp32x2_t>(branch_v[q]);
      if constexpr (kPostNormIsInputScale && kPostWeight) {
        v.x *= post_w[p][q].x;
        v.y *= post_w[p][q].y;
        v = round_bf16(v);
      }
      vals[p][q] = v;
      sum_sq = fmaf(v.x, v.x, sum_sq);
      sum_sq = fmaf(v.y, v.y, sum_sq);
    }
  }

  sum_sq = cta::reduce_sum<kNumThreads>(sum_sq, smem);
  const float post_rstd = math::rsqrt(sum_sq / kHidden + params.post_eps);

  fp32x2_t gate_a = {1.0f, 1.0f};
  fp32x2_t gate_b = {1.0f, 1.0f};
  if constexpr (kGate) {
    if (!channel_wise) {
      gate_a.x = gate_a.y = *alpha_ptr;
      gate_b.x = gate_b.y = *beta_ptr;
    }
  }

  float pre_sum_sq = 0.0f;
#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    F32Vec alpha_v[kFParts], beta_v[kFParts];
    if constexpr (kGate) {
      if (channel_wise) {
#pragma unroll
        for (int j = 0; j < kFParts; ++j) {
          if constexpr (kPrefetchWeights) {
            alpha_v[j] = alpha_pf[p][j];
            beta_v[j] = beta_pf[p][j];
          } else {
            alpha_v[j].load(alpha_ptr, idx * kFParts + j);
            beta_v[j].load(beta_ptr, idx * kFParts + j);
          }
        }
      }
    }
    F32Vec new_stream_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      const int h = q / kFPairs;
      const int l = q % kFPairs;
      fp32x2_t normed_b = vals[p][q];
      normed_b.x *= post_rstd;
      normed_b.y *= post_rstd;
      if constexpr (!kPostNormIsInputScale && kPostWeight) {
        normed_b.x *= post_w[p][q].x;
        normed_b.y *= post_w[p][q].y;
      }
      normed_b = round_bf16(normed_b);
      const fp32x2_t s = stream_v[p][h][l];
      fp32x2_t new_s;
      if constexpr (kGate) {
        const fp32x2_t ga = channel_wise ? alpha_v[h][l] : gate_a;
        const fp32x2_t gb = channel_wise ? beta_v[h][l] : gate_b;
        new_s.x = ga.x * s.x + gb.x * normed_b.x;
        new_s.y = ga.y * s.y + gb.y * normed_b.y;
      } else {
        new_s.x = s.x + normed_b.x;
        new_s.y = s.y + normed_b.y;
      }
      new_stream_v[h][l] = new_s;
      const fp32x2_t pre_v = pre_bf16 ? round_bf16(new_s) : new_s;
      vals[p][q] = pre_v;
      pre_sum_sq = fmaf(pre_v.x, pre_v.x, pre_sum_sq);
      pre_sum_sq = fmaf(pre_v.y, pre_v.y, pre_sum_sq);
    }
#pragma unroll
    for (int j = 0; j < kFParts; ++j)
      new_stream_v[j].store(new_stream, idx * kFParts + j);
  }

  pre_sum_sq = cta::reduce_sum<kNumThreads>(pre_sum_sq, kNumWarps == 1 ? smem : smem + kNumWarps);
  const float pre_rstd = math::rsqrt(pre_sum_sq / kHidden + params.pre_eps);

#pragma unroll
  for (int p = 0; p < kPacks; ++p) {
    const int idx = threadIdx.x + p * kNumThreads;
    BF16Vec weight_v;
    if (has_pre_weight) {
      if constexpr (kPrefetchWeights) {
        weight_v = pre_w_pf[p];
      } else {
        weight_v.load(pre_weight, idx);
      }
    }
    BF16Vec out_v;
    F32Vec router_v[kFParts];
#pragma unroll
    for (int q = 0; q < kPairs; ++q) {
      fp32x2_t v = vals[p][q];
      v.x *= pre_rstd;
      v.y *= pre_rstd;
      if (has_pre_weight) {
        const auto wf = cast<fp32x2_t>(weight_v[q]);
        v.x *= wf.x;
        v.y *= wf.y;
      }
      if (pre_bf16) v = round_bf16(v);
      out_v[q] = cast<bf16x2_t>(v);
      if constexpr (kHasRouter) {
        router_v[q / kFPairs][q % kFPairs] = v;
      }
    }
    out_v.store(out, idx);
    if constexpr (kHasRouter) {
#pragma unroll
      for (int j = 0; j < kFParts; ++j)
        router_v[j].store(router, idx * kFParts + j);
    }
  }

}


}
