#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <vector>

namespace dense_restart {
constexpr int THREADS = 160, MATH = 128, STAGES = 3, WB = 64 * 128, RP = 68;
__device__ __forceinline__ uint32_t shared_addr(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void init(uint32_t b, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(b), "r"(count) : "memory");
}
__device__ __forceinline__ void arm(uint32_t b, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(bytes) : "memory");
}
__device__ __forceinline__ void wait(uint32_t b, int phase) {
  asm volatile("{ .reg .pred p; W: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%0], %1;"
               " @!p bra W; }" :: "r"(b), "r"(phase) : "memory");
}
__device__ __forceinline__ void arrive(uint32_t b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(b) : "memory");
}
__device__ __forceinline__ void arrive_rank0(uint32_t b) {
  uint32_t remote;
  asm("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(remote) : "r"(b));
  asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ void math_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}
__device__ __forceinline__ void tma_w(uint32_t dst, const CUtensorMap* map, uint32_t bar, int k, int row) {
  uint64_t policy;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.L2::cache_hint "
               "[%0], [%1, {%3, %4}], [%2], %5;" ::
               "r"(dst), "l"(map), "r"(bar), "r"(k), "r"(row), "l"(policy) : "memory");
}
__device__ __forceinline__ void tma_x_pair(uint32_t dst, const CUtensorMap* map, uint32_t bar, int k) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
               "[%0], [%1, {%3, %4}], [%2], %5;" ::
               "r"(dst), "l"(map), "r"(bar), "r"(k), "r"(0), "h"((uint16_t)3) : "memory");
}
__device__ __forceinline__ uint64_t descriptor(uint32_t p) {
  return (uint64_t)((p & 0x3ffffu) >> 4) | (uint64_t(1) << 16)
         | (uint64_t(64) << 32) | (uint64_t(1) << 62);
}
template <int N> __device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory");
}
// Standard PTX SS-WGMMA operand lists are generated below. The scheduler,
// consumer-armed barrier protocol and work ownership are independent code.
template <int TOK> __device__ __forceinline__ void mma(float*, uint64_t, uint64_t);
template <> __device__ __forceinline__ void mma<16>(float* d, uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %10, 0; "
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7}, %8, %9, p, 1, 1, 0, 0; }"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]) : "l"(a), "l"(b), "r"(1));
}
template <> __device__ __forceinline__ void mma<32>(float* d, uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %18, 0; "
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, p, 1, 1, 0, 0; }"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]) : "l"(a), "l"(b), "r"(1));
}
template <> __device__ __forceinline__ void mma<128>(float* d, uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %66, 0; "
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, p, 1, 1, 0, 0; }"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "l"(a), "l"(b), "r"(1));
}
template <> __device__ __forceinline__ void mma<256>(float* d, uint64_t a, uint64_t b) {
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %130, 0; "
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, %128, %129, p, 1, 1, 0, 0; }"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]),"+f"(d[64]),"+f"(d[65]),"+f"(d[66]),"+f"(d[67]),"+f"(d[68]),"+f"(d[69]),"+f"(d[70]),"+f"(d[71]),"+f"(d[72]),"+f"(d[73]),"+f"(d[74]),"+f"(d[75]),"+f"(d[76]),"+f"(d[77]),"+f"(d[78]),"+f"(d[79]),"+f"(d[80]),"+f"(d[81]),"+f"(d[82]),"+f"(d[83]),"+f"(d[84]),"+f"(d[85]),"+f"(d[86]),"+f"(d[87]),"+f"(d[88]),"+f"(d[89]),"+f"(d[90]),"+f"(d[91]),"+f"(d[92]),"+f"(d[93]),"+f"(d[94]),"+f"(d[95]),"+f"(d[96]),"+f"(d[97]),"+f"(d[98]),"+f"(d[99]),"+f"(d[100]),"+f"(d[101]),"+f"(d[102]),"+f"(d[103]),"+f"(d[104]),"+f"(d[105]),"+f"(d[106]),"+f"(d[107]),"+f"(d[108]),"+f"(d[109]),"+f"(d[110]),"+f"(d[111]),"+f"(d[112]),"+f"(d[113]),"+f"(d[114]),"+f"(d[115]),"+f"(d[116]),"+f"(d[117]),"+f"(d[118]),"+f"(d[119]),"+f"(d[120]),"+f"(d[121]),"+f"(d[122]),"+f"(d[123]),"+f"(d[124]),"+f"(d[125]),"+f"(d[126]),"+f"(d[127]) : "l"(a), "l"(b), "r"(1));
}

template <int TOK, bool STREAM>
__device__ __forceinline__ void release_slot(uint32_t full, uint32_t empty, int rank, int seq, int length) {
  int slot = seq % STAGES;
  if (seq + STAGES < length) arm(full + slot * 8, WB + TOK * 128);
  // Re-arm BEFORE either empty arrival. Rank 0's empty phase cannot complete
  // until both destinations have armed the next full phase.
  arrive(empty + slot * 8);
  if (rank) arrive_rank0(empty + slot * 8);
}

__device__ __forceinline__ int owner(int unit, int total, int groups) {
  return (int)(((long long)(unit + 1) * groups - 1) / total);
}
__device__ __forceinline__ uint32_t publish_count(int* p) {
  uint32_t old;
  asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;" : "=r"(old) : "l"(p) : "memory");
  return old;
}

template <int TOK, bool STREAM>
__global__ void __launch_bounds__(THREADS, 1)
gemm(const __grid_constant__ CUtensorMap weights, const __grid_constant__ CUtensorMap activations,
     __nv_bfloat16* __restrict__ out, float4* __restrict__ partial,
     int* __restrict__ counts, int M, int N, int K, int groups, int maxparts) {
  constexpr int XB = TOK * 128, ST = WB + XB, ACC = TOK / 2;
  extern __shared__ __align__(1024) uint8_t raw[];
  const uint32_t sb = (shared_addr(raw) + 1023u) & ~1023u;
  const uint32_t full = sb + STAGES * ST, empty = full + STAGES * 8;
  int* last = (int*)(raw + (sb - shared_addr(raw)) + STAGES * ST + 48);
  float* transpose = (float*)(raw + (sb - shared_addr(raw)) + (STREAM ? STAGES * ST + 64 : 0));
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int rank = blockIdx.x & 1, group = blockIdx.x >> 1;
  const int kc = K / 64, pairs = N / 128, total = pairs * kc;
  const int begin = STREAM ? (int)((long long)total * group / groups) : group * kc;
  const int end = STREAM ? (int)((long long)total * (group + 1) / groups) : (group + 1) * kc;
  const int length = end - begin;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) { init(full + s * 8, 1); init(empty + s * 8, rank ? 1 : 2); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    for (int s = 0; s < STAGES && s < length; ++s) arm(full + s * 8, WB + XB);
  }
  __syncthreads();
  cluster_sync();

  if (warp == 4) {
    if (lane == 0) {
      asm volatile("prefetch.tensormap [%0];" :: "l"(&weights) : "memory");
      if (!rank) asm volatile("prefetch.tensormap [%0];" :: "l"(&activations) : "memory");
      for (int seq = 0; seq < length; ++seq) {
        const int slot = seq % STAGES;
        if (seq >= STAGES) wait(empty + slot * 8, ((seq / STAGES) - 1) & 1);
        const int unit = begin + seq, pair = unit / kc, col = (unit - pair * kc) * 64;
        const uint32_t dst = sb + slot * ST;
        tma_w(dst, &weights, full + slot * 8, col, (pair * 2 + rank) * 64);
        if (!rank) tma_x_pair(dst + WB, &activations, full + slot * 8, col);
      }
    }
  } else {
    int unit = begin, released = -1;
    while (unit < end) {
      const int pair = unit / kc, nb = pair * 2 + rank;
      const int segment_end = min(end, (pair + 1) * kc);
      float acc[ACC];
#pragma unroll
      for (int j = 0; j < ACC; ++j) acc[j] = 0.f;
      for (; unit < segment_end; ++unit) {
        const int seq = unit - begin, slot = seq % STAGES;
        wait(full + slot * 8, (seq / STAGES) & 1);
#pragma unroll
        for (int j = 0; j < ACC; ++j) asm volatile("" : "+f"(acc[j]) :: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
        for (int q = 0; q < 4; ++q)
          mma<TOK>(acc, descriptor(sb + slot * ST + q * 32), descriptor(sb + slot * ST + WB + q * 32));
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        wg_wait<1>();
#pragma unroll
        for (int j = 0; j < ACC; ++j) asm volatile("" : "+f"(acc[j]) :: "memory");
        if (seq - 1 > released) {
          if (tid == 0) release_slot<TOK, STREAM>(full, empty, rank, seq - 1, length);
          released = seq - 1;
        }
      }
      wg_wait<0>();
#pragma unroll
      for (int j = 0; j < ACC; ++j) asm volatile("" : "+f"(acc[j]) :: "memory");
      const int final_seq = unit - begin - 1;
      if (final_seq > released) {
        if (tid == 0) release_slot<TOK, STREAM>(full, empty, rank, final_seq, length);
        released = final_seq;
      }

      bool writer = true;
      if constexpr (STREAM) {
        const int first = owner(pair * kc, total, groups);
        const int contributors = owner((pair + 1) * kc - 1, total, groups) - first + 1;
        const int slot = group - first;
        float4* dst = partial + ((long long)nb * maxparts + slot) * (ACC / 4) * MATH;
#pragma unroll
        for (int j = 0; j < ACC / 4; ++j)
          __stcg(dst + j * MATH + tid, make_float4(acc[j * 4], acc[j * 4 + 1], acc[j * 4 + 2], acc[j * 4 + 3]));
        __threadfence();
        math_sync();
        if (tid == 0) *last = publish_count(counts + nb) == (uint32_t)(contributors - 1);
        math_sync();
        writer = *last != 0;
        if (writer) {
#pragma unroll
          for (int j = 0; j < ACC / 4; ++j) {
            float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int p = 0; p < contributors; ++p) {
              const float4 v = __ldcg(partial + (((long long)nb * maxparts + p) * (ACC / 4) + j) * MATH + tid);
              sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
            }
            acc[j * 4] = sum.x; acc[j * 4 + 1] = sum.y;
            acc[j * 4 + 2] = sum.z; acc[j * 4 + 3] = sum.w;
          }
        }
      }
      if (writer) {
        // Native WGMMA C -> padded token-major tile. The STREAM buffer is
        // disjoint from the ring, which may already hold the next row pair.
        const int row = warp * 16 + (lane >> 2), t = lane & 3;
#pragma unroll
        for (int j = 0; j < TOK / 8; ++j) {
          const int tok = j * 8 + 2 * t;
          transpose[tok * RP + row] = acc[j * 4];
          transpose[(tok + 1) * RP + row] = acc[j * 4 + 1];
          transpose[tok * RP + row + 8] = acc[j * 4 + 2];
          transpose[(tok + 1) * RP + row + 8] = acc[j * 4 + 3];
        }
        math_sync();
        for (int idx = tid; idx < M * 16; idx += MATH) {
          const int tok = idx / 16, col = (idx % 16) * 4;
          const float4 v = *reinterpret_cast<const float4*>(transpose + tok * RP + col);
          const __nv_bfloat162 a = __floats2bfloat162_rn(v.x, v.y), b = __floats2bfloat162_rn(v.z, v.w);
          *reinterpret_cast<uint2*>(out + (size_t)tok * N + nb * 64 + col) =
              make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
        }
        math_sync();
        if constexpr (STREAM) if (tid == 0) atomicExch(counts + nb, 0);
      }
      // Prevent a faster warp overwriting acc/red/last for the next segment
      // while another is still reading its private sums or output tile.
      math_sync();
    }
  }
  __syncthreads();
  // No rank may retire while its peer can still signal the remote empty bar.
  cluster_sync();
}

static PFN_cuTensorMapEncodeTiled_v12000 encode_fn() {
  static PFN_cuTensorMapEncodeTiled_v12000 fn = nullptr;
  if (!fn) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult query;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault, &query));
    TORCH_CHECK(p, "restart: missing cuTensorMapEncodeTiled");
    fn = reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(p);
  }
  return fn;
}
static CUtensorMap map(const void* ptr, int rows, int k, int boxrows) {
  CUtensorMap result;
  cuuint64_t dims[2] = {(cuuint64_t)k, (cuuint64_t)rows};
  cuuint64_t stride[1] = {(cuuint64_t)k * 2};
  cuuint32_t box[2] = {64, (cuuint32_t)boxrows}, elem[2] = {1, 1};
  CUresult status = encode_fn()(&result, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr),
      dims, stride, box, elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(status == CUDA_SUCCESS, "restart: tensor map status ", (int)status);
  return result;
}
template <int TOK, bool STREAM> constexpr int shared_bytes() {
  return 1024 + STAGES * (WB + TOK * 128) + 64 + (STREAM ? TOK * RP * 4 : 0);
}
template <int TOK, bool STREAM>
void launch(at::Tensor w, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters, int groups, int maxparts) {
  const int M = x.size(0), N = out.size(1), K = x.size(1);
  auto kernel = gemm<TOK, STREAM>;
  static bool ready = false;
  if (!ready) {
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes<TOK, STREAM>()));
    ready = true;
  }
  CUtensorMap wm = map(w.data_ptr(), N, K, 64), xm = map(x.data_ptr(), M, K, TOK);
  cudaLaunchConfig_t config = {};
  config.gridDim = dim3(groups * 2);
  config.blockDim = dim3(THREADS);
  config.dynamicSmemBytes = shared_bytes<TOK, STREAM>();
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attributes[1] = {};
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim.x = 2; attributes[0].val.clusterDim.y = 1; attributes[0].val.clusterDim.z = 1;
  config.attrs = attributes; config.numAttrs = 1;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, wm, xm,
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), reinterpret_cast<float4*>(ws.data_ptr()),
      counters.data_ptr<int>(), M, N, K, groups, maxparts));
}
template <int TOK, bool STREAM> std::vector<int64_t> attrs() {
  cudaFuncAttributes a;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&a, gemm<TOK, STREAM>));
  return {(int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)a.sharedSizeBytes,
          (int64_t)a.maxThreadsPerBlock, (int64_t)shared_bytes<TOK, STREAM>()};
}
} // namespace dense_restart

void rd_run(at::Tensor w, at::Tensor x, at::Tensor out, at::Tensor ws, at::Tensor counters,
            int64_t native, int64_t groups, int64_t maxparts, int64_t stream) {
  const int M = x.size(0), N = out.size(1), K = x.size(1);
  TORCH_CHECK(M > 0 && M <= native && K % 64 == 0 && N % 128 == 0 && maxparts > 0 && groups > 0 && groups <= (N / 128) * (K / 64),
              "restart: invalid tile/work geometry");
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16 && ws.scalar_type() == at::kFloat && counters.scalar_type() == at::kInt,
              "restart: dtype mismatch");
  TORCH_CHECK(x.is_contiguous() && w.is_contiguous() && out.is_contiguous() &&
              w.size(0) >= N && w.size(1) == K && out.size(0) == M, "restart: shape/stride mismatch");
  TORCH_CHECK(!stream || (counters.numel() >= N / 64 && ws.numel() >= (int64_t)(N / 64) * maxparts * native * 64),
              "restart: private workspace too small");
  TORCH_CHECK(stream || groups == N / 128, "restart: output-owner grid mismatch");
  using namespace dense_restart;
  if (native == 16 && !stream) { launch<16, false>(w,x,out,ws,counters,groups,maxparts); return; }
  if (native == 32 && !stream) { launch<32, false>(w,x,out,ws,counters,groups,maxparts); return; }
  if (native == 128) {
    if (stream) launch<128, true>(w,x,out,ws,counters,groups,maxparts);
    else launch<128, false>(w,x,out,ws,counters,groups,maxparts);
    return;
  }
  if (native == 256) {
    if (stream) launch<256, true>(w,x,out,ws,counters,groups,maxparts);
    else launch<256, false>(w,x,out,ws,counters,groups,maxparts);
    return;
  }
  TORCH_CHECK(false, "restart: unsupported native tile");
}
std::vector<int64_t> rd_attrs(int64_t native, int64_t stream) {
  using namespace dense_restart;
  if (native == 16 && !stream) return attrs<16, false>();
  if (native == 32 && !stream) return attrs<32, false>();
  if (native == 128) return stream ? attrs<128,true>() : attrs<128,false>();
  if (native == 256) return stream ? attrs<256,true>() : attrs<256,false>();
  TORCH_CHECK(false, "restart: unsupported resource query");
}
