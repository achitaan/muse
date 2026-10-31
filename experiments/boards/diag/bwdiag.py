# okc: problem=muse_prefill_layer
"""DIAGNOSTIC (not a real submission): H200 bandwidth microbenchmarks run inside prepare().
kernel_fn returns correctly shaped zeros; the tests are expected to fail. Read the BW_* log lines.

A: contiguous streaming read ceiling (ld.global.nc.v4), cold (L2 flushed) and warm.
B: TMA bulk (cp.async.bulk + mbarrier) smem-ring streaming read.
C: random 128-B row gather of K and V (attention KV pattern), register-prefetch and cp.async variants.
D: same as C with sorted indices (ceiling reference).
"""

import os
import statistics
import time

import torch

HIDDEN = 8192
BUDGET_S = 230.0  # skip remaining configs after this many seconds inside prepare()

CPP_A = r"""
#include <torch/extension.h>
void stream_read(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t unroll);
void gather_reg(torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int64_t grid, int64_t u);
void gather_cpa(torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int64_t grid, int64_t p);
"""

CUDA_A = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t x4(uint4 a) { return a.x ^ a.y ^ a.z ^ a.w; }

template <int U>
__global__ void stream_k(const uint4* __restrict__ p, long n, unsigned* out) {
  long per = (n + gridDim.x - 1) / gridDim.x;
  long s = per * blockIdx.x;
  long e = s + per < n ? s + per : n;
  uint32_t acc = 0;
  for (long base = s; base < e; base += (long)blockDim.x * U) {
    uint4 v[U];
#pragma unroll
    for (int j = 0; j < U; j++) {
      long i = base + (long)j * blockDim.x + threadIdx.x;
      v[j] = i < e ? __ldg(p + i) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < U; j++) acc ^= x4(v[j]);
  }
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) atomicXor(out, acc);
}

template <int P>
__device__ __forceinline__ void ld_idx(const int* idx, int t, int w, int W, int lane, int groups, int* r) {
#pragma unroll
  for (int j = 0; j < P; j++) {
    long g = ((long)t * W + w) * P + j;
    r[j] = g < groups ? __ldg(idx + g * 4 + (lane >> 3)) : -1;
  }
}

template <int U>
__global__ void __launch_bounds__(256) gather_reg_k(const uint4* __restrict__ K, const uint4* __restrict__ V,
                                                    const int* __restrict__ idx, int groups, unsigned* out) {
  int lane = threadIdx.x & 31;
  int w = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  int W = (gridDim.x * blockDim.x) >> 5;
  uint32_t acc = 0;
  int cur[U];
  ld_idx<U>(idx, 0, w, W, lane, groups, cur);
  for (int t = 0;; t++) {
    long g0 = ((long)t * W + w) * U;
    if (g0 >= groups) break;
    uint4 k[U], v[U];
#pragma unroll
    for (int j = 0; j < U; j++) {
      k[j] = make_uint4(0, 0, 0, 0);
      v[j] = make_uint4(0, 0, 0, 0);
      if (cur[j] >= 0) {
        size_t off = (size_t)cur[j] * 8 + (lane & 7);
        k[j] = __ldg(K + off);
        v[j] = __ldg(V + off);
      }
    }
    int nxt[U];
    ld_idx<U>(idx, t + 1, w, W, lane, groups, nxt);
#pragma unroll
    for (int j = 0; j < U; j++) {
      acc ^= x4(k[j]) ^ x4(v[j]);
      cur[j] = nxt[j];
    }
  }
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) atomicXor(out, acc);
}

__device__ __forceinline__ void cpa16(void* sm, const void* g) {
  unsigned a = (unsigned)__cvta_generic_to_shared(sm);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(a), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

#define CPA_S 4
template <int P>
__global__ void __launch_bounds__(256) gather_cpa_k(const uint4* __restrict__ K, const uint4* __restrict__ V,
                                                    const int* __restrict__ idx, int groups, unsigned* out) {
  extern __shared__ uint4 sm4[];
  int lane = threadIdx.x & 31;
  int tid = threadIdx.x;
  int w = (blockIdx.x * blockDim.x + tid) >> 5;
  int W = (gridDim.x * blockDim.x) >> 5;
  int T = (int)(((long)groups + (long)W * P - 1) / ((long)W * P));
  uint32_t acc = 0;
  int nx[P];
  ld_idx<P>(idx, 0, w, W, lane, groups, nx);
  int ti = 0;
#define SLOT(stage, j, kv) (sm4 + (((((stage) * P + (j)) * 2) + (kv)) * 256 + tid))
#define ISSUE()                                                              \
  {                                                                          \
    int st_ = ti % CPA_S;                                                    \
    _Pragma("unroll") for (int j = 0; j < P; j++) {                          \
      if (nx[j] >= 0) {                                                      \
        size_t off = (size_t)nx[j] * 8 + (lane & 7);                         \
        cpa16(SLOT(st_, j, 0), K + off);                                     \
        cpa16(SLOT(st_, j, 1), V + off);                                     \
      }                                                                      \
    }                                                                        \
    ld_idx<P>(idx, ti + 1, w, W, lane, groups, nx);                          \
  }
#pragma unroll
  for (int s = 0; s < CPA_S - 1; s++) {
    if (ti < T) ISSUE();
    cpa_commit();
    ti++;
  }
  for (int i = 0; i < T; i++) {
    if (ti < T) ISSUE();
    cpa_commit();
    ti++;
    cpa_wait<CPA_S - 1>();
    int st = i % CPA_S;
#pragma unroll
    for (int j = 0; j < P; j++) {
      long g = ((long)i * W + w) * P + j;
      if (g < groups) acc ^= x4(*SLOT(st, j, 0)) ^ x4(*SLOT(st, j, 1));
    }
  }
  cpa_wait<0>();
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) atomicXor(out, acc);
}

void stream_read(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t unroll) {
  auto st = at::cuda::getCurrentCUDAStream();
  long n = buf.numel() / 16;
  const uint4* p = reinterpret_cast<const uint4*>(buf.data_ptr());
  unsigned* o = reinterpret_cast<unsigned*>(out.data_ptr<int>());
  if (unroll == 4) stream_k<4><<<grid, threads, 0, st>>>(p, n, o);
  else if (unroll == 8) stream_k<8><<<grid, threads, 0, st>>>(p, n, o);
  else TORCH_CHECK(false, "bad unroll");
}

void gather_reg(torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int64_t grid, int64_t u) {
  auto st = at::cuda::getCurrentCUDAStream();
  int groups = (int)(idx.numel() / 4);
  const uint4* k = reinterpret_cast<const uint4*>(K.data_ptr());
  const uint4* v = reinterpret_cast<const uint4*>(V.data_ptr());
  const int* ix = idx.data_ptr<int>();
  unsigned* o = reinterpret_cast<unsigned*>(out.data_ptr<int>());
  if (u == 2) gather_reg_k<2><<<grid, 256, 0, st>>>(k, v, ix, groups, o);
  else if (u == 4) gather_reg_k<4><<<grid, 256, 0, st>>>(k, v, ix, groups, o);
  else if (u == 8) gather_reg_k<8><<<grid, 256, 0, st>>>(k, v, ix, groups, o);
  else if (u == 16) gather_reg_k<16><<<grid, 256, 0, st>>>(k, v, ix, groups, o);
  else TORCH_CHECK(false, "bad u");
}

void gather_cpa(torch::Tensor K, torch::Tensor V, torch::Tensor idx, torch::Tensor out, int64_t grid, int64_t p) {
  auto st = at::cuda::getCurrentCUDAStream();
  int groups = (int)(idx.numel() / 4);
  const uint4* k = reinterpret_cast<const uint4*>(K.data_ptr());
  const uint4* v = reinterpret_cast<const uint4*>(V.data_ptr());
  const int* ix = idx.data_ptr<int>();
  unsigned* o = reinterpret_cast<unsigned*>(out.data_ptr<int>());
  int smem = CPA_S * (int)p * 2 * 256 * 16;
  if (p == 1) {
    cudaFuncSetAttribute(gather_cpa_k<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    gather_cpa_k<1><<<grid, 256, smem, st>>>(k, v, ix, groups, o);
  } else if (p == 2) {
    cudaFuncSetAttribute(gather_cpa_k<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    gather_cpa_k<2><<<grid, 256, smem, st>>>(k, v, ix, groups, o);
  } else TORCH_CHECK(false, "bad p");
}
"""

CPP_B = r"""
#include <torch/extension.h>
void tma_read(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t chunk, int64_t stages);
"""

CUDA_B = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

__device__ __forceinline__ uint32_t x4(uint4 a) { return a.x ^ a.y ^ a.z ^ a.w; }

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t cnt) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(cnt) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t ok;
  do {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
        : "=r"(ok)
        : "r"(bar), "r"(parity)
        : "memory");
  } while (!ok);
}
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

__global__ void tma_k(const char* __restrict__ buf, long nch, int chunk, int S, unsigned* out) {
  extern __shared__ __align__(128) char sm[];
  uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
  uint32_t bar0 = sbase + S * chunk;  // S full barriers, 8 B each
  int tid = threadIdx.x;
  long c0 = nch * (long)blockIdx.x / gridDim.x;
  long c1 = nch * (long)(blockIdx.x + 1) / gridDim.x;
  long n = c1 - c0;
  if (tid == 0) {
    for (int s = 0; s < S; s++) mbar_init(bar0 + 8 * s, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    for (int s = 0; s < S && s < n; s++) {
      mbar_expect(bar0 + 8 * s, chunk);
      bulk_load(sbase + s * chunk, buf + (c0 + s) * (long)chunk, chunk, bar0 + 8 * s);
    }
  }
  uint32_t acc = 0;
  int n16 = chunk / 16;
  for (long i = 0; i < n; i++) {
    int s = (int)(i % S);
    mbar_wait(bar0 + 8 * s, (uint32_t)((i / S) & 1));
    const uint4* p = reinterpret_cast<const uint4*>(sm + (size_t)s * chunk);
    for (int k = tid; k < n16; k += blockDim.x) acc ^= x4(p[k]);
    __syncthreads();
    if (tid == 0 && i + S < n) {
      mbar_expect(bar0 + 8 * s, chunk);
      bulk_load(sbase + s * chunk, buf + (c0 + i + S) * (long)chunk, chunk, bar0 + 8 * s);
    }
  }
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  if ((tid & 31) == 0) atomicXor(out, acc);
}

void tma_read(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t threads, int64_t chunk, int64_t stages) {
  auto st = at::cuda::getCurrentCUDAStream();
  long nch = buf.numel() / chunk;
  int smem = (int)(stages * chunk + 64);
  cudaFuncSetAttribute(tma_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  tma_k<<<grid, threads, smem, st>>>(reinterpret_cast<const char*>(buf.data_ptr()), nch, (int)chunk, (int)stages,
                                     reinterpret_cast<unsigned*>(out.data_ptr<int>()));
}
"""

T0 = time.time()


def over():
    return time.time() - T0 > BUDGET_S


def log(s):
    print(s, flush=True)


def bench(fn, flush=None, n=20):
    """Capture one launch in a CUDA graph, replay n times, median ms (cuda events)."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        if flush is not None:
            flush.zero_()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e))
    del g
    return statistics.median(ts)


def gbps(nbytes, ms):
    return nbytes / (ms * 1e-3) / 1e9


def best_worst(d):
    b = max(d, key=d.get)
    w = min(d, key=d.get)
    return f"best={d[b]:.0f}@{b} worst={d[w]:.0f}@{w}"


def tab(d):
    return " ".join(f"{v:.0f}" for v in d.values())


def build(name, cpp, cuda, fns):
    from torch.utils.cpp_extension import load_inline

    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    t = time.time()
    ext = load_inline(name=name, cpp_sources=cpp, cuda_sources=cuda, functions=fns,
                      extra_cuda_cflags=["-O3"], verbose=False)
    log(f"BW_BUILD {name} {time.time() - t:.0f}s")
    return ext


def section_a(ext, dev, out, flush):
    cfgs = [(m, th, u) for m in (1, 2, 4) for th in (256, 512, 1024) for u in (4, 8)]
    log("BW_A cfg order (gridMult x threads u unroll): " + " ".join(f"{m}x{th}u{u}" for m, th, u in cfgs))
    tiny = torch.zeros(16384, dtype=torch.uint8, device=dev)
    ms = bench(lambda: ext.stream_read(tiny, out, 132, 256, 4), None)
    log(f"BW_FLOOR single 132x256 launch on 16KB in graph replay: {ms * 1000:.1f} us")
    for mb in (6, 12, 28, 50, 128):
        if over():
            log(f"BW_A skip {mb}MB (time)")
            continue
        nbytes = mb << 20
        buf = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=dev)
        cold, warm = {}, {}
        for m, th, u in cfgs:
            if over():
                break
            tag = f"{m}x{th}u{u}"
            fn = lambda: ext.stream_read(buf, out, 132 * m, th, u)
            cold[tag] = gbps(nbytes, bench(fn, flush))
            warm[tag] = gbps(nbytes, bench(fn, None))
        if cold:
            log(f"BW_A {mb}MB cold {best_worst(cold)} | warm {best_worst(warm)}")
            log(f"BW_A_tab {mb}MB cold: {tab(cold)}")
            log(f"BW_A_tab {mb}MB warm: {tab(warm)}")
        del buf


def section_b(ext, dev, out, flush):
    cfgs = [(c, s, th) for c in (16384, 32768) for s in (2, 3, 4) for th in (256, 512)]
    log("BW_B cfg order (chunkKB sStages tThreads): " + " ".join(f"{c // 1024}k s{s} t{th}" for c, s, th in cfgs))
    for mb in (28, 50):
        if over():
            log(f"BW_B skip {mb}MB (time)")
            continue
        nbytes = mb << 20
        buf = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device=dev)
        cold, warm = {}, {}
        for c, s, th in cfgs:
            if over():
                break
            tag = f"{c // 1024}k.s{s}.t{th}"
            fn = lambda: ext.tma_read(buf, out, 132, th, c, s)
            cold[tag] = gbps(nbytes, bench(fn, flush))
            warm[tag] = gbps(nbytes, bench(fn, None))
        if cold:
            log(f"BW_B {mb}MB cold {best_worst(cold)} | warm {best_worst(warm)}")
            log(f"BW_B_tab {mb}MB cold: {tab(cold)}")
            log(f"BW_B_tab {mb}MB warm: {tab(warm)}")
        del buf


def section_gather(ext, dev, out, label, K, V, idx):
    useful = 2 * 128 * idx.numel()
    reg, cpa = {}, {}
    for u in (2, 4, 8, 16):
        for m in (2, 4, 8):
            if over():
                break
            fn = lambda: ext.gather_reg(K, V, idx, out, 132 * m, u)
            reg[f"u{u}g{m}"] = gbps(useful, bench(fn, None))
    for p in (1, 2):
        for m in (2, 4, 8):
            if over():
                break
            fn = lambda: ext.gather_cpa(K, V, idx, out, 132 * m, p)
            cpa[f"p{p}g{m}"] = gbps(useful, bench(fn, None))
    if reg:
        log(f"BW_{label} reg {best_worst(reg)}")
        log(f"BW_{label}_tab reg (u2 g2 g4 g8, u4.., u8.., u16..): {tab(reg)}")
    if cpa:
        log(f"BW_{label} cpasync(4 stages) {best_worst(cpa)}")
        log(f"BW_{label}_tab cpa (p1 g2 g4 g8, p2 ..): {tab(cpa)}")


def run_all(dev):
    out = torch.zeros(1, dtype=torch.int32, device=dev)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    flush.zero_()
    log(f"BW_INFO {torch.cuda.get_device_name(dev)} torch {torch.__version__} cuda {torch.version.cuda}; MB=2^20 B, GB/s=1e9")
    ext = None
    try:
        ext = build("bwdiag_a", CPP_A, CUDA_A, ["stream_read", "gather_reg", "gather_cpa"])
    except Exception as e:
        log(f"BW_ERR build_a {str(e)[-600:]!r}")
    if ext is not None:
        try:
            section_a(ext, dev, out, flush)
        except Exception as e:
            log(f"BW_ERR A {str(e)[-300:]!r}")
    if not over():
        try:
            extb = build("bwdiag_b", CPP_B, CUDA_B, ["tma_read"])
            section_b(extb, dev, out, flush)
        except Exception as e:
            log(f"BW_ERR B {str(e)[-600:]!r}")
    del flush
    torch.cuda.empty_cache()
    if ext is None:
        return
    try:
        R = 16 * 80000
        L = R
        K = torch.randint(0, 256, (R * 128,), dtype=torch.uint8, device=dev)
        V = torch.randint(0, 256, (R * 128,), dtype=torch.uint8, device=dev)
        rnd = torch.randperm(R, device=dev)[:L].to(torch.int32).contiguous()
        srt = torch.arange(L, device=dev, dtype=torch.int32)
        log(f"BW_CD rows={R} (164MB per buffer), L={L}, useful bytes = 2x128xL = {2 * 128 * L / 1e6:.0f} MB")
        try:
            section_gather(ext, dev, out, "C", K, V, rnd)
        except Exception as e:
            log(f"BW_ERR C {str(e)[-300:]!r}")
        try:
            section_gather(ext, dev, out, "D", K, V, srt)
        except Exception as e:
            log(f"BW_ERR D {str(e)[-300:]!r}")
        del K, V, rnd, srt
    except Exception as e:
        log(f"BW_ERR CD_setup {str(e)[-300:]!r}")
    torch.cuda.empty_cache()


def prepare(weights):
    global T0
    T0 = time.time()
    try:
        run_all(torch.device("cuda"))
    except Exception as e:
        log(f"BW_ERR top {str(e)[-300:]!r}")
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    log(f"BW_DONE prepare {time.time() - T0:.0f}s")
    return weights


def kernel_fn(data, weights):
    """Pass-through: correctly shaped/dtyped zeros (tests fail; only the BW_* log matters)."""
    t = data["config"]["tokens"]
    dev = data["stream"].device
    hidden = torch.zeros((t, HIDDEN), dtype=torch.bfloat16, device=dev)
    stream = torch.zeros((t, HIDDEN), dtype=torch.float32, device=dev)
    ids = torch.zeros((t, 8), dtype=torch.int32, device=dev)
    return hidden, stream, ids
