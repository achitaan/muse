# okc: problem=muse_prefill_layer
"""DIAGNOSTIC (not a real submission): bulk-copy smem ring geometry microbenchmark, run inside prepare().
kernel_fn returns correctly shaped zeros; the tests are expected to fail. Read the RD_* / RD lines.

Question: is the ring geometry (NS slots x STAGE bytes, 1 persistent CTA/SM, one producer issuing
cp.async.bulk + mbarrier, consumers wait full / arrive empty) itself the ~3.7 TB/s ceiling vs ld.global.nc.v4 ~4.0?
Each CTA streams a chunk of a ~650 MB uint8 buffer (L2 flushed between reps by writing 256 MB).
"""

import os
import statistics
import time

import torch

HIDDEN = 8192
BUDGET_S = 230.0  # stop starting new work after this many seconds inside prepare()
TARGET_BYTES = 650_000_000
REPS = 25

CPP = r"""
#include <torch/extension.h>
void ring_read(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t NS, int64_t STAGE, int64_t CW,
               int64_t hint, int64_t inter, int64_t nst);
void stream_read(torch::Tensor buf, torch::Tensor out, int64_t nbytes, int64_t grid, int64_t threads, int64_t unroll);
"""

CUDA = r"""
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
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
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
__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar, int hint) {
  if (hint) {
    const uint64_t pol = 0x12F0000000000000ull;  // L2 evict_first
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
        ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "l"(pol)
        : "memory");
  } else {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                 "l"(src), "r"(bytes), "r"(bar)
                 : "memory");
  }
}

// threads = (CW + 1) * 32: warps 0..CW-1 consume, warp CW is the producer (lane 0 issues).
// Barriers live at sm + NS*STAGE: full[NS] (count 1 + tx), then empty[NS] (count CW).
__global__ void ring_k(const char* __restrict__ buf, int NS, int STAGE, int CW, int hint, int inter, int nst,
                       unsigned* out) {
  extern __shared__ __align__(128) char sm[];
  uint32_t sbase = (uint32_t)__cvta_generic_to_shared(sm);
  uint32_t full = sbase + NS * STAGE;
  uint32_t empty = full + 8 * NS;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int s = 0; s < NS; s++) {
      mbar_init(full + 8 * s, 1);
      mbar_init(empty + 8 * s, CW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == CW) {
    if (lane == 0) {
      for (int i = 0; i < nst; i++) {
        int s = i % NS;
        if (i >= NS) mbar_wait(empty + 8 * s, (uint32_t)(((i / NS) - 1) & 1));
        long g = inter ? (long)i * gridDim.x + blockIdx.x : (long)blockIdx.x * nst + i;
        mbar_expect(full + 8 * s, STAGE);
        bulk_load(sbase + s * STAGE, buf + g * (long)STAGE, STAGE, full + 8 * s, hint);
      }
    }
    return;
  }
  uint32_t acc = 0;
  int nchunk = STAGE / 512;  // one uint4 per lane per 512 B
  for (int i = 0; i < nst; i++) {
    int s = i % NS;
    mbar_wait(full + 8 * s, (uint32_t)((i / NS) & 1));
    const uint4* p = reinterpret_cast<const uint4*>(sm + (size_t)s * STAGE);
#pragma unroll 4
    for (int k = warp; k < nchunk; k += CW) acc ^= x4(p[k * 32 + lane]);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + 8 * s);
  }
  for (int o = 16; o; o >>= 1) acc ^= __shfl_xor_sync(0xffffffffu, acc, o);
  if (lane == 0) atomicXor(out, acc);
}

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

// ---HOST---
void ring_read(torch::Tensor buf, torch::Tensor out, int64_t grid, int64_t NS, int64_t STAGE, int64_t CW,
               int64_t hint, int64_t inter, int64_t nst) {
  auto st = at::cuda::getCurrentCUDAStream();
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  TORCH_CHECK(NS * STAGE + 8 * 2 * NS <= optin, "smem too large");
  TORCH_CHECK((int64_t)grid * nst * STAGE <= buf.numel(), "buffer too small");
  // Always request the max opt-in smem so exactly one CTA fits per SM (even for small rings).
  cudaFuncSetAttribute(ring_k, cudaFuncAttributeMaxDynamicSharedMemorySize, optin);
  ring_k<<<(unsigned)grid, (unsigned)((CW + 1) * 32), optin, st>>>(
      reinterpret_cast<const char*>(buf.data_ptr()), (int)NS, (int)STAGE, (int)CW, (int)hint, (int)inter, (int)nst,
      reinterpret_cast<unsigned*>(out.data_ptr<int>()));
}

void stream_read(torch::Tensor buf, torch::Tensor out, int64_t nbytes, int64_t grid, int64_t threads, int64_t unroll) {
  auto st = at::cuda::getCurrentCUDAStream();
  long n = nbytes / 16;
  const uint4* p = reinterpret_cast<const uint4*>(buf.data_ptr());
  unsigned* o = reinterpret_cast<unsigned*>(out.data_ptr<int>());
  if (unroll == 4) stream_k<4><<<grid, threads, 0, st>>>(p, n, o);
  else if (unroll == 8) stream_k<8><<<grid, threads, 0, st>>>(p, n, o);
  else TORCH_CHECK(false, "bad unroll");
}
"""

T0 = time.time()


def over():
    return time.time() - T0 > BUDGET_S


def log(s):
    print(s, flush=True)


def bench(fn, flush, n=REPS):
    """Capture one launch in a CUDA graph; replay n times with an L2 flush before each; list of us."""
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(3):
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
        ts.append(s.elapsed_time(e) * 1000.0)
    del g
    return ts


def report(cfg, hint, cw, order, nbytes, ts):
    med, mn = statistics.median(ts), min(ts)
    log(f"RD cfg={cfg} hint={hint} cw={cw} order={order} med_us={med:.1f} min_us={mn:.1f} "
        f"TBps={nbytes / (med * 1e-6) / 1e12:.3f} TBps_best={nbytes / (mn * 1e-6) / 1e12:.3f} MB={nbytes / 1e6:.0f}")


def build():
    from torch.utils.cpp_extension import load_inline

    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    t = time.time()
    ext = load_inline(name="ringdiag_b", cpp_sources=CPP, cuda_sources=CUDA,
                      functions=["ring_read", "stream_read"], extra_cuda_cflags=["-O3"], verbose=False)
    log(f"RD_BUILD {time.time() - t:.0f}s")
    return ext


def run_all(dev):
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    log(f"RD_INFO {torch.cuda.get_device_name(dev)} sms={sms} torch {torch.__version__} cuda {torch.version.cuda}; "
        f"TBps=1e12 B/s of bytes moved, median us over {REPS} reps, L2 flushed (256MB write) before each")
    ext = build()
    out = torch.zeros(1, dtype=torch.int32, device=dev)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
    flush.zero_()
    buf = torch.randint(0, 256, (TARGET_BYTES + (1 << 20),), dtype=torch.uint8, device=dev)
    grid = sms

    def ring(ns, kb, hint=1, cw=16, inter=0, nst=None):
        stage = kb * 1024
        nst = nst or TARGET_BYTES // (grid * stage)
        nbytes = grid * nst * stage
        tag = f"{ns}x{kb}s{nst}"
        if over():
            log(f"RD skip cfg={tag} hint={hint} cw={cw} order={'inter' if inter else 'contig'} (time)")
            return
        try:
            fn = lambda: ext.ring_read(buf, out, grid, ns, stage, cw, hint, inter, nst)
            report(tag, hint, cw, "inter" if inter else "contig", nbytes, bench(fn, flush))
        except Exception as e:
            log(f"RD_ERR cfg={tag} hint={hint} cw={cw} inter={inter} {str(e)[-200:]!r}")
            raise

    def ldg(m, th, u, nst=None):
        nbytes = grid * (nst or TARGET_BYTES // (grid * 36864)) * 36864
        if over():
            log(f"RD skip ldg (time)")
            return
        try:
            fn = lambda: ext.stream_read(buf, out, nbytes, grid * m, th, u)
            report(f"ldg{m}x{th}u{u}", "-", "-", "contig", nbytes, bench(fn, flush))
        except Exception as e:
            log(f"RD_ERR ldg{m}x{th}u{u} {str(e)[-200:]!r}")
            raise

    # floor: tiny launch
    try:
        tiny = lambda: ext.ring_read(buf, out, grid, 3, 36864, 16, 1, 0, 1)
        ts = bench(tiny, flush)
        log(f"RD_FLOOR 3x36 ring, 1 stage/CTA ({grid * 36864 / 1e6:.1f} MB): med_us={statistics.median(ts):.1f} "
            f"min_us={min(ts):.1f}")
    except Exception as e:
        log(f"RD_ERR floor {str(e)[-200:]!r}")

    jobs = []
    # MoE skinny volumes: stages of 36 KB per CTA = 12 (bs1 TINY, 57 MB), 36 (bs8 TINY2, 170 MB), 78 (bs16 MID, 368 MB)
    for n in (12, 36, 78):
        jobs += [lambda n=n: ring(3, 36, hint=0, cw=8, nst=n), lambda n=n: ring(3, 36, hint=1, cw=8, nst=n),
                 lambda n=n: ring(3, 36, hint=0, cw=16, nst=n), lambda n=n: ring(6, 36, hint=0, cw=8, nst=n),
                 lambda n=n: ldg(2, 512, 8, nst=n), lambda n=n: ldg(1, 1024, 8, nst=n)]
    for j in jobs:
        try:
            j()
        except Exception:
            break  # a CUDA error poisons the context; stop
    del buf, flush
    torch.cuda.empty_cache()


def prepare(weights):
    global T0
    T0 = time.time()
    try:
        run_all(torch.device("cuda"))
    except Exception as e:
        log(f"RD_ERR top {str(e)[-300:]!r}")
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    log(f"RD_DONE prepare {time.time() - T0:.0f}s")
    return weights


def kernel_fn(data, weights):
    """Pass-through: correctly shaped/dtyped zeros (tests fail; only the RD log matters)."""
    t = data["config"]["tokens"]
    dev = data["stream"].device
    hidden = torch.zeros((t, HIDDEN), dtype=torch.bfloat16, device=dev)
    stream = torch.zeros((t, HIDDEN), dtype=torch.float32, device=dev)
    ids = torch.zeros((t, 8), dtype=torch.int32, device=dev)
    return hidden, stream, ids
