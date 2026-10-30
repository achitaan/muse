
#include <cuda_runtime.h>
#include <stdint.h>
struct L2pRegions { unsigned long long p[6]; unsigned long long b[6]; int n; };
__global__ void l2p_prefetch(L2pRegions r) {
  const unsigned long long CH = 32768ull;
  const unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long G = (unsigned long long)gridDim.x * blockDim.x;
  for (int i = 0; i < r.n; ++i) {
    const unsigned long long nch = (r.b[i] + CH - 1) / CH;
    for (unsigned long long c = g; c < nch; c += G) {
      const unsigned long long off = c * CH;
      unsigned long long sz = r.b[i] - off < CH ? r.b[i] - off : CH;
      sz &= ~15ull;
      if (sz) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(r.p[i] + off), "r"((unsigned)sz) : "memory");
    }
  }
}
__global__ void l2p_spin(unsigned long long ns) {
  if (threadIdx.x) return;
  unsigned long long t0, t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
int l2p_launch(int64_t p0, int64_t b0, int64_t p1, int64_t b1, int64_t p2, int64_t b2, int64_t p3, int64_t b3,
               int64_t p4, int64_t b4, int64_t p5, int64_t b5, int n, int blocks, int64_t stream) {
  L2pRegions r{};
  const int64_t ps[6] = {p0, p1, p2, p3, p4, p5}, bs[6] = {b0, b1, b2, b3, b4, b5};
  if (n < 0 || n > 6 || blocks < 1 || blocks > 132) return -1;
  for (int i = 0; i < 6; ++i) {
    if (i < n && ((ps[i] & 15) || bs[i] < 0)) return -2;
    r.p[i] = (unsigned long long)ps[i];
    r.b[i] = (unsigned long long)bs[i];
  }
  r.n = n;
  l2p_prefetch<<<blocks, 128, 0, (cudaStream_t)stream>>>(r);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}
int l2p_spin_launch(int64_t ns, int64_t stream) {
  l2p_spin<<<1, 32, 0, (cudaStream_t)stream>>>((unsigned long long)ns);
  return cudaGetLastError() == cudaSuccess ? 0 : -8;
}
