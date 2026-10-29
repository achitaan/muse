#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t shared_addr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint32_t remote_addr(uint32_t p, uint32_t rank) {
  uint32_t r; asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(p), "r"(rank)); return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait(uint32_t bar) {
  asm volatile("{ .reg .pred done; AGAIN: mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 done, [%0], 0; @!done bra AGAIN; }" :: "r"(bar) : "memory");
}
__global__ void __cluster_dims__(2, 1, 1) cluster_x_probe(const unsigned char* src, uint32_t* out) {
  extern __shared__ __align__(128) unsigned char raw[];
  uint32_t full=shared_addr(raw), empty=shared_addr(raw+8), data=shared_addr(raw+128);
  uint32_t rank; asm volatile("mov.u32 %0, %cluster_ctarank;" : "=r"(rank));
  if (threadIdx.x==0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1; mbarrier.init.shared::cta.b64 [%1], 8; fence.mbarrier_init.release.cluster;" :: "r"(full), "r"(empty) : "memory");
  }
  __syncthreads(); cluster_sync();
  if (rank==0 && threadIdx.x==0) {
    uint32_t peer_full=remote_addr(full,1);
    asm volatile("mbarrier.arrive.expect_tx.release.cluster.shared::cta.b64 _, [%0], 4096; mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [%1], 4096;" :: "r"(full), "r"(peer_full) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], 4096, [%2], %3;" :: "r"(data), "l"(src), "r"(full), "h"((uint16_t)3) : "memory");
  }
  cluster_wait(full);
  if (threadIdx.x==0) out[rank]=*reinterpret_cast<const uint32_t*>(raw+128);
  __syncthreads();
  if ((threadIdx.x&31)==0) {
    uint32_t lead_empty=remote_addr(empty,0);
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" :: "r"(lead_empty) : "memory");
  }
  if (rank==0 && threadIdx.x==0) cluster_wait(empty);
  __syncthreads(); cluster_sync();
}
