typedef unsigned long long size_t;

// Prepare-only five-microsecond producer. All source values are copied by a
// preceding graph node; first-column sign bits are published only at its end.
__global__ void d16_spin_kernel(const unsigned short* src,unsigned short* dst,int m,int k,unsigned long long ns) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if(threadIdx.x==0) {
    unsigned long long begin,now;
    asm volatile("mov.u64 %0, %%globaltimer;":"=l"(begin));
    do { asm volatile("mov.u64 %0, %%globaltimer;":"=l"(now)); } while(now-begin<ns);
  }
  __syncthreads();
  if(threadIdx.x<m) {
    const size_t i=(size_t)threadIdx.x*k;
    dst[i]=src[i]^0x8000u;
  }
}
