#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
namespace fs {
constexpr int HD=4096, STAGE=36864, NS=4, NT=256, NTL=288;
constexpr int YSIZE=32*8*HD;
constexpr float TWO62=4.611686018427388e18f;
struct Params {
  const unsigned char* w;
  const float *ga,*gb,*tw;
  const __nv_bfloat16 *x,*nw;
  const int* ids;
  int *ctr;
  float *y;
  __nv_bfloat16* out;
  int M,lo,zero;
};
struct Shared {
  uint64_t full[NS],empty[NS];
  int experts[32],counts[32],tokens[32][8],ne,a,b,gen;
  float red[8][32][4];
  __nv_bfloat16 inter[8][16];
};
constexpr int HEADER=(sizeof(Shared)+127)&~127;
constexpr int SMEM=HEADER+NS*STAGE+4*HD*4;
static_assert(SMEM<=232448,"FS shared budget");
extern __shared__ __align__(128) unsigned char smem[];
// FS_HELPERS (verbatim arithmetic helpers from moe_b13)
__device__ __forceinline__ uint32_t su(const void* p) {return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(su(b)):"memory");
}
__device__ __forceinline__ void wait(uint64_t* b,int phase) {
  asm volatile("{ .reg .pred p; L%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra L%=; }"::"r"(su(b)),"r"(phase):"memory");
}
__device__ __forceinline__ int acquire(const int* p) {
  int v;asm volatile("ld.acquire.gpu.global.b32 %0,[%1];":"=r"(v):"l"(p):"memory");return v;
}
#define CSYNC() asm volatile("bar.sync 1, 256;":::"memory")
template<bool HIGH>
__global__ __launch_bounds__(NTL,1) void kernel(const __grid_constant__ Params p) {
  Shared& S=*reinterpret_cast<Shared*>(smem);
  unsigned char* ring=smem+HEADER;
  float* acc=reinterpret_cast<float*>(ring+NS*STAGE);
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,t=lane&3;
  asm volatile("griddepcontrol.wait;":::"memory");
  if(warp==0) {
    // One lane per expert: scan64 IDs in parallel, not2048 serial loads.
    int count=0;
    for(int tok=0;tok<p.M;++tok) {
      bool hit=false;
      for(int k=0;k<8;++k) hit|=p.ids[tok*8+k]==p.lo+lane;
      if(hit) S.tokens[lane][count++]=tok;
    }
    S.counts[lane]=count;
    unsigned mask=__ballot_sync(0xffffffffu,count!=0);
    if(count) S.experts[__popc(mask&((1u<<lane)-1))]=lane;
    if(lane==0) {S.ne=__popc(mask);S.gen=p.ctr[0];}
  }
  if(tid==0) {
    S.a=S.ne*256*(int)blockIdx.x/(int)gridDim.x;
    S.b=S.ne*256*((int)blockIdx.x+1)/(int)gridDim.x;
    for(int k=0;k<NS;++k) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;"::"r"(su(S.full+k)):"memory");
      asm volatile("mbarrier.init.shared::cta.b64 [%0],8;"::"r"(su(S.empty+k)):"memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  }
  __syncthreads();
  if(warp==8) {
    if(lane==0) {
      int seq=0;
      for(int task=S.a;task<S.b;++task) {
        const int e=S.experts[task>>8],it=task&255;
        for(int k=0;k<3;++k,++seq) {
          int sl=seq%NS;
          if(seq>=NS) wait(S.empty+sl,((seq/NS)&1)^1);
          const unsigned char* src=p.w+((size_t)e*256+it)*3*STAGE+k*STAGE;
          asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;"::"r"(su(S.full+sl)),"r"(STAGE):"memory");
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"(su(ring+sl*STAGE)),"l"(src),"r"(STAGE),"r"(su(S.full+sl)):"memory");
        }
      }
    }
    return;
  }
  const int gen=S.gen&1;
  float* y=p.y+(size_t)gen*YSIZE;
  // Inactive copy is ready for the next invocation at this invocation's grid
  // barrier. Never clear the copy receiving concurrent atomic reductions.
  for(int i=blockIdx.x*NT+tid;i<YSIZE;i+=gridDim.x*NT) p.y[(size_t)(gen^1)*YSIZE+i]=0.f;
  if(blockIdx.x==0 && tid==0) p.ctr[1+(gen^1)]=0;
  int task=S.a,seq=0;
  while(task<S.b) {
    const int e=S.experts[task>>8],cnt=S.counts[e];
    const int end=min(S.b,((task>>8)+1)*256);
    for(int i=tid;i<4*HD;i+=NT) acc[i]=0.f;
    float upper[HIGH?32:1][2];
    if constexpr(HIGH) {
#pragma unroll
      for(int z=0;z<32;++z) upper[z][0]=upper[z][1]=0.f;
    }
    uint4 x[16];
    if(g<cnt) {
      const uint4* src=reinterpret_cast<const uint4*>(p.x+(size_t)S.tokens[e][g]*HD+warp*512+16*t);
#pragma unroll
      for(int c=0;c<8;++c) {x[2*c]=__ldg(src+c*8);x[2*c+1]=__ldg(src+c*8+1);}
    } else {
#pragma unroll
      for(int c=0;c<16;++c)x[c]=make_uint4(0,0,0,0);
    }
    CSYNC();
    for(;task<end;++task) {
      // Two original full-K A units. Each produces eight inter indices.
      for(int half=0;half<2;++half,++seq) {
        int sl=seq%NS;wait(S.full+sl,(seq/NS)&1);
        const unsigned char* slot=ring+sl*STAGE;
        const uint4 sc=*reinterpret_cast<const uint4*>(slot+32768+warp*512+lane*16);
        float d[4]={0,0,0,0},d2[4]={0,0,0,0},d3[4]={0,0,0,0},d4[4]={0,0,0,0};
        const uint32_t k16=16u+p.zero;
#pragma unroll
        for(int c=0;c<8;++c) {
          const uint4 w=*reinterpret_cast<const uint4*>(slot+(warp*8+c)*512+lane*16);
          const uint32_t sw=c<4?(c<2?sc.x:sc.y):(c<6?sc.z:sc.w);
          const uint32_t sg=__byte_perm(sw,0u,0x4040u|((c&1)*2*0x0101u))*k16+0x5C005C00u;
          const uint32_t sg8=__byte_perm(sw,0u,0x4040u|(((c&1)*2+1)*0x0101u))*k16+0x5C005C00u;
          const uint32_t ww[4]={w.x,w.y,w.z,w.w};
          const uint32_t xx[8]={x[2*c].x,x[2*c].y,x[2*c].z,x[2*c].w,x[2*c+1].x,x[2*c+1].y,x[2*c+1].z,x[2*c+1].w};
#pragma unroll
          for(int k=0;k<4;++k) {
            uint32_t r0,r1,r2,r3;dq4(ww[k],r0,r1,r2,r3);
            mma_bf16(k==0?d:k==1?d2:k==2?d3:d4,bmul2(r0,sg),bmul2(r1,sg8),bmul2(r2,sg),bmul2(r3,sg8),xx[2*k],xx[2*k+1]);
          }
        }
#pragma unroll
        for(int k=0;k<4;++k)d[k]+=d2[k]+d3[k]+d4[k];
        __syncwarp();
        if(lane==0)arrive(S.empty+sl);
        *reinterpret_cast<float4*>(S.red[warp][lane])=make_float4(d[0],d[1],d[2],d[3]);
        CSYNC();
        if(warp==0) {
          float4 s=make_float4(0,0,0,0);
#pragma unroll
          for(int w=0;w<8;++w) {
            float4 z=*reinterpret_cast<float4*>(S.red[w][lane]);
            s.x+=z.x;s.y+=z.y;s.z+=z.z;s.w+=z.w;
          }
#pragma unroll
          for(int h=0;h<2;++h) {
            float2 z=achita_marlin_out2(h?s.y:s.x,h?s.w:s.z,p.ga[e]*TWO62);
            S.inter[2*t+h][half*8+g]=__float2bfloat16_rn(2*t+h<cnt?__fdividef(z.x,1.f+__expf(-z.x))*z.y:0.f);
          }
        }
        CSYNC();
      }
      int sl=seq%NS;wait(S.full+sl,(seq/NS)&1);
      const unsigned char* slot=ring+sl*STAGE;
      const uint32_t b0=*reinterpret_cast<uint32_t*>(&S.inter[g][2*t]);
      const uint32_t b1=*reinterpret_cast<uint32_t*>(&S.inter[g][8+2*t]);
#pragma unroll
      for(int z=0;z<32;++z) {
        const int rb=warp+8*z,row=rb*16+g;
        uint32_t r0,r1,r2,r3;dq4(*reinterpret_cast<const uint32_t*>(slot+rb*128+lane*4),r0,r1,r2,r3);
        uint32_t sg=(uint32_t(slot[32768+row])*16+0x5C00u)*0x10001u;
        uint32_t sg8=(uint32_t(slot[32768+row+8])*16+0x5C00u)*0x10001u;
        float d[4]={0,0,0,0};
        mma_bf16(d,bmul2(r0,sg),bmul2(r1,sg8),bmul2(r2,sg),bmul2(r3,sg8),b0,b1);
        if(t<2) {
          acc[(2*t)*HD+row]+=d[0];acc[(2*t+1)*HD+row]+=d[1];
          acc[(2*t)*HD+row+8]+=d[2];acc[(2*t+1)*HD+row+8]+=d[3];
        }
        if constexpr(HIGH) {
          // Rebalance upper-token results across ALL lanes: each lane owns
          // one of tokens4..7, two rows, 32 row blocks =>64 FP32 registers.
          const int from=(lane&~3)+2+(t>>1);
          float a=__shfl_sync(0xffffffffu,d[0],from),b=__shfl_sync(0xffffffffu,d[1],from);
          float c=__shfl_sync(0xffffffffu,d[2],from),dhi=__shfl_sync(0xffffffffu,d[3],from);
          upper[z][0]+=(t&1)?b:a;
          upper[z][1]+=(t&1)?dhi:c;
        }
      }
      __syncwarp();
      if(lane==0)arrive(S.empty+sl);
      ++seq;
    }
    CSYNC();
    for(int i=tid*4;i<min(cnt,4)*HD;i+=NT*4) {
      float4 z=*reinterpret_cast<float4*>(acc+i);
      float* dst=y+((size_t)e*8+S.tokens[e][i/HD])*HD+i%HD;
      asm volatile("red.global.add.v4.f32 [%0],{%1,%2,%3,%4};"::"l"(dst),"f"(z.x),"f"(z.y),"f"(z.z),"f"(z.w):"memory");
    }
    if constexpr(HIGH) {
      if(4+t<cnt) {
#pragma unroll
        for(int z=0;z<32;++z) {
          const int row=(warp+8*z)*16+g;
          float* dst=y+((size_t)e*8+S.tokens[e][4+t])*HD+row;
          // Shared accumulator uses segment token indices; its flush above
          // maps those to actual tokens. Register ownership does that here.
          asm volatile("red.global.add.f32 [%0],%1;"::"l"(dst),"f"(upper[z][0]):"memory");
          asm volatile("red.global.add.f32 [%0],%1;"::"l"(dst+8),"f"(upper[z][1]):"memory");
        }
      }
    }
    CSYNC();
  }
  asm volatile("griddepcontrol.launch_dependents;":::"memory");
  __threadfence();CSYNC();
  if(tid==0) {
    int old=atomicAdd(p.ctr+1+gen,1);
    if(old==(int)gridDim.x-1)atomicAdd(p.ctr,1);
    while(acquire(p.ctr+1+gen)<(int)gridDim.x)__nanosleep(64);
  }
  CSYNC();
  if((int)blockIdx.x<p.M) {
    const int tok=blockIdx.x;
    float out[16];
#pragma unroll
    for(int k=0;k<16;++k)out[k]=0.f;
    for(int r=0;r<8;++r) {
      const int e=p.ids[tok*8+r]-p.lo;
      if(e>=0 && e<32) {
        float v[16],ss=0.f;
#pragma unroll
        for(int k=0;k<8;++k) {
          int c=tid+2*k*NT;
          float2 z=achita_marlin_out2(__ldcg(y+((size_t)e*8+tok)*HD+c),__ldcg(y+((size_t)e*8+tok)*HD+c+NT),p.gb[e]*TWO62);
          v[2*k]=z.x*__bfloat162float(p.nw[c]);v[2*k+1]=z.y*__bfloat162float(p.nw[c+NT]);
          ss+=v[2*k]*v[2*k]+v[2*k+1]*v[2*k+1];
        }
        for(int d=16;d;d>>=1)ss+=__shfl_down_sync(0xffffffffu,ss,d);
        if(lane==0)S.red[warp][0][0]=ss;
        CSYNC();
        if(tid==0){float z=0;for(int w=0;w<8;++w)z+=S.red[w][0][0];S.red[0][0][1]=p.tw[tok*8+r]*rsqrtf(z*(1.f/4096.f)+1e-8f);}
        CSYNC();float scale=S.red[0][0][1];
#pragma unroll
        for(int k=0;k<16;++k)out[k]+=scale*v[k];
        CSYNC();
      }
    }
#pragma unroll
    for(int k=0;k<16;++k)p.out[tok*HD+tid+k*NT]=__float2bfloat16_rn(out[k]);
  }
}
} // namespace fs
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
static bool fs_pdl=false;
void fs_set_pdl(bool x){fs_pdl=x;}
void okc_set_pdl(int64_t x){fs_pdl=x!=0;}
std::vector<int64_t> fs_info(){
  cudaFuncAttributes a{},b{};
  TORCH_CHECK(cudaFuncGetAttributes(&a,fs::kernel<false>)==cudaSuccess,"FS attributes low");
  TORCH_CHECK(cudaFuncGetAttributes(&b,fs::kernel<true>)==cudaSuccess,"FS attributes high");
  return {a.numRegs,b.numRegs,(int64_t)a.localSizeBytes,(int64_t)b.localSizeBytes,fs::SMEM};
}
void fs_run(torch::Tensor packed, std::vector<torch::Tensor> aux, std::vector<torch::Tensor> scratch,
            torch::Tensor x, torch::Tensor ids, torch::Tensor tw, torch::Tensor out, int64_t lo, int64_t grid) {
  TORCH_CHECK(x.size(0)>=1 && x.size(0)<=8 && x.size(1)==4096,"FS shape");
  fs::Params p{};p.w=(const unsigned char*)packed.data_ptr();p.ga=aux[0].data_ptr<float>();p.gb=aux[1].data_ptr<float>();
  p.nw=(const __nv_bfloat16*)aux[2].data_ptr();p.x=(const __nv_bfloat16*)x.data_ptr();p.ids=ids.data_ptr<int>();p.tw=tw.data_ptr<float>();
  p.ctr=scratch[0].data_ptr<int>();p.y=scratch[1].data_ptr<float>();p.out=(__nv_bfloat16*)out.data_ptr();p.M=x.size(0);p.lo=lo;
  static bool attr=false;static int smcount=0;
  if(!attr){TORCH_CHECK(cudaFuncSetAttribute(fs::kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,fs::SMEM)==cudaSuccess,"FS shared attr");
    TORCH_CHECK(cudaFuncSetAttribute(fs::kernel<true>,cudaFuncAttributeMaxDynamicSharedMemorySize,fs::SMEM)==cudaSuccess,"FS shared attr");
    int a=0,b=0;
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&a,fs::kernel<false>,fs::NTL,fs::SMEM)==cudaSuccess && a>=1,"FS residency low");
    TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b,fs::kernel<true>,fs::NTL,fs::SMEM)==cudaSuccess && b>=1,"FS residency high");
    int device=0;cudaDeviceProp prop{};cudaGetDevice(&device);cudaGetDeviceProperties(&prop,device);smcount=prop.multiProcessorCount;attr=true;}
  TORCH_CHECK(grid>=p.M && grid<=smcount,"FS persistent grid must be resident and cover tokens");
  cudaStream_t stream=at::cuda::getCurrentCUDAStream();
  if(fs_pdl){cudaLaunchConfig_t c{};c.gridDim=dim3(grid);c.blockDim=dim3(fs::NTL);c.dynamicSmemBytes=fs::SMEM;c.stream=stream;
    cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=1;c.attrs=&a;c.numAttrs=1;
    if(p.M<=4)cudaLaunchKernelEx(&c,fs::kernel<false>,p);else cudaLaunchKernelEx(&c,fs::kernel<true>,p);
  }else{if(p.M<=4)fs::kernel<false><<<grid,fs::NTL,fs::SMEM,stream>>>(p);else fs::kernel<true><<<grid,fs::NTL,fs::SMEM,stream>>>(p);}
  TORCH_CHECK(cudaGetLastError()==cudaSuccess,"FS launch failed");
}
