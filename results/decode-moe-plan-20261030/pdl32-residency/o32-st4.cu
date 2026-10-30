
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#define OD __device__ __forceinline__
OD uint32_t ol_addr(void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
OD void ol_wait(uint32_t b,int ph) {

  asm volatile("{ .reg .pred ready, expired; .reg .b64 start, now, elapsed;\n"
    "mbarrier.try_wait.parity.shared::cta.b64 ready,[%0],%1; @ready bra DONE;\n"
    "mov.u64 start, %%globaltimer;\n"
    "WAIT: mbarrier.try_wait.parity.shared::cta.b64 ready,[%0],%1; @ready bra DONE;\n"
    "mov.u64 now, %%globaltimer; sub.u64 elapsed,now,start;\n"
    "setp.gt.u64 expired,elapsed,20000000; @expired trap; bra WAIT;\n"
    "DONE: }" :: "r"(b), "r"(ph) : "memory");

}
OD void ol_arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }
OD uint64_t ol_desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }
OD void ol_tma(uint32_t d,const void* map,int row,int kb,uint32_t b,uint64_t p) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{0,%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(row),"r"(kb),"r"(b),"l"(p) : "memory");
}
OD void ol_mma32(float* a,uint64_t w,uint64_t x) {
          asm volatile("wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, %16, %17, 1, 1, 1, 0, 0;" : "+f"(a[0]),"+f"(a[1]),"+f"(a[2]),"+f"(a[3]),"+f"(a[4]),"+f"(a[5]),"+f"(a[6]),"+f"(a[7]),"+f"(a[8]),"+f"(a[9]),"+f"(a[10]),"+f"(a[11]),"+f"(a[12]),"+f"(a[13]),"+f"(a[14]),"+f"(a[15]) : "l"(w),"l"(x));
        }

__device__ __noinline__ void pdlo32_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
__global__ __launch_bounds__(128*TW+((PIPE&2)?128:0)) void pdlo32_projection(
    const __grid_constant__ CUtensorMap tw,const __grid_constant__ CUtensorMap tx,
    const uint8_t* w,__nv_bfloat16* out,int m) {
  static_assert(BN==32 && BK==128 && ST==4 && PIPE==1 && PACK==1 && HINT==1 && TW==1, "fixed four-stage O32 ring");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  constexpr int T=BN*TW,R=PACK==2?32:64,CT=128*TW;
  constexpr int AB=64*BK*2,SB=(64+T)*BK*2,IT=1024/BK,TH=CT+((PIPE&2)?128:0);
  extern __shared__ __align__(1024) uint8_t sm[];
  uint32_t base=ol_addr(sm),full=base+ST*SB,empty=full+ST*8;
  int tid=threadIdx.x,nt=blockIdx.x,mt=blockIdx.y;
  uint64_t pw,px;
  if constexpr(HINT) {
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pw));
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(px));
  } else {
    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pw));px=pw;
  }
  auto loadw=[&](int i) {
    int s=i%ST;
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _,[%0],%1;" :: "r"(full+s*8),"r"(SB) : "memory");
    if constexpr(PACK==1) {
      const uint8_t* src=w+((long)nt*IT+i)*AB;
      asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1],%2,[%3],%4;" :: "r"(base+s*SB),"l"(src),"r"(AB),"r"(full+s*8),"l"(pw) : "memory");
    } else if constexpr(PACK==2) ol_tma(base+s*SB,&tw,0,nt*16+i*(BK/64),full+s*8,pw);
    else ol_tma(base+s*SB,&tw,nt*64,i*(BK/64),full+s*8,pw);
  };
  auto loadx=[&](int i) {
    const int s=i%ST;
    ol_tma(base+s*SB+AB,&tx,mt*T,i*(BK/64),full+s*8,px);
  };
  auto load=[&](int i) { loadw(i); loadx(i); };
  if(tid==((PIPE&2)?CT:0)) {
    #pragma unroll
    for(int s=0;s<ST;s++) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0],1;" :: "r"(full+s*8) : "memory");
      if constexpr(PIPE&2) asm volatile("mbarrier.init.shared::cta.b64 [%0],%1;" :: "r"(empty+s*8),"n"(TW) : "memory");
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) loadw(i);
    pdlo32_wait();
    #pragma unroll
    for(int i=0;i<ST && i<IT;i++) loadx(i);
  }
  __syncthreads();
  if((PIPE&2) && tid>=CT) {
    if(tid==CT) {
      #pragma unroll 1
      for(int i=ST;i<IT;i++) {ol_wait(empty+(i%ST)*8,((i/ST)-1)&1);load(i);}
    }
  } else {
    float a[BN/2];
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("mov.f32 %0,0f00000000;" : "=f"(a[j]) :: "memory");
    #pragma unroll 1
    for(int i=0;i<IT;i++) {
      int s=i%ST;
      ol_wait(full+s*8,(i/ST)&1);
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
      #pragma unroll
      for(int kk=0;kk<BK/16;kk++) {
        uint64_t aw=ol_desc(base+s*SB+(kk/4)*8192+(kk%4)*32);
        uint64_t ax=ol_desc(base+s*SB+AB+(kk/4)*T*128+(tid/128)*BN*128+(kk%4)*32);
        if constexpr(BN==32) ol_mma32(a,aw,ax);
      }
      asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
      if constexpr(PIPE&1) asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
      else asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      #pragma unroll
      for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
      int done=i-((PIPE&1)?1:0);
      if(done>=0 && done+ST<IT && tid%128==0) {
        if constexpr(PIPE&2) ol_arrive(empty+(done%ST)*8);
        else load(done+ST);
      }
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    #pragma unroll
    for(int j=0;j<BN/2;j++) asm volatile("" : "+f"(a[j]) :: "memory");
    int row=(tid%128/32)*16+(tid%32)/4;
    int col=2*(tid%4);
    #pragma unroll
    for(int j=0;j<BN/8;j++) {
      #pragma unroll
      for(int h=0;h<2;h++) {
        int token=mt*T+(tid/128)*BN+col+8*j;
        if(token<m) out[(long)token*8192+nt*R+row+8*h]=__float2bfloat16_rn(a[4*j+2*h]);
        if(token+1<m) out[(long)(token+1)*8192+nt*R+row+8*h]=__float2bfloat16_rn(a[4*j+2*h+1]);
      }
    }
  }

}

using OLEncode=decltype(&cuTensorMapEncodeTiled);
int ol_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static OLEncode encode=nullptr;
  if(!encode) {
    cudaDriverEntryPointQueryResult q;
    auto rc=cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);
    if(rc!=cudaSuccess || !encode) return -100;
  }
  uint64_t dims[]={64,uint64_t(tiled?32:rows),uint64_t(tiled?4096:16)},stride[]={uint64_t(tiled?128:2048),uint64_t(tiled?4096:128)};
  uint32_t b[]={64,uint32_t(box),uint32_t(bk/64)},es[]={1,1,1};
  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,3,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  constexpr int T=BN*TW,R=PACK==2?32:64;
  constexpr int MAIN=ST*((64+T)*BK*2+16),EPI=T*72*2,SMEM=MAIN>EPI?MAIN:EPI;
  auto fn=pdlo32_projection<BN,BK,ST,PIPE,PACK,HINT,TW>;
  static bool configured=false;
  if(!configured) {
    int rc=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM);if(rc)return rc;
    configured=true;
  }
  if(query) {cudaFuncAttributes a;int rc=cudaFuncGetAttributes(&a,fn);return rc?-rc:a.numRegs*10000+a.localSizeBytes;}
  CUtensorMap tw,tx;
  int rc=ol_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=ol_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
  cudaLaunchConfig_t lc = {};
  lc.gridDim=dim3(8192/R,(m+T-1)/T);
  lc.blockDim=dim3(128*TW+((PIPE&2)?128:0));
  lc.dynamicSmemBytes=SMEM; lc.stream=(cudaStream_t)stream;
  cudaLaunchAttribute attr[1];
  attr[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed=1;
  lc.attrs=attr; lc.numAttrs=1;
  const cudaError_t launch_rc=cudaLaunchKernelEx(&lc,fn,tw,tx,(const uint8_t*)w,(__nv_bfloat16*)out,m);
  if(launch_rc!=cudaSuccess) return int(launch_rc);
  return int(cudaGetLastError());
}
int pdlo32_info(int query) {
  if(query!=1) return -90;
  int blocks=0;
  const cudaError_t rc=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,pdlo32_projection<32,128,4,1,1,1,1>,128,98368);
  return rc==cudaSuccess ? blocks : -static_cast<int>(rc);
}
int ol_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {case 0: return ol_launch_t<32,128,4,1,1,1,1>(x,w,out,m,stream,query); default:return -99;}
}
