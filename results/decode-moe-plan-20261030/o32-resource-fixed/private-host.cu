
using O32REncode=decltype(&cuTensorMapEncodeTiled);
int o32r_map(CUtensorMap* tm,const void* ptr,int rows,int box,int bk,bool tiled=false) {
  static O32REncode encode=nullptr;
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
int o32r_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
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
  int rc=o32r_map(&tw,(void*)w,8192,64,BK,PACK==2);if(rc)return rc;
  rc=o32r_map(&tx,(void*)x,m,T,BK);if(rc)return rc;
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
int o32r_launch(int id,int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  switch(id) {case 0: return o32r_launch_t<32,128,8,1,1,1,1>(x,w,out,m,stream,query); default:return -99;}
}
