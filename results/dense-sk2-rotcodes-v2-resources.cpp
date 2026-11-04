
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <vector>
#include <stdexcept>
struct __nv_bfloat16 {};
struct cudaFuncAttributes { int numRegs; size_t localSizeBytes, sharedSizeBytes; };
enum { cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDevAttrMultiProcessorCount, cudaFuncAttributeMaxDynamicSharedMemorySize };
int cudaGetDevice(int* dev) { *dev=0; return 0; }
int cudaDeviceGetAttribute(int* v,int attr,int) { *v=attr==cudaDevAttrMultiProcessorCount?132:232448; return 0; }
template<class K> int cudaFuncGetAttributes(cudaFuncAttributes* fa,K) { *fa={64,0,0}; return 0; }
template<class K> int cudaFuncSetAttribute(K,int,int) { return 0; }
template<class K> int cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* active,K,int,size_t) { *active=1; return 0; }
#define C10_CUDA_CHECK(expr) do { assert((expr)==0); } while(0)
#define TORCH_CHECK(cond,...) do { if(!(cond)) throw std::runtime_error("check"); } while(0)
template<int MT,int TOKR,int NT,int NWR,int WSK,int UPW,int ST,bool COMP,bool HL,int NC,int CL,class Out,bool PK> void ref_sk2_kernel() {}
template<int MT,int TOKR,int NT,int NWR,int WSK,int UPW,int ST,bool COMP,bool HL,int NC,int CL,class Out,bool PK> void rot_sk2_kernel() {}
template<int MT,int TOKR,int NT,int NWR,int WSK,int UPW,int ST,bool COMP,bool HL,int NC,int CL,class Out,bool PK> void late_sk2_kernel() {}
// Prepare-time diagnostics only; these functions never run inside kernel_fn.
template <typename Kernel>
static std::vector<int64_t> sr_attributes(Kernel kern, size_t smem, int threads, int stages, int nesc) {
  int dev = 0, optin = 0, active = 0, sms = 0;
  C10_CUDA_CHECK(cudaGetDevice(&dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  cudaFuncAttributes fa;
  C10_CUDA_CHECK(cudaFuncGetAttributes(&fa, kern));
  TORCH_CHECK(smem + fa.sharedSizeBytes <= (size_t)optin, "SK2 resources exceed opt-in shared memory");
  C10_CUDA_CHECK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, optin - (int)fa.sharedSizeBytes));
  C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kern, threads, smem));
  return {fa.numRegs, (int64_t)fa.localSizeBytes, (int64_t)fa.sharedSizeBytes,
          (int64_t)smem, threads, active, stages, nesc, sms};
}
std::vector<int64_t> sk2rot_resources(int64_t cfg, int64_t tokr, int64_t pmax, int64_t mode) {
  TORCH_CHECK(pmax >= 0, "negative escape-table capacity");
  if (cfg == 0 && tokr == 32 && mode == 0) return sr_attributes(ref_sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 99920 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 0 && tokr == 32 && mode == 1) return sr_attributes(rot_sk2_kernel<2, 32, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 99920 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 2 && tokr == 32 && mode == 0) return sr_attributes(ref_sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, __nv_bfloat16, false>, 161520 + (size_t)pmax * 8, 288, 5, 4);
  if (cfg == 2 && tokr == 32 && mode == 1) return sr_attributes(rot_sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, __nv_bfloat16, false>, 161520 + (size_t)pmax * 8, 288, 5, 4);
  if (cfg == 2 && tokr == 32 && mode == 2) return sr_attributes(late_sk2_kernel<2, 32, 2, 2, 2, 2, 5, true, false, 1, 1, __nv_bfloat16, false>, 161520 + (size_t)pmax * 8, 288, 5, 4);
  if (cfg == 4 && tokr == 32 && mode == 0) return sr_attributes(ref_sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, __nv_bfloat16, false>, 218848 + (size_t)pmax * 8, 288, 4, 4);
  if (cfg == 4 && tokr == 32 && mode == 1) return sr_attributes(rot_sk2_kernel<2, 32, 3, 1, 4, 2, 4, true, false, 1, 1, __nv_bfloat16, false>, 218848 + (size_t)pmax * 8, 288, 4, 4);
  if (cfg == 7 && tokr == 32 && mode == 0) return sr_attributes(ref_sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 149072 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 7 && tokr == 32 && mode == 1) return sr_attributes(rot_sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 149072 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 7 && tokr == 32 && mode == 2) return sr_attributes(late_sk2_kernel<2, 32, 4, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 149072 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 8 && tokr == 16 && mode == 0) return sr_attributes(ref_sk2_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1, 1, __nv_bfloat16, false>, 173536 + (size_t)pmax * 8, 288, 4, 4);
  if (cfg == 8 && tokr == 16 && mode == 1) return sr_attributes(rot_sk2_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1, 1, __nv_bfloat16, false>, 173536 + (size_t)pmax * 8, 288, 4, 4);
  if (cfg == 8 && tokr == 16 && mode == 2) return sr_attributes(late_sk2_kernel<1, 16, 2, 2, 2, 4, 4, true, false, 1, 1, __nv_bfloat16, false>, 173536 + (size_t)pmax * 8, 288, 4, 4);
  if (cfg == 9 && tokr == 16 && mode == 0) return sr_attributes(ref_sk2_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 69008 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 9 && tokr == 16 && mode == 1) return sr_attributes(rot_sk2_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 69008 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 9 && tokr == 16 && mode == 2) return sr_attributes(late_sk2_kernel<1, 16, 2, 2, 2, 2, 3, true, false, 1, 1, __nv_bfloat16, false>, 69008 + (size_t)pmax * 8, 256, 3, 3);
  if (cfg == 11 && tokr == 16 && mode == 0) return sr_attributes(ref_sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 2, 1, __nv_bfloat16, false>, 181744 + (size_t)pmax * 8, 288, 5, 4);
  if (cfg == 11 && tokr == 16 && mode == 1) return sr_attributes(rot_sk2_kernel<1, 16, 3, 1, 4, 2, 5, true, false, 2, 1, __nv_bfloat16, false>, 181744 + (size_t)pmax * 8, 288, 5, 4);
  if (cfg == 12 && tokr == 16 && mode == 0) return sr_attributes(ref_sk2_kernel<1, 16, 2, 2, 1, 4, 3, true, false, 2, 1, __nv_bfloat16, false>, 69008 + (size_t)pmax * 8, 192, 3, 3);
  if (cfg == 12 && tokr == 16 && mode == 1) return sr_attributes(rot_sk2_kernel<1, 16, 2, 2, 1, 4, 3, true, false, 2, 1, __nv_bfloat16, false>, 69008 + (size_t)pmax * 8, 192, 3, 3);
  TORCH_CHECK(false, "SK2 resources: no specialization");
}

int main() {
{ auto a=sk2rot_resources(0,32,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==99920 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(0,32,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==99920 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(0,32,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==99936 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(0,32,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==99936 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(0,32,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==100432 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(0,32,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==100432 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(12,16,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69008 && a[4]==192 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(12,16,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69008 && a[4]==192 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(12,16,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69024 && a[4]==192 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(12,16,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69024 && a[4]==192 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(12,16,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69520 && a[4]==192 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(12,16,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69520 && a[4]==192 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149072 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149072 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,0,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149072 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149088 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149088 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,2,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149088 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149584 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149584 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(7,32,64,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==149584 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69008 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69008 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,0,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69008 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69024 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69024 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,2,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69024 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69520 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69520 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(9,16,64,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==69520 && a[4]==256 && a[5]==1 && a[6]==3 && a[7]==3 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==161520 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==161520 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,0,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==161520 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==161536 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==161536 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,2,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==161536 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==162032 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==162032 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(2,32,64,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==162032 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==173536 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==173536 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,0,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==173536 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==173552 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==173552 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,2,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==173552 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==174048 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==174048 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(8,16,64,2); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==174048 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(4,32,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==218848 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(4,32,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==218848 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(4,32,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==218864 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(4,32,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==218864 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(4,32,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==219360 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(4,32,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==219360 && a[4]==288 && a[5]==1 && a[6]==4 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(11,16,0,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==181744 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(11,16,0,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==181744 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(11,16,2,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==181760 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(11,16,2,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==181760 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(11,16,64,0); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==182256 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
{ auto a=sk2rot_resources(11,16,64,1); assert(a.size()==9 && a[0]==64 && a[1]==0 && a[2]==0 && a[3]==182256 && a[4]==288 && a[5]==1 && a[6]==5 && a[7]==4 && a[8]==132); }
bool bad=false; try { sk2rot_resources(0,32,-1,0); } catch (...) { bad=true; } assert(bad);
bad=false; try { sk2rot_resources(4,32,2,2); } catch (...) { bad=true; } assert(bad);
std::printf("60 resource dispatch cases passed (CUDA APIs mocked)\n");
}
