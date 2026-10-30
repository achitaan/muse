#include <stdint.h>
static int simulated_attribute = 49152;
template<int BN,int BK,int ST,int PIPE,int PACK,int HINT,int TW>
int ol_launch_t(int64_t x,int64_t w,int64_t out,int m,int64_t stream,int query) {
  static bool configured=false;
  if(!configured){simulated_attribute=196736;configured=true;}
  return simulated_attribute;
}
extern "C" int probe(){return ol_launch_t<32,128,8,1,1,1,1>(0,0,0,32,0,1);}
extern "C" int fixed_query(){simulated_attribute=196736;return simulated_attribute;}
extern "C" int module_tag(){return MODULE_ID;}
