
#include <cstdint>
#include <cstdio>
#include <cassert>
uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d=0;
  for (int i=0;i<4;i++) { uint32_t idx=(sel>>(4*i))&7, v=(idx<4?a:b)>>(8*(idx%4)); d|=(v&255)<<(8*i); }
  return d;
}
uint32_t lop3_sel(uint32_t a,uint32_t b,uint32_t c) { return (a&~c)|(b&c); }
uint32_t __funnelshift_r(uint32_t a,uint32_t b,int s) { return (a>>s)|(b<<(32-s)); }
inline void decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  uint32_t ha = lop3_sel(nw & 0x07070707u, nw << 4, 0x80808080u) + base4;
  uint32_t hb = lop3_sel((nw >> 4) & 0x07070707u, nw, 0x80808080u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
inline void rot_decode8(uint32_t lw0, uint32_t lw1, uint32_t nw, uint32_t base4, uint32_t* r) {
  // Signs were permuted within this word in prepare(); base <= 120, so no byte carries.
  uint32_t ha = (nw & 0x87878787u) + base4;
  uint32_t hb = (__funnelshift_r(nw, nw, 4) & 0x87878787u) + base4;
  r[0] = prmt(lw0, ha, 0x5140);
  r[1] = prmt(lw0, ha, 0x7362);
  r[2] = prmt(lw1, hb, 0x5140);
  r[3] = prmt(lw1, hb, 0x7362);
}
struct U4 { uint32_t x,y,z,w; }; struct U2 { uint32_t x,y; };
void late_decode(const U4* lw, const U2* cw, uint32_t base4, uint32_t* r) {
  const int nt=0;
  for (int j=0;j<4;j++) {
          const uint32_t code = j < 2 ? cw[nt].x : cw[nt].y;
          const uint32_t hi = ((j & 1) ? (__funnelshift_r(code, code, 4) & 0x87878787u)
                                      : (code & 0x87878787u)) + base4;
          const uint32_t low = j == 0 ? lw[nt].x : j == 1 ? lw[nt].y : j == 2 ? lw[nt].z : lw[nt].w;
          const uint32_t b0 = prmt(low, hi, 0x5140), b1 = prmt(low, hi, 0x7362);
    r[2*j]=b0; r[2*j+1]=b1;
  }
}

int main() {
  uint64_t count=0;
  for(uint32_t base=0;base<=120;base++) for(uint32_t p=0;p<65536;p++) {
    uint32_t low[2]={}, c=0, expected[4]={};
    for(int i=0;i<8;i++) {
      uint32_t v=(p+uint32_t(i)*7937u)&65535u, hi=v>>8;
      uint32_t o=(hi&127)<base?0:((hi&127)>base+7?7:(hi&127)-base);
      uint32_t code=(hi>>7)*8+o, dec=(((hi&128)|(base+o))<<8)|(v&255);
      low[i/4]|=(v&255)<<(8*(i%4)); c|=code<<(8*(i%4)+4*(i/4)); expected[i/2]|=dec<<(16*(i%2));
    }
    uint32_t sign=c&0x80808080u;
    uint32_t rot=(c&0x77777777u)|((c&0x08080808u)<<4)|(sign<<4)|(sign>>28), a[4],b[4];
    decode8(low[0],low[1],c,base*0x01010101u,a);
    rot_decode8(low[0],low[1],rot,base*0x01010101u,b);
    for(int i=0;i<4;i++) assert(a[i]==expected[i] && b[i]==expected[i]);
    U4 lw[1]={{low[0],low[1],~low[1],~low[0]}}; U2 cw[1]={{rot,~rot}};
    uint32_t late[8], second[4];
    late_decode(lw,cw,base*0x01010101u,late);
    rot_decode8(~low[1],~low[0],~rot,base*0x01010101u,second);
    for(int i=0;i<4;i++) assert(late[i]==b[i] && late[i+4]==second[i]);
    count++;
  }
  std::printf("%llu word/window cases\n",(unsigned long long)count);
}
