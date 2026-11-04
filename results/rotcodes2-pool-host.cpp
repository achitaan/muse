#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
struct uint2 { uint32_t x,y; }; struct uint4 { uint32_t x,y,z,w; };
float __uint_as_float(uint32_t u) { return std::bit_cast<float>(u); }
uint32_t __funnelshift_r(uint32_t a,uint32_t b,unsigned s) { return uint32_t(((uint64_t(b)<<32)|a)>>(s&31)); }
uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t sel) {
  uint64_t both=(uint64_t(b)<<32)|a; uint32_t out=0;
  for(int i=0;i<4;++i) { unsigned idx=(sel>>(4*i))&15; uint32_t v=(both>>(8*(idx&7)))&255;
    if(idx&8) v=(v&128)?255:0; out|=v<<(8*i); } return out;
}
template<bool ILP> std::array<float,2> run(uint4 (&lw)[2][1],uint2 (&cw)[2][1],float* xf,uint32_t base4,float a,float b) {
  constexpr int ROWS=2; int j=0; float acc[ROWS]={a,b};
  if constexpr(ILP) {

#pragma unroll
    for (int g = 0; g < 2; ++g) {
      uint32_t packed[ROWS][4];
#pragma unroll
      for (int r = 0; r < ROWS; ++r) {
        const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
        const uint32_t c = g ? cw[r][j].y : cw[r][j].x;
        const uint32_t ha = (c & 0x87878787u) + base4;
        const uint32_t hb = (__funnelshift_r(c, c, 4) & 0x87878787u) + base4;
        const uint32_t a = lws[g * 2], b = lws[g * 2 + 1];
        packed[r][0] = __byte_perm(a, ha, 0x5140);
        packed[r][1] = __byte_perm(a, ha, 0x7362);
        packed[r][2] = __byte_perm(b, hb, 0x5140);
        packed[r][3] = __byte_perm(b, hb, 0x7362);
      }
      // Alternate the two independent row chains. Within each row, the
      // sequence j -> g -> element is EXACTLY the original scalar GEMV.
#pragma unroll
      for (int e = 0; e < 8; ++e)
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
          const uint32_t p = packed[r][e / 2];
          const float w = __uint_as_float((e & 1) ? (p & 0xffff0000u) : (p << 16));
          acc[r] = fmaf(w, xf[g * 8 + e], acc[r]);
        }
    }

  } else {
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint32_t lws[4] = {lw[r][j].x, lw[r][j].y, lw[r][j].z, lw[r][j].w};
      const uint32_t cws[2] = {cw[r][j].x, cw[r][j].y};
      float v = acc[r];
#pragma unroll
      for (int g = 0; g < 2; ++g) {           // 8-element groups: low bytes lws[2g], lws[2g+1]; codes cws[g]
        // Signs are prepositioned: byte MSBs for values 0..3, ROR4's MSBs for 4..7.
        // base <= 120, so offset+base+sign <= 255: no cross-byte carries.
        const uint32_t ha = (cws[g] & 0x87878787u) + base4;
        const uint32_t hb = (__funnelshift_r(cws[g], cws[g], 4) & 0x87878787u) + base4;
        const uint32_t la = lws[2 * g], lb = lws[2 * g + 1];
        const uint32_t p01 = __byte_perm(la, ha, 0x5140), p23 = __byte_perm(la, ha, 0x7362);
        const uint32_t p45 = __byte_perm(lb, hb, 0x5140), p67 = __byte_perm(lb, hb, 0x7362);
        const float* xg = xf + 8 * g;
        v = fmaf(__uint_as_float(p01 << 16), xg[0], v); v = fmaf(__uint_as_float(p01 & 0xFFFF0000u), xg[1], v);
        v = fmaf(__uint_as_float(p23 << 16), xg[2], v); v = fmaf(__uint_as_float(p23 & 0xFFFF0000u), xg[3], v);
        v = fmaf(__uint_as_float(p45 << 16), xg[4], v); v = fmaf(__uint_as_float(p45 & 0xFFFF0000u), xg[5], v);
        v = fmaf(__uint_as_float(p67 << 16), xg[6], v); v = fmaf(__uint_as_float(p67 & 0xFFFF0000u), xg[7], v);
      }
      acc[r] = v;
    }
  }
  return {acc[0],acc[1]};
}
uint32_t state=0x91224151u;
uint32_t next() { state^=state<<13; state^=state>>17; state^=state<<5; return state; }
uint32_t rot(uint32_t c) { uint32_t h=c&0x80808080u; return (c&0x77777777u)|((c&0x08080808u)<<4)|(h<<4)|(h>>28); }
int main() {
  for(int i=0;i<100000;++i) {
    uint4 low[2][1]; uint2 codes[2][1]; float xf[16];
    for(int r=0;r<2;++r) {low[r][0]={next(),next(),next(),next()};codes[r][0]={rot(next()),rot(next())};}
    for(int e=0;e<16;++e) xf[e]=__uint_as_float(((next()&0x80000000u)|0x3f000000u|(next()&0xffu)<<16));
    uint32_t base=(40+i%71)*0x01010101u;
    float a=__uint_as_float((next()&0x80000000u)|0x3f000000u), b=__uint_as_float((next()&0x80000000u)|0x3f000000u);
    auto x=run<false>(low,codes,xf,base,a,b), y=run<true>(low,codes,xf,base,a,b);
    for(int r=0;r<2;++r) if(std::bit_cast<uint32_t>(x[r])!=std::bit_cast<uint32_t>(y[r]))
      throw std::runtime_error("per-row FP32 order mismatch");
  }
  std::cout<<"PASS 100000 two-row FMA-order cases\n";
}
