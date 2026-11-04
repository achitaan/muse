#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, unsigned s) {
  return uint32_t(((uint64_t(hi) << 32) | lo) >> (s & 31));
}
inline uint32_t cg_hib(uint32_t c, uint32_t base4) {
  return ((c & 0x07070707u) + base4) | ((c << 4) & 0x80808080u);
}

std::array<uint32_t, 2> original(uint32_t c, uint32_t base4) {
uint32_t cws[1] = {c}; int g = 0;
        const uint32_t ha = cg_hib(cws[g] & 0x0F0F0F0Fu, base4);         // hi bytes of elements 0..3
        const uint32_t hb = cg_hib((cws[g] >> 4) & 0x0F0F0F0Fu, base4);  // hi bytes of elements 4..7
return {ha, hb}; }

std::array<uint32_t, 2> candidate(uint32_t c, uint32_t base4) {
uint32_t cws[1] = {c}; int g = 0;
        // Signs are prepositioned: byte MSBs for values 0..3, ROR4's MSBs for 4..7.
        // base <= 120, so offset+base+sign <= 255: no cross-byte carries.
        const uint32_t ha = (cws[g] & 0x87878787u) + base4;
        const uint32_t hb = (__funnelshift_r(cws[g], cws[g], 4) & 0x87878787u) + base4;
return {ha, hb}; }

uint32_t rot(uint32_t c) {
  uint32_t h = c & 0x80808080u;
  return (c & 0x77777777u) | ((c & 0x08080808u) << 4) | (h << 4) | (h >> 28);
}
uint32_t unrot(uint32_t c) {
  uint32_t h = c & 0x08080808u;
  return (c & 0x77777777u) | ((c & 0x80808080u) >> 4) | (h >> 4) | (h << 28);
}
int main() {
  uint32_t seed = 0x95117821u;
  unsigned long long count = 0;
  for (unsigned b = 0; b <= 120; ++b) {
    for (unsigned i = 0; i < 65536; ++i) {
      seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
      uint32_t c = i < 32 ? 1u << i : seed;
      if (unrot(rot(c)) != c || original(c,b * 0x01010101u) != candidate(rot(c),b * 0x01010101u))
        throw std::runtime_error("compiled decoder mismatch");
      ++count;
    }
  }
  std::cout << "PASS " << count << " compiled word/window cases\n";
}
