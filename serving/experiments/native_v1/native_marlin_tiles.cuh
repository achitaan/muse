// Native Marlin NVFP4 tile access for the bounded SM90 WGS experiment.
// No persistent weight transformation. Tensor maps and scratch are caller-owned.
#include <cuda.h>

__device__ __forceinline__ void native_tma(
    void* dst, const CUtensorMap* map, int n, int k, uint64_t* barrier) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(uint32_t(__cvta_generic_to_shared(dst))), "l"(map),
         "r"(n), "r"(k), "r"(uint32_t(__cvta_generic_to_shared(barrier)))
      : "memory");
}

__device__ __forceinline__ int native_scale_col(int n) {
  int x = (n & 7) * 8 + ((n & 63) >> 3);
  return (n & ~63) + (x & ~3) + ((x & 1) << 1) + ((x & 2) >> 1);
}

// Marlin's encoded byte is a rebased FP16 exponent/mantissa fragment,
// NOT an E4M3 value. Decode its top bit into BF16 exponent bit 7.
// WGS uses operands 2^57 larger; undo at the accumulator rounding boundary.
__device__ __forceinline__ uint32_t native_scale(uint32_t s) {
  return s ? (((s & 128) << 7) | ((s & 127) << 4)) + (57 << 7) : 0;
}

__device__ __forceinline__ uint32_t native_pair(uint32_t q, int high_k) {
  q >>= high_k * 4;
  const uint32_t v = q & 0x000f000fu;
  return ((v & 0x00070007u) << 6) | ((v & 0x00080008u) << 12);
}

template <bool PB>
__device__ __forceinline__ void native_regs(
    const unsigned char* tile, int step, int wg, int wp, int lane,
    uint32_t (&r)[4]) {
  const int g = lane >> 2, t = lane & 3;
  const int n = PB ? wg * 64 + wp * 16 + g : wg * 32 + wp * 8 + g;
  constexpr int WIDTH = PB ? 512 : 256;
  const int wi = step * WIDTH + (n >> 6) * 128 + (g * 4 + t) * 4 + ((n & 63) >> 4);
  const uint32_t* words = reinterpret_cast<const uint32_t*>(tile);
  const uint32_t q0 = words[wi] >> (PB ? 0 : ((n & 8) ? 8 : 0));
  const uint32_t q1 = PB ? words[wi] >> 8 : words[4096 + wi] >> ((n & 8) ? 8 : 0);
  constexpr int NS = PB ? 256 : 128;
  const unsigned char* scales = tile + 32768 + step * NS;
  const uint32_t s0 = native_scale(scales[native_scale_col(n)]);
  const uint32_t s1 = native_scale(PB ? scales[native_scale_col(n + 8)]
                                     : scales[2048 + native_scale_col(n)]);
  const uint32_t ss = s0 | (s1 << 16);
  r[0] = bmul2l(native_pair(q0, 0), ss);
  r[1] = bmul2h(native_pair(q1, 0), ss);
  r[2] = bmul2l(native_pair(q0, 1), ss);
  r[3] = bmul2h(native_pair(q1, 1), ss);
}
