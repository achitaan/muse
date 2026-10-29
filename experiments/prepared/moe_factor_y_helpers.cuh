// Preserve the incumbent's two BF16 boundaries, retaining the product bits.
__device__ __forceinline__ __nv_bfloat162 factor_marlin_bits2(float a, float b, float scale) {
  const __nv_bfloat162 acc = __floats2bfloat162_rn(a * TWO62, b * TWO62);
  const __nv_bfloat16 gs = __float2bfloat16_rn(scale * 0x1p-62f);
  return __hmul2_rn(acc, __bfloat162bfloat162(gs));
}

// BF16-to-FP32 is exact bit extension. Explicit RN multiplication preserves
// the old FP32 product materialization before squares or route-weight FMAs.
__device__ __forceinline__ void factor_product(uint4 p, const float* gain, float4& a, float4& b) {
  a = make_float4(__fmul_rn(__uint_as_float(p.x << 16), gain[0]),
                 __fmul_rn(__uint_as_float(p.x & 0xffff0000u), gain[1]),
                 __fmul_rn(__uint_as_float(p.y << 16), gain[2]),
                 __fmul_rn(__uint_as_float(p.y & 0xffff0000u), gain[3]));
  b = make_float4(__fmul_rn(__uint_as_float(p.z << 16), gain[4]),
                 __fmul_rn(__uint_as_float(p.z & 0xffff0000u), gain[5]),
                 __fmul_rn(__uint_as_float(p.w << 16), gain[6]),
                 __fmul_rn(__uint_as_float(p.w & 0xffff0000u), gain[7]));
}
