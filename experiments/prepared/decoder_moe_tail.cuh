// Prototype: reuse the existing persistent tiny MoE launch for its Decode tail.
// The parent retains all packed-expert work and the original BF16 MoE output.
// Derivation of MMA ownership: NVIDIA PTX ISA, mma.m16n8k16 BF16 fragments.
struct DecoderTail {
  const uint4* weight;  // [64 output tiles][32 K steps][8 warps][32 lanes]
  __nv_bfloat16* branch;  // [M,1024], distinct from every expert scratch bank
  const float* stream;
  const __nv_bfloat16* post_weight;
  const __nv_bfloat16* pre_weight;
  const float* alpha;
  const float* beta;
  __nv_bfloat16* output;
  float* updated;
  int* ready;  // eight zero-initialized counters; cleared after tail consumption
};

// Sixteen original BF16 output rows per CTA. Every warp owns a disjoint
// 512-element K partition, then FP32 partials reduce before the BF16 store.
__device__ void decoder_tail_project(const Params& p, Smem& S) {
  static_assert(NT == 256 && OKC_WS && !OKC_WGS && OKC_FINCOL && OKC_FINPF,
                "tiny retained geometry and four-segment finalizer");
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rb = blockIdx.x;
  if (rb >= 64) return;
  // Other resident CTAs publish the four complete MoE output segments.
  // All grid CTAs must be resident before this protocol is admitted.
  if (tid == 0)
    for (int tk = 0; tk < p.M; ++tk)
      while (ld_acquire(p.tail.ready + tk) < 4) __nanosleep(32);
  CTA_SYNC();
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  const int token = lane >> 2, t = lane & 3;
#pragma unroll 1
  for (int ki = 0; ki < 32; ++ki) {
    const uint4 a = __ldg(p.tail.weight + (((rb * 32 + ki) * 8 + warp) * 32 + lane));
    const int k = warp * 512 + ki * 16 + 2 * t;
    uint32_t b0 = 0, b1 = 0;
    if (token < p.M) {
      const uint32_t* x = reinterpret_cast<const uint32_t*>(p.out + (size_t)token * HD + k);
      b0 = __ldg(x); b1 = __ldg(x + 4);
    }
    mma_bf16(acc, a.x, a.y, a.z, a.w, b0, b1);
  }
  *reinterpret_cast<float4*>(&S.red[0][warp][lane][0]) = make_float4(acc[0], acc[1], acc[2], acc[3]);
  CTA_SYNC();
  if (warp == 0) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int w = 0; w < NW; ++w) {
      const float4 v = *reinterpret_cast<const float4*>(&S.red[0][w][lane][0]);
      sum.x = __fadd_rn(sum.x, v.x); sum.y = __fadd_rn(sum.y, v.y);
      sum.z = __fadd_rn(sum.z, v.z); sum.w = __fadd_rn(sum.w, v.w);
    }
    const int row = rb * 16 + (lane >> 2), tk = 2 * t;
    if (tk < p.M) {
      p.tail.branch[(size_t)tk * 1024 + row] = __float2bfloat16_rn(sum.x);
      p.tail.branch[(size_t)tk * 1024 + row + 8] = __float2bfloat16_rn(sum.z);
    }
    if (tk + 1 < p.M) {
      p.tail.branch[(size_t)(tk + 1) * 1024 + row] = __float2bfloat16_rn(sum.y);
      p.tail.branch[(size_t)(tk + 1) * 1024 + row + 8] = __float2bfloat16_rn(sum.w);
    }
  }
}

// Match the retained Triton distribution: 32 strided elements per logical
// thread, balanced adjacent tree, XOR warp reduction, eight warp partials.
__device__ __forceinline__ float decoder_tail_sum(float (&v)[32], float* red) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#pragma unroll
  for (int step = 1; step < 32; step *= 2)
#pragma unroll
    for (int i = 0; i < 32; i += step * 2) v[i] = __fadd_rn(v[i], v[i + step]);
  float sum = v[0];
#pragma unroll
  for (int o = 16; o; o >>= 1) sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, o));
  if (lane == 0) red[warp] = sum;
  CTA_SYNC();
  if (warp == 0) {
    float total = lane < NW ? red[lane] : 0.f;
#pragma unroll
    for (int o = 16; o; o >>= 1) total = __fadd_rn(total, __shfl_xor_sync(0xffffffffu, total, o));
    if (lane == 0) red[32] = total;
  }
  CTA_SYNC();
  return red[32];
}

__device__ void decoder_tail_boundary(const Params& p, Smem& S) {
  const int tid = threadIdx.x, row = blockIdx.x;
  if (row >= p.M) return;
  float v[32], squares[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int col = tid + i * NT;
    const float branch = col < 1024 ? __bfloat162float(p.tail.branch[(size_t)row * 1024 + col]) : 0.f;
    const float gain = __bfloat162float(__ldg(p.tail.post_weight + col));
    v[i] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(branch, gain)));
    squares[i] = __fmul_rn(v[i], v[i]);
  }
  float* red0 = &S.red[0][0][0][0];
  float* red1 = &S.red[1][0][0][0];  // distinct: no read/write alias across reductions
  const float post_inv = rsqrtf(__fadd_rn(__fmul_rn(decoder_tail_sum(squares, red0), 1.f / 8192.f), 1e-8f));
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int col = tid + i * NT;
    const size_t off = (size_t)row * 8192 + col;
    const float branch = __bfloat162float(__float2bfloat16_rn(__fmul_rn(v[i], post_inv)));
    v[i] = __fadd_rn(__fmul_rn(__ldg(p.tail.alpha + col), __ldg(p.tail.stream + off)),
                    __fmul_rn(__ldg(p.tail.beta + col), branch));
    p.tail.updated[off] = v[i];
    squares[i] = __fmul_rn(v[i], v[i]);
  }
  const float pre_inv = rsqrtf(__fadd_rn(__fmul_rn(decoder_tail_sum(squares, red1), 1.f / 8192.f), 1e-5f));
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int col = tid + i * NT;
    p.tail.output[(size_t)row * 8192 + col] = __float2bfloat16_rn(
        __fmul_rn(__fmul_rn(v[i], pre_inv), __bfloat162float(__ldg(p.tail.pre_weight + col))));
  }
  if (tid == 0) p.tail.ready[row] = 0;  // every projection consumer passed the second grid barrier
}
