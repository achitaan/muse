
#define __device__
#define __forceinline__ inline
using uint32_t = unsigned int;
using size_t = unsigned long;
struct uint4 { uint32_t x,y,z,w; };
struct uint2 { uint32_t x,y; };
struct Dim { int x; } threadIdx;
#define WARP_IDX(tid) ((tid) >> 5)
constexpr int WGS_PIECE=9216, WGS_NS=3, WGS_NX=3, UNIT_BYTES=36864;
constexpr int WGS_XS=32768, WGS_FIXW=4096, BW_F=0, BW_E=3, BX_F=6, BX_E=9;
constexpr int OKC_WGSDFIN=2;
struct Params { uint32_t zero; };
struct Smem { unsigned long long bars[24]; };
struct WgsPend { int kind,u,slot,nst; };
void wgs_wait(unsigned long long*, uint32_t, int);
void mbar_arrive1(unsigned long long*);
void __syncwarp();
void dq4(uint32_t,uint32_t&,uint32_t&,uint32_t&,uint32_t&);
uint32_t __byte_perm(uint32_t,uint32_t,uint32_t);
uint32_t bmul2l(uint32_t,uint32_t), bmul2h(uint32_t,uint32_t);
void mma_bf16(float*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
template <bool B> void wgs_flush(const Params&,Smem&,WgsPend&);
template <bool B> void wgs_epi_tile(const Params&,const Smem&,int,int,int,const float*);
float* wgs_fix(const Params&,int);
// Verify experiment: use the warp-local tensor instruction only for N=8.
// Each warp already owns exactly the m16n8 accumulator/register-A fragment
// of one m64n8 WGMMA. No weight, activation, routing or output layout changes.
template <bool PB>
__device__ __forceinline__ void wgs_seg_sync8(
    const Params& p, Smem& S, unsigned char* ring, int u, int s0, int s1,
    int slot, int& seq, WgsPend& pd) {
  const int tid = threadIdx.x, lane = tid & 31;
  const int warp = WARP_IDX(tid), wg = warp >> 2, wp = warp & 3;
  const int g = lane >> 2, t = lane & 3, wt = tid & 127;
  constexpr int UBL = PB ? 4 : 5;
  const uint32_t k16c = 16u + p.zero;
  const unsigned char* pc0 = ring + wg * WGS_PIECE;
  const unsigned char* xring = ring + WGS_NS * UNIT_BYTES;
  float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
  for (int st = s0; st < s1; ++st) {
    const int ws = seq % WGS_NS, xs = seq % WGS_NX;
    wgs_wait(S.bars + BW_F + ws, (uint32_t)(seq / WGS_NS) & 1u, 3000000 + seq);
    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const unsigned char* xb = xring + xs * WGS_XS;
    uint4 cwn = *reinterpret_cast<const uint4*>(pc + (wp * 32 + lane) * 16);
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cwn;
      const uint2 scg = scn;
      if (lg < 3) {
        cwn = *reinterpret_cast<const uint4*>(pc + (((lg + 1) * 4 + wp) * 32 + lane) * 16);
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
      } else {
        // All 16 code/scale words have been captured. One arrival per warp,
        // exactly like WGS; the producer's empty barrier expects NW arrivals.
        __syncwarp();
        if (lane == 0) mbar_arrive1(S.bars + BW_E + ws);
      }
#pragma unroll
      for (int js = 0; js < 4; ++js) {
        const uint32_t q = js == 0 ? cwg.x : js == 1 ? cwg.y : js == 2 ? cwg.z : cwg.w;
        uint32_t r0, r1, r2, r3;
        dq4(q, r0, r1, r2, r3);
        const uint32_t sw = js < 2 ? scg.x : scg.y, sb = (uint32_t)(js & 1) * 2u;
        const uint32_t sgp = __byte_perm(sw, 0u, 0x4040u | sb | ((sb + 1u) << 8)) * k16c + 0x5C005C00u;
        const uint32_t a0 = bmul2l(r0, sgp), a1 = bmul2h(r1, sgp);
        const uint32_t a2 = bmul2l(r2, sgp), a3 = bmul2h(r3, sgp);
        // x is [n8][k8][token-in-n8][K-in-k8]. For lane (g,t),
        // mma.sync B owns tokens g, K=(2t,2t+1,8+2t,9+2t).
        // Each scalar warp load touches 32 consecutive, distinct banks.
        const unsigned char* bp = xb + (lg * 4 + js) * 256 + g * 16 + t * 4;
        const uint32_t b0 = *reinterpret_cast<const uint32_t*>(bp);
        const uint32_t b1 = *reinterpret_cast<const uint32_t*>(bp + 128);
        mma_bf16(d, a0, a1, a2, a3, b0, b1);
      }
      // Match the incumbent's deferred-work position (CI=3, GS=2).
      if (st == s0 && lg == 1 && pd.kind) wgs_flush<PB>(p, S, pd);
    }
    // Synchronous MMA has consumed this stage before it can be overwritten.
    __syncwarp();
    if (lane == 0) mbar_arrive1(S.bars + BX_E + xs);
    ++seq;
  }
  if (slot >= 0) {
    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;
    asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};\n"
                 ::"l"(fb), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]) : "memory");
    pd = WgsPend{(((OKC_WGSDFIN >> (PB ? 1 : 0)) & 1) && (PB ? s0 == 0 : s1 == 16)) ? 3 : 2,
                 u, slot, s1 - s0};
    return;
  }
  const int j = u >> UBL, r16 = (u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp;
  wgs_epi_tile<PB>(p, S, j, r16, 0, d);
  if (!PB) pd = WgsPend{1, u, -1, 0};
}

template void wgs_seg_sync8<false>(const Params&, Smem&, unsigned char*,int,int,int,int,int&,WgsPend&);
template void wgs_seg_sync8<true>(const Params&, Smem&, unsigned char*,int,int,int,int,int&,WgsPend&);
