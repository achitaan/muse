"""Assemble MAIN-only L1 from exact moe_b13; other builds retain parent code."""
from pathlib import Path
import ast,hashlib
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'experiments/boards/moe/moe_b13.py').read_bytes().decode()
node=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
src=ast.literal_eval(node.value)
def rep(a,b):
    global src
    assert src.count(a)==1,(a[:100],src.count(a))
    src=src.replace(a,b)
rep('constexpr int NW = OKC_NW, NT = NW * 32;', '''#ifndef OKC_L1
#define OKC_L1 1
#endif
#define WGS_L1 (OKC_L1 && OKC_WGS)
constexpr int NW = OKC_NW, NT = NW * 32;
static_assert(!WGS_L1 || NT == 512, "L1 requires four complete consumer warpgroups");
static_assert(512 * 112 + 128 * 32 == 640 * 96, "L1 redistributes, never overcommits registers");''')
rep('constexpr int NTL = NT + (OKC_WGS ? 64 : OKC_WS ? 32 : 0);',
    'constexpr int NTL = NT + (OKC_WGS ? (WGS_L1 ? 128 : 64) : OKC_WS ? 32 : 0);')
rep('''#if OKC_WGS
  if (tid >= NT) {
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    return;
  }
#endif''','''#if OKC_WGS
  if (tid >= NT) {
#if WGS_L1
    // Uniform across the FULL producer warpgroup, before the two warp roles split.
    asm volatile("setmaxnreg.dec.sync.aligned.u32 32;" ::: "memory");
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else if (tid < NT + 64) wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else { WS_JOIN(); }  // padding warps remain alive through the routing/gather join
#else
    if (tid < NT + 32) wgs_wproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
    else wgs_xproducer(p, S, reinterpret_cast<unsigned char*>(xbuf));
#endif
    return;
  }
#if WGS_L1
  // Four complete consumer WGs; at most8192 registers requested, exactly the
  // number donated above. Launch allocation remains96 registers at640 threads.
  asm volatile("setmaxnreg.inc.sync.aligned.u32 112;" ::: "memory");
#endif
#endif''')
a=src.index('    const unsigned char* pc = pc0 + ws * UNIT_BYTES;',src.index('void wgs_seg('))
b=src.index('#pragma unroll\n      for (int h = 0; h < 4 / GS;',a)
old=src[a:b]
new='''    const unsigned char* pc = pc0 + ws * UNIT_BYTES;
#if WGS_L1
    // Capture a complete stage:16 packed-code +8 packed-scale registers/thread.
    // No decoding or x wait until the weights slot has been returned.
    uint4 cw[4]; uint2 sc[4];
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      cw[lg] = *reinterpret_cast<const uint4*>(pc + ((lg * 4 + wp) * 32 + lane) * 16);
      sc[lg] = *reinterpret_cast<const uint2*>(pc + 8192 + ((lg * 4 + wp) * 8 + g) * 8);
    }
    uint32_t ready = 0;
#pragma unroll
    for (int lg = 0; lg < 4; ++lg)
      ready ^= cw[lg].x ^ cw[lg].y ^ cw[lg].z ^ cw[lg].w ^ sc[lg].x ^ sc[lg].y;
    // p.zero is set to0 by the unchanged host launcher, opaque to ptxas.
    // The FULL warp mask depends on EVERY load: a real instruction operand,
    // not empty inline-asm inputs that ptxas can erase. Each lane waits for all
    // its loads before the warp rendezvous; lane0 then releases one of16 credits.
    const uint32_t mask = 0xffffffffu ^ (ready & p.zero);
    asm volatile("bar.warp.sync %0;" :: "r"(mask) : "memory");
    if (lane == 0) mbar_arrive1(S.bars + BW_E + ws);
    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);
    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));
#pragma unroll
    for (int lg = 0; lg < 4; ++lg) {
      const uint4 cwg = cw[lg];
      const uint2 scg = sc[lg];
#else
'''+old[len('    const unsigned char* pc = pc0 + ws * UNIT_BYTES;\n'):]+'''#endif // WGS_L1: the decode/MMA sequence below is unchanged
'''
src=src[:a]+new+src[b:]
# Fix an existing declaration-order issue for strict offline/front-end builds.
# This is a declaration only; it does not alter arithmetic or generated code.
anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
rep(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'+anchor)
lines=base.splitlines(True)
text=''.join(lines[:node.lineno-1])+'CUDA_SRC = '+repr(src)+'\n'+''.join(lines[node.end_lineno:])
text=text.replace('okc_moe_w4a16_v216x_pair1dq8_', 'okc_moe_l1_v216x_pair1dq8_')
ast.parse(text)
out=ROOT/'experiments/boards/moe/moe_l1.py';out.write_bytes(text.encode())
(ROOT/'.l1_offline/l1.cu').write_text(src)
print(out,len(text.encode()),hashlib.sha256(text.encode()).hexdigest())
