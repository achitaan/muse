"""MAIN native N96 attempt, final N80 fallback on exact moe_b13 without L1."""
import ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'experiments/boards/moe/moe_b13.py').read_bytes().decode()
node=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
src=ast.literal_eval(node.value)
def rep(a,b):
    global src
    assert src.count(a)==1,(a[:100],src.count(a))
    src=src.replace(a,b)
def section(name,func):
    global src
    a=src.index(name);b=src.index('\n}',a)+2
    old=src[a:b];new=func(old);assert old!=new,name;src=src[:a]+new+src[b:]
rep('constexpr int HD = 4096, ID = 4096;', '''#ifndef OKC_L2
#define OKC_L2 1
#endif
#ifndef OKC_L2_CAP
#define OKC_L2_CAP 80
#endif
#ifndef OKC_L2_GS
#define OKC_L2_GS 1
#endif
#define WGS_L2 (OKC_L2 && OKC_WGS)
static_assert(OKC_L2_CAP == 64 || OKC_L2_CAP == 80 || OKC_L2_CAP == 96, "L2 cap");
static_assert(OKC_L2_GS == 1 || OKC_L2_GS == 2, "L2 wide commit grouping");
constexpr int HD = 4096, ID = 4096;''')
rep('    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;', '''#if WGS_L2
    const bool wide = tot > FT_T && tot <= OKC_L2_CAP;
    const unsigned wm = __ballot_sync(0xffffffffu, wide);
    if (lane == 0) S.preissuedB = wm != 0;  // WGS-only alias: launch-wide x layout
    const int nfj = wide ? 1 : (tot > SK_T ? (tot + FT_T - 1) / FT_T : 0);
#else
    const int nfj = tot > SK_T ? (tot + FT_T - 1) / FT_T : 0;
#endif''')
rep('      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T; S.jC[fj + q] = min(FT_T, tot - q * FT_T);', '''      S.jE[fj + q] = lane; S.jS[fj + q] = off + q * FT_T;
#if WGS_L2
      S.jC[fj + q] = wide ? tot : min(FT_T, tot - q * FT_T);
#else
      S.jC[fj + q] = min(FT_T, tot - q * FT_T);
#endif''')
rep('constexpr int WGS_NT = FT_T / 8;', 'constexpr int WGS_NT = WGS_L2 ? OKC_L2_CAP / 8 : FT_T / 8;')
rep('constexpr int WGS_XS = WGS_NT * 4096;', 'constexpr int WGS_XS = (FT_T / 8) * 4096;')
rep('static_assert(BX_E + WGS_NX <= 24, "WGS barriers");', '''static_assert(BX_E + WGS_NX <= 24, "WGS barriers");
constexpr int WGS_XSW = 12 * 4096;
static_assert(3 * WGS_XS == 2 * WGS_XSW, "L2 x layouts have identical footprint");
__device__ __forceinline__ bool wgs_wide_x(const Smem& S) {
#if WGS_L2 && OKC_L2_CAP > 64
  return S.preissuedB != 0;
#else
  return false;
#endif
}
__device__ __forceinline__ int wgs_xslot(const Smem& S, int seq) {
  return wgs_wide_x(S) ? (seq & 1) : seq % WGS_NX;
}
__device__ __forceinline__ uint32_t wgs_xphase(const Smem& S, int seq) {
  return wgs_wide_x(S) ? ((uint32_t)seq >> 1) & 1u : (uint32_t)(seq / WGS_NX) & 1u;
}''')
rep('''  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);''', '''#if WGS_L2
  const unsigned long long costs = pb ? WGS_PB : WGS_PA;
  if (cnt > FT_T) {
    // Preserve baseline total cost and every narrow job's cost coordinate.
    // The native job replaces64+tail: this keeps narrow CTA/K splits identical.
    return (int)((costs >> 56) & 255ull) + (int)((costs >> (8 * (((cnt - FT_T + 7) >> 3) - 1))) & 255ull);
  }
  return (int)((costs >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
#else
  return (int)(((pb ? WGS_PB : WGS_PA) >> (8 * (((cnt + 7) >> 3) - 1))) & 255ull);
#endif''')
# Native instructions: preserve the existing N8..64 definitions verbatim.
def mma(nt):
    out=f'template <> struct WGSMMA<{nt}> {{\n'
    for fn,sd,cstr in [('mma0',0,'=f'),('mma',1,'+f')]:
        outputs=', '.join('%'+str(i) for i in range(4*nt));areg=', '.join('%'+str(4*nt+i) for i in range(4))
        regs=', '.join(f'"{cstr}"(d[{i//4}][{i%4}])' for i in range(4*nt))
        out+=f'''  static __device__ __forceinline__ void {fn}(float (&d)[WGS_NT][4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{8*nt}k16.f32.bf16.bf16 {{{outputs}}}, {{{areg}}}, %{4*nt+4}, {sd}, 1, 1, 0;\\n"
                 : {regs} : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc));
  }}
'''
    return out+'};\n'
anchor='// two producer warps: the weights warp streams every stage\'s 36 KB'
extra='\n'.join(f'#if WGS_L2 && OKC_L2_CAP >= {8*n}\n'+mma(n)+'#endif' for n in range(9,13))
rep(anchor,extra+'\n'+anchor)
section('void wgs_xproducer(',lambda t:t.replace('  int xs = 0, n = 0;',
    '  const int nx = wgs_wide_x(S) ? 2 : WGS_NX;\n  const int xpitch = wgs_wide_x(S) ? WGS_XSW : WGS_XS;\n  int xs = 0, n = 0;').replace('n >= WGS_NX','n >= nx').replace('xs * WGS_XS','xs * xpitch').replace('++xs == WGS_NX','++xs == nx'))
rep('  constexpr int GS = OKC_WGSGS;', '  constexpr int GS = (WGS_L2 && NTT > 8) ? OKC_L2_GS : OKC_WGSGS;')
rep('    const int ws = seq % WGS_NS, xs = seq % WGS_NX;', '    const int ws = seq % WGS_NS, xs = wgs_xslot(S, seq);')
rep('    wgs_wait(S.bars + BX_F + xs, (uint32_t)(seq / WGS_NX) & 1u, 4000000 + seq);',
    '    wgs_wait(S.bars + BX_F + xs, wgs_xphase(S, seq), 4000000 + seq);')
rep('    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * (WGS_XS >> 4));',
    '    const uint64_t xd = xd0 + (uint64_t)(uint32_t)(xs * ((wgs_wide_x(S) ? WGS_XSW : WGS_XS) >> 4));')
# Narrow deferred flush remains after the same8 k16 steps. Wide segments drain
# earlier (below), before their accumulator becomes live.
rep('if (FIRST && ci == OKC_WGSFCI && pd.kind)',
    'if (FIRST && NTT <= 8 && ci == ((OKC_WGSFCI + 1) * OKC_WGSGS / GS - 1) && pd.kind)')
# Drain the previous segment before creating wide accumulators. This avoids
# carrying40 live result registers through the deferred epilogue in N72/N80.
rep('  const uint32_t k16c = 16u + p.zero;',
    '  if constexpr (WGS_L2 && NTT > 8) { if (pd.kind) wgs_flush<PB>(p, S, pd); }\n  const uint32_t k16c = 16u + p.zero;')
rep('      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;',
    '#if WGS_L2\n      case 8: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n'+
    '\n'.join(f'#if OKC_L2_CAP >= {8*n}\n      case {n}: wgs_seg<PB, {n}>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n#endif' for n in range(9,13))+
    '\n      default: __trap(); break;\n#else\n      default: wgs_seg<PB, 8>(p, S, ring, u, s0, s1, slot, seq, pd); break;\n#endif')
anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase'
rep(anchor,'__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'+anchor)
lines=base.splitlines(True)
text=''.join(lines[:node.lineno-1])+'CUDA_SRC = '+repr(src)+'\n'+''.join(lines[node.end_lineno:])
text=text.replace('WGS_FIXW = 4096 ', 'WGS_FIXW = 6144 ')
text=text.replace('okc_moe_w4a16_v216x_pair1dq8_', 'okc_moe_l2_v216x_pair1dq8_')
ast.parse(text)
assert 'setmaxnreg' not in src and 'uint4 cw[4]' not in src
p=ROOT/'experiments/boards/moe/moe_l2.py';p.write_bytes(text.encode())
(ROOT/'.l2_offline/l2.cu').write_text(src)
print(p,len(text.encode()),hashlib.sha256(text.encode()).hexdigest())
