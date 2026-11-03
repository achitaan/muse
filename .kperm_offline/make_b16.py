"""Reimplement KPERM3 in b15, never substitute the reference submission."""
import ast,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=(ROOT/'experiments/boards/moe/moe_b15.py').read_bytes().decode()
node=next(n for n in ast.parse(base).body if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='CUDA_SRC' for v in n.targets))
src=ast.literal_eval(node.value)
def rep(a,b):
    global src
    assert src.count(a)==1,(a[:100],src.count(a))
    src=src.replace(a,b)
rep('#ifndef OKC_FINREG', '''// KPERM3 scale-sharing idea by alygg77 (Apache-2.0), published in
// okc-muse-spark-kernels/kernels/muse_moe_w4a16.py. Local integration keeps
// our dq4 packing, Marlin boundaries, HINT, ERL, FINREG and empty-range fix.
#ifndef OKC_KPERM
#define OKC_KPERM 3
#endif
static_assert(OKC_KPERM == 0 || OKC_KPERM == 3, "KPERM supports legacy0 or KPERM3");
#ifndef OKC_FINREG''')
rep('          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);',
'''#if OKC_KPERM == 3
          // Reimplement alygg77's KPERM3 activation gather. Each destination
          // vector is four intact BF16 pairs, one from each64-column band.
          const int group = tid & 31;
          const int pair = ((group & 6) << 2) | ((group >> 2) & 6) | (group & 1);
          const uint32_t* row = reinterpret_cast<const uint32_t*>(
              p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD + (tid >> 5) * 256);
          v[k] = make_uint4(__ldg(row + pair), __ldg(row + pair + 32),
                            __ldg(row + pair + 64), __ldg(row + pair + 96));
#else
          v[k] = __ldg(reinterpret_cast<const uint4*>(p.x + (size_t)(S.ent[S.jS[j] + r] >> 3) * HD) + tid);
#endif''')
rep('    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);',
'''#if OKC_KPERM == 3
    // One8-byte load per stage: [j=0..3][gate/up row] for block4*t+j.
    // Invariant across lg, allowing the four PRMT/IMAD pairs to be CSE'd.
    const uint2 stage_sc = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 32 + lane) * 8);
#else
    uint2 scn = *reinterpret_cast<const uint2*>(pc + 8192 + (wp * 8 + g) * 8);
#endif''')
rep('      const uint2 scg = scn;', '''#if OKC_KPERM == 3
      const uint2 scg = stage_sc;
#else
      const uint2 scg = scn;
#endif''')
rep('        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);',
'''#if OKC_KPERM == 0
        scn = *reinterpret_cast<const uint2*>(pc + 8192 + (((lg + 1) * 4 + wp) * 8 + g) * 8);
#endif''')
rep('          uint32_t u = cwn.x ^ cwn.y ^ cwn.z ^ cwn.w ^ scn.x ^ scn.y;',
'''#if OKC_KPERM == 3
          // Preserve our ERL release at lg2. All scale words have already
          // been consumed by lg0; only the last code load is outstanding.
          // Do not extend raw scale-register lifetimes solely for this check.
          uint32_t u = cwn.x ^ cwn.y ^ cwn.z ^ cwn.w;
#else
          uint32_t u = cwn.x ^ cwn.y ^ cwn.z ^ cwn.w ^ scn.x ^ scn.y;
#endif''')
lines=base.splitlines(True)
text=''.join(lines[:node.lineno-1])+'CUDA_SRC = '+repr(src)+'\n'+''.join(lines[node.end_lineno:])
text=text.replace('okc_moe_w4a16_v216x_pair1dq8_', 'okc_moe_b16_kperm3_')
# Python controls the corresponding packing AND define, so a supported mode
# change is made before prepare(), never by changing only the CUDA flag.
text=text.replace('MAIN_NS = 3', '''# KPERM3 algorithm/design: alygg77, Apache-2.0. This is a local port into b15,
# not their submission. Set KPERM=0 BEFORE prepare for the legacy WGS layout.
KPERM = 3
MAIN_NS = 3''',1)
anchor='    defines = tuple(MAIN_DEFINES if defines is None else defines)'
assert text.count(anchor)==1
text=text.replace(anchor,anchor+'''
    if "-DOKC_WGS=1" in defines:
        assert KPERM in (0, 3), "KPERM must be0 or3 before prepare"
        mode = f"-DOKC_KPERM={KPERM}"
        specified = [d for d in defines if d.startswith("-DOKC_KPERM=")]
        assert all(d == mode for d in specified), "KPERM compile flag disagrees with prepared layout"
        if not specified:
            defines = defines + (mode,)''')
helpers='''# KPERM3 technique credited to alygg77 (Apache-2.0). Independently expressed
# as a packed-BYTE transpose: adjacent FP4 nibbles never separate, so no nibble
# unpack/requantization is needed for this additional K permutation.
def _wgs_kperm_codes(codes):
    rows, half_k = codes.shape
    assert codes.dtype == torch.uint8 and half_k % 128 == 0
    # Original byte axes inside256 K: [t4,j4,lg4,pair2]. New physical axes:
    # [lg4,j4,pair2,t4]. Every original byte occurs exactly once.
    return (codes.reshape(rows, half_k // 128, 4, 4, 4, 2)
            .permute(0, 1, 4, 3, 5, 2).contiguous().view(rows, half_k))


def _wgs_kperm_index(size, device):
    """Physical K position -> original K; independent256-column permutations."""
    assert size % 256 == 0
    p = torch.arange(size, device=device)
    return ((p & ~255) | ((p & 6) << 5) | (p & 48) | ((p >> 4) & 12)
            | ((p >> 2) & 2) | (p & 1))


def _wgs_kperm_rows(rows):
    """Reorder gate/up TOGETHER: A output position P holds inter[perm(P)].

    B reads its input columns in perm(P) order. Keeping the original A epilogue
    then directly writes B's desired xB layout; no additional runtime transform.
    """
    planes = rows.view(-1, 2, 8).permute(1, 0, 2).reshape(2, -1)
    index = _wgs_kperm_index(planes.shape[1], rows.device)
    return (planes.index_select(1, index).view(2, -1, 8)
            .permute(1, 0, 2).contiguous().view(-1))


'''
text=text.replace('def _wgs_copy(codes, sf, rows):',helpers+'def _wgs_copy(codes, sf, rows):',1)
anchor='    wn = _pack_codes_nat(codes, rows)'
assert text.count(anchor)==1
text=text.replace(anchor,'''    assert KPERM in (0, 3)
    wn = _pack_codes_nat(_wgs_kperm_codes(codes) if KPERM == 3 else codes, rows)''')
anchor='    s = s.permute(0, 5, 1, 6, 2, 4, 7, 3).contiguous().view(nub, 16, 4, 1024)  # ub, st, q, grp, warp, g, j, h (OKC_WGSSC2)'
assert text.count(anchor)==1
text=text.replace(anchor,'''    if KPERM == 3:
        # Original scales are not numerically changed: [ub,st,q,warp,g,t,j,h],
        # where original16-K block =16*st+4*t+j. One uint2 per lane per stage.
        s = s.permute(0, 5, 1, 2, 4, 6, 7, 3).contiguous().view(nub, 16, 4, 1024)
    else:
'''+anchor.replace('    s =','        s ='))
anchor='''        state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], rowsA) for e in range(lo, lo + cnt)]))'''
assert text.count(anchor)==1
text=text.replace(anchor,'''        wgs_rowsA = _wgs_kperm_rows(rowsA) if KPERM == 3 else rowsA
        state["w"].append(torch.cat([_wgs_copy(weights["w13"][e], weights["w13_sf"][e], wgs_rowsA) for e in range(lo, lo + cnt)]))''')
# Keep the layout docstring accurate for both packing modes.
fn=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='_wgs_copy')
doc=fn.body[0];parts=text.splitlines(True)
text=''.join(parts[:doc.lineno-1])+'''    """WGS bytes [256-row unit][K256 stage][warpgroup4][9216 bytes].
    Each WG piece has8192 code bytes [lg,warp,lane,j] using OUR dq4 bit encoding,
    then1024 scale bytes. Mode0 uses the original natural K and scale axes
    [lg,warp,g,j,h]. Mode3 permutes K within256, with scales [warp,g,t,j,h].
    prepare() additionally permutes the paired gate/up row order only for A.
    No code or scale value is requantized; all transformations are bijections.
    """
'''+''.join(parts[doc.end_lineno:])
ast.parse(text);assert len(text.encode())<1_000_000
p=ROOT/'experiments/boards/moe/moe_b16.py';p.write_bytes(text.encode())
(ROOT/'.kperm_offline/b16.cu').write_text(src)
print(p,len(text.encode()),hashlib.sha256(text.encode()).hexdigest())
