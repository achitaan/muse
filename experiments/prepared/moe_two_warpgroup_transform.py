"""Pure source transform for a two-consumer-group, two-CTA WGS design."""
import re


def transform(original):
    source=original;ledger=[]
    def replace(before,after,count=1):
        nonlocal source
        positions=[m.start() for m in re.finditer(re.escape(before),source)]
        assert len(positions)==count,(before,len(positions),count)
        for pos in reversed(positions):
            ledger.append(dict(position=pos,before=before,after=after))
            source=source[:pos]+after+source[pos+len(before):]
    def function(marker):
        start=source.index(marker)
        while True:
            op=source.index('{',start)
            if ';' not in source[start:op]:break
            start=source.index(marker,start+len(marker))
        depth=0
        token=r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]'
        for match in re.finditer(token,source[op:]):
            s=match[0]
            if s=='{':depth+=1
            elif s=='}':
                depth-=1
                if depth==0:return source[start:op+match.end()]
        raise AssertionError(marker)

    replace('#define OKC_FT_T 64','#define OKC_FT_T 32')
    replace('#define OKC_WGSNX 3','#define OKC_WGSNX 2')
    replace('UNIT_BYTES = UNIT_CODES + UNIT_SC','UNIT_BYTES = 2 * 9216')
    replace('constexpr int WGS_SLOTS = 160;','constexpr int WGS_SLOTS = 272;')
    replace('WGS_SLOTS * 4','WGS_SLOTS * 2',7)
    replace('pd.slot * 4 + wg','pd.slot * 2 + wg',2)
    replace('slot * 4 + wg','slot * 2 + wg')
    replace('const int ubl = pb ? 4 : 5;','const int ubl = pb ? 5 : 6;',2)
    replace('constexpr int UBL = PB ? 4 : 5;','constexpr int UBL = PB ? 5 : 6;',3)
    replace('ub * 16 + wg * 4 + wp','ub * 8 + wg * 4 + wp')
    replace('(u & ((1 << UBL) - 1)) * 16 + wg * 4 + wp','(u & ((1 << UBL) - 1)) * 8 + wg * 4 + wp')
    replace('(ub >> 1)','(ub >> 2)',2)
    replace('wgs_cost(0, S.jC[j]) << 9','wgs_cost(0, S.jC[j]) << 10')
    replace('wgs_cost(1, S.jC[j]) << 8','wgs_cost(1, S.jC[j]) << 9')
    replace('(9 - pb)','(10 - pb)',4)

    # Remove impossible NTT5..8 variants, including the unused older fat family.
    for begin,end in (('template <> struct WG<5> {','// one 36 KB k-stage of a fat unit'),
                      ('template <> struct WGSMMA<5> {','// two producer warps:')):
        a=source.index(begin);b=source.index(end,a);replace(source[a:b],'')
    for name in ('fat_compute_wg','fat_compute'):
        a=source.index('          case 5: '+name+'<5>');b=source.index('\n        }',a)
        replace(source[a:b],'          default: __trap(); break;')
    a=source.index('      case 5: wgs_seg<PB, 5>');b=source.index('\n    }',a)
    replace(source[a:b],'      default: __trap(); break;')

    gather=function('__device__ void wgs_gather(')
    changed=gather.replace('static_assert(NT * 8 == HD,','static_assert(NT * 16 == HD,')
    changed=changed.replace('  int j = 0;\n','')
    a=changed.index('  for (int qb =');b=changed.index('\n  CTA_SYNC();',a)
    loop=re.sub(r'\btid\b','vtid',changed[a:b])
    changed=changed[:a]+'  #pragma unroll 1\n  for (int turn = 0; turn < 2; ++turn) {\n    const int vtid = tid + turn * NT;\n    int j = 0;\n'+loop+'\n  }'+changed[b:]
    replace(gather,changed)

    # Full4096-column FP32 norm reduction with256physical consumers. Each
    # thread contributes two old logical chunks, then writes both chunks.
    a=source.index('  // item = one token\'s whole row')
    b=source.index('\n#elif OKC_FINCOL',a)
    old_final=source[a:b]
    final=old_final.replace('static_assert(HD == NT * 8,','static_assert(HD == NT * 16,')
    start=final.index('          const float4* src =')
    stop=final.index('\n        }',start)
    summation=final[start:stop]
    assert 'ps[sl] =' in summation
    summation=summation.replace(' + tid * 2;',' + (tid + turn * NT) * 2;').replace('ps[sl] =','ps[sl] +=')
    final=final[:start]+'          #pragma unroll\n          for (int turn = 0; turn < 2; ++turn) {\n'+summation+'\n          }'+final[stop:]
    start=final.index('      float acc[8];');stop=final.index('\n      CTA_SYNC();',start)
    stores=final[start:stop]
    stores=stores.replace(' + tid * 2;',' + vtid * 2;').replace(' + tid * 8)', ' + vtid * 8)')
    final=final[:start]+'      #pragma unroll 1\n      for (int turn = 0; turn < 2; ++turn) {\n        const int vtid = tid + turn * NT;\n'+stores+'\n      }'+final[stop:]
    replace(old_final,final)

    old_range=function('__device__ __forceinline__ void wgs_range(')
    helper='''// Exact integer floor: FP32 is only an initial estimate, never the answer.
// The new geometry can exceed uint32 products. Preserve the full numerator.
__device__ __forceinline__ uint32_t w2g_floor(unsigned long long n, uint32_t d) {
  uint32_t q = __float2uint_rz(__fdividef((float)n, (float)d));
  while ((unsigned long long)q * d > n) --q;
  while ((unsigned long long)(q + 1u) * d <= n) ++q;
  return q;
}
'''
    new_range=old_range.replace('tot * (uint32_t)c / (uint32_t)G','w2g_floor((unsigned long long)tot * (uint32_t)c, (uint32_t)G)').replace('tot * (uint32_t)(c + 1) / (uint32_t)G','w2g_floor((unsigned long long)tot * (uint32_t)(c + 1), (uint32_t)G)')
    replace(old_range,helper+new_range)
    old_owner=function('__device__ __forceinline__ int wgs_cta_of(')
    prefix=old_owner[:old_owner.index('  // largest c')]
    new_owner=prefix+'''  // floor(tot*c/G)<xe iff tot*c<xe*G. Largest integer c is
  // floor((xe*G-1)/tot), clamped to the last CTA. Exact wide arithmetic.
  const uint32_t c = w2g_floor((unsigned long long)xe * (uint32_t)G - 1ull, tot);
  return min((int)c, G - 1);
}'''
    replace(old_owner,new_owner)
    old_ranges=function('__device__ __noinline__ void wgs_ranges(')
    replace(old_ranges,old_ranges.replace('  wgs_prefix(S, nj);',
        '  if (nj == 0) { for (int k = 0; k < 4; ++k) S.wgr[k] = 0; return; }\n  wgs_prefix(S, nj);'))
    replace('__launch_bounds__(NTL, 1)','__launch_bounds__(NTL, 2)')
    replace('static int g_okc_pdl = 0;', 'static int g_w2g_capacity = 0;\nstatic int g_okc_pdl = 0;')
    replace('  okc::Params p;',
            '  TORCH_CHECK(g_w2g_capacity > 0 && grid <= g_w2g_capacity && grid < okc::WGS_SLOTS, "unsafe WGS2 cooperative grid");\n  okc::Params p;')
    resources='''
// Static resources only, prepared before timed calls; no data-dependent tuning.
std::vector<int64_t> okc_w2g_resources() {
  cudaFuncAttributes a{}, b{};
  const int sm = (int)okc_smem_bytes();
  TORCH_CHECK(cudaFuncSetAttribute(okc::moe_fused<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess, "WGS2 shared opt-in failed");
  TORCH_CHECK(cudaFuncSetAttribute(okc::moe_fused<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm) == cudaSuccess, "WGS2 shared opt-in failed");
  TORCH_CHECK(cudaFuncGetAttributes(&a, okc::moe_fused<false>) == cudaSuccess, "WGS2 resource query failed");
  TORCH_CHECK(cudaFuncGetAttributes(&b, okc::moe_fused<true>) == cudaSuccess, "WGS2 resource query failed");
  int oa = 0, ob = 0, dev = 0; cudaDeviceProp prop{};
  TORCH_CHECK(cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess, "WGS2 device query failed");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&oa, okc::moe_fused<false>, okc::NTL, sm) == cudaSuccess, "WGS2 occupancy failed");
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ob, okc::moe_fused<true>, okc::NTL, sm) == cudaSuccess, "WGS2 occupancy failed");
  g_w2g_capacity = prop.multiProcessorCount * (oa < ob ? oa : ob);
  return {(int64_t)okc::NTL, (int64_t)sizeof(okc::Smem), (int64_t)sm,
    (int64_t)a.numRegs, (int64_t)a.localSizeBytes, (int64_t)oa,
    (int64_t)b.numRegs, (int64_t)b.localSizeBytes, (int64_t)ob,
    (int64_t)prop.multiProcessorCount, (int64_t)prop.regsPerMultiprocessor,
    (int64_t)prop.sharedMemPerMultiprocessor, (int64_t)g_w2g_capacity};
}
'''
    # Record the appended helper in the same reversible edit ledger.
    ledger.append(dict(position=len(source),before='',after=resources));source+=resources
    replace('namespace okc {','namespace okc_w2g {')
    positions=len(re.findall(re.escape('okc::'),source));replace('okc::','okc_w2g::',positions)
    recovered=source
    for item in reversed(ledger):
        p,b,a=item['position'],item['before'],item['after']
        assert recovered[p:p+len(a)]==a
        recovered=recovered[:p]+b+recovered[p+len(a):]
    assert recovered==original
    return source,ledger
