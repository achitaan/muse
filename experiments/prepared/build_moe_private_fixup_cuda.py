"""Fixed disjoint-partial MAIN CUDA draft, exact inverse ledger, no GPU execution."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
BASE='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
OUT=ROOT/'results/moe-private-fixup-preflight-20261029/v4'


def build():
    raw=(ROOT/'best/moe_verified.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==BASE
    original=next(n.value.value for n in ast.parse(raw).body if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    cuda=original;edits=[]
    def replace(old,new):
        nonlocal cuda
        assert cuda.count(old)==1,(old[:100],cuda.count(old))
        pos=cuda.index(old);cuda=cuda[:pos]+new+cuda[pos+len(old):]
        edits.append(dict(offset=pos,old=old,new=new))
    anchor='// per-stage cost by n-tiles (1..8), phase A and phase B'
    helper='''// Private endpoint values: leading shares reuse the old value area;
// trailing shares use an extra bank after all original values/counters/doneAs.
constexpr size_t WGS_PRIVATE_EXTRA = (size_t)2 * WGS_SLOTS * 4 * WGS_FIXW;
__device__ __forceinline__ float* wgs_private(const Params& p, int pb, int c, bool trailing, int wg, int wt) {
  float* root = trailing ? p.ybuf + (size_t)EMAX * HD + WGS_FIXF +
                          (size_t)pb * WGS_SLOTS * 4 * WGS_FIXW : wgs_fix(p, pb);
  return root + (size_t)(c * 4 + wg) * WGS_FIXW + wt * 4;
}
__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);
'''
    replace(anchor,helper+anchor)
    anchor='#define OKC_WGSREORD 1'
    last_owner='''// Starting at the known first contributor avoids a redundant64-bit
// inverse-owner division while accumulator registers are live. The original
// bound tot*grid<2^32 makes both products below exact unsigned32-bit values.
__device__ __forceinline__ int wgs_last_from_first(const Smem& S, int pb, int u, int first, int G) {
  const int ubl = pb ? 4 : 5, j = u >> ubl, ub = u & ((1 << ubl)-1);
  const uint32_t total = pb ? S.wtotB : S.wtot;
  const uint32_t end = (uint32_t)(pb ? S.wcpB : S.wcp)[j] +
                       (uint32_t)((ub+1)*16) * (uint32_t)wgs_cost(pb,S.jC[j]);
  const uint32_t limit = end * (uint32_t)G;
  int last = first;
  while (last+1 < G && total*(uint32_t)(last+1) < limit) ++last;
  return last;
}
'''
    replace(anchor,last_owner+anchor)
    replace('  short pos[EMAX];   // entry -> pair position, -1 if not local\n};',
            '  short pos[EMAX];   // entry -> pair position, -1 if not local\n'
            '#if OKC_WGS\n  int private_bounds[4]; // phase x endpoint; packed first/last CTA\n#endif\n};')
    old='  if (wt == 0) { while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32); }'
    new='''  if (wt == 0) {
    while (ld_acquire(fc) < 16 - pd.nst) __nanosleep(32);
    // Reuse this WG's scratch only after the original completion wait.
    S.wglast[wg] = S.private_bounds[2*PB + ((pd.u << 4) >= S.wgr[2*PB])];
  }'''
    replace(old,new)
    replace('    S.wglast[wg] = old + pd.nst == 16;',
            '    S.wglast[wg] = old + pd.nst == 16 ?\n'
            '        S.private_bounds[2*PB + ((pd.u << 4) >= S.wgr[2*PB])] : 0;')
    replace('    const int slot = (s1 - s0 == 16) ? -1 : (s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x));',
            '    const int slot = (s1 - s0 == 16) ? -1 : (S.private_bounds[2*PB + (s0 == 0)] & 255);')
    anchor='  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);\n}'
    precompute='''  wgs_range(S, nj, 1, blockIdx.x, gridDim.x, S.wgr[2], S.wgr[3]);
  for (int pb=0; pb<2; ++pb) {
    const int lo=S.wgr[2*pb], hi=S.wgr[2*pb+1];
    S.private_bounds[2*pb]=S.private_bounds[2*pb+1]=0;
    if (lo < hi) {
      for (int side=0; side<2; ++side) {
        const bool needed = side ? (hi & 15) != 0 : (lo & 15) != 0;
        if (needed) {
          const int u = side ? (hi-1)>>4 : lo>>4;
          const int first=wgs_cta_of(S,pb,u<<4,gridDim.x);
          const int last=wgs_last_from_first(S,pb,u,first,gridDim.x);
          S.private_bounds[2*pb+side]=first | (last<<8);
        }
      }
    }
  }
}'''
    replace(anchor,precompute)
    begin=cuda.index('  float* fb = wgs_fix(p, PB) + (size_t)(pd.slot * 4 + wg) * WGS_FIXW + wt * 4;')
    end=cuda.index('  if (wt == 0) *fc = 0;',begin)
    old=cuda[begin:end]
    new='''  const int endpoints = S.wglast[wg];
  const int first = endpoints & 255, last = endpoints >> 8;
  const int ntt = (S.jC[j] + 7) >> 3, r16 = ub * 16 + wg * 4 + wp;
  for (int nt = 0; nt < ntt; ++nt) {
    float a0=0.f,a1=0.f,a2=0.f,a3=0.f;
#pragma unroll 1
    for (int c = first; c <= last; ++c) {
      const float* part = wgs_private(p, PB, c, c == first, wg, wt);
      const float4 q = __ldcg(reinterpret_cast<const float4*>(part + nt*512));
      a0 += q.x; a1 += q.y; a2 += q.z; a3 += q.w;
    }
    const float d4[4] = {a0,a1,a2,a3};
    wgs_epi_tile<PB>(p, S, j, r16, nt, d4);
  }
'''
    replace(old,new)
    begin=cuda.index('    float* fb = wgs_fix(p, PB) + (size_t)(slot * 4 + wg) * WGS_FIXW + wt * 4;')
    end=cuda.index('    // kind 3:',begin)
    old=cuda[begin:end]
    new='''    float* fb = wgs_private(p, PB, blockIdx.x, s0 == 0, wg, wt);
#pragma unroll
    for (int nt = 0; nt < NTT; ++nt)
      asm volatile("st.global.v4.f32 [%0], {%1, %2, %3, %4};\\n" ::"l"(fb + nt * 512),
                   "f"(d[nt][0]), "f"(d[nt][1]), "f"(d[nt][2]), "f"(d[nt][3]) : "memory");
'''
    replace(old,new)
    replace('(size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF,',
            '(size_t)okc::EMAX + (size_t)okc::EMAX * okc::HD + okc::WGS_FIXF + okc::WGS_PRIVATE_EXTRA,')
    replace('"OKC_WGS needs the stream-K fixup area after ybuf (scratch[1])"',
            '"Private WGS needs both endpoint value banks after ybuf (scratch[1])"')
    # Distinct symbol makes actual new-path execution externally observable.
    renamed=cuda.replace('moe_fused','moe_fused_privatefix')
    assert renamed!=cuda
    edits.append(dict(offset=0,kind='symbol',old='moe_fused',new='moe_fused_privatefix'))
    restored=renamed.replace('moe_fused_privatefix','moe_fused')
    for e in reversed(edits[:-1]):
        p=e['offset'];assert restored[p:p+len(e['new'])]==e['new']
        restored=restored[:p]+e['old']+restored[p+len(e['new']):]
    assert restored==original
    assert 'red.global.add.v4.f32' not in renamed
    assert renamed.count('p.doneAs = reinterpret_cast<int*>')==original.count('p.doneAs = reinterpret_cast<int*>')
    report=dict(baseline_sha256=BASE,baseline_CUDA_sha256=hashlib.sha256(original.encode()).hexdigest(),
                CUDA_sha256=hashlib.sha256(renamed.encode()).hexdigest(),inverse_exact=True,
                edits=edits,extra_scratch_bytes=20971520,original_shared_field_offsets_unchanged=True,
                appended_shared_metadata_bytes=16,
                geometry_and_weight_formats_unchanged=True,FP32_reduction_order_changed=True,
                original_publication_counter_and_doneAs_protocol_retained=True,
                requires_separate_scratch_from_incumbent=True,GPU_jobs=0,
                restrictions='Compile and use for EP8 PRMT MAIN only; keep all original fallbacks.')
    report['revision']=4
    report['prior_proxy_failure']='v1:184/768/1216 stack/store/load fields; v2 removes redundant inverse-owner division:88/612/992; v3 shortens reduction live ranges:32/136/304. This revision computes endpoint owners once in range setup, before matrix accumulators are live. Ownership, phase readiness and FP32 contribution order remain unchanged.'
    OUT.mkdir(exist_ok=True,parents=True)
    for name,data in [('baseline.cu',original.encode()),('kernel.cu',renamed.encode()),
                      ('source-audit.json',(json.dumps(report,indent=2)+'\n').encode())]:
        p=OUT/name;assert not p.exists() or p.read_bytes()==data,'Frozen private-fixup collision';p.write_bytes(data)
    print(json.dumps({k:v for k,v in report.items() if k!='edits'}))


if __name__=='__main__':build()
