"""Uncommitted, CPU-only route study while the frozen FP32 trial waits."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())

HELPER=r'''
#if OKC_TINY && FAST1_ON && !OKC_SPLIT && !OKC_WGS
template <bool SECOND>
__device__ __forceinline__ void route_small(const Params& p, Smem& S) {
  const int tid=threadIdx.x, lane=tid&31, E=p.M*8;
  if (tid<32) {
    S.cnt[lane]=0;
    int e0=lane<E ? __ldg(p.topk_ids+lane)-p.lo : -1;
    if (e0<0 || e0>=p.nloc) e0=-1;
    if (lane<E) S.eid[lane]=(signed char)e0;
    const unsigned peers0=__match_any_sync(0xffffffffu,e0);
    int e1=-1;
    unsigned peers1=0;
    if constexpr (SECOND) {
      e1=lane+32<E ? __ldg(p.topk_ids+lane+32)-p.lo : -1;
      if (e1<0 || e1>=p.nloc) e1=-1;
      if (lane+32<E) S.eid[lane+32]=(signed char)e1;
      peers1=__match_any_sync(0xffffffffu,e1);
    }
    __syncwarp();
    if (e0>=0 && lane==__ffs(peers0)-1) S.cnt[e0]=__popc(peers0);
    __syncwarp();
    const int first_count=S.cnt[lane];
    if constexpr (SECOND) {
      // Capture all first counts before any second-slice writer changes them.
      __syncwarp();
      if (e1>=0 && lane==__ffs(peers1)-1) S.cnt[e1]+=__popc(peers1);
      __syncwarp();
    }
    const int tot=lane<p.nloc ? S.cnt[lane] : 0;
__JOB_PREFIX__
    early_issue(p,S); // FAST1_ON makes this exact retained helper a no-op.
    if (lane<E) {
      if (e0>=0) {
        const int q=S.off[e0]+__popc(peers0&((1u<<lane)-1u));
        S.ent[q]=(short)lane; S.pos[lane]=(short)q;
      } else S.pos[lane]=-1;
    }
    if constexpr (SECOND) {
      // Full-mask shuffle stays warp-uniform, even for invalid route lanes.
      const int first_e1=__shfl_sync(0xffffffffu,first_count,e1&31);
      if (lane+32<E) {
        if (e1>=0) {
          const int q=S.off[e1]+first_e1+__popc(peers1&((1u<<lane)-1u));
          S.ent[q]=(short)(lane+32); S.pos[lane+32]=(short)q;
        } else S.pos[lane+32]=-1;
      }
    }
  }
  CTA_SYNC();
}
#endif
'''


def main():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
    tree=ast.parse(raw)
    parent=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
    assert hashlib.sha256(parent.encode()).hexdigest()=='1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c'
    start=parent.index('    int incl = tot;',parent.index('__device__ void route('))
    end=parent.index('\n  }\n  CTA_SYNC();\n  early_issue',start)
    jobs=parent[start:end]
    offsets='''    int o2 = off;
#pragma unroll
    for (int w = 0; w < NW; ++w) { S.wc[w][lane] = o2; o2 += c[w]; }
'''
    assert jobs.count(offsets)==1
    jobs=jobs.replace(offsets,'')
    helper=HELPER.replace('__JOB_PREFIX__',jobs)
    anchor='__device__ void route(const Params& p, Smem& S) {'
    branch='''
#if OKC_TINY && FAST1_ON && !OKC_SPLIT && !OKC_WGS
  if (p.M<=8) {
    if (p.M<=4) route_small<false>(p,S);
    else route_small<true>(p,S);
    return;
  }
#endif
'''
    source=parent.replace(anchor,helper+'\n'+anchor+branch)
    assert source.count(helper)==1
    restored=source.replace(helper+'\n'+anchor+branch,anchor)
    assert restored==parent
    source=source.replace('namespace okc {','namespace okc_route_small {').replace('okc::','okc_route_small::').replace('// namespace okc','// namespace okc_route_small')
    output=Path(__file__).resolve().parent
    for name,text in [('full.cu',source),('device.cu',source.split('#include <torch/extension.h>',1)[0]),('baseline.cu',parent.split('#include <torch/extension.h>',1)[0])]:
        (output/name).write_bytes(text.encode())
    report=dict(full_CUDA_sha256=hashlib.sha256(source.encode()).hexdigest(),
                inserted_helper_is_only_device_edit_except_namespace=True,
                original_job_prefix_and_fat_queue_text_reused_except_unused_warp_offsets=True,
                original_route_retained_as_fallback=True,eligibility='TINY, FAST1, !SPLIT, !WGS and M<=8',
                no_weight_or_matmul_or_epilogue_or_finalizer_change=True,
                CTA_routing_barriers_old=4,CTA_routing_barriers_new=1,
                route_entry_order='expert ascending, then original entry index ascending',
                GPU_jobs=0,kernel_execution=False,submission_built=False)
    (output/'source-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
