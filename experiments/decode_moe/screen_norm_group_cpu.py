"""One in-CTA norm aggregation prototype; no GPU payload/admission."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/decode-moe-plan-20261030/norm-group'

def main():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
    vals={n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse(raw).body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')in ('CUDA_SRC','TINY_DEFINES','TINY2_DEFINES')}
    parent=vals['CUDA_SRC'];src=parent;edits=[]
    def edit(old,new):
        nonlocal src
        assert src.count(old)==1,old[:80]
        src=src.replace(old,new,1);edits.append((old,new))
    edit('struct Smem {','static_assert(OKC_TINY && OKC_WS && !SKDYN && SK_T == 8, "norm grouping requires static tiny WS");\nstruct Smem {\n  float norm_group[JMAX * 8];')
    for begin,end in [('__device__ void skinny_ring_ws(', '__device__ __forceinline__ void ws_issue('),('__device__ void ws_helper(', '__device__ __forceinline__ void skinny_split(')]:
        a=src.index(begin);b=src.index(end,a);old=src[a:b]
        target='if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);'
        replacement='''if (gg == 0 && tk < cnt) {
          // Match global FP32 atomic input FTZ before shared aggregation.
          const float nsq = (__float_as_uint(sq) & 0x7f800000u) == 0 ? 0.f : sq;
          atomicAdd(&S.norm_group[j * 8 + tk], nsq);
        }'''
        assert old.count(target)==1
        edit(old,old.replace(target,replacement,1))
    anchor='    S.wsr[2] = muldiv_g(nB, c, G, rG); S.wsr[3] = muldiv_g(nB, c + 1, G, rG);'
    edit(anchor,anchor+'''
    if (S.wsr[2] < S.wsr[3]) {
      const int lo = (S.wsr[2] / RB_B) * 8;
      const int hi = ((S.wsr[3] - 1) / RB_B + 1) * 8;
      for (int i = lo; i < hi; ++i) S.norm_group[i] = 0.f;
    } // tid0 publishes initialization at the existing first WS_JOIN
''')
    anchor='  WS_JOIN();  // the helper\'s last epilogue stores are visible'
    edit(anchor,anchor+'''
  const int ng_b0 = S.wsr[2], ng_b1 = S.wsr[3];
  const int ng_lo = (ng_b0 / RB_B) * 8;
  const int ng_hi = ng_b0 < ng_b1 ? ((ng_b1 - 1) / RB_B + 1) * 8 : ng_lo;
  for (int i = ng_lo + tid; i < ng_hi; i += NT) {
    const int j = i >> 3, tk = i & 7;
    const float sum = S.norm_group[i];
    if (tk < S.jC[j] && sum != 0.f) atomicAdd(&ss_cur()[S.jS[j] + tk], sum);
  }
''')
    inverse=src
    for old,new in reversed(edits):inverse=inverse.replace(new,old,1)
    assert inverse==parent
    src=src.replace('namespace okc','namespace norm_group').replace('okc::','norm_group::').replace('moe_fused','moe_group_fused')
    OUT.mkdir(exist_ok=True);(OUT/'candidate.cu').write_bytes(src.encode())
    # Disjoint static unit ranges; each unit contributes once,then each CTA
    # publishes at most one sum per route/job instead of one per16columns.
    models=[]
    for jobs in (1,2,4,8,16,32):
        spans=[(jobs*256*c//132,jobs*256*(c+1)//132)for c in range(132)]
        counts=[0]*jobs
        for a,b in spans:
            for j in range(jobs):
                if max(a,j*256)<min(b,(j+1)*256):counts[j]+=1
        assert all(0<c<=132 for c in counts)
        models.append(dict(jobs=jobs,old_atomics_per_route=256,new_max_atomics_per_route=max(counts)))
    sys.path.insert(0,str(ROOT.parent/'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv
    results={}
    for name in ('TINY','TINY2'):
        for prmt in (False,True):
            label=name+('_PRMT'if prmt else '_LUT')
            flags=list(vals[name+'_DEFINES'])+['-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']+(['-DOKC_PRMTSC=1']if prmt else [])
            r=nv.NVRTC().compile(src.split('#include <torch/extension.h>',1)[0].encode(),['&norm_group::moe_group_fused<false>'],flags,tags=True,lineinfo=False)
            (OUT/(label+'.log')).write_bytes(r['log'].encode());assert r['rc']==0,r['log'][-3000:]
            results[label]=nv.parse_ptxas(r['log']);print(label,json.dumps(results[label]),flush=True)
    clean=all(k['stack_frame']==k['spill_stores']==k['spill_loads']==0 for ks in results.values()for k in ks)
    report=dict(CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),exact_inverse=True,models=models,compiler=results,zero_stack_spills=clean,
      costs='2816extra shared bytes;tid0 initializes intersecting job buckets before existing WS_JOIN;shared atomics;sparse publication after final WS_JOIN. No added barrier;original finalizer and ybuf traffic retained.',
      precision='Unit norm addends flush subnormals before shared atomic to match original global atomic FTZ;association changes need approved norm bounds.',
      GPU_calls=0,admitted_for_GPU=False,remaining='Numerical/order,helper-tail/lifetime and resource proofs;exact source packaging and H200 gates remain required.')
    (OUT/'cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())

if __name__=='__main__':main()
