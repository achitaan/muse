"""Preserve logical K ranges while modeling/rebinding B work to ready CTAs.

An offline prototype only. No Python submission or GPU admission is created.
"""

import ast
import hashlib
import itertools
import json
import random
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/decode-moe-plan-20261030/moe-dynamic-b'


def derive():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
    tree=ast.parse(raw)
    parent=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')=='CUDA_SRC')
    src=parent;edits=[]
    for start,end in [('__device__ void wgs_wproducer(', '// x producer:'),
                      ('__device__ void wgs_xproducer(', "// phase A's x gather")]:
        a=src.index(start);b=src.index(end,a);old=src[a:b];body=old.rstrip('\n')
        early='  if ((threadIdx.x & 31) != 0) return;\n'
        assert body.count(early)==1;body=body.replace(early,'',1)
        loop='  for (int pb = 0; pb < 2; ++pb) {\n'
        replacement=loop+'    if (pb == 1) { WS_JOIN(); if ((threadIdx.x & 31) != 0) return; }\n    if ((threadIdx.x & 31) == 0) {\n'
        assert body.count(loop)==1;body=body.replace(loop,replacement,1)
        tail='    }\n  }\n}'
        assert body.endswith(tail)
        body=body[:-len(tail)]+'    }\n    } // lane0 producer work; all lanes survive to the phase join\n  }\n}'
        new=body+'\n'*(len(old)-len(old.rstrip('\n')))
        src=src[:a]+new+src[b:];edits.append((old,new))
    old='    wgs_phase<false>(p, S, ring, S.wgr[0], S.wgr[1], seq);\n    DBG(2);\n'
    new=old+'''    CTA_SYNC();
    if (tid == 0) {
      const int logical_b = atomicAdd(p.ctr + 1, 1);
      if (logical_b >= (int)gridDim.x) __trap();
      const int nj = S.njF + S.njS;
      if (nj) wgs_range(S, nj, 1, logical_b, gridDim.x, S.wgr[2], S.wgr[3]);
      else S.wgr[2] = S.wgr[3] = 0;
    }
    WS_JOIN();  // 512 consumers and both complete producer warps; publishes B range
'''
    assert src.count(old)==1;src=src.replace(old,new,1);edits.append((old,new))
    old='(s0 == 0 ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x))'
    new='(s0 == 0 && !PB ? (int)blockIdx.x : wgs_cta_of(S, PB, u << 4, gridDim.x))'
    assert src.count(old)==1;src=src.replace(old,new,1);edits.append((old,new))
    inverse=src
    for old,new in reversed(edits):
        assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==parent
    return raw,tree,parent,src


def model():
    rng=random.Random(20261030);ownership_cases=0;bad_physical_owner=0
    # Any bijection of logical ranges preserves the original K partitions.
    # Fixed uniform stages here exercise split units and endpoint cases; the
    # kernel reuses the original weighted wgs_range/wgs_cta_of functions.
    for grid in (1,2,4,7,32,132):
        for units in (0,1,3,16,32,128):
            total=units*16
            old=[(total*c//grid,total*(c+1)//grid)for c in range(grid)]
            for trial in range(8):
                order=list(range(grid));rng.shuffle(order)
                new=[None]*grid
                for logical,physical in enumerate(order):new[physical]=(logical,old[logical])
                assert sorted((logical,span)for logical,span in new)==list(enumerate(old))
                for physical,(logical,(a,b))in enumerate(new):
                    for unit in range(a//16,(b+15)//16):
                        first,last=max(a,unit*16),min(b,(unit+1)*16)
                        if first>=last:continue
                        holders=[c for c,(x,y)in enumerate(old)if x<=unit*16<y]
                        if not holders:continue
                        owner=holders[0]
                        assert old[owner][0]<=unit*16<old[owner][1]
                        if first==unit*16 and owner!=physical:bad_physical_owner+=1
                ownership_cases+=1
    # Four128-thread consumer groups plus two32-thread producer warps. All
    # producers must remain alive until the mid-CTA join; early lane returns
    # would leave only514 arrivals instead of576.
    arrivals=0
    for permutation in itertools.permutations(range(6)):
        done=set();count=0;claimed=False
        for role in permutation:
            done.add(role)
            if role>=4:count+=32
            if not claimed and set(range(4))<=done:claimed=True;count+=512
            if count==576:assert claimed and len(done)==6
        assert count==576 and claimed;arrivals+=1
    assert 512+2!=576
    return dict(ownership_cases=ownership_cases,logical_fixup_owner_mutants_detected=bad_physical_owner,
        role_arrival_orders=arrivals,consumer_join_count=512,mid_join_count=576,
        producer_warps_retained_until_mid_join=2,
        caveat='These are ownership/liveness models,not latency forecasts,CUDA race validation or a proof of FP32 atomic summation stability.')


def producer_range_source(src):
    original=src;edits=[]
    old='__device__ void wgs_ranges(const Params& p, Smem& S);'
    new='__device__ __forceinline__ void wgs_range(const Smem& S, int nj, int pb, int c, int G, int& g0, int& g1);\n'+old
    assert src.count(old)==1;src=src.replace(old,new,1);edits.append((old,new))
    old='  int wgr[4];';new=old+' int wgs_blogical;'
    assert src.count(old)==1;src=src.replace(old,new,1);edits.append((old,new))
    old='''      const int logical_b = atomicAdd(p.ctr + 1, 1);
      if (logical_b >= (int)gridDim.x) __trap();
      const int nj = S.njF + S.njS;
      if (nj) wgs_range(S, nj, 1, logical_b, gridDim.x, S.wgr[2], S.wgr[3]);
      else S.wgr[2] = S.wgr[3] = 0;'''
    new='''      S.wgs_blogical = atomicAdd(p.ctr + 1, 1);
      if (S.wgs_blogical >= (int)gridDim.x) __trap();'''
    assert src.count(old)==1;src=src.replace(old,new,1);edits.append((old,new))
    old='    WS_JOIN();  // 512 consumers and both complete producer warps; publishes B range'
    new='    WS_JOIN();  // publishes logical B ID to the producer\n    WS_JOIN();  // producer publishes the computed range'
    assert src.count(old)==1;src=src.replace(old,new,1);edits.append((old,new))
    for begin,end,calc in [('__device__ void wgs_wproducer(', '// x producer:',False),
                           ('__device__ void wgs_xproducer(', "// phase A's x gather",True)]:
        a=src.index(begin);b=src.index(end,a);old=src[a:b]
        marker='    if (pb == 1) { WS_JOIN(); if ((threadIdx.x & 31) != 0) return; }'
        new='''    if (pb == 1) {
      WS_JOIN();
'''
        if calc:new+='''      if ((threadIdx.x & 31) == 0) {
        const int nj = S.njF + S.njS;
        if (nj) wgs_range(S, nj, 1, S.wgs_blogical, gridDim.x, S.wgr[2], S.wgr[3]);
        else S.wgr[2] = S.wgr[3] = 0;
      }
'''
        new+='''      WS_JOIN();
      if ((threadIdx.x & 31) != 0) return;
    }'''
        assert old.count(marker)==1;changed=old.replace(marker,new,1)
        src=src[:a]+changed+src[b:];edits.append((old,changed))
    inverse=src
    for old,new in reversed(edits):
        assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==original
    return src


def main():
    global OUT
    variant='--producer-range'in sys.argv
    if variant:OUT=OUT/'producer-range'
    OUT.mkdir(exist_ok=True);raw,tree,parent,src=derive();models=model()
    if variant:src=producer_range_source(src)
    (OUT/'parent.cu').write_bytes(parent.encode());(OUT/'candidate.cu').write_bytes(src.encode())
    sys.path.insert(0,str(ROOT.parent/'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv
    compiler=nv.NVRTC()
    flags=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')=='MAIN_DEFINES')
    flags=list(flags)+['-DOKC_PRMTSC=1','-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
    results={}
    for arm,source in [('parent',parent),('dynamic',src)]:
        results[arm]={}
        device=source.split('#include <torch/extension.h>',1)[0]
        # Same declaration-only NVRTC adaptation used by the pinned compiler
        # screen; the official NVCC source resolves the later definition.
        anchor='template <bool PB>\n__device__ __forceinline__ void wgs_phase('
        forward='__device__ __forceinline__ int wgs_cta_of(const Smem& S, int pb, int s, int G);\n'
        assert device.count(anchor)==1;device=device.replace(anchor,forward+anchor,1)
        (OUT/(arm+'-device.cu')).write_bytes(device.encode())
        for big in ('false','true'):
            r=compiler.compile(device.encode(),
                ['&okc::moe_fused<'+big+'>'],flags,tags=True,lineinfo=False)
            (OUT/(arm+'-'+big+'.log')).write_bytes(r['log'].encode())
            result=dict(rc=r['rc'],kernels=nv.parse_ptxas(r['log'])if r['rc']==0 else [],
                C751_serialization=bool(re.search(r'C751[0-9]',r['log'])))
            results[arm][big]=result;print(arm,big,json.dumps(result),flush=True)
            assert r['rc']==0,r['log'][-3000:]
    no_regression=True
    for big in ('false','true'):
        a,b=results['parent'][big],results['dynamic'][big]
        no_regression=no_regression and not b['C751_serialization']
        for ka,kb in zip(a['kernels'],b['kernels']):
            no_regression=no_regression and all(kb[k]<=ka[k]for k in ('registers','stack_frame','spill_stores','spill_loads'))
    report=dict(parent_sha256=hashlib.sha256(raw).hexdigest(),CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),
        exact_inverse=True,model=models,compiler=results,resources_no_worse=no_regression,
        scope='Original logical A/B K partitions,all math,weights,scratch geometry and fixup slots. B ranges assigned once to physical CTAs after A completion.',
        tradeoff='Adds a576-thread CTA join and loses early B prefetch; input-stage A/B readiness remains original. No time saving established.',
        decision='CPU reject'if not no_regression else 'CPU screen only; numerical/resource/latency review required',
        producer_range_variant=variant,mid_CTA_joins=2 if variant else 1,
        GPU_jobs=0,admitted_for_GPU=False,Decode_checks_used=0)
    (OUT/'cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items()if k!='compiler'},indent=2))


if __name__=='__main__':main()
