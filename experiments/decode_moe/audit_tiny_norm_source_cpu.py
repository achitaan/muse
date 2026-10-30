"""Compile/resource and row-ownership screen without loading a GPU kernel."""

import hashlib
import json
import sys

from build_tiny_norm_source import OUT, derive
from tinyroute_source import HERE


def main():
    parent,src,cpp,vals=derive()
    assert (OUT/'candidate.cu').read_bytes()==src.encode()
    assert (OUT/'candidate.cpp').read_bytes()==cpp.encode()
    # Independent complete-row and packed-output ownership model.
    columns=[16*tid+4*k+j for tid in range(256)for k in range(4)for j in range(4)]
    assert sorted(columns)==list(range(4096))
    for grid in (1,2,132,148):
        for m in range(1,9):
            owners=[0]*m
            for block in range(grid):
                for token in range(block,m,grid):owners[token]+=1
            assert owners==[1]*m
    # Every local route is reduced by eight disjoint warps, and receives
    # exactly one statistic writer (lane=slot in its token-owning CTA).
    assert {warp*32+lane for warp in range(8)for lane in range(32)}==set(range(256))
    assert max((warp*8+slot)*4 for warp in range(8)for slot in range(8))<256
    assert max((8*8+slot+1)*4 for slot in range(8))==288
    sys.path.insert(0,str(HERE.parents[2]/'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv
    compiler=nv.NVRTC();compiles={}
    for name in ('TINY','TINY2'):
        for prmt in (False,True):
            defs=list(vals[name+'_DEFINES'])+['-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
            if prmt:defs+=['-DOKC_PRMTSC=1']
            label=name+('_PRMT'if prmt else '_LUT')
            r=compiler.compile(src.split('#include <torch/extension.h>',1)[0].encode(),
                ['&tiny_norm_whole::moe_tnorm_fused<false>'],defs,tags=True,lineinfo=False)
            (OUT/(label+'.log')).write_text(r['log'],encoding='utf-8')
            compiles[label]=dict(rc=r['rc'],seconds=r['seconds'],cubin_sha256=r['cubin_sha256'],
                kernels=nv.parse_ptxas(r['log'])if r['rc']==0 else [])
            print(label,json.dumps(compiles[label]),flush=True)
            assert r['rc']==0,r['log'][-4000:]
    report=dict(candidate_CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),
        row_columns_exactly_once=True,token_owner_cases=32,shared_reduction_bytes=288,
        compiles=compiles,GPU_jobs=0,admitted_for_GPU=False,
        limitation='Compilation/layout evidence only; H200 numerical gates,barrier lifetime,resources and whole-kernel performance remain unverified.')
    (OUT/'cpu-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
