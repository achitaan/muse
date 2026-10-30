"""Prepare one private whole-row tiny-norm CUDA hypothesis; no GPU launch."""

import ast
import hashlib
import json

from tinyroute_source import HERE, ROOT, BASE_SHA

OUT=ROOT/'results/decode-moe-plan-20261030/tiny-norm-whole'


def derive():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==BASE_SHA
    tree=ast.parse(raw)
    vals={n.targets[0].id:ast.literal_eval(n.value)for n in tree.body
          if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)
          and n.targets[0].id in ('CUDA_SRC','CPP_SRC','TINY_DEFINES','TINY2_DEFINES')}
    parent=vals['CUDA_SRC'];src=parent;edits=[]
    # These are the only two phase-B epilogue owners in the retained WS builds:
    # helper prefix and compute tail. Other GEMM/routing bodies stay exact.
    for begin,end in [('__device__ void skinny_ring_ws(', '__device__ __forceinline__ void ws_issue('),
                      ('__device__ void ws_helper(', '__device__ __forceinline__ void skinny_split(')]:
        first=src.index(begin);last=src.index(end,first+len(begin));old=src[first:last]
        sq='float sq = v0f * v0f + v1f * v1f;'
        assert old.count(sq)==1;new=old.replace(sq,'// Norm sums are formed by the final row owner.',1)
        for statement in ['sq += __shfl_xor_sync(0xffffffffu, sq, 4);',
                          'sq += __shfl_xor_sync(0xffffffffu, sq, 8);',
                          'sq += __shfl_xor_sync(0xffffffffu, sq, 16);',
                          'if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);']:
            assert new.count(statement)==1;new=new.replace(statement,'',1)
        src=src[:first]+new+src[last:];edits.append((old,new))
    old='constexpr int FSEGP = HD / (NT * 4);'
    new='constexpr int FSEGP = 1;  // whole-row token owner'
    assert src.count(old)==1;src=src.replace(old,new,1);edits.append((old,new))
    first=src.index('#if OKC_WGS && OKC_WGSFINSS');last=src.index('#elif OKC_FINCOL',first)
    old=src[first:last];new=(HERE/'tiny_norm_whole_finalizer.cuh').read_text(encoding='utf-8')
    src=src[:first]+new+src[last:];edits.append((old,new))
    inverse=src
    for old,new in reversed(edits):
        assert inverse.count(new)==1;inverse=inverse.replace(new,old,1)
    assert inverse==parent
    # Unique device namespace and host API avoid cross-module template/static
    # identity collisions. The retained parent module is never replaced.
    for old,new in [('namespace okc','namespace tiny_norm_whole'),('okc::','tiny_norm_whole::'),
                    ('moe_fused','moe_tnorm_fused'),('okc_run','okc_tnorm_run'),
                    ('okc_set_pdl','okc_tnorm_set_pdl'),('okc_smem_bytes','okc_tnorm_smem_bytes'),
                    ('g_okc_pdl','g_okc_tnorm_pdl')]:
        src=src.replace(old,new)
    src='#if !OKC_TINY || !OKC_WS\n#error Tiny WS builds only\n#endif\n'+src
    cpp=vals['CPP_SRC'].replace('okc_run','okc_tnorm_run').replace('okc_set_pdl','okc_tnorm_set_pdl')
    return parent,src,cpp,vals


def main():
    OUT.mkdir(exist_ok=True);parent,src,cpp,vals=derive()
    (OUT/'candidate.cu').write_bytes(src.encode());(OUT/'candidate.cpp').write_bytes(cpp.encode())
    record=dict(retained_MoE_sha256=BASE_SHA,parent_CUDA_sha256=hashlib.sha256(parent.encode()).hexdigest(),
        candidate_CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),CUDA_bytes=len(src.encode()),
        exact_inverse_before_private_names=True,scope='TINY/TINY2 WS epilogue norm and whole-row finalization only; original routing and GEMM/ybuf/inter arithmetic retained.',
        hypothesis='Remove256 norm atomics per route; transfer the retained MAIN two-pass row-norm ownership to NT256 tiny builds.',
        added_launches=0,grid_join_unchanged=True,precision_boundaries_unchanged=True,
        extra_logical_read_bytes_per_route=4096*4,maximum_M8_extra_bytes=64*4096*4,
        shared_ring_reduction_bytes=(8*8+8)*4,
        unresolved='Whole-row finalizer lowers useful CTA count and doubles logical ybuf reads; no speed benefit established.',
        performance_gate='Existing greater-than3% paired cold-L2 graph gate required later; no GPU admission.',
        GPU_jobs=0,admitted_for_GPU=False)
    (OUT/'source-binding.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
