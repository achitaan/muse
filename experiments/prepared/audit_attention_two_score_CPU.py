"""Independent source inverse, partition coverage and epoch models; no GPU."""
import ast
import hashlib
import json
import random

from build_attention_two_score_pipeline import ROOT, OUT, PARENT, freeze


def literal(tree, name):
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and
        any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))


def source_proof():
    parent=(ROOT/'best/attention_checked.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==PARENT
    tree=ast.parse(parent)
    raw=(OUT.parent/'admission-v3/submission.py').read_bytes(); trial=ast.parse(raw)
    old=literal(tree,'CUDA_SRC'); new=literal(trial,'CUDA_SRC')
    segment_anchor='template <int L, bool FLAT, int ROLE, int MODE = 0>\n'
    a=new.index(segment_anchor)
    b=new.index(segment_anchor,a+len(segment_anchor))
    cloned_segment=new[a:b]
    assert 'bool segment_two_score(' in cloned_segment
    recovered=new[:a]+new[b:]
    kernel_anchor='template <int L, bool FLAT, int MODE = 0>\n__global__'
    k0=recovered.index(kernel_anchor)
    k1=recovered.index(kernel_anchor,k0+len(kernel_anchor))
    stop=recovered.index('\n}  // namespace',k1)
    clone_kernel=recovered[k1:stop]
    recovered=recovered[:k1-1]+recovered[stop:]
    recovered=recovered.replace('  static int pipeline_occ = 0, pipeline_regs = 0;\n','',1)
    a=recovered.index('    if constexpr (L == 8) {\n      auto pipe =')
    b=recovered.index('    }\n',a)+6
    recovered=recovered[:a-1]+recovered[b:]
    a=recovered.index('  if constexpr (L == 8) {\n    const int pairs = p.Hkv / 2;')
    b=recovered.index('  if (cl || pdl)',a)
    dispatch=recovered[a:b]
    recovered=recovered[:a]+recovered[b:]
    a=recovered.index('\nstd::vector<int64_t> two_score_resources()')
    recovered=recovered[:a]
    if recovered!=old:
        import difflib
        print(''.join(list(difflib.unified_diff(old.splitlines(True),recovered.splitlines(True)))[:30]))
    assert recovered==old,'Deleting explicitly identified additions must restore original CUDA exactly'
    # Independently compare every statement on either side of the new loop.
    original_segment=old[old.index(segment_anchor):old.index('template <int L, bool WS, int NT>\n')]
    pre=original_segment[:original_segment.index('  const int nit = ntiles;')]
    assert cloned_segment.startswith(pre.replace('bool segment(', 'bool segment_two_score(',1))
    assert cloned_segment[cloned_segment.index('  PROF_DUMP;',len(pre)):]==original_segment[original_segment.index('  PROF_DUMP;',len(pre)):]
    anchor='    if constexpr (WS) {   // K (wgmma) and V (conversion) of the stage are consumed'
    ma=original_segment.index(anchor)
    mb=original_segment.index('      float* od = &o[0][0][0];\n      \n           // done before the next tile',ma)
    arithmetic=original_segment[ma:mb]
    assert cloned_segment.count(arithmetic)==1
    # Restore cloned outer kernel to its exact original rather than trusting a
    # builder manifest about which finalizer/producer it calls.
    old_kernel=old[old.index(kernel_anchor):old.index('\n}  // namespace',old.index(kernel_anchor))]
    ck=clone_kernel.replace('__launch_bounds__(Cfg<L, FLAT>::NTX, 1) attn_kernel_two_score(',
        '__launch_bounds__(Cfg<L, FLAT>::NTX, Cfg<L, FLAT>::MINB) attn_kernel_fp8pv(',1)
    ck=ck.replace('"n"(232)','"n"(Cfg<L, FLAT>::CREGS)',1)
    a=ck.index('        if constexpr (R == 1) {\n')
    b=ck.index('        }',ck.index('        } else {',a)+len('        } else {'))+9
    wrapper=ck[a:b]
    preserved_call=wrapper.split('\n',1)[1].split('\n        } else {')[0]
    assert 'segment<L, FLAT, R>' in preserved_call
    assert wrapper.count('segment_two_score<L, FLAT, R>')==1
    ck=ck[:a]+preserved_call+ck[b:]
    assert ck==old_kernel
    expected=('p.B > 1 && p.B <= 64','pipeline_occ == 1','pipeline_regs == 168',
        'const int splits = nsm / pairs;','q.n_splits = splits;',
        'const dim3 pipeline_grid(splits,1,pairs);','q.ns_magic = splits >= 2')
    assert all(s in dispatch for s in expected)
    # Python interfaces/fallbacks remain original except extension admission.
    for i,(p,t) in enumerate(zip(tree.body,trial.body)):
        if isinstance(p,ast.FunctionDef) and p.name=='_build':trial.body[i]=p
        elif isinstance(p,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('CUDA_SRC','CPP_SRC') for x in p.targets):trial.body[i]=p
        else:assert ast.dump(p)==ast.dump(t)
    assert ast.dump(tree)==ast.dump(trial)
    return dict(source_sha256=hashlib.sha256(raw).hexdigest(),CUDA_sha256=hashlib.sha256(new.encode()).hexdigest(),
        exact_original_CUDA_restored=True,exact_tile_arithmetic=True,exact_segment_prefix_suffix=True,
        exact_original_outer_kernel_after_reversing_three_changes=True,other_Python_AST_exact=True)


def partitions(lengths,grid):
    cum=[0]
    for length in lengths: cum.append(cum[-1]+length)
    total=cum[-1]+(len(lengths)-1)*600
    def mapping(v):
        # Independent linear virtual-position mapping; CUDA uses ballots or a
        # binary search. Padding maps to the preceding request's right edge.
        b=max(i for i in range(len(lengths)) if cum[i]+600*i<=v)
        return min(v-600*b,cum[b+1])
    intervals=[]
    for c in range(grid):
        lo=mapping(c*total//grid);hi=mapping((c+1)*total//grid)
        segments=[(b,max(lo,cum[b])-cum[b],min(hi,cum[b+1])-cum[b])
                  for b in range(len(lengths)) if cum[b+1]>lo and cum[b]<hi]
        assert len(segments)<=64
        intervals.append(segments)
    for b,length in enumerate(lengths):
        coverage=sorted((a,z) for segments in intervals for bb,a,z in segments if bb==b)
        end=0
        for a,z in coverage:assert a==end and a<z<=length;end=z
        assert end==length
    return intervals


def main():
    proof=source_proof();rng=random.Random(20261029);cases=0;max_segments=0
    for batch in (2,3,4,8,16,31,32,33,63,64):
        for pairs in (1,2,3,4,8,16,32,64,128):
            grid=132//pairs
            assert grid*pairs<=132 and grid>0
            for kind in ('tiny','mixed','long','skewed'):
                lengths=([8]*batch if kind=='tiny' else [rng.randrange(8,80008) for _ in range(batch)]
                    if kind=='mixed' else [524288+8]*batch if kind=='long' else [8]*(batch-1)+[80008])
                intervals=partitions(lengths,grid)
                max_segments=max(max_segments,max(map(len,intervals)));cases+=1
    # The unchanged double-counter epoch supports changing grid size because
    # each launch uses its own G, clears the alternate bank and advances epoch.
    epoch=0;counts=[0,0]
    for replay in range(256):
        grid=rng.choice((1,33,66,128,132,264))
        parity=epoch&1
        assert counts[parity]==0
        arrivals=list(range(grid));rng.shuffle(arrivals)
        for number,cta in enumerate(arrivals):
            previous=counts[parity];counts[parity]+=1
            if previous+1==grid:
                assert number==grid-1
                counts[parity^1]=0;epoch+=1
        assert epoch==replay+1
    proof.update(CPU_only=True,GPU_jobs=0,partition_cases=cases,largest_segments_per_CTA=max_segments,
        changing_grid_epochs=256,seed=20261029,
        limitations='CPU/source proof only: changed FP8 split rounding, CUDA memory order, actual resources, graph replay and performance still require official GPU validation')
    freeze(OUT.parent/'source-coverage-CPU-proof.json',(json.dumps(proof,indent=2)+'\n').encode())
    print(json.dumps(proof,indent=2))


if __name__=='__main__':main()
