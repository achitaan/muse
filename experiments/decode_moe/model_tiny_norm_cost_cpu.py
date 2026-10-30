"""Source-bound ownership/traffic screen; no GPU or timing prediction."""

import ast
import hashlib
import json

from tinyroute_source import HERE, ROOT, BASE_SHA


def main():
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==BASE_SHA
    tree=ast.parse(raw)
    values={n.targets[0].id:ast.literal_eval(n.value)for n in tree.body
            if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('CUDA_SRC','TINY_DEFINES','TINY2_DEFINES')}
    src=values['CUDA_SRC']
    assert 'RB_B = HD / 16;'in src
    assert 'if (nF > 0) __trap();'in src
    assert 'if (gg == 0 && tk < cnt) atomicAdd(&ss_cur()[st + tk], sq);'in src
    assert 'const int Nh = (OKC_WSB && OKC_WSBL) ? N - min(OKC_WSBLK, b1 - b0) : N;'in src
    assert 'WS_JOIN();  // every epilogue store of this CTA is done before the grid barrier'in src
    for name,tail in [('TINY_DEFINES',2),('TINY2_DEFINES',1)]:
        assert '-DOKC_NW=8'in values[name]and '-DOKC_WS=1'in values[name]
        assert '-DOKC_WSBLK='+str(tail)in values[name]
    # Model one warp's two-token-per-lane phase-B epilogue. The helper
    # executes both h values; a compute tail uses the same lane/token mapping.
    counts=[]
    for count in range(1,9):
        writes=set();arrivals=[0]*count
        for rb in range(256):
            for lane in range(32):
                gg,tt=lane>>2,lane&3
                for h in range(2):
                    tk=2*tt+h
                    if tk<count:
                        for col in (rb*16+gg,rb*16+gg+8):
                            key=tk,col
                            assert key not in writes;writes.add(key)
                        if gg==0:arrivals[tk]+=1
        assert len(writes)==count*4096 and arrivals==[256]*count
        counts.append(dict(expert_tokens=count,FP32_y_elements=len(writes),norm_atomic_arrivals=arrivals))
    # No fat/dynamic units in these tiny WS builds. Each unit belongs either
    # to the helper prefix or exactly one compute-tail warp, never both.
    partitions=0
    for grid in (1,2,132,148):
        for jobs in range(33):
            total=jobs*256
            for tail in (1,2):
                owners=[0]*total
                for cta in range(grid):
                    first,last=total*cta//grid,total*(cta+1)//grid
                    boundary=last-min(tail,last-first)
                    for u in range(first,boundary):owners[u]+=1
                    for warp in range(min(tail,last-first)):
                        u=last-1-warp;assert boundary<=u<last;owners[u]+=1
                assert owners==[1]*total;partitions+=1
    # Proposed full-row scans retain the original segment's values for output.
    # Their extra logical requests are not measured HBM traffic or elapsed time.
    route_bytes=4096*4
    report=dict(retained_MoE_sha256=BASE_SHA,CUDA_sha256=hashlib.sha256(src.encode()).hexdigest(),
        expert_count_cases=counts,partition_cases=partitions,all_ownership_checks_pass=True,
        norm_atomics_per_valid_route=256,
        current_finalizer_read_bytes_per_route=route_bytes,
        proposed_four_CTA_read_bytes_per_route=4*route_bytes,
        proposed_extra_bytes_per_route_with_segment_reuse=3*route_bytes,
        maximum_M8_routes=64,maximum_M8_extra_bytes_with_reuse=64*3*route_bytes,
        maximum_M8_extra_bytes_without_reuse=64*4*route_bytes,
        proposed_shared_reduction_bytes=(8*8+8)*4,
        proposed_norm_statistic_writer='Only segment0,thread0 writes each valid route statistic after the existing grid join; other segments use local sums.',
        limitations=['No kernel implementation or GPU job.','Counts are source-derived logical work,not exposed latency or measured traffic.',
                    'Reusing the freed ring still requires an implementation/resource/lifetime audit.',
                    'Full-row reads and two CTA barriers per item may outweigh removing the atomics.'],
        GPU_jobs=0,admitted_for_GPU=False)
    out=ROOT/'results/decode-moe-plan-20261030/tiny-norm-model';out.mkdir(exist_ok=True)
    (out/'cpu-model.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items()if k!='expert_count_cases'},indent=2))


if __name__=='__main__':main()
