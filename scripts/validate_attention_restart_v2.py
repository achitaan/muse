"""Offline checks for 256-key ping-pong and normalized FP16 split means.

Uses the unchanged v1 fragment emulator and pristine public reference as
independent expected-value sources. Does not execute candidate CUDA kernels.
"""
import ast
import hashlib
import json
from pathlib import Path

import torch
import validate_attention_restart as v1

ROOT=Path(__file__).resolve().parents[1]


def pipeline_tests():
    iterations=0
    for pipe in (2,3):
        for tiles in (0,1,2,3,4,7,9,32,33,128,337,1344):
            resident={i%pipe:i for i in range(min(pipe-1,tiles))}
            for i in range(tiles):
                assert resident[i%pipe]==i
                nxt=i+pipe-1
                if nxt<tiles:
                    assert resident.get(nxt%pipe,-1)<=i-1
                    assert nxt%pipe!=i%pipe
                    resident[nxt%pipe]=nxt
                iterations+=1
    return dict(checked_iterations=iterations,pipeline_depths=[2,3],
                limitation='logical stage reuse only; GPU memory ordering unverified')


def partial_tests():
    # Analytical expected values for arbitrarily large raw numerators: constant
    # FP8 V=448. A normalized partial stays 448, while raw FP16 overflows.
    lengths=torch.tensor([0.,1.,128.,65536.,1048576.])
    numer=lengths*448
    mean=torch.where(lengths>0,numer/lengths,0).half()
    assert torch.isfinite(mean).all()
    assert torch.equal(mean.float(),torch.tensor([0.,448.,448.,448.,448.]))
    assert torch.isinf(numer.half()[3:]).all()
    # Round-to-nearest nonnegative FP16 probabilities satisfy round(p)<=2p,
    # including the first subnormal bin. Therefore |O/l| is bounded near
    # 2*max|V| even when many tiny probabilities round upwards. This is far
    # below FP16 overflow for finite E4M3 V; raw O has no such length bound.
    threshold=torch.tensor(2.0**-25,dtype=torch.float32)
    p=torch.cat((torch.logspace(-40,0,40001),threshold[None],
                 torch.nextafter(threshold,torch.tensor(float('inf')))[None]))
    ph=p.half().float()
    assert bool((ph<=2*p).all())
    # Check one lane's eight half elements fit the aligned uint4 load and the
    # producer's permuted 16-value ownership writes exactly that row once.
    offsets=[(16*t+8*v)*2 for t in range(4) for v in range(2)]
    assert sorted(offsets)==list(range(0,128,16))
    assert all((r*128+off)%16==0 for r in range(128) for off in offsets)
    return dict(normalized_half_max=float(mean.max()),overflow_stress='pass',
                probability_rounding_ratio_max=float((ph/p).max()),
                half_row_stride_bytes=128,float_row_stride_bytes=256)


def cases(tile_override=None):
    # Early/late causal windows, over-splitting, exact/partial gathers, mixed
    # lengths, large FP32 RoPE angles, nonunit scales, and coherent low logits.
    specs=[
        (1,1,0,'sliding','normal',1,1,37,'fixed'),
        (1,1,65,'full','normal',64,1/64,11,'fixed'),
        (2,1,2048,'sliding','edges',1,1,33,'fixed'),
        (2,8,1,'full','normal',1,1,11,'fixed'),
        (2,8,255,'full','soft',64,1/64,13,'fixed'),
        (2,8,256,'full','normal',1,1,5,'fixed'),
        (2,8,257,'full','coherent',1,1,3,'fixed'),
        (2,8,2049,'sliding','edges',1,1,17,'fixed'),
        (3,8,5000,'sliding','soft',1,1/64,9,'mixed'),
        (3,1,4000,'full','soft',64,1,17,'mixed'),
        (1,1,80000,'full','soft',1,1,252,'fixed'),
        (1,8,80000,'full','coherent',1,1,128,'fixed'),
        (1,8,80000,'sliding','edges',1,1,9,'fixed'),
    ]
    result=[]
    for i,(B,L,ctx,layer,dist,page,scale,splits,spread) in enumerate(specs):
        d=v1.public.generate_input(B,ctx,'tp8',layer,7200+i,L,dist=dist,page=page,
                                   kv_scale=scale,spread=spread,device='cpu')
        ref=v1.public.ref_kernel(d,torch.float8_e4m3fn)[0].bfloat16()
        tiles=(tile_override or {}).get(L,(64,128) if L==1 else (64,256))
        for tile in tiles:
            for half in (False,True):
                out,k,v,_=v1.emulate(d,tile,splits,half=half,micro=True)
                err=float((out.float()-ref.float()).norm()/ref.float().norm().clamp(min=1e-12))
                assert err<0.005, (i,tile,half,err)
                raw=torch.cat([t.contiguous().view(torch.uint8).flatten() for t in (out,k,v)])
                ok,message=v1.public.check_implementation(d,raw)
                assert ok,message
                result.append(dict(B=B,L=L,ctx=ctx,layer=layer,dist=dist,page=page,scale=scale,
                                   spread=spread,tile=tile,splits=splits,half=half,
                                   relative_l2_vs_fp8_reference=err,judge=message))
    return result


def manufactured_tests(tile_override=None):
    n=0
    for L in (1,8):
        for scale in (1.,1/64):
            for value in (0.,1.):
                d=v1.public.generate_input(1,257,'tp8','full',67,L,page=1,
                                           kv_scale=scale,device='cpu')
                d['v_cache'].fill_(value/scale)
                d['qkv'][...,1152:1280]=value
                d['qkv'][...,1280:]=0
                tiles=(tile_override or {}).get(L,(64,128) if L==1 else (64,256))
                for tile in tiles:
                    for half in (False,True):
                        out,*_=v1.emulate(d,tile,7,half=half,micro=True)
                        # Independent analytical solution: constant V survives
                        # attention, zero gate gives sigmoid=1/2, BF16 rounding.
                        torch.testing.assert_close(out,torch.full_like(out,value/2),rtol=0,atol=0)
                        n+=1
    return dict(exact_analytical_cases=n)


def main():
    torch.set_num_threads(4)
    candidate=ROOT/'candidates/attention_restart_v2.py'
    base=ROOT/'board_best/attention_kb1h2_scored_2.6536.py'
    tail=ROOT/'.tools/attention_restart_v2_tail.py'
    candidate.write_text(base.read_text()+tail.read_text())
    assert candidate.read_bytes().startswith(base.read_bytes())
    report=dict(candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                fallback_prefix_byte_identical=True)
    report['static']=v1.static_tests(candidate,v2=True)
    report['pipeline']=pipeline_tests()
    report['partials']=partial_tests()
    report['manufactured']=manufactured_tests()
    report['fragments']=[dict(L=L,tile=T,max_abs_pv_error=v1.fragment_test(L,T,123+L+T))
                         for L,T in ((1,64),(1,128),(8,64),(8,256))]
    report['numerics']=cases()
    # A 256-key gather must not allocate a 256-key score register tile.
    src=next(ast.literal_eval(n.value) for n in ast.parse(candidate.read_text()).body
             if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='RR_CUDA')
    assert 'MICRO=KEYS<64?KEYS:64' in src and 'N8=MICRO/8' in src
    assert 'PIPE = TILE==256 ? 2 : 3' in src
    report['gpu']='unavailable: real CUDA compilation, execution, races and speed are unverified'
    (ROOT/'results/attn_restart_v2_cpu.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='numerics'},indent=2))
    print('PASS %d numerical variants, max FP8-reference relative L2 %.7f'%
          (len(report['numerics']),max(x['relative_l2_vs_fp8_reference'] for x in report['numerics'])))


if __name__=='__main__': main()
