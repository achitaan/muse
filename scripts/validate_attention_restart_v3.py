"""CPU validation of the v3 bank, prefetch, merge and dispatch changes.

The public-judge tests are numerical emulation, not GPU execution. CUDA
ordering, compiled registers/spills, SFU error and performance remain untested.
"""
import ast
from collections import defaultdict
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import torch
import validate_attention_restart as v1
import validate_attention_restart_v2 as v2

ROOT=Path(__file__).resolve().parents[1]


def bank_degree(offsets):
    banks=defaultdict(set)
    for word in offsets:
        banks[word%32].add(word)
    return max(map(len,banks.values()))


def exchange_tests():
    old=[(lane//4)*64+16*(lane%4) for lane in range(32)]
    new=[18*lane+2 for lane in range(32)]
    assert bank_degree(old)==16 and bank_degree(new)==2
    # Actual raw K scalar loads are conflict-free; V's uint2 loads are also
    # conflict-free within each mandatory 128-byte, sixteen-lane wavefront.
    assert bank_degree([v1.ko(l//4,4*(l%4))//4 for l in range(32)])==1
    for lo in (0,16):
        off=[v1.ko(2*(l%4),8*(l//4))//4+d for l in range(lo,lo+16) for d in (0,1)]
        assert bank_degree(off)==1
    gen=np.random.default_rng(915)
    a=gen.normal(size=(2,32,16)).astype(np.float32)
    m=gen.normal(size=(2,32)).astype(np.float32)
    length=gen.random((2,32)).astype(np.float32)
    s=np.full(1152,np.nan,np.float32)
    writes=[]
    for h in range(2):
        for lane in range(32):
            p=(h*32+lane)*18
            s[p:p+2]=m[h,lane],length[h,lane]
            for d in range(8):
                s[p+2+2*d],s[p+3+2*d]=a[h,lane,d],a[h,lane,d+8]
            writes.extend(range(p,p+18))
    assert sorted(writes)==list(range(1152))
    for h in range(2):
        for lane in range(32):
            p=(h*32+lane)*18
            # Expected source: independent per-lane state, including the
            # unusual old C-register -> natural-output column permutation.
            np.testing.assert_array_equal(s[p:p+2],[m[h,lane],length[h,lane]])
            got=np.concatenate((s[p+2:p+18:2],s[p+3:p+18:2]))
            np.testing.assert_array_equal(got,a[h,lane])
    return dict(old_decode_scalar_bank_degree=16,new_decode_scalar_bank_degree=2,
                raw_K_bank_degree=1,raw_V_128byte_wave_bank_degree=1,
                peer_exchange_bytes=4608,old_data_exchange_bytes=8192,
                lane_state_roundtrip='bit exact')


def copy_tests():
    rng=np.random.default_rng(31)
    count=0
    for page in (1,64):
        lp=page.bit_length()-1
        for seq in (0,1,63,64,127,128,2048,4097):
            end=seq+8
            n=(end+page-1)//page
            table=rng.permutation(n).astype(np.int32)
            cache=rng.integers(0,256,(n*page,128),dtype=np.uint8)
            for tile in (64,128):
                for nt in (128,256):
                    for begin in sorted({0,max(0,seq-63),max(0,seq-127)}):
                        passes=tile//(nt//8)
                        pages=np.zeros((nt,passes),np.int64)
                        # Producer phase: only lane-group leaders load; all
                        # independent page loads finish before the copy phase.
                        for t in range(nt):
                            for p in range(passes):
                                pos=begin+p*(nt//8)+t//8
                                if t%8==0 and pos<end and pos<seq:
                                    pages[t,p]=table[pos>>lp]
                        out=np.zeros(tile*128,np.uint8)
                        for t in range(nt):
                            leader=(t//32)*32+(t%32//8)*8
                            for p in range(passes):
                                r=p*(nt//8)+t//8
                                pos=begin+r
                                chunk=t%8
                                pg=int(pages[leader,p])
                                if page==1 and begin+tile<=seq:
                                    byte=((pg<<7)+chunk*16)&0xffffffff
                                    row=byte//128
                                else:
                                    row=(pg<<lp)+(pos&(page-1))
                                if pos<end and pos<seq:
                                    at=v1.ko(r,chunk*16)
                                    out[at:at+16]=cache[row,chunk*16:chunk*16+16]
                        for r in range(tile):
                            pos=begin+r
                            expected=cache[int(table[pos//page])*page+pos%page] if pos<seq else np.zeros(128,np.uint8)
                            got=np.array([out[v1.ko(r,c)] for c in range(128)])
                            np.testing.assert_array_equal(got,expected)
                        count+=1
    return dict(copy_layout_cases=count,batched_page_broadcast='pass',
                page1_uint32_offsets='pass',generic_page64='pass')


def pipeline_and_merge_tests():
    iters=0
    for pipe in (3,4):
        for n in (0,1,2,3,4,9,33,128,1344):
            resident={i%pipe:i for i in range(min(pipe-1,n))}
            for i in range(n):
                assert resident[i%pipe]==i
                k=i+pipe-1
                if k<n:
                    assert resident.get(k%pipe,-1)<=i-1
                    resident[k%pipe]=k
                iters+=1
    for n in (1,7,8,9,16,33,132,264,528):
        visited=[]
        for k0 in range(0,n,8):
            visited.extend(k0+u for u in range(8) if k0+u<n)
        assert visited==list(range(n))
    for L in (1,8):
        writes=[]
        lanes=8 if L==1 else 32
        rows=128//lanes
        for block in range(1 if L==1 else L*4):
            for group in range(rows):
                row=block*rows+group
                writes.extend((row,lane*(64//lanes)+d) for lane in range(lanes) for d in range(64//lanes))
        assert len(writes)==len(set(writes))==L*16*64
    return dict(stage_reuse_iterations=iters,pipelines=[3,4],
                batch8_tails='pass',warp_row_merge_ownership='pass')


def dispatch_tests(candidate):
    spec=importlib.util.spec_from_file_location('attention_restart_v3_cpu',candidate)
    module=importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(module)
    module._num_sms=lambda device:132
    rows=[]
    for B in (1,16,32):
        for L in (1,8):
            for w in (0,2048):
                choices=module._rr_choices(B,L,w,None)
                expected=(B,L,w) in ((1,1,2048),(16,1,2048),(1,8,2048))
                assert bool(choices)==expected
                assert all(p['last'] in (0,1) and p['tile'] in (64,128) for p in choices)
                rows.append(dict(B=B,L=L,window=w,arms=len(choices)))
    assert not module._RR_SELECTED and not module._RR_WS
    return rows


def main():
    torch.set_num_threads(4)
    base=ROOT/'board_best/attention_kb1h2_scored_2.6536.py'
    tail=ROOT/'.tools/attention_restart_v3_tail.py'
    candidate=ROOT/'candidates/attention_restart_v3.py'
    candidate.write_text(base.read_text()+tail.read_text())
    assert candidate.read_bytes().startswith(base.read_bytes())
    result=dict(candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                fallback_prefix_byte_identical=True)
    result['static']=v1.static_tests(candidate,v3=True)
    text=candidate.read_text().split('# Attention restart v3:',1)[1]
    assert 'void attention(const __grid_constant__ Params p)' in text
    assert 'void merge(const __grid_constant__ Params p)' in text
    assert 'void attention_plain(Params p)' in text
    result['banks_and_exchange']=exchange_tests()
    result['copy']=copy_tests()
    result['pipeline_and_merge']=pipeline_and_merge_tests()
    result['dispatch']=dispatch_tests(candidate)
    override={1:(64,128),8:(64,128)}
    result['numerics']=v2.cases(override)
    result['manufactured']=v2.manufactured_tests(override)
    result['fragments']=[dict(L=L,tile=T,max_abs_pv_error=v1.fragment_test(L,T,126+L+T))
                         for L,T in ((1,64),(1,128),(8,64),(8,128))]
    result['gpu']='unverified: no CUDA compiler/GPU; parent v1/v2 checks do NOT validate v3'
    (ROOT/'results/attn_restart_v3_cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='numerics'},indent=2))
    print('PASS %d CPU judge variants; worst FP8-reference L2 %.7f'%
          (len(result['numerics']),max(r['relative_l2_vs_fp8_reference'] for r in result['numerics'])))


if __name__=='__main__': main()
