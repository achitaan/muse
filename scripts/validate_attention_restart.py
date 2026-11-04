"""CPU validation of the restart's exact thread/fragment and reduction layouts.

Expected values come from matrix multiplication and the pristine public
reference, not from previously observed candidate outputs. No GPU is required.
This does NOT emulate CUDA memory ordering, ptxas, or tensor-core rounding.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.tools/attention-spec'))
import reference as public


def ko(r, c):
    return r*128 + (((c//16) ^ (r & 7))*16) + c%16


def qo(r, c):
    return r*64 + (((c//16) ^ ((r//2) & 3))*16) + c%16


def byte4(x):
    return np.frombuffer(np.uint32(x).tobytes(), dtype=np.uint8)


def word(x):
    return int(np.frombuffer(np.asarray(x, dtype=np.uint8).tobytes(), dtype=np.uint32)[0])


def half_word(x):
    return int(np.frombuffer(np.asarray(x, dtype=np.float16).tobytes(), dtype=np.uint32)[0])


def halves(x):
    return np.frombuffer(np.uint32(x).tobytes(), dtype=np.float16).astype(np.float32)


def prmt(a, b, selector):
    src = np.concatenate((byte4(a), byte4(b)))
    return word([src[(selector >> (4*i)) & 7] for i in range(4)])


FP8_TABLE = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().numpy()


def mma(a, b, fp8):
    """Independently reconstruct documented mma.sync operands from all lanes.

    PTX m16n8k32 FP8 and m16n8k16 FP16: lane group g=lane/4,
    thread-in-group t=lane%4. C lane = rows g,g+8 and columns 2t,2t+1.
    """
    k = 32 if fp8 else 16
    A, B = np.empty((16,k), np.float32), np.empty((k,8), np.float32)
    for lane in range(32):
        g, t = divmod(lane,4)
        for reg in range(4):
            r = g+(reg%2)*8
            c = (4*t if fp8 else 2*t)+(reg//2)*(k//2)
            v = FP8_TABLE[byte4(a[lane,reg])] if fp8 else halves(a[lane,reg])
            A[r,c:c+len(v)] = v
        for reg in range(2):
            r = (4*t if fp8 else 2*t)+reg*(k//2)
            v = FP8_TABLE[byte4(b[lane,reg])] if fp8 else halves(b[lane,reg])
            B[r:r+len(v),g] = v
    C = A @ B
    result = np.empty((32,4), np.float32)
    for lane in range(32):
        g,t = divmod(lane,4)
        result[lane] = C[g,2*t],C[g,2*t+1],C[g+8,2*t],C[g+8,2*t+1]
    return result


def shared_kv(raw):
    n = raw.shape[0]
    s = np.empty(n*128, np.uint8)
    flat = raw.reshape(n,128)
    for r in range(n):
        for c in range(0,128,16):
            s[ko(r,c):ko(r,c)+16] = flat[r,c:c+16]
    return s


def shared_q(raw):
    n = raw.shape[0]
    s = np.empty(n*64, np.uint8)
    for r in range(n):
        for c in range(0,64,8):
            s[qo(r,c):qo(r,c)+8] = raw[r,c:c+8]
    return s


def load_word(s, at):
    return word(s[at:at+4])


def value_frag(s, kb, hk):
    b0,b1 = np.empty((32,8),np.uint32),np.empty((32,8),np.uint32)
    for lane in range(32):
        g,t = divmod(lane,4)
        x = []
        for u in range(4):
            r = kb+2*t+(u&1)+8*(u//2)
            c = hk*64+8*g
            at = ko(r,c)
            x.append((load_word(s,at),load_word(s,at+4)))
        for h in range(2):
            for dst,u in ((b0,0),(b1,2)):
                for j,sel in enumerate((0x5140,0x7362)):
                    bits = byte4(prmt(x[u][h],x[u+1][h],sel))
                    dst[lane,4*h+2*j] = half_word(FP8_TABLE[bits[:2]])
                    dst[lane,4*h+2*j+1] = half_word(FP8_TABLE[bits[2:]])
    return b0,b1


def fragment_test(length, tile, seed):
    """Full QK and PV reconstruction using the candidate's byte addresses."""
    gen = torch.Generator().manual_seed(seed)
    qrows = 16 if length == 1 else 64
    q = torch.randn(2,qrows,64,generator=gen).to(torch.float8_e4m3fn)
    if length == 1:
        q[:,8:] = 0
    k = torch.randn(tile,2,64,generator=gen).to(torch.float8_e4m3fn)
    v = torch.randn(tile,2,64,generator=gen).to(torch.float8_e4m3fn)
    sk,sv = shared_kv(k.view(torch.uint8).numpy()),shared_kv(v.view(torch.uint8).numpy())
    stripes = 2 if length == 1 else 1
    per = tile//stripes
    largest = 0.0
    for warp in range(4 if length == 1 else 8):
        hk,sub = warp%2,warp//2
        qs = shared_q(q[hk].view(torch.uint8).numpy())
        rbase = 0 if length == 1 else 16*sub
        keybase = sub*per if length == 1 else 0
        score = []
        for nt in range(per//8):
            c = np.zeros((32,4),np.float32)
            for ks in range(2):
                a,b = np.empty((32,4),np.uint32),np.empty((32,2),np.uint32)
                for lane in range(32):
                    g,t = divmod(lane,4)
                    for hf in range(2):
                        d = 32*ks+16*hf+4*t
                        a[lane,2*hf] = load_word(qs,qo(rbase+g,d))
                        a[lane,2*hf+1] = load_word(qs,qo(rbase+g+8,d))
                    r = keybase+8*nt+g
                    b[lane,0] = load_word(sk,ko(r,hk*64+32*ks+4*t))
                    b[lane,1] = load_word(sk,ko(r,hk*64+32*ks+16+4*t))
                c += mma(a,b,True)
            score.append(c)
        score = np.stack(score)
        full = np.empty((16,per),np.float32)
        for nt in range(per//8):
            for lane in range(32):
                g,t = divmod(lane,4)
                for e in range(4):
                    full[g+(e//2)*8,8*nt+2*t+e%2] = score[nt,lane,e]
        # Expected values: independently decoded FP8 matrix product.
        expected = q[hk,rbase:rbase+16].float().numpy() @ k[keybase:keybase+per,hk].float().numpy().T
        np.testing.assert_allclose(full,expected,rtol=1e-6,atol=2e-5)
        # Arbitrary positive probabilities test PV independently of softmax/QK.
        probs = np.random.default_rng(seed+warp).random((16,per)).astype(np.float16)
        for nt in range(per//8):
            for lane in range(32):
                g,t = divmod(lane,4)
                for e in range(4):
                    score[nt,lane,e] = probs[g+(e//2)*8,8*nt+2*t+e%2]
        o = np.zeros((8,32,4),np.float32)
        for ks in range(per//16):
            b0,b1 = value_frag(sv,keybase+16*ks,hk)
            a = np.empty((32,4),np.uint32)
            for lane in range(32):
                a[lane] = (half_word(score[2*ks,lane,:2]),half_word(score[2*ks,lane,2:]),
                           half_word(score[2*ks+1,lane,:2]),half_word(score[2*ks+1,lane,2:]))
            for d in range(8):
                o[d] += mma(a,np.stack((b0[:,d],b1[:,d]),1),False)
        got = np.empty((16,64),np.float32)
        for lane in range(32):
            g,t = divmod(lane,4)
            for rr in range(2):
                for d in range(8):
                    got[g+8*rr,16*t+d] = o[d,lane,2*rr]
                    got[g+8*rr,16*t+8+d] = o[d,lane,2*rr+1]
        # Expected values: independent P@V with FP16 P and exact FP8 V.
        want = probs.astype(np.float32) @ v[keybase:keybase+per,hk].float().numpy()
        np.testing.assert_allclose(got,want,rtol=2e-5,atol=2e-5)
        largest = max(largest,float(np.abs(got-want).max()))
    return largest


def merge(parts):
    """Stable merge of (m,l,numerator), including empty splits."""
    m = torch.stack([p[0] for p in parts]).amax(0)
    weights = [torch.where(torch.isfinite(m),torch.exp2(p[0]-m),0) for p in parts]
    out = sum(w[...,None]*p[2] for w,p in zip(weights,parts))
    den = sum(w*p[1] for w,p in zip(weights,parts))
    return m,den,out


def emulate(data,tile,splits,*,half=False,micro=False):
    """Numerical model of every tile/stripe/split, not a call to softmax."""
    cfg = data['config']
    B,L = cfg['batch'],cfg['q_len']
    qi,ki,vi,gate = public.split_qkv(data)
    q,k = public._rms(qi),public._rms(ki)
    pos = data['positions'].view(B,L)
    if cfg['rope_theta']:
        q,k = public._rope(q,pos,cfg['rope_theta']),public._rope(k,pos,cfg['rope_theta'])
    q = q.to(torch.float8_e4m3fn).float().view(B,L,2,8,64)
    kn,vn = public._quant(k,data['k_scale']),public._quant(vi,data['v_scale'])
    kc,vc = data['k_cache'].view(-1,2,64).clone(),data['v_cache'].view(-1,2,64).clone()
    kc[data['cache_loc']],vc[data['cache_loc']] = kn.flatten(0,1),vn.flatten(0,1)
    all_y = []
    partitions = []
    for b in range(B):
        seq = int(data['seq_lens'][b])
        start = max(0,seq-cfg['window']+1) if cfg['window'] else 0
        end = seq+L
        positions = torch.arange(start,end)
        rows = data['page_table'][b,positions//cfg['page_size']].long()*cfg['page_size']+positions%cfg['page_size']
        K,V = kc[rows].float(),vc[rows].float()
        S = torch.einsum('lkgd,tkd->lkgt',q[b],K)*cfg['qk_scale_factor']*cfg['sm_scale']*data['k_scale']*math.log2(math.e)
        pquery = seq+torch.arange(L)
        visible = positions[None,:] <= pquery[:,None]
        if cfg['window']:
            visible &= positions[None,:] > pquery[:,None]-cfg['window']
        S.masked_fill_(~visible[:,None,None,:],-math.inf)
        tiles = (end-start+tile-1)//tile
        coverage = torch.zeros(end-start,dtype=torch.int32)
        parts = []
        for split in range(splits):
            lo,hi = tiles*split//splits,tiles*(split+1)//splits
            stripe_parts = []
            for stripe in range(2 if L == 1 else 1):
                m = torch.full((L,2,8),-math.inf)
                l = torch.zeros(L,2,8)
                o = torch.zeros(L,2,8,64)
                for it in range(lo,hi):
                    count = tile//2 if L==1 else tile
                    step = min(64,count) if micro else count
                    for offset in range(0,count,step):
                        at = it*tile+stripe*count+offset
                        to = min(at+step,end-start)
                        if at >= to:
                            continue
                        coverage[at:to] += 1
                        score = S[...,at:to]
                        mx = score.amax(-1)
                        mn = torch.maximum(m,mx)
                        alpha = torch.where(torch.isfinite(mn),torch.exp2(m-mn),0)
                        P = torch.where(torch.isfinite(mn)[...,None],torch.exp2(score-mn[...,None]),0)
                        o *= alpha[...,None]
                        o += torch.einsum('lkgt,tkd->lkgd',P.half().float(),V[at:to])
                        l = l*alpha+P.sum(-1)
                        m = mn
                stripe_parts.append((m,l,o))
            parts.append(merge(stripe_parts))
        assert torch.equal(coverage,torch.ones_like(coverage)), 'token dropped or read twice across splits'
        if half:
            m = torch.stack([p[0] for p in parts]).amax(0)
            out,den = torch.zeros_like(parts[0][2]),torch.zeros_like(parts[0][1])
            for pm,pl,po in parts:
                mean = torch.where(pl[...,None]>0,po/pl[...,None],0).half().float()
                assert torch.isfinite(mean).all(), 'normalized FP16 partial overflow'
                weight = torch.where(pl>0,torch.exp2(pm-m)*pl,0)
                out += weight[...,None]*mean
                den += weight
        else:
            _,den,out = merge(parts)
        y = out*(data['v_scale']/den[...,None])
        # The CUDA kernel honors the real BF16 activation boundary before RMS.
        y = y.bfloat16().float().reshape(L,16,64)
        all_y.append(y)
        partitions.append([tiles*s//splits for s in range(splits+1)])
    y = torch.stack(all_y)
    output = (public._rms(y)*torch.sigmoid(gate.float()).view(B,L,16,64)).flatten(2).bfloat16()
    return output,kn,vn,partitions


def integration_tests():
    summaries = []
    cases = [
        (1,1,0,'sliding','normal',1,1,64,37),
        (1,1,1,'full','normal',64,1,128,11),
        (3,1,65,'full','normal',64,1/64,64,7),
        (2,8,127,'full','normal',1,1,64,13),
        (2,8,129,'full','coherent',64,1,64,3),
        (2,1,2047,'sliding','edges',1,1,64,33),
        (3,1,2048,'sliding','soft',64,1/64,128,7),
        (2,8,2049,'sliding','edges',1,1,64,37),
        (3,8,4097,'sliding','soft',1,1/64,64,11),
        (2,8,4097,'full','coherent',1,1,64,9),
        (1,1,80000,'full','soft',1,1,128,252),
        (1,8,80000,'full','coherent',1,1,64,128),
        (1,8,80000,'sliding','edges',1,1,64,37),
        (3,8,5000,'sliding','soft',1,1,64,11),
        (3,1,4000,'full','soft',64,1,128,17),
    ]
    torch.set_num_threads(4)
    for i,(B,L,ctx,layer,dist,page,scale,tile,splits) in enumerate(cases):
        spread = 'mixed' if i >= 13 else 'fixed'
        d = public.generate_input(B,ctx,'tp8',layer,1234+i,L,spread=spread,
                                  dist=dist,page=page,kv_scale=scale,device='cpu')
        # Reference accepts empty context, although its generator sizes the
        # cache pool to hold only new tokens in that case.
        out,k,v,parts = emulate(d,tile,splits)
        # Expected: pristine reference FP8-query computation, attending exactly
        # the candidate's newly quantized cache entries, as the judge does.
        ref = public.ref_kernel(d,torch.float8_e4m3fn,new_kv=(k,v))[0].bfloat16()
        rel = float((out.float()-ref.float()).norm()/ref.float().norm().clamp(min=1e-12))
        assert torch.isfinite(out).all()
        assert rel < 0.005, (i,rel)
        _,rk,rv = public.ref_kernel(d)
        assert torch.equal(k.view(torch.uint8),rk.view(torch.uint8))
        assert torch.equal(v.view(torch.uint8),rv.view(torch.uint8))
        # Expected-value source: public check_implementation's independent
        # BF16-query reference, precision allowance, row bounds and cache check.
        raw = torch.cat([t.contiguous().view(torch.uint8).flatten() for t in (out,k,v)])
        judge_ok,judge_message = public.check_implementation(d,raw)
        assert judge_ok, judge_message
        summaries.append(dict(B=B,L=L,ctx=ctx,layer=layer,dist=dist,page=page,scale=scale,
                              spread=spread,tile=tile,splits=splits,judge=judge_message,
                              relative_l2_vs_fp8_reference=rel,
                              split_boundaries=parts[0]))
    return summaries


def static_tests(candidate,*,v2=False):
    tree = ast.parse(candidate.read_text())
    strings = {n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
               if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
               and n.targets[0].id in ('RR_CUDA','RR_CPP')}
    assert 'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32' in strings['RR_CUDA']
    assert 'wgmma' not in strings['RR_CUDA']
    assert 'atomicAdd' not in strings['RR_CUDA']
    assert 'memory_order_acq_rel' in strings['RR_CUDA']
    assert 'getCurrentCUDAStream' in strings['RR_CUDA']
    finish = 'finish<L,NT,HALF>' if v2 else 'finish<L,NT>'
    assert finish+'(p,blockIdx.y,begin,min(begin+NT/8,L*16))' in strings['RR_CUDA']
    # Prove all shared byte addresses are bijections and naturally aligned.
    for tile in (64,128,256):
        addresses = [ko(r,c) for r in range(tile) for c in range(128)]
        assert sorted(addresses) == list(range(tile*128))
        assert all(ko(r,c)%16 == 0 for r in range(tile) for c in range(0,128,16))
        assert all(ko(r,c)%8 == 0 for r in range(tile) for c in range(0,128,8))
    for rows in (16,64):
        assert sorted(qo(r,c) for r in range(rows) for c in range(64)) == list(range(rows*64))
    # Every output element has exactly one warp/lane owner, including the
    # nonstandard column permutation introduced by the register V transpose.
    for L in (1,8):
        written = []
        for warp in range(4 if L==1 else 8):
            hk,sub=warp%2,warp//2
            if L==1 and sub:
                continue
            for lane in range(32):
                g,t=divmod(lane,4)
                for rr in range(1 if L==1 else 2):
                    qr=(0 if L==1 else 16*sub)+g+8*rr
                    j,h=qr//8,hk*8+qr%8
                    written.extend((j,h,16*t+d) for d in range(16))
        assert len(written)==len(set(written))==L*16*64
        assert set(written)=={(j,h,d) for j in range(L) for h in range(16) for d in range(64)}
    # Import must be CPU-safe and must not select a route on an untested device.
    spec=importlib.util.spec_from_file_location('restart_candidate',candidate)
    module=importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    assert module._RR_SELECTED=={} and module._RR_MODULE is None
    assert not module._RR_WS
    return dict(python_ast='pass',cpu_import='pass',shared_address_bijection='pass',
                output_ownership='pass',new_core_uses_wgmma=False,new_core_float_atomics=False)


def rope_and_pipeline_tests():
    # Expected source: sin/cos of the reference's FP32 angle, evaluated in FP64.
    freq = 1/(500000.0**(torch.arange(0,64,2).float()/64))
    angles=(torch.tensor([0,1,2047,80000,1000000,2097151]).float()[:,None]*freq).numpy()
    k=np.rint(angles*np.float32(0.15915493667125702)).astype(np.float32)
    r=angles.copy()
    for c in (6.25,0.03318530693650246,2.430837753308879e-10):
        # A product of FP32 inputs is exact in FP64; one conversion models FMA.
        r=(-k.astype(np.float64)*np.float64(np.float32(c))+r.astype(np.float64)).astype(np.float32)
    error=max(float(np.abs(np.sin(r.astype(np.float64))-np.sin(angles.astype(np.float64))).max()),
              float(np.abs(np.cos(r.astype(np.float64))-np.cos(angles.astype(np.float64))).max()))
    assert error<2e-6, error
    schedules=0
    for tiles in (0,1,2,3,4,7,32,33,126,127,128,1344):
        # Copy issue follows cp.wait_group<1> + CTA barrier. At iteration i,
        # consumers through i-1 have all finished, so stage i+2 is safe to reuse.
        resident={i%3:i for i in range(min(2,tiles))}
        for i in range(tiles):
            assert resident[i%3]==i
            nxt=i+2
            if nxt<tiles:
                old=resident.get(nxt%3,-1)
                assert old<=i-1 and nxt%3!=i%3
                resident[nxt%3]=nxt
            schedules+=1
    return dict(rope_angle_max_abs_error=error,ring_slot_iterations=schedules,
                limitation='does not model GPU instruction timing, SFU error or memory ordering')


def manufactured_tests():
    result=[]
    for scale in (1.0,1/64):
        for value in (0.0,1.0):
            d=public.generate_input(1,129,'tp8','full',44,q_len=8,page=1,
                                    kv_scale=scale,device='cpu')
            d['v_cache'].fill_(value/scale)
            d['qkv'][...,1152:1280]=value
            d['qkv'][...,1280:]=0
            out,*_=emulate(d,64,11)
            # Analytical source: a constant V is invariant under any normalized
            # attention weights. Zero gate => sigmoid=1/2. RMS(ones)=ones (up
            # to epsilon, which rounds away in BF16); RMS(zeros)=zeros.
            expected=torch.full_like(out,value/2)
            torch.testing.assert_close(out,expected,rtol=0,atol=0)
            result.append(dict(scale=scale,constant_value=value,result='exact analytical BF16 output'))
    return result


def main():
    candidate=ROOT/'candidates/attention_restart.py'
    # Assembly is byte-for-byte incumbent plus an independently readable core.
    incumbent=ROOT/'board_best/attention_kb1h2_scored_2.6536.py'
    tail=ROOT/'.tools/attention_restart_tail.py'
    candidate.write_text(incumbent.read_text()+tail.read_text())
    assert candidate.read_bytes().startswith(incumbent.read_bytes())
    result={'candidate':str(candidate),
            'candidate_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
            'fallback_prefix_byte_identical':True,
            'validation_scope':'CPU algorithm/layout only; not GPU execution'}
    result['static']=static_tests(candidate)
    result['rope_and_pipeline']=rope_and_pipeline_tests()
    result['manufactured']=manufactured_tests()
    result['fragments']=[dict(L=L,tile=T,max_abs_pv_error=fragment_test(L,T,900+L+T))
                         for L,T in ((1,64),(1,128),(8,64))]
    result['numerics']=integration_tests()
    result['cuda_compile']='skipped: no CUDA toolkit, NVRTC, ptxas or GPU on this ARM macOS host'
    (ROOT/'results/attn_restart_cpu.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='numerics'},indent=2))
    print('PASS %d numerical cases; worst relative L2 %.7f' %
          (len(result['numerics']),max(c['relative_l2_vs_fp8_reference'] for c in result['numerics'])))


if __name__=='__main__':
    main()
