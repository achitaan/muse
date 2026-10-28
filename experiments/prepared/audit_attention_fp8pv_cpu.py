"""Independent byte permutation/PV fragment and scaled FP8 probability evidence."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_attention_fp8pv import ROOT,BASE_SHA,literal,transform,parent_source
sys.path.insert(0,str(ROOT/'scripts'))
from verify_attention_precision import verify

torch.set_num_threads(2)
rng=np.random.default_rng(20261028)
source=(ROOT/'.tools/attention-fp8pv-variant.py').read_bytes()
cuda=literal(source)[1]
verify(source)
assert transform(literal(parent_source())[1])[0]==cuda
assert 'const float pv = ex2(s[nt][e] - m_new) * 256.f;' in cuda
helper=cuda[cuda.index('__device__ __forceinline__ void load_v_frags_f8'):cuda.index('// the merge / barrier fences')]
assert '__shfl' not in helper and 'cvt2(' not in helper
assert 'const int k0 = kb + 16 * h + 2 * tig;' in helper
assert '0x5140' in helper and '0x7362' in helper and '0x5410' in helper and '0x7632' in helper

def pack(values):return sum(int(v)<<(8*i) for i,v in enumerate(values))
def unpack(word):return np.array([(word>>(8*i))&255 for i in range(4)],np.int64)
def prmt(a,b,selector):
    values=list(unpack(a))+list(unpack(b))
    return pack([values[(selector>>(4*i))&7] for i in range(4)])
def transpose4(v):
    dst=np.empty(8,np.uint32)
    for z in range(2):
        a,b,c,d=[pack(row[4*z:4*z+4]) for row in v]
        e01,e23=prmt(a,b,0x5140),prmt(c,d,0x5140)
        o01,o23=prmt(a,b,0x7362),prmt(c,d,0x7362)
        dst[4*z:4*z+4]=[prmt(e01,e23,0x5410),prmt(e01,e23,0x7632),
                         prmt(o01,o23,0x5410),prmt(o01,o23,0x7632)]
    assert np.array_equal(np.stack([unpack(x) for x in dst]),np.asarray(v).T)
    return dst

matrices=0;byte_transposes=0
for trial in range(8):
    p=rng.integers(0,256,size=(2,8,64),dtype=np.int64)
    v=rng.integers(0,256,size=(64,2,64),dtype=np.int64)
    for hk in range(2):
        for wsub in range(2):
            a=np.zeros((16,32),np.int64)
            bs=np.empty((8,32,8),np.int64)
            counts=np.zeros((8,32),np.int32)
            bcounts=np.zeros((32,8),np.int32)
            for lane in range(32):
                g,tig=lane>>2,lane&3
                for h in range(2):
                    # Actual candidate A is packed directly from s[0/1]
                    # or s[2/3]: these hold 8*nt+2*tig+{0,1} keys.
                    physical=32*wsub+16*h+2*tig+np.array([0,1,8,9])
                    apack=p[hk,g,physical]
                    # Same four physical V rows, eight adjacent dimensions.
                    words=transpose4(v[physical,hk,8*g:8*g+8])
                    byte_transposes+=1
                    for byte in range(4):
                        logical=16*h+4*tig+byte
                        a[g,logical]=apack[byte]
                        counts[g,logical]+=1
                        for t in range(8):bs[t,logical,g]=unpack(words[t])[byte]
                        bcounts[logical,g]+=1
            assert (counts==1).all() and (bcounts==1).all()
            permutation=np.array([16*h+2*t+e+8*j for h in range(2) for t in range(4) for j in range(2) for e in range(2)])
            assert sorted(permutation.tolist())==list(range(32))
            assert np.array_equal(a[:8],p[hk][:,32*wsub+permutation])
            got=np.empty((8,64),np.int64)
            for t in range(8):
                assert np.array_equal(bs[t],v[32*wsub+permutation,hk][:,8*np.arange(8)+t])
                acc=a@bs[t]
                assert (acc[8:]==0).all()
                for lane in range(32):
                    g,tig=lane>>2,lane&3
                    for e in range(2):got[g,8*(2*tig+e)+t]=acc[g,2*tig+e]
                matrices+=1
            assert np.array_equal(got,p[hk,:,32*wsub:32*wsub+32]@v[32*wsub:32*wsub+32,hk])

def bf(x):return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)).bfloat16().float().numpy()
def f8(x):return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)).to(torch.float8_e4m3fn).float().numpy()
def gated(x,gate):
    y=bf(x)
    return bf(y/np.sqrt(np.mean(y*y,axis=-1,keepdims=True)+np.float32(1e-5))/(1+np.exp(-gate)))

errors=[];near_zero=[]
for scale in (1.,1./64):
    for coherent in (False,True):
        for context in (1,10,63,64,65,257,4096,80000):
            # Model retained QK FP16 input, which includes its score scale.
            qraw=bf(rng.normal(size=(8,64)))
            q=bf(qraw/np.sqrt(np.mean(qraw*qraw,-1,keepdims=True)+np.float32(1e-5)))
            qs=(q*np.float32((5.473/8)*np.log2(np.e)*scale)).astype(np.float16).astype(np.float32)
            k=f8(rng.normal(size=(context,64))/scale)
            v=f8(rng.normal(size=(context,64))/scale)
            if coherent:
                # Low-attention values share a direction; preserve their
                # collective contribution rather than dropping small P.
                direction=rng.normal(size=(1,64)).astype(np.float32)
                k=f8((.8*rng.normal(size=(1,64))+.2*rng.normal(size=(context,64)))/scale)
                v=f8((direction+.1*rng.normal(size=(context,64)))/scale)
            scores=qs@k.T
            # Stress rare maximum updates and a long coherent low-P tail.
            if context>=257:
                scores[:,:min(3,context)]+=np.float32(12.)
                scores[:,context//2:context//2+1]+=np.float32(18.)
            partials=[]
            for wsub in range(2):
                m=np.full((8,1),-np.inf,np.float32);den=np.zeros((8,1),np.float32);num=np.zeros((8,64),np.float32)
                for start in range(32*wsub,context,64):
                    end=min(start+32,context)
                    s=scores[:,start:end];newm=np.maximum(m,s.max(-1,keepdims=True))
                    alpha=np.exp2(m-newm);p=np.exp2(s-newm)*np.float32(256.)
                    den=den*alpha+p.sum(-1,keepdims=True)
                    num=num*alpha+f8(p)@v[start:end]
                    m=newm
                partials.append((m,den,num))
            maxm=np.maximum(partials[0][0],partials[1][0])
            den=np.zeros((8,1),np.float32);num=np.zeros((8,64),np.float32)
            for m,d,n in partials:
                alpha=np.exp2(m-maxm)
                den+=d*alpha;num+=n*alpha
            raw=num/den*np.float32(scale)
            p=np.exp2(scores.astype(np.float64)-scores.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            oracle=p@v.astype(np.float64)*scale
            gate=bf(rng.normal(size=(8,64)))
            got,expected=gated(raw,gate),gated(oracle,gate)
            rel=float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-12))
            assert np.isfinite(got).all() and rel<.03,(scale,coherent,context,rel)
            errors.append(rel)
            near_zero.append(float(np.max(np.linalg.norm(got-expected,axis=-1)/np.maximum(np.linalg.norm(expected,axis=-1),.1*np.mean(np.linalg.norm(expected,axis=-1))))))

# Exactly scaled representable E4M3 probabilities have a 256x smaller tail
# threshold than unscaled conversion; no thresholding of any key is added.
tails=np.exp2(np.arange(-25,1,dtype=np.float32))
assert np.count_nonzero(f8(tails*256))>np.count_nonzero(f8(tails))
report=dict(source_sha256=hashlib.sha256(source).hexdigest(),exact_integer_pv=True,
            operand_matrices=matrices,byte_transposes=byte_transposes,numerical_cases=len(errors),
            max_output_relative_l2=max(errors),max_per_row_relative_l2=max(near_zero),
            bf16_guard=True,unchanged_fp16_qk=True,no_extra_shared_storage=True,
            scope='CPU evidence with synthetic inputs; official held-out GPU coverage still required')
(ROOT/'.tools/attention-fp8pv-cpu-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report,indent=2))
