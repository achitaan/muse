"""CPU fragment/layout and scaled-softmax evidence; GPU coverage is still required.

Independent fragment oracle: NVIDIA PTX 9.7.16.5.10, m16n8k32.
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
The formulas below reconstruct complete A/B operands and C destinations. This
does not emulate tensor-core accumulation order or prove hardware performance.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_attention_fp8q import ROOT, BASE_SHA, literal, transform
sys.path.insert(0,str(ROOT/'scripts'))
from verify_attention_precision import verify

torch.set_num_threads(2)
rng = np.random.default_rng(20261028)
source = (ROOT/'.tools/attention-fp8q-variant.py').read_bytes()
cuda = literal(source)[1]
verify(source)
base = (ROOT/'best/attention_checked.py').read_bytes()
assert hashlib.sha256(base).hexdigest() == BASE_SHA
assert transform(literal(base)[1])[0] == cuda
assert 'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32' in cuda
assert '"r"(a0), "r"(0u), "r"(a2), "r"(b0), "r"(b1)' in cuda
newqk = cuda[cuda.index('  // PTX m16n8k32:'):cuda.index('    if (!interior) {',cuda.index('  // PTX m16n8k32:'))]
assert 'cvt2(' not in newqk and 'mma_f16(' not in newqk
# These are the actual candidate's address expressions, evaluated for every lane.
def expression(prefix):
    line = next(l for l in newqk.splitlines() if prefix in l)
    return line.split(prefix,1)[1].split(';',1)[0]
dexpr = expression('const int d = ')
offexpr = expression('const int off = ')
qaddress = next(l for l in newqk.splitlines() if 'qa[h][r] =' in l).split('(qs + ',1)[1].split(');',1)[0]
def evaluate(expr, **values):
    return eval(expr, {'__builtins__':{}}, values)

fragments = 0
bank_checks = 0
write_owners = 0
for trial in range(12):
    # Unique integer labels expose row/head/K-coordinate swaps exactly.
    q = rng.integers(-1000,1000,size=(2,8,64),dtype=np.int32)
    k = rng.integers(-1000,1000,size=(64,2,64),dtype=np.int32)
    qsmem = np.full((2*8*128,), -777777, np.int32)
    ksmem = np.full((64*128,), -888888, np.int32)
    for hk in range(2):
        for g in range(8):
            for part in range(8):
                slot = (hk*8+g)*128+part*16
                # The store stays inside its raw-read slot. Distinct owners
                # never overwrite bytes another thread still needs to read.
                own_read = set(range(slot,slot+16))
                own_write = set(range(slot,slot+8))
                assert own_write <= own_read
                qsmem[slot:slot+8] = q[hk,g,part*8:part*8+8]
                write_owners += 1
        for key in range(64):
            for dim in range(64):
                off = key*128+(((hk*4+dim//16)^(key&7))*16)+dim%16
                ksmem[off] = k[key,hk,dim]
    for hk in range(2):
        for wsub in range(2):
            got = np.empty((8,32),np.int64)
            for nt in range(4):
                acc = np.zeros((16,8),np.int64)
                for h in range(2):
                    a = np.zeros((16,32),np.int64)
                    b = np.empty((32,8),np.int64)
                    acount = np.zeros((8,32),np.int32)
                    bcount = np.zeros((32,8),np.int32)
                    for lane in range(32):
                        group, tig = lane>>2, lane&3
                        g = group
                        key = 32*wsub+8*nt+g
                        for r in range(2):
                            d = evaluate(dexpr,h=h,r=r,tig=tig)
                            qaoff = evaluate(qaddress,g=g,d=d) + hk*8*128
                            apack = qsmem[qaoff:qaoff+4]
                            rtotal = 2*h+r
                            bpoff = evaluate(offexpr,key=key,ROWB=128,hk=hk,r=rtotal,tig=tig)
                            bpack = ksmem[bpoff:bpoff+4]
                            # PTX A register0/2 has row group, 4*tig+byte
                            # plus16 for register2. Registers1/3 are zero.
                            for byte in range(4):
                                col = 4*tig+byte+16*r
                                a[group,col] = apack[byte]
                                b[col,group] = bpack[byte]
                                acount[group,col] += 1
                                bcount[col,group] += 1
                    assert (acount == 1).all() and (bcount == 1).all()
                    assert np.array_equal(a[:8],q[hk,:,h*32:h*32+32])
                    assert (a[8:] == 0).all()
                    assert np.array_equal(b,k[32*wsub+8*nt:32*wsub+8*nt+8,hk,h*32:h*32+32].T)
                    acc += a@b
                    fragments += 1
                # Independent PTX C mapping: row group, col 2*tig+element.
                for lane in range(32):
                    g,tig = lane>>2,lane&3
                    for e in range(2):
                        got[g,8*nt+2*tig+e] = acc[g,2*tig+e]
                assert (acc[8:]==0).all()
            assert np.array_equal(got,q[hk].astype(np.int64)@k[32*wsub:32*wsub+32,hk].astype(np.int64).T)
            for nt in range(4):
                for r in range(4):
                    banks=[]
                    for lane in range(32):
                        g,tig=lane>>2,lane&3
                        key=32*wsub+8*nt+g
                        off=evaluate(offexpr,key=key,ROWB=128,hk=hk,r=r,tig=tig)
                        assert off%4==0 and 0<=off<=64*128-4
                        banks.append((off//4)%32)
                    assert len(set(banks))==32
                    bank_checks += 1

def bf(x):
    return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)).bfloat16().float().numpy()
def f8(x):
    return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)).to(torch.float8_e4m3fn).float().numpy()

# FP8 query quantization is allowed. Compare the complete modeled path with an
# FP64 oracle using that same allowed query, retaining BF16 boundaries. Include
# two scales, RoPE/no-RoPE, coherent keys, edge/partial and single-token ranges.
errors=[]
for scale in (1.,1./64):
    for rope in (False,True):
        for context in (1,10,63,64,65,127,256,2049):
            qraw=bf(rng.normal(size=(8,64)))
            qr=bf(qraw/np.sqrt((qraw*qraw).mean(-1,keepdims=True)+np.float32(1e-5)))
            if rope:
                angle=np.float32(79999)*(1/(10000.**(np.arange(32,dtype=np.float32)/32)))
                c,s=np.cos(angle),np.sin(angle)
                x,y=qr[:,::2].copy(),qr[:,1::2].copy()
                qr[:,::2]=bf(x*c-y*s);qr[:,1::2]=bf(x*s+y*c)
            q8=f8(qr)
            kr=bf(rng.normal(size=(context,64))+.7*rng.normal(size=(1,64)))
            kr=bf(kr/np.sqrt((kr*kr).mean(-1,keepdims=True)+np.float32(1e-5)))
            k8=f8(kr/scale)
            v8=f8(rng.normal(size=(context,64))/scale)
            sl2=np.float32((5.473/8)*np.log2(np.e)*scale)
            m=np.full((8,1),-np.inf,np.float32);den=np.zeros((8,1),np.float32);num=np.zeros((8,64),np.float32)
            for start in range(0,context,64):
                end=min(start+64,context)
                score=(q8@k8[start:end].T)*sl2
                newm=np.maximum(m,score.max(-1,keepdims=True))
                alpha=np.exp2(m-newm)
                p=np.exp2(score-newm)
                den=den*alpha+p.sum(-1,keepdims=True)
                num=num*alpha+p.astype(np.float16).astype(np.float32)@v8[start:end]
                m=newm
            got=bf(num/den*scale)
            logs=(q8.astype(np.float64)@k8.astype(np.float64).T)*float(sl2)
            p=np.exp2(logs-logs.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            expected=bf(p@v8.astype(np.float64)*scale)
            error=float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-12))
            assert error<.01,(scale,rope,context,error)
            errors.append(error)

report=dict(source_sha256=hashlib.sha256(source).hexdigest(),fragment_matrices=fragments,
            exact_integer_qk=True,write_owner_checks=write_owners,conflict_free_warp_loads=bank_checks,
            numerical_cases=len(errors),max_relative_l2_vs_fp8q_oracle=max(errors),
            native_bf16_boundary_guard=True,scope='CPU evidence; official GPU check still required')
(ROOT/'.tools/attention-fp8q-cpu-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps(report,indent=2))
