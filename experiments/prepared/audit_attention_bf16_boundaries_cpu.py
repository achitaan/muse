"""CPU evidence for BF16 stage boundaries, not a substitute for GPU tests.

Oracle: pinned SGLang a55ab7d5 fused_qk_norm.py lines 75..119, 299..308;
baseline.py's BF16 cast before rms_norm_gate. Torch BF16 conversion is compared
with an independent NumPy round-to-nearest-even implementation. The arithmetic
model uses separate FP32 operations; it does not claim CUDA FMA bit equivalence.
"""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
torch.set_num_threads(2)
rng = np.random.default_rng(20261028)


def bf16(x):
    values = np.ascontiguousarray(x, dtype=np.float32)
    bits = values.view(np.uint32)
    assert np.isfinite(values).all()
    rounded = ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000))
    return rounded.view(np.float32)


def torch_bf16(x):
    return torch.from_numpy(np.ascontiguousarray(x)).bfloat16().float().numpy()


def rotate(x, cs, sn):
    a, b = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = a * cs - b * sn
    out[:, 1::2] = a * sn + b * cs
    return out


def fp8bits(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()


rows, d = 8192, 64
raw = bf16(rng.normal(size=(rows,d)).astype(np.float32))
raw[:4] = 0
raw[4:8] = 0
raw[np.arange(4,8), [0,1,31,63]] = [8,-8,1e-6,1e6]
norm = raw * np.reciprocal(np.sqrt(np.mean(raw * raw, axis=-1, keepdims=True) + np.float32(1e-5)))
angle = rng.uniform(-np.pi,np.pi,size=(rows,d//2)).astype(np.float32)
cs, sn = np.cos(angle), np.sin(angle)
norm_bf = bf16(norm)
assert np.array_equal(norm_bf.view(np.uint32),torch_bf16(norm).view(np.uint32))
rope_float = rotate(norm_bf,cs,sn)
rope_bf = bf16(rope_float)
assert np.array_equal(rope_bf.view(np.uint32),torch_bf16(rope_float).view(np.uint32))
# Independent tensor composition for full and sliding Q/K materializations.
tn = torch.from_numpy(norm).bfloat16().float()
tc,ts = torch.from_numpy(cs),torch.from_numpy(sn)
tr = torch.stack([tn[:,0::2]*tc-tn[:,1::2]*ts,tn[:,0::2]*ts+tn[:,1::2]*tc],dim=-1).flatten(-2).bfloat16().float().numpy()
assert np.array_equal(rope_bf.view(np.uint32),tr.view(np.uint32))
mutants = {}
for scale in (1.,1./64):
    for name, full, wrong in (
        ('full_missing_norm',norm_bf,norm),
        ('sliding_missing_norm',rope_bf,bf16(rotate(norm,cs,sn))),
        ('sliding_missing_rope',rope_bf,rope_float)):
        expected = fp8bits(full/np.float32(scale))
        mutants[f'{name}_scale_{scale}'] = int(np.count_nonzero(expected != fp8bits(wrong/np.float32(scale))))
        assert mutants[f'{name}_scale_{scale}'] > 0
        # Power-of-two scaling of normal BF16 activations preserves BF16 exactly.
        assert np.array_equal((full/np.float32(scale)).view(np.uint32), bf16(full/np.float32(scale)).view(np.uint32))
out = rng.normal(0,.07,size=(rows,d)).astype(np.float32)
rounded_out = bf16(out)
assert np.array_equal(rounded_out.view(np.uint32),torch_bf16(out).view(np.uint32))
gate = rng.normal(size=(rows,d)).astype(np.float32)
sigmoid = 1/(1+np.exp(-gate))
def gated_norm(y):
    return bf16(y / np.sqrt(np.mean(y*y,axis=-1,keepdims=True)+np.float32(1e-5))*sigmoid)
mutants['missing_attention_output'] = int(np.count_nonzero(gated_norm(out).view(np.uint32) != gated_norm(rounded_out).view(np.uint32)))
assert mutants['missing_attention_output'] > 0
# RNE ties, sign and immediate FP32 neighbors around representable BF16 midpoints.
bases = np.arange(0x3e00,0x4200,dtype=np.uint32)<<16
ties = (bases + np.uint32(0x8000)).view(np.float32)
neighbors = np.concatenate([ties, np.nextafter(ties,np.float32(np.inf)),np.nextafter(ties,np.float32(-np.inf))])
neighbors = np.concatenate([neighbors,-neighbors,np.array([0.,-0.],np.float32)])
assert np.array_equal(bf16(neighbors).view(np.uint32),torch_bf16(neighbors).view(np.uint32))

source = (ROOT/'.tools/attention-bf16-boundaries-variant.py').read_bytes()
tree = ast.parse(source)
cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
assert cuda.count('x[e] = bf16_round(x[e] * r);') == 3
assert cuda.count('x[2 * e] = bf16_round(a0 * c - a1 * sn);') == 3
assert cuda.count('x[2 * e + 1] = bf16_round(a0 * sn + a1 * c);') == 3
assert cuda.count('attn_out_bf16(') == 16
assert 'pr8(' not in cuda and '__float2bfloat16_rn(x)' in cuda
# Execute only the standalone PyTorch fallback on nearby unsupported geometry.
scope = {'torch':torch,'_inv_freq':lambda theta,device,n: 1/(theta**(torch.arange(0,2*n,2,device=device).float()/(2*n)))}
functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_rms','_rope','_fallback')]
fallback_text=ast.unparse(ast.Module(body=functions,type_ignores=[]))
# Torch 2.3 CPU lacks Float8 advanced indexing; index the same bytes as uint8.
# The submitted GPU path is unchanged. Arithmetic and conversion stay identical.
for name,value in [('k_cache','k / k_scale'),('v_cache','v_in.reshape(batch, length, hkv, d).float() / v_scale')]:
    before=f'{name}[slots] = ({value}).to(torch.float8_e4m3fn)'
    after=f'{name}.view(torch.uint8)[slots] = ({value}).to(torch.float8_e4m3fn).view(torch.uint8)'
    assert fallback_text.count(before)==1
    fallback_text=fallback_text.replace(before,after)
    fallback_text=fallback_text.replace(f'{name}[rows].float()',f'{name}.view(torch.uint8)[rows].view(torch.float8_e4m3fn).float()')
exec(compile(fallback_text,'<fallback-byte-indexing>','exec'),scope)
gen = torch.Generator().manual_seed(20261028)
fallback_results = []
for window in (0,4):
    for page in (1,8):
        for length in (1,3,8):
            B,HQ,HK,D,S = 2,8,1,16,32
            positions=torch.arange(4,4+length).repeat(B,1)
            page_table=torch.arange(B*S//page).reshape(B,-1)
            slots=page_table.gather(1,positions//page)*page+positions%page
            qkv=torch.randn(B,length,(2*HQ+2*HK)*D,generator=gen).bfloat16()
            kc=torch.randn(B*S,HK,D,generator=gen).to(torch.float8_e4m3fn)
            vc=torch.randn(B*S,HK,D,generator=gen).to(torch.float8_e4m3fn)
            cfg=dict(batch=B,q_len=length,num_q_heads=HQ,num_kv_heads=HK,head_dim=D,page_size=page,window=window,rope_theta=10000. if window else 0,sm_scale=D**-.5,qk_scale_factor=5.473)
            data=dict(qkv=qkv,positions=positions,cache_loc=slots,page_table=page_table,k_cache=kc,v_cache=vc,k_scale=torch.tensor(1.),v_scale=torch.tensor(1.),config=cfg)
            got=scope['_fallback'](data)
            assert got.shape==(B,length,HQ*D) and got.dtype==torch.bfloat16 and torch.isfinite(got.float()).all()
            # Independent per-token/per-head FP64 mathematical oracle, with BF16
            # materializations at the pinned operation boundaries.
            q,k,v,g=qkv.split([HQ*D,HK*D,HK*D,HQ*D],-1)
            def qkn(x,h):
                y=x.reshape(B,length,h,D).double()
                return (y/torch.sqrt((y*y).mean(-1,keepdim=True)+1e-5)).bfloat16().double()
            q,k=qkn(q,HQ),qkn(k,HK)
            if window:
                inv=1/(10000.**(torch.arange(0,D,2).double()/D))
                a=positions.double()[...,None,None]*inv
                def rot(y):
                    z=torch.stack([y[...,0::2]*a.cos()-y[...,1::2]*a.sin(),y[...,0::2]*a.sin()+y[...,1::2]*a.cos()],-1).flatten(-2)
                    return z.bfloat16().double()
                q,k=rot(q),rot(k)
            # Check fresh cache writes as well as outputs; old cache remains the
            # actual independently generated FP8 tensor on every case.
            expected_k=k.float().to(torch.float8_e4m3fn)
            assert torch.equal(kc.view(torch.uint8)[slots],expected_k.view(torch.uint8))
            result=[]
            for b in range(B):
                cache_rows=page_table[b,torch.arange(S)//page]*page+torch.arange(S)%page
                ks=kc.view(torch.uint8)[cache_rows].view(torch.float8_e4m3fn).double()
                vs=vc.view(torch.uint8)[cache_rows].view(torch.float8_e4m3fn).double()
                qh=q[b].permute(1,0,2)
                logits=qh@ks[:,0].T*(D**-.5*5.473)
                visible=torch.arange(S)[None,:]<=positions[b,:,None]
                if window: visible &= torch.arange(S)[None,:]>positions[b,:,None]-window
                weights=logits.masked_fill(~visible[None],float('-inf')).softmax(-1)
                y=(weights@vs[:,0]).permute(1,0,2).bfloat16().double()
                y=y/torch.sqrt((y*y).mean(-1,keepdim=True)+1e-5)
                result.append((y*torch.sigmoid(g[b].reshape(length,HQ,D).double())).bfloat16().float())
            expected=torch.stack(result).reshape_as(got)
            relative_l2=float(torch.linalg.vector_norm(got.float()-expected)/torch.linalg.vector_norm(expected))
            assert relative_l2<.012,relative_l2
            fallback_results.append(dict(window=window,page=page,length=length,relative_l2=relative_l2))

report=dict(source_sha256=hashlib.sha256(source).hexdigest(),seed=20261028,rows=rows,head_dim=d,rounding_values=3*rows*d+len(neighbors),mutant_differences=mutants,fallback=fallback_results,cpu_float8_indexing='uint8 view of identical bytes; arithmetic unchanged',torch=torch.__version__,numpy=np.__version__,gpu_verified=False)
(ROOT/'.tools/attention-bf16-boundaries-cpu-audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
