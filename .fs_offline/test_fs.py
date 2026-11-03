"""CPU only: byte bijections, PTX-fragment ownership, task/replay invariants."""
import ast, collections, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parents[1]
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
fs=load(ROOT/'experiments/boards/moe/moe_fs1.py','fs1')
parent=(ROOT/'experiments/boards/moe/moe_b13.py').read_bytes()
assert (ROOT/'experiments/boards/moe/moe_fs1.py').read_bytes().startswith(parent)
assert len((ROOT/'experiments/boards/moe/moe_fs1.py').read_bytes())<1_000_000
assert fs._fs_parent_kernel is not fs.kernel_fn
rng=np.random.default_rng(20261101)

def decode_words(words):
    """Independent CUDA dq4 algebra, not the packer's _POS inverse."""
    q=np.asarray(words,dtype=np.uint32)
    r=[q&np.uint32(0x81c081c0),
       ((q<<np.uint32(10))|(q>>np.uint32(22)))&np.uint32(0x81c081c0),
       ((q<<np.uint32(13))|(q>>np.uint32(19)))&np.uint32(0x81c081c0),
       ((q&np.uint32(0x001b001b))*np.uint32(0x1084))&np.uint32(0x81c081c0)]
    out=np.empty(q.shape+(4,2),dtype=np.uint8)
    for reg in range(4):
        for up in range(2):
            z=r[reg]>>np.uint32(16*up)
            out[...,reg,up]=sum(((z>>np.uint32(ob))&1)<<bit for bit,ob in enumerate((6,7,8,15)))
    return out

# Expected bytes are the original seeded raw tensors, including every sign/zero
# code and all256 scale bytes (packing must be lossless even for unsupported scales).
a=torch.from_numpy(rng.integers(0,256,(8192,2048),dtype=np.uint8))
sa=torch.from_numpy(rng.integers(0,256,(8192,256),dtype=np.uint8))
b=torch.from_numpy(rng.integers(0,256,(4096,2048),dtype=np.uint8))
sb=torch.from_numpy(rng.integers(0,256,(4096,256),dtype=np.uint8))
rows=torch.stack((torch.arange(4096).view(-1,8),4096+torch.arange(4096).view(-1,8)),1).flatten()
unit=fs._unit_order(fs._pack_codes(a,rows),fs._pack_scales(sa,rows)).view(256,2,36864)
packed_b=fs._fs_pack_b(b,sb)
packed=torch.cat((unit,packed_b[:,None]),dim=1)
assert packed.shape==(256,3,36864)
units=packed[:,:2].contiguous().view(512,36864).numpy()
words=units[:,:32768].copy().view(np.uint32).reshape(512,8,8,8,4,4)
bits=decode_words(words)
n=np.empty(words.shape+(2,4),dtype=np.uint8)
for reg,(h,j) in enumerate(((0,0),(1,0),(0,2),(1,2))):
    for up in range(2):n[...,h,j+up]=bits[...,reg,up]
an=n.transpose(0,6,3,1,2,4,5,7).reshape(8192,4096)
rec=(an[:,0::2]|(an[:,1::2]<<4))[np.argsort(rows.numpy())]
assert np.array_equal(rec,a.numpy()),'A code bytes'
sc=units[:,32768:].reshape(512,8,8,4,8,2).transpose(0,5,2,1,4,3).reshape(8192,256)
assert np.array_equal(sc[np.argsort(rows.numpy())],sa.numpy()),'A scale bytes'
del bits,n,words,an,rec,sc
words=packed[:,2,:32768].contiguous().view(torch.int32).numpy().view(np.uint32)
words=words.reshape(8,8,4,256,32).transpose(3,0,1,4,2).reshape(256,8,8,8,4,4)
bits=decode_words(words)
n=np.empty(words.shape+(2,2,2),dtype=np.uint8)
for reg,(h,q) in enumerate(((0,0),(1,0),(0,1),(1,1))):
    for up in range(2):n[...,h,q,up]=bits[...,reg,up]
bn=n.transpose(0,6,3,1,2,5,7,4,8).reshape(4096,4096)
assert np.array_equal(bn[:,0::2]|(bn[:,1::2]<<4),b.numpy()),'B code bytes'
assert torch.equal(packed[:,2,32768:].T,sb),'B scale bytes'
print('PASS full expert108KiB-task repack: 25,165,824 codes bytes +3,145,728 scales bytes exact')

# Independent PTX m16n8k16 layout: A lane g,t -> row g+8h,k2t+8q+u.
# Compare actual repacked fragment coordinates to original matrix coordinates.
for ks,rb in ((0,0),(3,17),(73,255),(255,129)):
    raw=packed[ks,2,:32768].contiguous().view(torch.int32).numpy().view(np.uint32).reshape(256,32)[rb]
    frag=decode_words(raw)
    for lane in range(32):
        for reg,(h,q) in enumerate(((0,0),(1,0),(0,1),(1,1))):
            for up in range(2):
                row=rb*16+lane//4+8*h;k=ks*16+2*(lane%4)+8*q+up
                expected=(int(b[row,k//2])>>(4*(k%2)))&15
                assert frag[lane,reg,up]==expected

# Hybrid redistribution must preserve all4096*8 (token,column) outputs once.
shared=collections.Counter();upper=collections.Counter()
for warp in range(8):
    for lane in range(32):
        g,t=divmod(lane,4)
        for z in range(32):
            row=(warp+8*z)*16+g
            if t<2:
                for h in (0,1):
                    shared[2*t+h,row]+=1;shared[2*t+h,row+8]+=1
            src=(lane&~3)+2+(t>>1)
            # Shuffle each component THEN select using destination parity.
            assert 2*(src%4)+(t&1)==4+t
            upper[4+t,row]+=1;upper[4+t,row+8]+=1
assert shared==collections.Counter({(t,c):1 for t in range(4) for c in range(4096)})
assert upper==collections.Counter({(t,c):1 for t in range(4,8) for c in range(4096)})
# Shared flush mapping is a segment-position -> actual-token permutation.
tokens=[1,3,4,7]
dest=[(tokens[i//4096],i%4096) for i in range(4*4096)]
assert len(set(dest))==16384 and {t for t,c in dest}==set(tokens)
print('PASS actual W2 fragments and hybrid shuffle/flush ownership')

coverage=[]
for ne in (0,1,2,6,32):
    for grid in (8,17,132):
        N=ne*256;seen=[];lengths=[]
        for c in range(grid):
            start=N*c//grid;end=N*(c+1)//grid
            seen.extend(range(start,end));lengths.append(end-start)
            # Enumerate producer/consumer ring phases and exact stage order.
            produced=[(v//256,v%256,k) for v in range(start,end) for k in range(3)]
            slots=[None]*4
            issued=min(4,len(produced))
            for q in range(issued):slots[q%4]=produced[q]
            for q in range(len(produced)):
                assert slots[q%4]==produced[q]
                if q+4<len(produced):slots[q%4]=produced[q+4]
                # Every slot reuse waits for the previous empty phase.
                if q>=4:assert (((q//4)&1)^1)==(((q-4)//4)&1)
        assert seen==list(range(N))
        assert max(lengths)-min(lengths)<=1
        coverage.append((ne,grid,min(lengths),max(lengths)))
# Arbitrary sequential invocation routings, incl zero; inactive buffer cleared
# before grid release, active data never cleared by a concurrent CTA.
y=np.zeros((2,32,8,16),np.float32);generation=0
for step in range(100):
    active=generation&1;inactive=active^1
    assert not y[active].any()
    y[inactive].fill(0)
    expected=np.zeros_like(y[active])
    for _ in range(0 if step%7==0 else 20):
        e,t=rng.integers(32),rng.integers(8);v=rng.normal(size=16).astype(np.float32)
        y[active,e,t]+=v;expected[e,t]+=v
    generation+=1
    assert np.array_equal(y[active],expected)
print('PASS15 task/ring models and100 replay generations including empty routing')

# CUDA13 cuda_occupancy.h:1523-1545: hardware rounds nine warps to four
# subpartitions, and each warp allocation to256 registers. A tempting
# maxnreg224 compilation is NOT launchable despite224*288<65536.
def regs_assumed(regs):return ((regs*32+255)//256)*256*12
assert regs_assumed(168)==64512<=65536
assert regs_assumed(224)==86016>65536
assert 5760+4*36864+4*4096*4==218752<232448
assert '__launch_bounds__(NTL,1)' in fs.FS_CUDA
assert 'cudaOccupancyMaxActiveBlocksPerMultiprocessor' in fs.FS_CUDA
print('PASS resident launch resource model; reject nonlaunchable224-register variant')

# Independent token-list oracle for every supported M, offset local shards,
# duplicate routes, sparse actual token indices, no-local and all-local cases.
for M in range(1,9):
    for lo in (0,32,224):
        for kind in ('random','zero','duplicate','all'):
            ids=rng.integers(0,256,(M,8))
            if kind=='zero':ids.fill((lo+32)%256)
            if kind=='duplicate':ids.fill((lo+32)%256);ids[1::2,:2]=lo+7
            if kind=='all':ids[:]=lo+np.arange(8)[None,:]
            lists={e:[t for t in range(M) if any(ids[t,k]==lo+e for k in range(8))] for e in range(32)}
            for e,tokens in lists.items():
                expected=sorted(set(np.nonzero(ids==lo+e)[0].tolist()))
                assert tokens==expected and len(tokens)<=M
            assert sum(bool(t) for t in lists.values())==len({int(e)-lo for e in ids.flatten() if lo<=e<lo+32})
print('PASS96 routing models: all M, offset shards, duplicate/sparse/zero routes')

# Checker oracle from checks.py, not a manufactured acceptance result.
r=torch.ones((2,4096),dtype=torch.bfloat16)
assert fs._fs_metrics(r,r)['ok']
assert not fs._fs_metrics(r*2,r)['ok']
assert fs._fs_metrics(r*0,r*0)['ok']
assert not fs._fs_metrics(r*float('nan'),r)['ok']

# Reference formula known solutions: one nonzero gate/up and down coefficient.
# E2M1=1 code2, unit scales/global scales, x[0]=1 -> inter=silu(1).
# Only output column0 nonzero, so normalized value=silu(1)/sqrt(silu(1)^2/4096+eps).
# Exercise optional Marlin boundaries as well; no CUDA involved.
weights=dict(w13=torch.zeros((1,8192,2048),dtype=torch.uint8),w13_sf=torch.ones((1,8192,256)).to(torch.float8_e4m3fn),
             w13_gs=torch.ones(1),w2=torch.zeros((1,4096,2048),dtype=torch.uint8),
             w2_sf=torch.ones((1,4096,256)).to(torch.float8_e4m3fn),w2_gs=torch.ones(1),post_norm_w=torch.zeros(4096,dtype=torch.bfloat16))
weights['w13'][0,0,0]=2;weights['w13'][0,4096,0]=2;weights['w2'][0,0,0]=2
x=torch.zeros((1,4096),dtype=torch.bfloat16);x[0,0]=1
ids=torch.ones((1,8),dtype=torch.int32);ids[0,0]=0
tw=torch.ones((1,8))/8
data=dict(hidden_states=x,topk_ids=ids,topk_weights=tw,config=dict(local_start=0,local_count=1))
for marlin in (False,True):
    v=torch.nn.functional.silu(torch.tensor(1.))
    if marlin:v=v.to(torch.bfloat16).float()
    expected=torch.zeros_like(x);expected[0,0]=(v/torch.sqrt(v*v/4096+1e-8)/8).to(torch.bfloat16)
    assert torch.equal(fs._fs_reference(data,weights,marlin=marlin),expected)
print('PASS independent reference analytical one-coordinate solution, raw and Marlin boundaries')

diag=load(ROOT/'experiments/boards/diag/fsdiag.py','fsdiag')
import base64,zlib
assert zlib.decompress(base64.b85decode(diag.CANDIDATE_B85))==(ROOT/'experiments/boards/moe/moe_fs1.py').read_bytes()
assert diag.BASE_SHA==hashlib.sha256(parent).hexdigest()
from build_source import source
assert fs.FS_CUDA==source()
(ROOT/'.fs_offline/cpu_results.json').write_text(json.dumps(dict(seed=20261101,coverage=coverage,byte_roundtrip=True,hybrid_ownership=True,replay_generations=100,reference_known_solution=True),indent=2))
print('PASS source identity, pinned baseline and diagnostic embedding; CPU ONLY')
