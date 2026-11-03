"""CPU only: inverse bytes, actual CUDA fragment addresses and algebraic invariants."""
import ast,hashlib,importlib.util,json,sys,types
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
sys.dont_write_bytecode=True;torch.set_num_threads(4)
ROOT=Path(__file__).resolve().parents[1]
def load(p,name):
    sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
base=load(ROOT/'experiments/boards/moe/moe_b15.py','b15');port=load(ROOT/'experiments/boards/moe/moe_b16.py','b16')
rng=np.random.default_rng(20261102)
# Scalar oracle from the published fragment coordinates, not our byte transpose.
perm=np.empty(256,dtype=np.int64)
for lg in range(4):
    for j in range(4):
        for t in range(4):
            for q in range(2):
                for u in range(2):perm[(4*lg+j)*16+8*q+2*t+u]=16*(4*t+j)+4*lg+2*q+u
assert sorted(perm.tolist())==list(range(256)) and not np.array_equal(perm,np.arange(256))
assert np.array_equal(perm[1::2],perm[::2]+1) and (perm[::2]%2==0).all()
byte_inv=np.argsort(perm[::2]//2)
P=(np.arange(4096)//256)*256+perm[np.arange(4096)%256]
assert np.array_equal(port._wgs_kperm_index(4096,'cpu').numpy(),P)

# Read ONLY the reference's pure permutation helpers as an additional oracle;
# never import its extension/context bindings or submit its file.
rp=ROOT.parent/'alygg/kernels/muse_moe_w4a16.py';rt=rp.read_text()
rn={n.name:n for n in ast.parse(rt).body if isinstance(n,ast.FunctionDef)}
oracle={'torch':torch}
for name in ('_kperm_src','_kperm_codes','_kperm_rows'):
    exec(compile(ast.get_source_segment(rt,rn[name]),'<read-only reference permutation>','exec'),oracle)
assert np.array_equal(oracle['_kperm_src']().numpy(),perm)

def decode_words(words):
    """Independent b15 CUDA dq4 algebra; not the host _POS bit scatter inverse."""
    q=np.asarray(words,dtype=np.uint32)
    rr=[q&np.uint32(0x81c081c0),
        ((q<<np.uint32(10))|(q>>np.uint32(22)))&np.uint32(0x81c081c0),
        ((q<<np.uint32(13))|(q>>np.uint32(19)))&np.uint32(0x81c081c0),
        ((q&np.uint32(0x001b001b))*np.uint32(0x1084))&np.uint32(0x81c081c0)]
    out=np.empty(q.shape+(4,2),dtype=np.uint8)
    for r in range(4):
        for u in range(2):
            v=rr[r]>>np.uint32(16*u)
            out[...,r,u]=sum(((v>>np.uint32(ob))&1)<<bit for bit,ob in enumerate((6,7,8,15)))
    return out

def invert(packed,rows):
    r=len(rows);nub=r//256
    stage=packed.numpy().reshape(nub,16,4,9216)
    words=stage[...,:8192].copy().view(np.uint32).reshape(nub,16,4,4,4,32,4)
    words=words.transpose(0,2,4,1,3,6,5).reshape(r//16,8,8,4,8,4).transpose(0,1,2,4,5,3)
    decoded=decode_words(words)
    n=np.empty(words.shape+(2,2,2),dtype=np.uint8)
    for reg,(h,q) in enumerate(((0,0),(1,0),(0,1),(1,1))):
        for u in range(2):n[...,h,q,u]=decoded[...,reg,u]
    n=n.transpose(0,6,3,1,2,5,7,4,8).reshape(r,4096)
    physical=(n[:,::2]|(n[:,1::2]<<4))
    row_inv=np.argsort(rows.numpy())
    codes=physical.reshape(r,16,128)[:,:,byte_inv].reshape(r,2048)[row_inv]
    scales=stage[...,8192:].copy().reshape(nub,16,4,4,8,4,4,2)
    scales=scales.transpose(0,2,3,7,4,1,5,6).reshape(r,256)[row_inv]
    return codes,scales

records=[]
for kind,r in (('A',8192),('B',4096)):
    raw=rng.integers(0,256,(r,2048),dtype=np.uint8)
    # Include all possible scale bytes: this is a lossless-layout test even for
    # bytes that production PRMT validation rejects as nonpositive/nonnormal.
    sf=rng.integers(0,256,(r,256),dtype=np.uint8)
    codes=torch.from_numpy(raw);scales=torch.from_numpy(sf)
    before=(hashlib.sha256(raw.tobytes()).digest(),hashlib.sha256(sf.tobytes()).digest())
    if kind=='A':
        rows=torch.stack((torch.arange(4096).view(-1,8),4096+torch.arange(4096).view(-1,8)),1).flatten()
        physical_rows=port._wgs_kperm_rows(rows)
        assert torch.equal(physical_rows,oracle['_kperm_rows'](rows))
    else:rows=physical_rows=torch.arange(4096)
    assert torch.equal(port._wgs_kperm_codes(codes),oracle['_kperm_codes'](codes))
    packed=port._wgs_copy(codes,scales,physical_rows)
    recovered,rs=invert(packed,physical_rows)
    # Expected bytes are the original seeded inputs, not a generated golden file.
    assert np.array_equal(recovered,raw) and np.array_equal(rs,sf),(kind,'inverse')
    stage=packed.numpy().reshape(r//256,16,4,9216)
    # Test the ACTUAL consumer addresses on32768 independently sampled fragments.
    count=32768
    ub=rng.integers(r//256,size=count);st=rng.integers(16,size=count);wg=rng.integers(4,size=count)
    wp=rng.integers(4,size=count);lane=rng.integers(32,size=count);lg=rng.integers(4,size=count)
    j=rng.integers(4,size=count);reg=rng.integers(4,size=count);u=rng.integers(2,size=count)
    h=reg&1;qh=reg//2;t=lane&3;g=lane//4
    word_index=((lg*4+wp)*32+lane)*4+j
    cw=stage[...,:8192].copy().view(np.uint32).reshape(r//256,16,4,2048)[ub,st,wg,word_index]
    got=decode_words(cw)[np.arange(count),reg,u]
    row=physical_rows.numpy()[ub*256+wg*64+wp*16+g+8*h]
    k=st*256+16*(4*t+j)+4*lg+2*qh+u
    expected=(raw[row,k//2]>>(4*(k%2)))&15
    scale=stage[ub,st,wg,8192+(wp*32+lane)*8+2*j+h]
    assert np.array_equal(got,expected)
    assert np.array_equal(scale,sf[row,k//16])
    # Every old dequantization code/scale pair is preserved at its new position.
    # Normal scales: encoded BF16 weight value times2^62 equals E2M1*E4M3.
    valid=(scale>=8)&(scale<=126)
    grid=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
    fp4=np.where((got[valid]&8)!=0,-grid[got[valid]&7],grid[got[valid]&7])
    fp8=torch.from_numpy(scale[valid].copy()).view(torch.float8_e4m3fn).float().numpy()
    bfscale=((scale[valid].astype(np.uint32)*16+0x5c00)<<16).view(np.float32)
    assert np.array_equal((fp4*np.float32(2.**-126))*bfscale*np.float32(2.**62),fp4*fp8)
    # Mode0 remains byte-identical to b15's original WGS repack.
    port.KPERM=0
    assert torch.equal(port._wgs_copy(codes,scales,rows),base._wgs_copy(codes,scales,rows))
    port.KPERM=3
    assert before==(hashlib.sha256(raw.tobytes()).digest(),hashlib.sha256(sf.tobytes()).digest()),'prepare repack modified inputs'
    records.append(dict(kind=kind,code_bytes=raw.size,scale_bytes=sf.size,fragments=count))
    print('PASS',kind,'full expert round-trip',raw.size,'code bytes',sf.size,'scale bytes;32768 native fragments; legacy mode exact')
    del packed,raw,sf,stage,codes,scales,recovered,rs

# CUDA gather addresses: each vector of8 BF16 values must equal x[perm(P)].
gather=[]
for tid in range(512):
    G=tid&31;pair=((G&6)<<2)|((G>>2)&6)|(G&1)
    gather.extend((tid>>5)*256+2*(pair+32*t)+u for t in range(4) for u in range(2))
assert np.array_equal(gather,P)
# Row permutation never crosses a256-intermediate block: B readiness still
# waits for exactly two A units/four warpgroup epilogues per unit.
assert np.array_equal(P//256,np.arange(4096)//256)
for st in range(16):
    produced=P[st*256:(st+1)*256]
    assert sorted(produced.tolist())==list(range(st*256,(st+1)*256))
print('PASS all4096 activation gather coordinates and all16 A->B readiness blocks')

# Algebraic composition oracle: integer-valued matrix products in FP64 are exact
# here. Permute both K operands and gate/up rows, then compare the composed MLP.
gen=torch.Generator().manual_seed(20261102)
x=torch.randint(-2,3,(3,256),generator=gen).double()
A=torch.randint(-3,4,(512,256),generator=gen).double()
B=torch.randint(-3,4,(256,256),generator=gen).double()
p=torch.from_numpy(perm);ar=torch.stack((torch.arange(256).view(-1,8),256+torch.arange(256).view(-1,8)),1).flatten()
ar3=port._wgs_kperm_rows(ar)
h=x@A.T;h3=x[:,p]@A[ar3][:,p].T
assert torch.equal(h3,h[:,ar3])
g3=h3.view(3,-1,2,8)[:,:,0,:].reshape(3,256)/512
u3=h3.view(3,-1,2,8)[:,:,1,:].reshape(3,256)/512
inter=(torch.nn.functional.silu(h[:,:256]/512)*(h[:,256:]/512)).to(torch.bfloat16).double()
inter3=(torch.nn.functional.silu(g3)*u3).to(torch.bfloat16).double()
assert torch.equal(inter3,inter[:,p])
assert torch.equal(inter@B.T,inter3@B[:,p].T)
print('PASS composed gate/up permutation, SwiGLU and B-column invariant (exact FP64 test, not GPU tolerance evidence)')

# Compiler mode and pack mode are locked together; test flags without building.
mods=[]
def fake_load(**kw):mods.append(kw);return types.SimpleNamespace(okc_set_pdl=lambda x:None)
with patch.object(port,'load_inline',fake_load),patch.object(torch.cuda,'get_device_capability',lambda *a:(9,0)):
    port._ext(port.MAIN_DEFINES)
    assert '-DOKC_KPERM=3' in mods[-1]['extra_cuda_cflags']
    try:port._ext(tuple(port.MAIN_DEFINES)+('-DOKC_KPERM=0',))
    except AssertionError:pass
    else:raise AssertionError('packing/compile mismatch accepted')
    port.KPERM=0;port._ext(port.MAIN_DEFINES)
    assert '-DOKC_KPERM=0' in mods[-1]['extra_cuda_cflags']
port.KPERM=3
(ROOT/'.kperm_offline/cpu.json').write_text(json.dumps(dict(seed=20261102,roundtrip=records,activation_gather=True,readiness=True,mathematical_composition=True),indent=2))
print('PASS mode guards; CPU only, no GPU context/build/run')
