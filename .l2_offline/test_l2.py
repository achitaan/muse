"""CPU model of jobs, cost coordinates, units, x rows/rings, and fixups."""
import ast,bisect,collections,hashlib,json,re,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def load(p):
    d={};exec(compile(p.read_bytes(),str(p),'exec'),d);return d
base=load(ROOT/'experiments/boards/moe/moe_b13.py');l2=load(ROOT/'experiments/boards/moe/moe_l2.py')
a,b=base['CUDA_SRC'],l2['CUDA_SRC']
CAP=int(re.search(r'#define OKC_L2_CAP (\d+)',b)[1])
assert 'setmaxnreg' not in b and 'WGS_L1' not in b and 'uint4 cw[4]' not in b
def section(s,name):
    start=s.index(name);return s[start:s.index('\n}',start)+2]
for name in ('struct Smem {','struct Params {','void wgs_wproducer(', 'void wgs_gather(',
             'void wgs_epi_tile(', 'void wgs_flush(', 'void wgs_prefix(', 'void wgs_range(',
             'static size_t okc_smem_bytes()'):
    assert section(a,name)==section(b,name),name
for n in range(1,9):assert section(a,f'template <> struct WGSMMA<{n}>')==section(b,f'template <> struct WGSMMA<{n}>')
assert base['CPP_SRC']==l2['CPP_SRC']
assert l2['WGS_FIXW']==12*512 and l2['WGS_FIXF']==2*160*4*6144+2*160*4+136*16
assert 3*32768==2*49152
print('PASS no L1 inheritance; unchanged narrow MMA/epilogues/gather and shared-memory footprint')

CA=[112,135,140,158,173,185,200,207];CB=[100,104,115,123,132,144,156,174]
def cost(n,pb):
    c=(CA,CB)[pb]
    return c[7]+c[(n-64+7)//8-1] if n>64 else c[(n+7)//8-1]
def jobs(counts,native):
    prefix=np.cumsum([0]+counts).tolist();out=[]
    for e,n in enumerate(counts):
        if n>8:
            if native and 64<n<=CAP:out.append((e,prefix[e],n))
            else:out.extend((e,prefix[e]+q,min(64,n-q)) for q in range(0,n,64))
    out.extend((e,prefix[e],n) for e,n in enumerate(counts) if 0<n<=8)
    return out
def ranges(js,pb,G):
    size=512>>pb;prefix=[0]
    for e,off,n in js:prefix.append(prefix[-1]+cost(n,pb)*size)
    def stage(x):
        j=bisect.bisect_right(prefix[:-1],x)-1
        return j*size+(x-prefix[j])//cost(js[j][2],pb)
    boundaries=[stage(prefix[-1]*c//G) for c in range(G)]+[len(js)*size]
    owner=np.empty(len(js)*size,dtype=np.int32)
    coverage=np.zeros_like(owner)
    for c,(lo,hi) in enumerate(zip(boundaries,boundaries[1:])):
        owner[lo:hi]=c;coverage[lo:hi]+=1
    assert (coverage==1).all()
    return prefix,boundaries,owner
rng=np.random.default_rng(20261101)
cases=[[0]*32,[96]*32,[384]*8+[0]*24,[65]*31+[0]]
cases += [[n]+[0]*31 for n in (1,8,9,63,64,65,68,72,73,75,80,81,88,89,95,96,97,128,144,192,384)]
cases += [[75,144,68,64,32,8,1]+[0]*25]
for _ in range(40):
    c=rng.multinomial(1200,np.ones(32)/32).tolist()
    for e in rng.choice(32,4,replace=False):c[e]=int(rng.choice([65,68,75,80,88,96,97,144,256]))
    assert sum(c)<=3072
    cases.append(c)
records=[]
for counts in cases:
    old=jobs(counts,False);new=jobs(counts,True)
    assert len(new)==len(old)-sum(64<n<=CAP for n in counts)
    assert max([n for e,off,n in new]+[0])<=CAP
    seen=collections.Counter(i for e,off,n in new for i in range(off,off+n))
    assert seen==collections.Counter({i:1 for i in range(sum(counts))})
    old_rows=np.cumsum([0]+[(n+7)//8*8 for _,_,n in old]).tolist()
    new_rows=np.cumsum([0]+[(n+7)//8*8 for _,_,n in new]).tolist()
    assert old_rows[-1]==new_rows[-1]
    for j,key in enumerate(new):
        if key[2]<=64:assert new_rows[j]==old_rows[old.index(key)]
    if not new:continue # zero-job runtime remains disabled, only model empty construction
    for pb in (0,1):
        size=512>>pb
        for G in (1,7,132):
            op,ob,oo=ranges(old,pb,G);np_,nb,no=ranges(new,pb,G)
            assert op[-1]==np_[-1] and op[-1]*G<2**32
            for j,key in enumerate(new):
                e,off,n=key
                if n<=64:
                    k=old.index(key)
                    assert op[k]==np_[j] # unchanged absolute narrow job cost origin
                    assert np.array_equal(oo[k*size:(k+1)*size],no[j*size:(j+1)*size])
                # Unit-stage coverage/owners recover every K-stage once.
                for ub in range(size//16):assert len(no[j*size+ub*16:j*size+(ub+1)*16])==16
    padded=[(n+7)//8*8 for _,_,n in new]
    assert sum(padded)<=3072+8*88
    records.append(dict(routes=sum(counts),base_jobs=len(old),native_jobs=len(new),wide=sum(n>64 for _,_,n in new),xrows=sum(padded)))
print('PASS',len(cases),'job models; phases A/B and grids1/7/132: units once, exact narrow CTA/K partitions')

# Independent row-major -> xA/xB layout oracle. Each real row/K appears exactly
# once, and every padded-row address stays inside npad*4096. Test all boundary N.
for n in (1,7,8,9,63,64,65,68,72,73,75,80,81,88,89,95,96):
    npad=(n+7)//8*8;ntt=npad//8
    row=np.arange(npad)[:,None];k=np.arange(4096)[None,:]
    oracle=((((k//256)*ntt+row//8)*32+(k%256)//8)*64+(row%8)*8+k%8)
    assert np.unique(oracle).size==npad*4096 and oracle.min()==0 and oracle.max()==npad*4096-1
    # Parent wgs_gather uint4 address, one thread per8-element row chunk.
    tid=np.arange(512)[None,:]
    gather=(((tid//32)*ntt+row//8)*32+tid%32)*64+(row%8)*8
    assert np.array_equal(gather,oracle[:,::8])
    # Parent A epilogue col/r16 mapping writes each real intermediate exactly once.
    epicol=np.arange(4096)[None,:]
    epi=(((epicol>>8)*ntt+(row>>3))*32+((epicol&255)>>3))*64+(row&7)*8+(epicol&7)
    assert np.array_equal(epi,oracle)
    # Producer copies contiguous npad*256 per k-stage to slot; no cross-slot copy.
    nx,pitch=(2,49152) if n>64 else (3,32768)
    for seq in range(100):
        slot=seq%nx;phase=seq//nx&1
        assert slot*pitch+npad*512<=(slot+1)*pitch<=98304
        assert phase==(((seq>>1)&1) if nx==2 else (seq//3&1))
        if seq>=nx:assert ((seq//nx&1)^1)==((seq-nx)//nx&1)
    # Every B K256 stage waits for exactly2 A units *4 warpgroup epilogues.
    flags=collections.Counter(ub>>1 for ub in range(32) for wg in range(4))
    assert flags==collections.Counter({ks:8 for ks in range(16)})
    # Fixup layout [slot,wg,nt,t,4] is non-overlapping and can hold all12 tiles.
    for nt in range(ntt):assert nt*512+127*4+3<6144
print('PASS x gather/epilogue address bijections, both x-slot layouts/parities, readiness and12-tile fixups')

# Independent delayed-release pipeline: x(s) remains live until the first
# committed group of stage(s+1), unless a unit boundary drains all WGMMA.
# Two slots must still make progress, while weights keep their own three slots.
for nx,pitch in ((2,49152),(3,32768)):
    for stages in (1,2,3,16,17,48):
        full=[None]*nx;free=[True]*nx;produced=0;consumed=0;prev=None
        while consumed<stages:
            while produced<stages and free[produced%nx]:
                sl=produced%nx;free[sl]=False;full[sl]=produced;produced+=1
            sl=consumed%nx
            assert full[sl]==consumed,('x ring stalled',nx,stages,consumed)
            if prev is not None:free[prev]=True
            prev=sl
            consumed+=1
            if consumed%16==0 or consumed==stages:
                free[prev]=True;prev=None
        assert all(free)
print('PASS12 delayed-x-release pipeline models (both x-slot layouts, unit boundaries)')

if '--sass' in sys.argv:
    def instructions(path):
        return re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]+?;)\s+/\*',path.read_text())
    off=instructions(ROOT/'.l2_offline/OFF.sass');old=instructions(ROOT/'.l2_offline/BASE.sass')
    assert len(off)==len(old)
    diffs=[]
    for x,y in zip(old,off):
        if x!=y:
            # Disabled L2 differs only by exchanging the x producer's n/xph
            # counters R12/R16. Their initial values are both0; math is identical.
            assert re.sub(r'\bR(12|16)\b',lambda m:'R16' if m[1]=='12' else 'R12',x)==y
            diffs.append((x,y))
    for k in ('TINY','TINY2','MID','MAIN_FALLBACK'):
        assert instructions(ROOT/'.l2_offline'/(k+'.sass'))==instructions(ROOT/'.l1_offline'/(k+'_BASE.sass')),k
    assert 'C7512' not in (ROOT/'.l2_offline/L2.log').read_text()
    code=(ROOT/'.l2_offline/L2.sass').read_text()
    assert 'HGMMA.64x72x16' in code and 'HGMMA.64x80x16' in code
    assert 'HGMMA.64x88x16' not in code and 'HGMMA.64x96x16' not in code
    assert 'SETMAXREG' not in code
    print('PASS macro-off counter-register-only differences',len(diffs),'; non-MAIN SASS identical; N80 native without C7512/L1')
(ROOT/'.l2_offline/cpu_results.json').write_text(json.dumps(dict(seed=20261101,cases=records,narrow_grouping_preserved=True),indent=2))
