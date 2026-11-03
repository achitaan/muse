"""CPU source/layout/protocol tests; no device execution."""
from pathlib import Path
import ast,collections,hashlib,json,random,re,sys
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
def load(p):
    n={};exec(compile(p.read_bytes(),str(p),'exec'),n);return n
base=load(ROOT/'experiments/boards/moe/moe_b13.py');l1=load(ROOT/'experiments/boards/moe/moe_l1.py')
a,b=base['CUDA_SRC'],l1['CUDA_SRC']
def section(src,name):
    start=src.index(name);return src[start:src.index('\n}',start)+2]
for name in ('struct Params {','struct Smem {','void wgs_wproducer(','void wgs_xproducer(',
             'void wgs_gather(','void wgs_flush(', 'void wgs_epi_tile(', 'void wgs_ranges('):
    assert section(a,name)==section(b,name),name
assert a.split('#include <torch/extension.h>')[1]==b.split('#include <torch/extension.h>')[1]
for name in ('MAIN','TINY','TINY2','MID','MAIN_FALLBACK'):assert base[name+'_DEFINES']==l1[name+'_DEFINES']
assert base['CPP_SRC']==l1['CPP_SRC']
# Every arithmetic/decode/MMA operation after selecting cwg/scg is verbatim.
tail='#pragma unroll\n      for (int h = 0; h < 4 / GS;'
aa=section(a,'void wgs_seg(');bb=section(b,'void wgs_seg(')
assert aa[aa.index(tail):]==bb[bb.index(tail):]
start=a.index('template <int NTT> struct WGSMMA;');end=a.index('// Phase A runs',start)
start2=b.index('template <int NTT> struct WGSMMA;');end2=b.index('// Phase A runs',start2)
assert a[start:end]==b[start2:end2]
print('PASS unchanged data/host ABI, producers, epilogues, arithmetic and WGMMA sequence')

# Expected coordinates are the parent's four sequential load groups, evaluated
# independently of the candidate's whole-stage capture order.
rng=np.random.default_rng(20261101)
stage=rng.integers(0,256,36864,dtype=np.uint8)
addresses=collections.Counter()
for wg in range(4):
    for wp in range(4):
        for lane in range(32):
            g=lane//4;code=[];scale=[]
            for group in range(4):
                c=wg*9216+((group*4+wp)*32+lane)*16
                s=wg*9216+8192+((group*4+wp)*8+g)*8
                code.append(stage[c:c+16].copy().view(np.uint32))
                scale.append(stage[s:s+8].copy().view(np.uint32))
                addresses.update(range(c,c+16));addresses.update(range(s,s+8))
            ready=np.uint32(0)
            for group in range(4):
                for v in np.r_[code[group],scale[group]]:ready^=v
            assert (np.uint32(0xffffffff)^(ready&np.uint32(0)))==np.uint32(0xffffffff)
            for lg in range(4):
                for h in range(2):
                    for jj in range(2):
                        js=h*2+jj
                        c=wg*9216+((lg*4+wp)*32+lane)*16+4*js
                        s=wg*9216+8192+((lg*4+wp)*8+g)*8+4*(js//2)
                        assert code[lg][js]==stage[c:c+4].copy().view(np.uint32)[0]
                        assert scale[lg][js//2]==stage[s:s+4].copy().view(np.uint32)[0]
for wg in range(4):
    assert all(addresses[i]==1 for i in range(wg*9216,wg*9216+8192))
    assert all(addresses[i]==4 for i in range(wg*9216+8192,(wg+1)*9216))
assert len(addresses)==36864
print('PASS whole packed stage byte coverage; codes once, scales4-way replicated exactly as parent')

# Expected hardware roles from accepted plan; only complete WGs change limits.
groups=[set(range(wg*128,(wg+1)*128)) for wg in range(5)]
cons=set(range(512));producer=set(range(512,640));wp=set(range(512,544));xp=set(range(544,576));pad=set(range(576,640))
assert set.union(*groups[:4])==cons and groups[4]==producer
assert sum(112 if tid in cons else 32 for tid in range(640))==640*96==61440
assert len(cons|wp|xp|pad)==640 and sum(map(len,(cons,wp,xp,pad)))==640
assert len(cons|xp)==544 and not ((cons|xp)&pad)
assert len(cons)//32==16
assert b.count('setmaxnreg.dec.sync.aligned.u32 32;')==1
assert b.count('setmaxnreg.inc.sync.aligned.u32 112;')==1
assert 'else { WS_JOIN(); }' in b
assert 'p.zero = 0;' in b
# Adversarial ring progress: weights may overwrite a slot immediately after the
# last warp capture, long BEFORE x becomes ready. Captured bytes must remain.
for N in (1,2,3,4,16,31,64):
    randomizer=random.Random(20261101+N)
    ring=[None]*3;released=[16]*3;pending=[0]*16;captures={};produced=0;done=[0]*16
    for iteration in range(100000):
        actions=list(range(17));randomizer.shuffle(actions)
        for actor in actions:
            if actor==16:
                if produced<N and released[produced%3]==16:
                    sl=produced%3;ring[sl]=produced;released[sl]=0;produced+=1
            else:
                seq=pending[actor]
                if seq<N and seq<produced and ring[seq%3]==seq:
                    captures[actor,seq]=ring[seq%3];released[seq%3]+=1;pending[actor]+=1
                if done[actor]<pending[actor] and randomizer.randrange(3)==0:
                    assert captures[actor,done[actor]]==done[actor];done[actor]+=1
        if all(d==N for d in done):break
    else:raise AssertionError('ring model stuck')
    assert len(captures)==16*N
print('PASS register/CTA/barrier role accounting and7 asynchronous early-release ring models')

if '--sass' in sys.argv:
    def insns(name):
        text=(ROOT/'.l1_offline'/(name+'.sass')).read_text()
        return {part.splitlines()[0]:re.findall(r'/\*[0-9a-f]+\*/\s+(.+?;)',part)
                for part in text.split('Function : ')[1:]}
    assert insns('BASE')==insns('OFF'),'MAIN macro-off must retain exact SASS'
    for k in ('TINY','TINY2','MID','MAIN_FALLBACK'):
        assert insns(k+'_BASE')==insns(k+'_L1'),k
    sas=(ROOT/'.l1_offline/L1.sass').read_text()
    assert 'USETMAXREG' in sas
    print('PASS MAIN-off and all non-MAIN SASS identical; dynamic register instruction present')
