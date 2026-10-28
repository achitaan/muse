"""Resource ABI model, complete partitions, public-shape bounds and warpgroup budgets."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import bisect
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_attention_three_cta import ROOT,BASE,literal,parent_source,transform
sys.path.insert(0,str(ROOT/'scripts'))
from verify_attention_precision import verify
raw=(ROOT/'.tools/attention-three-cta-variant.py').read_bytes()
cuda=literal(raw)[1];old=literal(parent_source())[1]
assert transform(old)[0]==cuda
verify(raw)
# No new barrier, atomic, queue, cache operation or numerical instruction.
for token in ('asm volatile(', '__syncthreads()', 'mbar_arrive(', 'csync<', 'attn_out_bf16(', 'mma_f8pv(', '256.f;'):
    assert cuda.count(token)==old.count(token),token

def smem_layout(source,L,F):
    block=source.split('struct Smem {',1)[1].split('\n};',1)[0]
    types={'uint8_t':(1,1),'uint64_t':(8,8),'int':(4,4),'float':(4,4),'__nv_bfloat16':(2,2),'long long':(8,8)}
    stages=3 if L==1 or not F else 4
    special={'kvq':2048 if L==1 else 18432,'q8s':16 if L==1 else 8192,
             'cum':513 if F else 1,'pseg':64 if F else 1,
             'dq':16 if F else 1,'dqf':16 if F else 1,'dqe':16 if F else 1,
             'seg':((512 if L==1 else 64) if '((L == 1) ? MAXB : MAXSEG)' in block else 64) if F else 1}
    offset=0;maxalign=1;rows=[]
    for line in block.splitlines():
        line=line.split('//',1)[0].strip()
        match=re.fullmatch(r'(?:alignas\((\d+)\)\s+)?(uint8_t|uint64_t|int|float|__nv_bfloat16|long long)\s+(.+);',line)
        if not match:continue
        declaredalign,typename,declaration=match.groups()
        # Split declarators, excluding commas inside dimensions / Cfg<L,F>.
        declarations=[];depth=0;start=0
        for i,ch in enumerate(declaration):
            if ch=='[':depth+=1
            elif ch==']':depth-=1
            elif ch==',' and depth==0:declarations.append(declaration[start:i]);start=i+1
        declarations.append(declaration[start:])
        for decl in declarations:
            m=re.fullmatch(r'\s*(\w+)((?:\[[^\]]+\])*)\s*',decl);assert m,decl
            name,dimensions=m.groups();dims=re.findall(r'\[([^\]]+)\]',dimensions)
            count=1
            for i,dim in enumerate(dims):
                if name in special and i==0:count*=special[name]
                else:
                    dim=dim.replace('Cfg<L, F>::STAGES',str(stages))
                    count*=eval(dim,{'__builtins__':{}},dict(L=L,F=F,STAGE_BYTES=8192,ROWB=128,TILE=64,MAXB=512,MAXSEG=64))
            width,align=types[typename];align=max(align,int(declaredalign or 1));maxalign=max(maxalign,align)
            offset=(offset+align-1)//align*align
            rows.append(dict(name=name,offset=offset,bytes=width*count,alignment=align))
            offset+=width*count
    assert [r['name'] for r in rows]==('k v mbar pbar pgc dbgn vidx pidx full empty vtf vte qdone claim nxt idx '
                                      'kvq q8s kvst gst kn vn cs sn ifreq scl cloc flag sense cum seg nseg fcs fce '
                                      'pseg pnseg pchunk dq dqf dqe').split()
    size=(offset+maxalign-1)//maxalign*maxalign
    return dict(size=size,dynamic=size+1024,fields=rows)

layouts={}
for L in (1,8):
    for F in (False,True):
        a,b=smem_layout(old,L,F),smem_layout(cuda,L,F)
        if (L,F)!=(1,True):assert a==b
        layouts[f'L{L}_flat{int(F)}']=b
dynamic=layouts['L1_flat1']['dynamic']
assert smem_layout(old,1,True)['size']==61440
assert 3*dynamic<=233472
configs=[]
for L in (1,8):
    for F in (False,True):
        nt=128 if L==1 else 256;np_=128 if F else 0;ntx=nt+np_
        minb=3 if L==1 and F else 2
        avg=min(248,(65536//(minb*ntx))//8*8)
        producer=56 if L==1 else 32
        consumer=min(248,((avg*ntx-producer*np_)//nt)//8*8)
        if L==1 and F:
            assert (avg,producer,consumer)==(80,56,104)
            assert minb*(nt*consumer+np_*producer)==61440
            # Each CTA's producer releases exactly what its consumer needs.
            assert np_*(avg-producer)==nt*(consumer-avg)==3072
        else:
            originalavg=min(248,(65536//(2*ntx))//8*8)
            originalconsumer=min(248,((originalavg*ntx-producer*np_)//nt)//8*8)
            assert avg==originalavg and consumer==originalconsumer
        assert producer%8==0 and consumer%8==0 and 24<=consumer<=256
        configs.append(dict(L=L,flat=F,ntx=ntx,minblocks=minb,average=avg,producer=producer,consumer=consumer))

def partition(lengths,G,cap):
    cum=np.r_[0,np.cumsum(lengths,dtype=np.int64)]
    virtualstarts=cum[:-1]+np.arange(len(lengths))*400
    total=int(cum[-1]+(len(lengths)-1)*400)
    cuts=np.arange(G+1,dtype=np.int64)*total//G
    owners=np.searchsorted(virtualstarts,cuts,side='right')-1
    realcuts=np.minimum(cuts-owners*400,cum[owners+1])
    assert realcuts[0]==0 and realcuts[-1]==cum[-1]
    assert np.all(realcuts[1:]>=realcuts[:-1])
    segments=[];lost=0;maxseg=0
    for csg,ceg in zip(realcuts[:-1],realcuts[1:]):
        if csg==ceg:continue
        b=int(np.searchsorted(cum,csg,side='right')-1);n=0
        while b<len(lengths) and cum[b]<ceg:
            s=max(int(csg),int(cum[b]))-int(cum[b]);e=min(int(ceg),int(cum[b+1]))-int(cum[b])
            assert 0<=s<e<=lengths[b]
            if n<cap:segments.append((b,s,e))
            else:lost+=e-s
            b+=1;n+=1
        maxseg=max(maxseg,n)
    return segments,lost,maxseg

rng=np.random.default_rng(20261028)
cases=[]
for B in (2,3,8,16,31,32,33,64,128,256,512):
    for kind in ('zero','uniform','mixed','one_long','agentic'):
        if kind=='zero':seq=np.zeros(B,np.int64)
        elif kind=='uniform':seq=np.full(B,79999,np.int64)
        elif kind=='mixed':seq=rng.integers(0,100000,size=B,dtype=np.int64)
        elif kind=='one_long':seq=np.zeros(B,np.int64);seq[0]=4*1048576
        else:seq=np.maximum(0,np.exp(rng.uniform(0,np.log(100000),B)).astype(np.int64)-1)
        for pairs in (1,2,4,8):
            if B*pairs<=4096:cases.append((B,pairs,kind,seq+1))
segments_checked=0;partitions=0;old_truncations=0;max_segments=0;pages_checked=0
for B,pairs,kind,lengths in cases:
    for sms in (80,120,132):
        G=max(1,(3*sms)//pairs)
        segments,lost,maxseg=partition(lengths,G,512)
        assert lost==0
        max_segments=max(max_segments,maxseg)
        buckets=[[] for _ in range(B)]
        for b,s,e in segments:
            buckets[b].append((s,e));segments_checked+=1
            # Arbitrary, unaligned split ends are clipped before cache access.
            tiles=(e-s+63)//64
            assert s+64*(tiles-1)<e<=s+64*tiles
            for page in (1,8,64):
                assert (e-1)//page<=(int(lengths[b])-1)//page
                pages_checked+=1
        for b,items in enumerate(buckets):
            items.sort();assert items[0][0]==0 and items[-1][1]==lengths[b]
            assert all(a[1]==z[0] for a,z in zip(items,items[1:]))
            # Global cache write and key-counter finalization happen once.
            assert sum(s<=lengths[b]-1<e for s,e in items)==1
            assert sum(e-s for s,e in items)==lengths[b]
        if kind=='one_long':
            _,oldlost,_=partition(lengths,max(1,2*sms//pairs),64)
            old_truncations+=oldlost>0
        partitions+=1
assert old_truncations>0,'Audit must expose the inherited segment-capacity defect'

# Execute only public-shape planning, with immutable device properties mocked.
trees=[ast.parse(s) for s in (parent_source(),raw)]
scopes=[]
for tree in trees:
    names={'_plan','_splits','_copies','_headroom','_fits'}
    code=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
    constants={n.targets[0].id:n.value.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    limit=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INDEX_LIMIT' for t in n.targets))
    assert not any(isinstance(n,ast.Name) for n in ast.walk(limit))
    constants['INDEX_LIMIT']=eval(compile(ast.Expression(limit),'<index-limit>','eval'),{'__builtins__':{}})
    constants['_num_sms']=lambda device:132
    exec(compile(code,'<shape-plans>','exec'),constants);scopes.append(constants)
planning_cases=0
for B in (1,2,3,16,32,33,128,512):
    for pairs in (1,2,4,8):
        for L in (1,8):
            for window in (0,2048):
                for page in (1,64):
                    oldplan=scopes[0]['_plan'](B,pairs,L,window,100000//page+1,page,'mock')
                    newplan=scopes[1]['_plan'](B,pairs,L,window,100000//page+1,page,'mock')
                    if L==1 and window==0 and B>1:
                        assert newplan==dict(oldplan,n_splits=max(1,396//pairs))
                    else:assert oldplan==newplan
                    assert newplan['n_splits']<=1024
                    planning_cases+=1

class TensorShape:
    def __init__(self,n,shape):self.n,self.shape=n,shape
    def numel(self):return self.n
for page in (1,64):
    for B in (1,32,512):
        for extent in (100000,1048576,2**31):
            cfg=dict(batch=B,q_len=1,num_q_heads=16,head_dim=64,page_size=page)
            pages=(extent+page-1)//page
            data=dict(config=cfg,page_table=TensorShape(B*pages,(B,pages)),qkv=TensorShape(B*2304,(B,2304)))
            fit=scopes[1]['_fits'](data)
            if fit:assert B*(pages*page+1+600)<2**31

report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),layout=layouts,configurations=configs,
            dynamic_shared_bytes=dynamic,three_cta_shared_bytes=3*dynamic,three_cta_registers=61440,
            partitions=partitions,segments_checked=segments_checked,paged_end_checks=pages_checked,
            max_segment_count=max_segments,old_truncation_cases=old_truncations,
            shape_plans=planning_cases,full_key_coverage=True,unchanged_math=True,
            scope='CPU model; actual CUDA resource attributes and official GPU coverage required')
(ROOT/'.tools/attention-three-cta-cpu-audit.json').write_bytes(json.dumps(report,indent=2).encode())
print(json.dumps({k:v for k,v in report.items() if k not in ('layout','configurations')},indent=2))
