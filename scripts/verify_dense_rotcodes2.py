"""Offline verification of the rotated-code extension. No GPU or OKC calls.

The unchanged parent tests supply independent bitfield/CG references. New
tests exercise SK2 B-fragment streams, router A-fragment hi/lo planes, scalar
per-output FMA order, small-M/tail dispatch, and unchanged admission policy.
"""
from __future__ import annotations

import ast
import contextlib
import copy
import ctypes.util
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import torch

import verify_dense_cg_rotcodes as v1
from rotcodes2_syntax import check as syntax_check

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / 'candidates/dense_rotcodes2.py'
PARENT = ROOT / 'candidates/dense_cg_rotcodes.py'


def setup():
    tree, parent = ast.parse(CANDIDATE.read_text()), ast.parse(PARENT.read_text())
    prefix = copy.deepcopy(tree.body[:len(parent.body)])
    for node, old in zip(prefix, parent.body):
        if isinstance(node, ast.FunctionDef) and node.name in ('_make_vjeux', '_make_nb'):
            assert 'r2_parts' in ast.unparse(node.body[-1])
            node.body[-1] = copy.deepcopy(old.body[-1])
    assert ast.dump(ast.Module(body=prefix,type_ignores=[])) == ast.dump(parent)
    ns = dict(_PRUNE=True,_TABLE={},_DEFAULT='chenwainuo',_NB_C12=False,_dt_torch=torch,_dt_check=lambda d:None)
    for name in ('_make_vjeux','_make_nb'):
        module=copy.deepcopy(v1.defs(tree,{name}))
        if name=='_make_nb':
            # The incumbent eagerly builds its extension at factory exit.
            # Omit only that build expression in this offline extraction;
            # never import/run or patch the real submission/Torch module.
            fn=module.body[0]
            builds=[n for n in fn.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
                    and isinstance(n.value.func,ast.Name) and n.value.func.id=='ext']
            assert len(builds)==1
            fn.body=[n for n in fn.body if n not in builds]
        exec(compile(module,str(CANDIDATE),'exec'),ns)
    ns['_K'] = {'vjeux':ns['_make_vjeux'](),'nb':ns['_make_nb']()}
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id.startswith(('_RC_','_R2_')) for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANDIDATE),'exec'),ns)
        if isinstance(node,ast.FunctionDef) and node.name.startswith(('_rc_','_r2_')):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(CANDIDATE),'exec'),ns)
    final = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'][-1]
    exec(compile(ast.Module(body=[final],type_ignores=[]),str(CANDIDATE),'exec'),ns)
    return tree,ns


def source_gates(tree,ns):
    core,router = ns['_r2_core_source'](),ns['_r2_router_source']()
    assert '@@' not in core+router
    assert core.count('launch_r2_sk2<') == 12
    assert 'template <int ROWS, int NTH, int K, bool ILP>' in core
    assert 'r2_pool_kernel<2,512,8192,true>' in core
    assert 'r2_pool_kernel<2,512,8192,false>' in core
    assert router.count('r2_rt12_decode8<ROT>(') == 4
    assert 'r2_routertail_c12_dispatch<true>' in router and 'r2_routertail_c12_dispatch<false>' in router
    # The entire SK2 kernel and host launcher are untouched, not its epilogue,
    # split geometry, barriers, escape walker or MMA order.
    p = ns['_K']['vjeux']['r2_parts']
    got = core[core.index('#define SK2_WAIT'):]
    got = got.replace('r2_sk2_kernel','sk2_kernel').replace('launch_r2_sk2','launch_sk2').replace('void r2_sk2(','void sk2(')
    assert got[:got.index('  if(cfg==')].rstrip() == p['sk2'][:p['sk2'].index('  @@SK2_DISPATCH@@')].strip()
    # In pool mode, reversing only the ILP block reproduces the scalar kernel
    # byte-for-byte (apart from template/function names and trailing blanks).
    pool,old = ns['_r2_pool_components'](ns['_K']['vjeux']['rot_parts'])
    new = '    if constexpr (ILP) {\n'+ns['_R2_POOL_INNER']+'\n    } else {\n'+old+'\n    }'
    pool = pool[:pool.index('void r2_pool(')].replace(new,old)
    pool = pool.replace('template <int ROWS, int NTH, int K, bool ILP>','template <int ROWS, int NTH, int K>').replace('r2_pool_kernel','rot_cgemv_kernel')
    original = ns['_rc_source'](ns['_K']['vjeux']['rot_parts'])
    original = original[original.index('template <int ROWS, int NTH, int K>'):original.index('void rot_cgemv(')]
    assert pool.rstrip() == original.rstrip()
    # Router FP32 hi/lo arithmetic and exact escape overwrite remain intact.
    rt = ns['_K']['nb']['r2_parts']['source']
    marker = '    // exact escapes: overwrite'
    a = router[router.index(marker):router.index('template<int MT,int S,int WK,bool ROT>\nint r2_routertail_c12_t')]
    b = rt[rt.index(marker):rt.index('template<int MT,int S,int WK>\nint routertail_c12_t')]
    assert a.replace('r2_routertail_','routertail_') == b
    final = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='kernel_fn'][-1]
    call = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_r2_call')
    for node in (final,call):
        for c in ast.walk(node):
            if isinstance(c,ast.Call):
                attr = c.func.attr if isinstance(c.func,ast.Attribute) else c.func.id if isinstance(c.func,ast.Name) else ''
                assert attr not in {'item','cpu','numpy','double','synchronize','print','open','any'}
    (ROOT/'results/rotcodes2-core.cu').write_text(core)
    (ROOT/'results/rotcodes2-router.cu').write_text(router)
    return dict(core_sha256=hashlib.sha256(core.encode()).hexdigest(),router_sha256=hashlib.sha256(router.encode()).hexdigest(),
                core_syntax=syntax_check(core,ROOT/'results/rotcodes2-core-syntax.cpp'),
                router_syntax=syntax_check(router,ROOT/'results/rotcodes2-router-syntax.cpp'))


def pool_cpp_gate(ns):
    compiler = shutil.which('clang++') or shutil.which('g++')
    if not compiler:
        return dict(result='SKIPPED: no CPU C++ compiler')
    _,old = ns['_r2_pool_components'](ns['_K']['vjeux']['rot_parts'])
    source = r'''#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
struct uint2 { uint32_t x,y; }; struct uint4 { uint32_t x,y,z,w; };
float __uint_as_float(uint32_t u) { return std::bit_cast<float>(u); }
uint32_t __funnelshift_r(uint32_t a,uint32_t b,unsigned s) { return uint32_t(((uint64_t(b)<<32)|a)>>(s&31)); }
uint32_t __byte_perm(uint32_t a,uint32_t b,uint32_t sel) {
  uint64_t both=(uint64_t(b)<<32)|a; uint32_t out=0;
  for(int i=0;i<4;++i) { unsigned idx=(sel>>(4*i))&15; uint32_t v=(both>>(8*(idx&7)))&255;
    if(idx&8) v=(v&128)?255:0; out|=v<<(8*i); } return out;
}
template<bool ILP> std::array<float,2> run(uint4 (&lw)[2][1],uint2 (&cw)[2][1],float* xf,uint32_t base4,float a,float b) {
  constexpr int ROWS=2; int j=0; float acc[ROWS]={a,b};
  if constexpr(ILP) {
'''+ns['_R2_POOL_INNER']+'\n  } else {\n'+old+r'''
  }
  return {acc[0],acc[1]};
}
uint32_t state=0x91224151u;
uint32_t next() { state^=state<<13; state^=state>>17; state^=state<<5; return state; }
uint32_t rot(uint32_t c) { uint32_t h=c&0x80808080u; return (c&0x77777777u)|((c&0x08080808u)<<4)|(h<<4)|(h>>28); }
int main() {
  for(int i=0;i<100000;++i) {
    uint4 low[2][1]; uint2 codes[2][1]; float xf[16];
    for(int r=0;r<2;++r) {low[r][0]={next(),next(),next(),next()};codes[r][0]={rot(next()),rot(next())};}
    for(int e=0;e<16;++e) xf[e]=__uint_as_float(((next()&0x80000000u)|0x3f000000u|(next()&0xffu)<<16));
    uint32_t base=(40+i%71)*0x01010101u;
    float a=__uint_as_float((next()&0x80000000u)|0x3f000000u), b=__uint_as_float((next()&0x80000000u)|0x3f000000u);
    auto x=run<false>(low,codes,xf,base,a,b), y=run<true>(low,codes,xf,base,a,b);
    for(int r=0;r<2;++r) if(std::bit_cast<uint32_t>(x[r])!=std::bit_cast<uint32_t>(y[r]))
      throw std::runtime_error("per-row FP32 order mismatch");
  }
  std::cout<<"PASS 100000 two-row FMA-order cases\n";
}
'''
    path,binary = ROOT/'results/rotcodes2-pool-host.cpp',ROOT/'results/.rotcodes2-pool-host'
    path.write_text(source)
    command = [compiler,'-O2','-std=c++20','-Wno-unknown-pragmas',str(path),'-o',str(binary)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    output = subprocess.run([str(binary)],check=True,capture_output=True,text=True).stdout.strip()
    binary.unlink()
    assert output=='PASS 100000 two-row FMA-order cases'
    return dict(result=output,command=command)


def sk_functions(tree):
    factory = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_make_vjeux')
    ns = dict(torch=torch,EXPONENT_WINDOW=8,EXPONENT_VALUES=128,MAX_ESCAPE_DENOMINATOR=200,BASE4_REPLICATE=0x01010101)
    exec(compile(v1.defs(factory,{'_exponent_window','_escapes','sk2_tileunits','sk2_compress','sk2_stream'}),str(CANDIDATE),'exec'),ns)
    return ns


def sk_fragments(stream,b4,rotated):
    raw = stream.numpy().reshape(-1,768)
    low = raw[:,:512].reshape(-1,32,16).astype(np.uint16)
    words = raw[:,512:].copy().view(np.uint32).reshape(-1,32,2)
    hi = v1.decode_high(words,b4&255,rotated)
    high = np.empty_like(low)
    for q in range(2):
        for i in range(4):
            high[:,:,q*8+i] = (hi[:,:,q,0]>>(8*i))&255
            high[:,:,q*8+4+i] = (hi[:,:,q,1]>>(8*i))&255
    return (low | (high<<8)).reshape(*stream.shape[:-1],32,16)


def sk_dense(frag,cfg,k):
    _,_,_,R,nt,nwr,wsk,upw,_,_ = cfg
    shape = frag.shape
    s=np.arange(shape[0])[:,None,None,None,None,None,None]
    c=np.arange(shape[1])[None,:,None,None,None,None,None]
    w=np.arange(shape[2])[None,None,:,None,None,None,None]
    u=np.arange(upw)[None,None,None,:,None,None,None]
    t=np.arange(nt)[None,None,None,None,:,None,None]
    lane=np.arange(32)[None,None,None,None,None,:,None]
    v=np.arange(16)[None,None,None,None,None,None,:]
    row = c*R+(w//wsk)*nt*8+t*8+lane//4
    col = (s*wsk*upw+(w%wsk)*upw+u)*64+16*(v//4)+8*((v//2)&1)+2*(lane&3)+(v&1)
    rows,cols = np.broadcast_to(row,shape),np.broadcast_to(col,shape)
    out = np.empty((shape[1]*R,k),dtype=np.uint16)
    count = np.zeros_like(out,dtype=np.int16)
    out[rows,cols] = frag
    np.add.at(count,(rows.ravel(),cols.ravel()),1)
    assert np.all(count==1)
    return out


def sk_gates(tree,ns):
    f=sk_functions(tree); parts=ns['_K']['vjeux']['r2_parts']
    gen=torch.Generator().manual_seed(512994)
    records=[]; exact_weights=0; escapes=0
    for cfg in parts['configs']:
        name,layer,_,R,nt,nwr,wsk,upw,nacc,cl=cfg
        if layer=='router': continue
        buckets=[t for t in (8,16,32) if parts['pick'].get((layer,t))==name]
        if not buckets: continue
        k,n=1024,R*2
        w=(torch.randn(n,k,generator=gen)/k**.5).bfloat16()
        w[0,0],w[1,3],w[2,4],w[3,7]=0.,2.**-24,-2.**-25,-0.
        comp=f['sk2_compress'](w); assert comp is not None
        stream=f['sk2_stream'](comp['packed'],n,R,nt,nwr,wsk,upw)
        codes=ns['_r2_skpack'](stream)
        oldfrag,newfrag=sk_fragments(stream,comp['base4'],False),sk_fragments(codes,comp['base4'],True)
        assert np.array_equal(oldfrag,newfrag)
        d=sk_dense(newfrag,cfg,k)
        restored=d.copy();restored[comp['rows'].numpy(),comp['cols'].numpy()]=comp['wbits'].numpy().astype(np.uint16)
        assert np.array_equal(restored,w.view(torch.int16).numpy().view(np.uint16))
        exact_weights+=w.numel();escapes+=len(comp['rows'])
        dec=torch.from_numpy((d.astype(np.uint32)<<16).view(np.float32).copy())
        for bucket in buckets:
            _,stages=parts['geom'](cfg,bucket); nesc=min(stages,4)
            na=1 if bucket==32 and stages>=4 and nt==4 else nacc
            for m in (range(2,9) if bucket==8 else sorted(set((bucket,bucket-1)))):
                for dist in ('normal','outliers','probes'):
                    x=torch.randn(m,k,generator=gen)
                    if dist=='outliers':x[:,::97]*=30
                    if dist=='probes':
                        x.zero_(); rows=torch.arange(m);x[rows,(rows*63)%k]=torch.where(rows%2==0,8.,-32.)
                    x=x.bfloat16(); corr=torch.zeros(nesc,m,n)
                    order=torch.argsort(comp['cols'],stable=True)
                    for e in order.tolist():
                        row,col=int(comp['rows'][e]),int(comp['cols'][e]);stage=col//(wsk*upw*64)
                        delta=w[row,col].float()-dec[row,col]
                        corr[stage%nesc,:,row]+=delta*x[:,col].float()
                    result=torch.zeros(m,n)
                    for e in range(nesc):result+=corr[e]
                    cols=torch.arange(k)
                    for wk in range(wsk):
                        val=None
                        for chain in range(na):
                            take=((cols//64)%(wsk*upw)//upw==wk)&((((cols//64)%upw)*4+(cols//16)%4)%na==chain)
                            p=x[:,take].float()@dec[:,take].t()
                            val=p if val is None else val+p
                        result+=val
                    out=result.bfloat16();ref=x.double()@w.double().t()
                    err=float((out.double()-ref).norm()/ref.norm().clamp_min(1e-30))
                    rn=ref.norm(dim=-1);rr=float(((out.double()-ref).norm(dim=-1)/torch.maximum(rn,.1*rn.mean()).clamp_min(1e-30)).max())
                    assert err<=1e-2 and rr<=5e-2
                    if dist=='probes':assert torch.equal(out,ref.bfloat16())
                    records.append(dict(cfg=name,m=m,dist=dist,relative_l2=err,row_error=rr))
    return dict(roundtrip_weights=exact_weights,escapes=escapes,cases=records)


def multi_gates(tree,ns):
    pack=v1.packing_functions(tree);gen=torch.Generator().manual_seed(118588)
    records=[]
    for k,nth in ((1024,64),(4096,128)):
        w=(torch.randn(8,k,generator=gen)/k**.5).bfloat16()
        w[0,0],w[1,3],w[2,7]=0.,2.**-24,-2.**-25
        cg=pack(w,2);assert cg is not None
        codes=ns['_rc_codes'](cg['C'])
        a,b=v1.decoded(cg,cg['C'],False),v1.decoded(cg,codes,True)
        assert np.array_equal(a.view(np.uint32),b.view(np.uint32))
        for m in range(2,9):
            for dist in ('normal','outliers','probes'):
                x=torch.randn(m,k,generator=gen)
                if dist=='outliers':x[:,::97]*=30
                if dist=='probes':
                    x.zero_();r=torch.arange(m);x[r,(r*63)%k]=torch.where(r%2==0,8.,-32.)
                x=x.bfloat16()
                old=torch.stack([v1.gemv_model(a,cg,row[None,:],nth) for row in x])
                new=torch.stack([v1.gemv_model(b,cg,row[None,:],nth) for row in x])
                assert torch.equal(old,new)
                ref=x.double()@w.double().t();err=float((new.double()-ref).norm()/ref.norm())
                assert err<=1e-2
                if dist=='probes':assert torch.equal(new,ref.bfloat16())
                # Explicit trace: interleaving tokens/rows or using a one-j
                # x register window does not change ANY dot's K sequence.
                for tid in (0,nth-1):
                    reference=[j*16*nth+16*tid+e for j in range(k//(16*nth)) for e in range(16)]
                    fused=[j*16*nth+16*tid+g*8+e for j in range(k//(16*nth)) for g in range(2) for e in range(8)]
                    assert reference==fused
                records.append(dict(k=k,m=m,dist=dist,relative_l2=err))
    return records


def router_restore(data,etab,bases,s,wk,rotated):
    ku=128//s//wk
    a=data.numpy().reshape(ku,s,16,wk,2,3,32,4).transpose(2,1,3,0,4,5,6,7).reshape(16,128,2,3,32,4)
    raw=a.copy().view(np.uint8).reshape(16,128,2,3,32,16)
    low=raw[:,:,:,:2].transpose(0,1,2,4,3,5).reshape(16,128,2,32,32).astype(np.uint16)
    words=a[:,:,:,2].copy().view(np.uint32)
    high=np.empty_like(low)
    for h,base in enumerate(bases):
        hh=v1.decode_high(words[:,:,h],base,rotated)
        for q in range(4):
            for b in range(4):
                high[:,:,h,:,q*8+b]=(hh[:,:,:,q,0]>>(8*b))&255
                high[:,:,h,:,q*8+4+b]=(hh[:,:,:,q,1]>>(8*b))&255
    bits=low|(high<<8)
    table=etab.numpy().view(np.uint32)
    for key in range(len(table)):
        cta,warp=key//wk,key%wk;rg,rank=cta//s,cta%s
        for i in range(1,int(table[key,0])+1):
            e=int(table[key,i]);lane=e&31;q=(e>>5)&3;reg=(e>>7)&3;half=(e>>9)&1;h=(e>>10)&1;kl=(e>>11)&1
            chunk=(rank*wk+warp)*ku+kl
            bits[rg,chunk,h,lane,q*8+2*reg+half]=e>>16
    dense=[]
    for h in range(2):
        dense.append(bits[:,:,h].reshape(16,128,8,4,4,2,2,2).transpose(0,6,2,1,3,4,5,7).reshape(256,8192))
    return dense


def router_gates(ns):
    gen=torch.Generator().manual_seed(51291)
    w=torch.randn(256,8192,generator=gen)/8192**.5
    hi=w.bfloat16();lo=(w-hi.float()).bfloat16()
    total=escapes=0;records=[]
    for s,wk in ((16,8),(16,4)):
        pk=ns['_K']['nb']['r2_parts']['pack'](w,s,wk);assert pk is not None
        data,et,bh,bl,ne=pk;new=ns['_r2_rtpack'](data)
        a=router_restore(data,et,(bh,bl),s,wk,False);b=router_restore(new,et,(bh,bl),s,wk,True)
        for h,target in enumerate((hi,lo)):
            assert np.array_equal(a[h],b[h]) and np.array_equal(b[h],target.view(torch.int16).numpy().view(np.uint16))
            total+=target.numel()
        escapes+=ne
        for m in (2,7,8,16,32):
            x=torch.randn(m,8192,generator=gen).bfloat16();x[0].zero_();x[0,63]=8.
            out=x.float()@hi.float().t()+x.float()@lo.float().t()
            ref=x.double()@w.double().t();err=float((out.double()-ref).norm()/ref.norm())
            probe=float((out[0].double()-ref[0]).norm()/ref[0].norm())
            assert err<=1e-4 and probe<=1e-4
            records.append(dict(s=s,wk=wk,m=m,relative_l2=err,probe_error=probe))
    return dict(exact_bf16_payloads=total,escapes=escapes,cases=records)


def wrapper_policy_gates(ns):
    calls=[]
    class Mod:
        def r2_pool(self,*args):
            calls.append(('pool',args));args[5].copy_(args[2].float().sum(1,keepdim=True).expand_as(args[5]).bfloat16())
        def r2_multi(self,*args):
            calls.append(('multi',args));args[5].copy_(args[2].float().sum(1,keepdim=True).expand_as(args[5]).bfloat16())
        def r2_sk2(self,*args):
            calls.append(('sk2',args));args[2].copy_(args[1].float().sum(1,keepdim=True).expand_as(args[2]).bfloat16())
        def r2_routertail_c12(self,*args):calls.append(('router',args));return 0
    mod=Mod();ns.update(_ctx_bind=lambda x:None,_r2_parent_kernel_fn=lambda d,s:'fallback')
    state={'vjeux':{'dummy':torch.empty(0)},'rot2':{}}
    cg=dict(L=torch.empty(0),C=torch.empty(0),tab=torch.empty(0),pmax=2,base4=63*0x01010101)
    for kind,m in (('pool',1),('multi',2),('multi',7),('multi',8),('sk2',8),('sk2',16),('sk2',32)):
        layer='pooler' if kind=='pool' else 'unpooler'
        arm=dict(kind=kind,n=128,k=64,mod=mod,cg=cg,codes=torch.ones(1),orig=torch.zeros(1),
                 tab=torch.empty(0),pm=2,b4=0,ci=1,bucket=8 if m<=8 else m)
        state['rot2'][layer,m]=arm
        x=torch.zeros(m,64,dtype=torch.bfloat16);cfg=dict(layer=layer,m=m,n=128,k=64)
        a=ns['kernel_fn'](dict(x=x,config=cfg),state);x[:,0]=1;b=ns['kernel_fn'](dict(x=x,config=cfg),state)
        assert a.data_ptr()!=b.data_ptr() and not bool(a.any()) and bool((b==1).all())
        for bad in (dict(x=x.float(),config=cfg),dict(x=x,config=dict(cfg,n=256)),dict(x=x,config=dict(cfg,k=128)),
                    dict(x=torch.zeros(m,128,dtype=torch.bfloat16)[:,::2],config=cfg)):
            assert ns['kernel_fn'](bad,state)=='fallback'
    # Router launch ABI and standard/rotated control selection, without CUDA.
    saved=ns['_dt_torch'];ns['_dt_torch']=SimpleNamespace(empty=torch.empty,float32=torch.float32,bfloat16=torch.bfloat16,
        cuda=SimpleNamespace(current_stream=lambda:SimpleNamespace(cuda_stream=17)))
    try:
        arm=dict(kind='router',n=256,k=64,mod=mod,orig=torch.zeros(1),codes=torch.ones(1),etab=torch.zeros(1),bh=61,bl=50,s=16,wk=8)
        x=torch.zeros(8,64,dtype=torch.bfloat16);data=dict(x=x,config=dict(layer='router',m=8,n=256,k=64))
        a=ns['_r2_call'](data,arm,state,True);b=ns['_r2_call'](data,arm,state)
        assert a.dtype==b.dtype==torch.float32 and a.data_ptr()!=b.data_ptr()
        assert len(calls[-1][1])==11 and calls[-2][1][-1]==0 and calls[-1][1][-1]==1
    finally:ns['_dt_torch']=saved
    saved=dict(ns)
    try:
        ns.update(_dt_inputs=lambda *a:[torch.zeros(1)],_rc_capture=lambda fn:(None,torch.zeros(1)),_dt_close=lambda *a:True)
        arm=dict(kind='pool',tag='test',n=128,k=64)
        for draws,expect in (([(.974,.974)]*3,True), ([(.975,.975)]*3,True),
                             ([(.974,.977)],False), ([(.974,.974),(.976,.974)],False), ([(float('nan'),.9)],False)):
            it=iter(draws);ns['_dt_abba']=lambda *args:(17.,16.5,*next(it))
            got=ns['_r2_measure']('pooler',1,arm,{}, {'pooler':torch.zeros(1)},None,None,None,float('inf'))
            assert (got is not None)==expect
        ns['_dt_close']=lambda *a:False;ns['_dt_abba']=lambda *a:(10.,9.,.9,.9)
        try:ns['_r2_measure']('pooler',1,arm,{}, {'pooler':torch.zeros(1)},None,None,None,float('inf'))
        except RuntimeError:pass
        else:raise AssertionError('incorrect numerical result admitted')
    finally:ns.clear();ns.update(saved)
    return dict(fresh_bf16_calls=14,router_launches=2,fallbacks=28,admission_cases=6)


def main():
    torch.set_num_threads(4)
    assert b'\r' not in CANDIDATE.read_bytes()
    tree,ns=setup()
    report=dict(candidate_sha256=hashlib.sha256(CANDIDATE.read_bytes()).hexdigest(),parent_prefix_preserved=True,
        source=source_gates(tree,ns),bitfield=v1.exhaustive_gates(ns),pool_fma=pool_cpp_gate(ns),
        sk2=sk_gates(tree,ns),scalar_multi=multi_gates(tree,ns),router=router_gates(ns),
        wrappers=wrapper_policy_gates(ns),
        cuda_tools={**{k:shutil.which(k) for k in ('nvcc','ptxas')},'nvrtc_library':ctypes.util.find_library('nvrtc')},
        cuda_compile='SKIPPED: no CUDA/NVRTC toolchain; C++ syntax checks stub CUDA/PTX',
        gpu='NOT RUN: no GPU; no OKC operations authorized')
    (ROOT/'results/rotcodes2-offline.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
