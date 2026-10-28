"""CPU ABI layout, wide scheduler arithmetic and local ring liveness audit.

Models built-in C++ fields from the retained CUDA declaration. This is not a
CUDA compiler/resource check. The retained dynamic shared size is independently
anchored to the prior H200 resource report (231168bytes).
"""
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
from collections import deque
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
raw=(ROOT/'best/moe_verified.py').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50'
tree=ast.parse(raw)
cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SRC' for t in n.targets))
body=cuda.split('struct Smem {',1)[1].split('\n};',1)[0]

def smem(nw,ft):
    env=dict(NW=nw,FT_T=ft,MMAX=384,EMAX=3072,JMAX=32+3072//ft+8,RED_W=4,NRB=2)
    types={'uint32_t':C.c_uint32,'uint64_t':C.c_uint64,'int':C.c_int32,'float':C.c_float,'void':None,'signed char':C.c_int8,'short':C.c_int16}
    active=True;stack=[];decl=[]
    for line in body.splitlines():
        line=line.split('//')[0].strip()
        if line.startswith('#if '):
            condition=line[4:].strip();value={'!OKC_WGS':False,'OKC_WGS':True,'OKC_WS':False}[condition]
            stack.append((active,value));active=active and value;continue
        if line=='#else':
            parent,value=stack[-1];active=parent and not value;continue
        if line=='#endif':active=stack.pop()[0];continue
        if not active or not line:continue
        match=re.fullmatch(r'(uint32_t|uint64_t|int|float|void|signed char|short)(\*)?\s+(.+);',line)
        assert match,line
        base=C.c_void_p if match[2] else types[match[1]]
        for field in match[3].split(','):
            field=field.strip();name=re.match(r'\w+',field)[0]
            typ=base
            for size in reversed(re.findall(r'\[([^\]]+)\]',field)):
                # Expressions consist only of the pinned declaration's constants.
                assert re.fullmatch(r'[A-Z_0-9+ ]+',size)
                typ=typ*eval(size,{'__builtins__':{}},env)
            decl.append((name,typ))
    assert not stack
    cls=type('Smem', (C.Structure,), {'_fields_':decl})
    return cls,dict(size=C.sizeof(cls),alignment=C.alignment(cls),base=(C.sizeof(cls)+127)//128*128,
                    offsets={n:getattr(cls,n).offset for n,_ in decl},jmax=env['JMAX'])

old,old_layout=smem(16,64);new,new_layout=smem(8,32)
old_dynamic=old_layout['base']+3*4*9216+3*8*4096
new_dynamic=new_layout['base']+3*2*9216+2*4*4096
assert old_dynamic==231168,'CPU ABI model differs from observed retained H200 layout'
assert C.sizeof(C.c_void_p)==8 and C.sizeof(C.c_uint64)==8
assert new_dynamic*2<228*1024
# The original padded host allocations already cover this proposed main bank;
# it still must be separate from old banks because counter offsets change.
slots,groups,fixw,jmax=272,2,4*512,new_layout['jmax']
fix_floats=2*slots*groups*fixw+2*slots*groups+jmax*16
old_alloc_fix=2*160*4*(8*512)+2*160*4+136*16
assert fix_floats<=old_alloc_fix
assert 8+jmax<=8+512 and 8+512+jmax<=8+1024
assert 3072+8*jmax==3072+8*(32+3072//32+8)

ca=[112,135,140,158];cb=[100,104,115,123]
def jobs(counts):
    fat=[min(32,c-first) for c in counts if c>8 for first in range(0,c,32)]
    skinny=[c for c in counts if 0<c<=8]
    return fat+skinny

# Valid cardinalities: all3072routes of384tokens are local, counts97on31
# experts and65onone. The enlarged A range product exceeds uint32.
edge=jobs([97]*31+[65]);assert sum(edge)==3072
edge_total=sum(ca[(c+7)//8-1] for c in edge)*1024
wide=edge_total*264
assert wide>2**32 and (wide%(2**32))//264!=edge_total

rng=np.random.default_rng(20261028)
distributions=[[],[32],edge,[32]*136,[1]*136]
for m in (1,8,17,32,33,64,128,256,383,384):
    for p in (np.ones(32)/32,np.r_[.8,np.ones(31)*.2/31],np.r_[1.,np.zeros(31)]):
        distributions.append(jobs(rng.multinomial(m*8,p).tolist()))

checks=0;max_correction=0
def floor_estimate(n,d):
    global checks,max_correction
    n=np.asarray(n,dtype=np.int64)
    expected=n//d
    nominal=(n.astype(np.float32)/np.float32(d))
    for direction in (-1,0,1):
        value=nominal.copy()
        toward=np.float32(np.inf if direction>0 else -np.inf)
        if direction:
            for _ in range(2):value=np.nextafter(value,toward)
        q=np.maximum(value,0).astype(np.int64);initial=q.copy()
        # Same monotone exact integer correction proposed for CUDA; no clipping
        # to the correct answer. All products here fit uint64 and signed int64.
        bad=q*d>n
        while np.any(bad):q[bad]-=1;bad=q*d>n
        bad=(q+1)*d<=n
        while np.any(bad):q[bad]+=1;bad=(q+1)*d<=n
        assert np.array_equal(q,expected)
        max_correction=max(max_correction,int(np.max(np.abs(q-initial))))
        checks+=q.size
    return expected

for counts in distributions:
    if not counts:continue # Explicit zero-job ranges, no divide-by-zero.
    assert len(counts)<=136 and all(1<=c<=32 for c in counts)
    for pb,costs in enumerate((ca,cb)):
        units=64>>pb;stages=units*16
        costs=np.array([costs[(c+7)//8-1] for c in counts],np.int64)
        prefix=np.r_[0,np.cumsum(costs*stages)];total=int(prefix[-1])
        xe=np.concatenate([prefix[j]+(np.arange(0,stages,16,dtype=np.int64)+1)*cost for j,cost in enumerate(costs)])
        for grid in (1,2,8,31,64,127,132,160,263,264,271):
            boundaries=floor_estimate(total*np.arange(grid+1,dtype=np.int64),grid)
            assert boundaries[0]==0 and boundaries[-1]==total and np.all(np.diff(boundaries)>=0)
            owner=np.minimum(floor_estimate(xe*grid-1,total),grid-1)
            assert np.all(boundaries[owner]<xe)
            assert np.all((owner==grid-1)|(boundaries[np.minimum(owner+1,grid)]>=xe))

# Local pipeline BFS. A consumer releases stage(n-1)'s X slot only after
# starting stage(n); segment ends release their current X slot immediately.
# Two X slots are the minimum that avoids circular producer/consumer waiting.
# Deferred global fixups and CUDA memory ordering are outside this model.
def ring_model(nx,ends,n=6):
    initial=(0,0,(0,0),tuple([0]*n),tuple([0]*n))
    queue=deque([initial]);seen={initial};dead=0;transitions=0
    while queue:
        pw,px,cons,wr,xr=state=queue.popleft();next_states=[]
        if pw<n and (pw<3 or wr[pw-3]==3):next_states.append((pw+1,px,cons,wr,xr))
        if px<n and (px<nx or xr[px-nx]==3):next_states.append((pw,px+1,cons,wr,xr))
        for wg,s in enumerate(cons):
            if s>=n or s>=pw or s>=px:continue
            cn=list(cons);ww=list(wr);xx=list(xr);bit=1<<wg
            cn[wg]+=1;ww[s]|=bit
            if s>0:xx[s-1]|=bit
            if s+1 in ends:xx[s]|=bit
            next_states.append((pw,px,tuple(cn),tuple(ww),tuple(xx)))
        terminal=cons==(n,n) and pw==px==n
        if not next_states and not terminal:dead+=1
        transitions+=len(next_states)
        for value in next_states:
            if value not in seen:seen.add(value);queue.append(value)
    return dict(states=len(seen),transitions=transitions,deadends=dead)
rings=[ring_model(2,ends) for ends in ({6},{2,4,6},{1,3,4,6})]
assert all(x['deadends']==0 for x in rings)
assert ring_model(1,{6})['deadends']>0,'Missing-slot mutant must be detected'

report=dict(parent_sha256=hashlib.sha256(raw).hexdigest(),old_layout=old_layout,new_layout=new_layout,
    old_observed_dynamic_shared_bytes=old_dynamic,proposed_dynamic_shared_bytes=new_dynamic,
    proposed_register_upper_bound=2*320*96,actual_cuda_resources_unknown=True,
    old_allocation_fix_floats=old_alloc_fix,new_required_fix_floats=fix_floats,
    integer_overflow_counterexample=dict(jobs=len(edge),routes=3072,total=edge_total,grid=264,product=wide),
    corrected_division_checks=checks,max_tested_float_correction=max_correction,
    local_ring_models=rings,missing_x_slot_mutant_detected=True,
    unverified=['CUDA ABI/resource attributes','cross-CTA fixup liveness and memory order','GPU implementation and all numerical behavior','performance'])
(ROOT/'experiments/prepared/moe_two_warpgroup_control_cpu_audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps({k:v for k,v in report.items() if k not in ('old_layout','new_layout')},indent=2))
