"""Validate packaged source wiring, actual Torch repack and finalizer geometry."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from bisect import bisect_right
import numpy as np
import torch
from moe_two_warpgroup_transform import transform

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').is_file())
torch.set_num_threads(2)
base=(ROOT/'best/moe_verified.py').read_bytes();raw=(ROOT/'.tools/moe-two-warpgroup-variant.py').read_bytes()
bt,nt=ast.parse(base),ast.parse(raw)
def constants(t):return {n.targets[0].id:ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
bc,nc=constants(bt),constants(nt)
assert nc['CUDA_SRC']==bc['CUDA_SRC']
expected,ledger=transform(bc['CUDA_SRC']);assert expected==nc['W2G_CUDA']
original_functions={n.name:n for n in bt.body if isinstance(n,ast.FunctionDef)}
packaged_functions={n.name:n for n in nt.body if isinstance(n,ast.FunctionDef)}
changed={'_ext','kernel_fn'}
for node,newnode in zip(bt.body,nt.body):
    if isinstance(node,ast.FunctionDef) and node.name in changed:continue
    assert ast.dump(node)==ast.dump(newnode),'Unexpected base AST change'
assert len(nt.body)>len(bt.body)
# The original prepare function is retained byte-for-byte as an AST, before
# the final wrapper shadows its public name. Kernel dispatch remains original.
assert ast.dump(original_functions['prepare'])==ast.dump(next(n for n in nt.body if isinstance(n,ast.FunctionDef) and n.name=='prepare'))

cuda=nc['W2G_CUDA']
assert '#define OKC_FT_T 32' in cuda and '#define OKC_WGSNX 2' in cuda
assert '__launch_bounds__(NTL, 2)' in cuda and 'namespace okc_w2g' in cuda
assert all(f'struct {family}<{n}>' not in cuda for family in ('WG','WGSMMA') for n in range(5,9))
assert cuda.count('achita_marlin_out2(')==12
assert '__floats2bfloat162_rn(encoded_acc0 * TWO62, encoded_acc1 * TWO62)' in cuda
assert '__bfloat1622float2(__hmul2_rn(acc, __bfloat162bfloat162(gs)))' in cuda
assert '* gsc' not in cuda
assert 'for (int turn = 0; turn < 2; ++turn) {\n    const int vtid = tid + turn * NT;\n    int j = 0;' in cuda
assert 'if (nj == 0) { for (int k = 0; k < 4; ++k) S.wgr[k] = 0; return; }' in cuda
assert 'grid <= g_w2g_capacity && grid < okc_w2g::WGS_SLOTS' in cuda

# Compile-hook inspection executes only _ext with a fake CUDA device/compiler.
# It verifies exactly what would be sent to load_inline, not actual CUDA code.
scope={};wanted={'MAIN_DEFINES','TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES','TINY_MAX_M','TINY2_MAX_M','MID_MAX_M','MAIN_NS','DQ7','PDL'}
for n in bt.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in wanted for t in n.targets):exec(compile(ast.Module(body=[n],type_ignores=[]),'<constants>','exec'),scope)
scope.update(CUDA_SRC=bc['CUDA_SRC'],CPP_SRC=bc['CPP_SRC'],ALT_SRCS={'w2g':cuda},_EXTS={})
scope['MAIN_DEFINES']=tuple(d for d in scope['MAIN_DEFINES'] if not d.startswith('-DOKC_NW='))+('-DOKC_NW=8','@w2g')
compiled=[]
scope['torch']=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda:(9,0)))
scope['load_inline']=lambda **kw: (compiled.append(kw) or SimpleNamespace(okc_set_pdl=lambda p:None))
exec(compile(ast.Module(body=[packaged_functions['_ext'],original_functions['_main_defines']],type_ignores=[]),'<compile-wiring>','exec'),scope)
scope['_ext'](scope['MAIN_DEFINES']+('-DOKC_PRMTSC=1',))
assert compiled[-1]['cuda_sources']==cuda and 'okc_w2g_resources' in compiled[-1]['functions']
assert 'std::vector<int64_t> okc_w2g_resources();' in compiled[-1]['cpp_sources']
assert '-DOKC_NW=8' in compiled[-1]['extra_cuda_cflags']
for key in ('TINY_DEFINES','TINY2_DEFINES','MID_DEFINES','MAIN_FALLBACK_DEFINES'):
    scope['_ext'](scope[key]+('-DOKC_PRMTSC=1',))
    assert compiled[-1]['cuda_sources']==bc['CUDA_SRC']
    assert compiled[-1]['functions']==['okc_run','okc_set_pdl']
assert scope['_main_defines'](())==scope['MAIN_FALLBACK_DEFINES']

# Actual packaged Torch repack matches an independent scalar piece-owner oracle.
ns={'torch':torch};exec(compile(ast.Module(body=[packaged_functions['_w2g_repack']],type_ignores=[]),'<repack>','exec'),ns)
rng=np.random.default_rng(20261028);torch_pack_bytes=0
for units in (32,16):
    old=rng.integers(0,256,size=(2,units,16,4,9216),dtype=np.uint8)
    got=ns['_w2g_repack'](torch.from_numpy(old.reshape(-1)),2,units).numpy().reshape(2,2*units,16,2,9216)
    for e in range(2):
        for u in range(2*units):
            for s in range(16):
                for g in range(2):assert np.array_equal(got[e,u,s,g],old[e,u//2,s,2*(u%2)+g])
    torch_pack_bytes+=old.nbytes

# Source-derived virtual gather must reset its per-pass job cursor. The
# independent oracle uses padded-row binary search; stale-cursor mutant fails.
gather_checks=0;mutant_detected=False
for counts in ([1,9,32,3],[31,1,8,17,2],[32]):
    bases=[0]
    for c in counts:bases.append(bases[-1]+((c+7)//8)*8)
    for grid in (1,2,8,31,264):
        for block in range(grid):
            q0,q1=bases[-1]*block//grid,bases[-1]*(block+1)//grid
            stale=0
            for turn in range(2):
                j=0
                for q in range(q0,q1):
                    while bases[j+1]<=q:j+=1
                    expected=bisect_right(bases,q)-1
                    assert j==expected and 0<=q-bases[j]<bases[j+1]-bases[j]
                    gather_checks+=1
                    while bases[stale+1]<=q:stale+=1
                    if stale!=expected:mutant_detected=True
assert mutant_detected

# Independently compare the two-pass, eight-warp FP32 sum of squares to
# FP64 over every column. Model separate operations, not CUDA FMA bit identity.
max_norm_relative=0.;missing_half_detected=False
vectors=list(rng.normal(size=(32,4096)).astype(np.float32))
vectors.extend([np.zeros(4096,np.float32)])
for index in (0,2047,2048,4095):
    v=np.zeros(4096,np.float32);v[index]=1.;vectors.append(v)
for y in vectors:
    chunks=y.reshape(512,8)
    terms=chunks*chunks
    sums=terms[:,0].copy()
    for i in range(1,8):sums=sums+terms[:,i]
    physical=(sums[:256]+sums[256:]).reshape(8,32)
    for offset in (16,8,4,2,1):physical=physical+physical[:,np.arange(32)^offset]
    total=np.float32(0)
    for warp in range(8):total=np.float32(total+physical[warp,0])
    exact=float(np.dot(y.astype(np.float64),y.astype(np.float64)))
    if exact:max_norm_relative=max(max_norm_relative,abs(float(total)-exact)/exact)
    assert abs(float(total)-exact)<=2e-6*max(exact,1e-30)
    if y[4095]==1. and np.count_nonzero(y)==1:missing_half_detected=np.sum(y[:2048]*y[:2048])!=exact
assert missing_half_detected

# Mock only extension calls; actual Torch CPU tensors exercise all public host
# branches, chunk tails and bank identity. No mocked numeric result is tested.
calls=[]
class Extension:
    def __init__(self,defs):self.defs=defs
    def okc_run(self,*args):calls.append((self.defs,args[0],args[1],args[2].shape[0],args[-1]))
host=dict(scope);host.update(torch=torch,HIDDEN=4096,CHUNK_TOKENS=384,_ext=lambda d:Extension(d))
exec(compile(ast.Module(body=[packaged_functions['kernel_fn']],type_ignores=[]),'<dispatch>','exec'),host)
oldbank=[object()];newbank=[object()];oldscratch=[object()];newscratch=[object()]
dispatches=0
for active in (True,False):
    for m in (1,4,5,8,9,16,17,32,33,64,384,385):
        calls.clear();sfx=('-DOKC_PRMTSC=1',) if active else ()
        state=dict(w=oldbank,scratch=oldscratch,grid=132,sfx=sfx,lo=0,cnt=32,w2g=dict(w=newbank,scratch=newscratch,grid=264) if active else None)
        data=dict(hidden_states=torch.zeros(m,4096,dtype=torch.bfloat16),topk_ids=torch.zeros(m,8,dtype=torch.int32),topk_weights=torch.ones(m,8,dtype=torch.float32),config=dict(local_start=0,local_count=32))
        host['kernel_fn'](data,state)
        assert sum(c[3] for c in calls)==m
        for defs,bank,scratch,size,grid in calls:
            selected=active and m>16
            assert (bank is newbank)==selected and (scratch is newscratch)==selected
            assert grid==(264 if selected else 132)
            assert ('@w2g' in defs)==selected
        dispatches+=1

report=dict(source_sha256=hashlib.sha256(raw).hexdigest(),parent_sha256=hashlib.sha256(base).hexdigest(),
    cuda_inverse_exact=True,unchanged_original_cuda=True,unrelated_base_AST_unchanged=True,
    unchanged_BF16_epilogues_and_FP32_norm=True,compile_wiring_tests=len(compiled),
    torch_repack_bytes_verified=torch_pack_bytes,gather_cursor_checks=gather_checks,
    stale_cursor_mutant_detected=mutant_detected,missing_half_finalizer_mutant_detected=bool(missing_half_detected),
    max_FP32_norm_relative_error_to_FP64=max_norm_relative,host_dispatch_cases=dispatches,
    gpu_verified=False,scope='CPU wiring, integer/layout and separate-FP32 arithmetic model only')
(ROOT/'.tools/moe-two-warpgroup-cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
print(json.dumps(report,indent=2))
