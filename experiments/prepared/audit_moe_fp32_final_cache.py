"""Finite FP32 finalizer and assembly screen; no GPU execution."""
import ast
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())


def finalize(y,valid,weights,cache=False,mutant=None):
    # Mirrors 512 lanes x8 columns, XOR32 and serial16-warp reductions.
    ps=np.zeros((8,512),np.float32)
    saved=np.empty_like(y)
    for sl in range(8):
        if valid[sl] or mutant=='load_invalid':
            a=y[sl].copy()
            if cache:
                if mutant=='premature_BF16':
                    u=a.view(np.uint32);a=((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
                saved[sl]=a
            total=a[:,0]*a[:,0]
            for i in range(1,8):total=total+a[:,i]*a[:,i]
            ps[sl]=total
    lanes=np.arange(32)
    warps=ps.reshape(8,16,32)
    for offset in (16,8,4,2,1):warps=warps+warps[:,:,lanes^offset]
    ss=np.zeros(8,np.float32)
    for w in range(16):ss=ss+warps[:,w,0]
    out=np.zeros((512,8),np.float32)
    for sl in range(8):
        if valid[sl] or mutant=='load_invalid':
            scale=np.float32(weights[sl]/np.sqrt(np.float32(ss[sl]*np.float32(1/4096)+np.float32(1e-8))))
            a=saved[sl] if cache else y[sl]
            if mutant=='wrong_route' and valid[(sl+1)%8]:a=saved[(sl+1)%8]
            out=out+scale*a
    return out


def main():
    work=ROOT/'.tools/moe-fp32-final-cache'
    out=ROOT/'results/moe-fp32-final-cache-preflight-20261029'
    out.mkdir(exist_ok=True)
    def freeze(name,data):
        path=out/name
        assert not path.exists() or path.read_bytes()==data,('Frozen collision',name)
        path.write_bytes(data)
    source=json.loads((work/'source-audit.json').read_bytes())
    cuda=(work/'full.cu').read_bytes()
    assert hashlib.sha256(cuda).hexdigest()==source['full_CUDA_sha256']
    cases=0; digest=hashlib.sha256();rejected=set()
    for count in range(9):
        for seed in range(8):
            rng=np.random.default_rng(20261029+count*100+seed)
            valid=np.zeros(8,dtype=bool);valid[rng.permutation(8)[:count]]=True
            y=rng.normal(size=(8,512,8)).astype(np.float32)
            y[~valid]=np.nan
            weights=rng.random(8).astype(np.float32);weights/=weights.sum()
            old=finalize(y,valid,weights)
            got=finalize(y,valid,weights,cache=True)
            assert np.isfinite(got).all() and np.array_equal(old.view(np.uint32),got.view(np.uint32))
            if not valid.any():assert not np.count_nonzero(got)
            digest.update(got.tobytes());cases+=1
            for mutant in ('load_invalid','premature_BF16','wrong_route'):
                with np.errstate(invalid='ignore',divide='ignore'):
                    bad=finalize(y,valid,weights,cache=True,mutant=mutant)
                if not np.array_equal(old.view(np.uint32),bad.view(np.uint32)):rejected.add(mutant)
    assert rejected=={'load_invalid','premature_BF16','wrong_route'}
    binroot=ROOT/'.tools/cuda-compiler-13.0/nvidia/cu13/bin'
    summaries={}
    for name,tag in [('parent','incumbent-main4'),('candidate','moe-fp32-final-cache-empty-guard')]:
        base=ROOT/'.tools/nvrtc-preflight'/tag
        metadata=json.loads(base.with_suffix('.json').read_bytes())
        if name=='candidate':assert metadata['original_source_sha256']==source['device_prefix_sha256']
        assert metadata['return_code']==0 and not metadata['gpu_kernels_loaded_or_launched']
        sass=subprocess.check_output([str(binroot/'nvdisasm.exe'),'--print-code','--print-line-info',str(base.with_suffix('.cubin'))])
        resources=subprocess.check_output([str(binroot/'cuobjdump.exe'),'--dump-resource-usage',str(base.with_suffix('.cubin'))])
        log=base.with_suffix('.log').read_bytes()
        assert b'C751' not in log
        assert log.count(b'16 bytes stack frame, 56 bytes spill stores, 124 bytes spill loads')==2
        assert log.count(b'Used 96 registers, used 16 barriers')==2
        metrics={op:len(re.findall(rb'\b'+re.escape(op.encode())+rb'\b',sass)) for op in ('LDG.E.128','LDL','STL','HGMMA','WARPGROUP.ARRIVE','WARPGROUP.DEPBAR')}
        summaries[name]=dict(cubin_sha256=hashlib.sha256(base.with_suffix('.cubin').read_bytes()).hexdigest(),
                              SASS_sha256=hashlib.sha256(sass).hexdigest(),static_instructions=metrics,
                              base_registers_each=96,offline_stack_bytes_each=16,
                              static_spill_store_bytes_each=56,static_spill_load_bytes_each=124)
        for suffix in ('.cu','.json','.log'):freeze(name+'-compile'+suffix,base.with_suffix(suffix).read_bytes())
        freeze(name+'-resources.txt',resources)
        freeze(name+'-complete.sass.gz',gzip.compress(sass,mtime=0))
    a,b=summaries['parent']['static_instructions'],summaries['candidate']['static_instructions']
    assert a['LDG.E.128']-b['LDG.E.128']==32
    assert all(a[k]==b[k] for k in a if k!='LDG.E.128')
    masks=[((0x81c081c0<<k)|(0x81c081c0>>(32-k if k else 32)))&0xffffffff for k in range(32)]
    checked=0
    for shifts in itertools.combinations(range(1,32),3):
        checked+=1;bits=masks[0]
        for shift in shifts:
            if bits&masks[shift]:break
            bits|=masks[shift]
        else:raise AssertionError(('Unexpected four-rotate cover',shifts,bits))
    assert checked==4495
    report=dict(full_CUDA_sha256=source['full_CUDA_sha256'],device_prefix_sha256=source['device_prefix_sha256'],
                finite_finalizer_cases=cases,CPU_same_bits_with_identical_FP32_arithmetic=True,
                output_fixture_sha256=digest.hexdigest(),rejected_mutants=sorted(rejected),
                all_nonlocal_zero_and_NaN_poison_masking=True,
                compiler=summaries,vector_load_instructions_removed_each_BIG=16,
                spills_and_matrix_wait_static_counts_equal=True,
                actual_runtime_attributes_verified=False,GPU_numerical_validation=False,
                GPU_performance_validation=False,GPU_jobs=0,
                separate_codec_screen=dict(four_rotate_mask=hex(0x81c081c0),shift_triples_checked=4495,
                                           exact_disjoint_covers=0,conclusion='No four-rotation-only bijection for this fixed output mask; other decoders not ruled out'))
    for name in ('full.cu','device.cu','edits.json','source-audit.json'):freeze(name,(work/name).read_bytes())
    freeze('cpu-and-compiler-audit.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(dict(CUDA_sha256=source['full_CUDA_sha256'],CPU_cases=cases,mutants_rejected=sorted(rejected),
                          static_instruction_counts={k:v['static_instructions'] for k,v in summaries.items()},GPU_jobs=0)))


if __name__=='__main__':main()
