"""Preserve CPU-only routing/public-source/Decode ownership screens."""
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())


def main():
    work=ROOT/'.tools/moe-small-route-study'
    out=ROOT/'results/moe-small-route-screen-20261029';out.mkdir(exist_ok=True)
    files={}
    names=['build.py','model.py','source-audit.json','model-audit.json','opportunity-screen.json',
           'public-source-review.json','REVIEW.md','full.cu','device.cu','baseline.cu',
           'vjeux-moe-upstream.py','vjeux-Apache-2.0.txt','posttrain-provenance.json',
           'decoder-v18d.cu','decoder-qbig.cu']
    for n in names:files[n]=(work/n).read_bytes()
    assert json.loads(files['model-audit.json'])['cases_checked']==10611
    assert json.loads(files['opportunity-screen.json'])['GPU_jobs']==0
    tools=ROOT/'.tools/cuda-compiler-13.0/nvidia/cu13/bin'
    compiles={}
    for mode in ('tiny1','tiny2'):
        for arm in ('baseline','candidate'):
            tag=arm+'-'+mode;base=work/tag
            metadata=json.loads(base.with_suffix('.json').read_bytes())
            log=base.with_suffix('.log').read_bytes()
            assert metadata['return_code']==0 and not metadata['gpu_kernels_loaded_or_launched']
            assert log.count(b'0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads')==2
            assert log.count(b'Used 146 registers, used 3 barriers')==2
            sass=subprocess.check_output([str(tools/'nvdisasm.exe'),'--print-code','--print-line-info',str(base.with_suffix('.cubin'))])
            resources=subprocess.check_output([str(tools/'cuobjdump.exe'),'--dump-resource-usage',str(base.with_suffix('.cubin'))])
            for suffix in ('.cu','.json','.log'):files[tag+suffix]=base.with_suffix(suffix).read_bytes()
            files[tag+'-complete.sass.gz']=gzip.compress(sass,mtime=0)
            files[tag+'-resources.txt']=resources
            compiles[tag]={'cubin_sha256':hashlib.sha256(base.with_suffix('.cubin').read_bytes()).hexdigest(),
                           'SASS_sha256':hashlib.sha256(sass).hexdigest(),'BIG_variants':2,
                           'proxy_base_registers_each':146,'stack_and_spill_bytes_each':0}
    binding={'files_sha256':{n:hashlib.sha256(d).hexdigest() for n,d in files.items()},
             'compiler':compiles,'GPU_jobs':0,'kernel_execution':False,'challenge_submission_built':False,
             'decision':'Park pending credible routing critical-path benefit; retain verified source and conserve GPU quota',
             'full_CUDA_sha256':hashlib.sha256(files['full.cu']).hexdigest(),
             'retained_Decode_source_sha256':'d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988'}
    files['binding.json']=(json.dumps(binding,indent=2)+'\n').encode()
    for n,data in files.items():
        p=out/n
        assert not p.exists() or p.read_bytes()==data,('Frozen collision',n)
        p.write_bytes(data)
    print(json.dumps({'directory':str(out.relative_to(ROOT)),'files':len(files),'CPU_cases':10611,'GPU_jobs':0}))


if __name__=='__main__':main()
