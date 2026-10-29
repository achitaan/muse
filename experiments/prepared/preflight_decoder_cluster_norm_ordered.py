"""Compile the single source-order repair; preserve the first trial unchanged."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from preflight_decoder_cluster_norm import ROOT,BASE,freeze


def main():
    assert hashlib.sha256((ROOT/'best/decoder_verified.py').read_bytes()).hexdigest()==BASE
    path=ROOT/'experiments/prepared/decoder_cluster_norm_ordered.cu'
    raw=path.read_bytes();work=ROOT/'.tools/decoder-cluster-norm-ordered-v1/main'
    if not work.with_suffix('.json').exists():
        cmd=[sys.executable,str(ROOT/'experiments/prepared/local_nvrtc_compile.py'),
             '--source',str(path),'--out',str(work)]
        for kind in range(4):cmd+=['--name',f'&okc_cluster_norm_ordered::boundary<{kind}>']
        subprocess.run(cmd,cwd=ROOT,check=True)
    meta=json.loads(work.with_suffix('.json').read_bytes());assert meta['return_code']==0
    assert meta['original_source_sha256']==hashlib.sha256(raw).hexdigest()
    log=work.with_suffix('.log').read_text()
    fields=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
    assert fields and all(tuple(map(int,x))==(0,0,0) for x in fields)
    registers=list(map(int,re.findall(r'Used (\d+) registers',log)))
    assert len(registers)==4 and max(registers)<=128
    out=ROOT/'results/decoder-cluster-norm-ordered-preflight-20261029/v1'
    for ext in ('.cu','.json','.log'):freeze(out/('compiler'+ext),work.with_suffix(ext).read_bytes())
    report=dict(parent_source_sha256=BASE,CUDA_sha256=hashlib.sha256(raw).hexdigest(),
        repaired_failure_source='f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf',
        repair='Original256logical threads/vec8/fourpacks/eight-warp XOR tree distributed over four64thread CTAs',
        offline_register_counts=registers,zero_stack_and_spills=True,static_shared_bytes=16,
        CPU_only=True,GPU_jobs=0,native_resource_and_performance_unverified=True,
        numerical_limits_unchanged=True,Python_submission_written=False)
    freeze(out/'binding.json',(json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
