"""Compile the fixed external-finalizer design and selected control bodies."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from build_attention_external_finalize import ROOT,OUT,freeze


def main():
    work=ROOT/'.tools/attention-external-finalize-preflight-v1';work.mkdir(exist_ok=True)
    controls=['&attn_kernel_fp8pv<1,true,0>','&attn_kernel_fp8pv<8,true,0>',
              '&attn_kernel_fp8pv<1,false,9>','&attn_kernel_fp8pv<1,false,13>',
              '&attn_kernel_fp8pv<8,false,3>','&attn_kernel_fp8pv<8,false,15>']
    for arm,file in [('baseline','baseline.cu'),('external','kernel.cu')]:
        source=(OUT/file).read_text();anchor='template <int L>\nvoid launch('
        assert source.count(anchor)==1
        prefix=source.split(anchor)[0]
        for header in ('#include <torch/extension.h>\n','#include <ATen/cuda/CUDAContext.h>\n','#include <cuda.h>\n'):
            assert prefix.count(header)==1;prefix=prefix.replace(header,'')
        prefix+='\ntemplate<int N> __global__ void layout_bytes_probe() {}\n'
        names=controls+['&layout_bytes_probe<sizeof(Smem<8,true>)+1024>']
        if arm=='external':names+=['&attn_kernel_fp8pv<8,true,16>','&attn_external_finalize8']
        path=work/(arm+'.cu');freeze(path,prefix.encode())
        if not path.with_suffix('.json').exists():
            command=[sys.executable,str(ROOT/'experiments/prepared/local_nvrtc_compile.py'),
                     '--source',str(path),'--minimal-std-tags','--out',str(path.with_suffix('')),
                     '--define','INFINITY=__int_as_float(0x7f800000)']
            for name in names:command+=['--name',name]
            subprocess.run(command,cwd=ROOT,check=True)
        report=json.loads(path.with_suffix('.json').read_bytes())
        assert report['return_code']==0 and report['original_source_sha256']==hashlib.sha256(prefix.encode()).hexdigest()
        log=path.with_suffix('.log').read_text()
        fields=re.findall(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',log)
        assert fields and all(tuple(map(int,row))==(0,0,0) for row in fields)
        assert not re.search(r'warning.*C751[45]',log)
        print(json.dumps(dict(arm=arm,kernels=len(names),zero_stack_spill_fields=True,
                             GPU_jobs=0,exact_judge_toolchain=False)))


if __name__=='__main__':main()
