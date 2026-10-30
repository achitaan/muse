"""CPU screen of the exact two-namespace R2f diagnostic package."""

import json
import re
import subprocess

from build_norm_pair_diag import OUT
from check_r2_cpu import PRE,device_part
from check_pdl_diag_cpu import Capture,MAIN,nv


def main():
    binding=json.loads((OUT/'source-binding.json').read_text())
    original=OUT.parent/'pdli-cpu'
    proxy=PRE+'\n#include <cooperative_groups.h>\n';expressions=[]
    for name,spec in binding['R2f_specs'].items():
        body=device_part((original/(name+'.cu')).read_text())[len(PRE):]
        body=re.sub(r'^\s*#include\s+<[^>]+>[^\n]*\n','',body,flags=re.M)
        proxy+='\nnamespace pdli_'+name+' {\n'+body+'\n}\n#undef SMD\n'
        expressions.append('&pdli_'+name+'::'+spec['prefix']+'_gemv<'+','.join(map(str,spec['instance']))+'>')
    compiler=Capture();result=compiler.compile(proxy.encode(),expressions,[],tags=True,lineinfo=True)
    (OUT/'r2f-device-proxy.cu').write_bytes(proxy.encode())
    (OUT/'r2f-compile.log').write_bytes(result['log'].encode())
    assert result['rc']==0,result['log'][-2500:]
    kernels=nv.parse_ptxas(result['log']);assert len(kernels)==2
    assert all(not any(k[f]for f in ['stack_frame','spill_stores','spill_loads'])for k in kernels)
    cubin=OUT/'r2f.cubin';cubin.write_bytes(compiler.cubin)
    sass=subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(cubin)],text=True)
    (OUT/'r2f.sass').write_bytes(sass.encode())
    record=dict(GPU_jobs=0,R2f_CUDA_sha256=binding['R2f_CUDA_sha256'],kernels=kernels,
        no_stack_or_spills=True,limitation='Exact device bodies in two namespaces; host SDK/pybind compilation and H200 resources remain unverified.')
    (OUT/'cpu-compile.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
