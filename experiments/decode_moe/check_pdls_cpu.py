"""Bounded CPU compiler screen of the two exact HasRouter=true bodies."""

import hashlib
import json
import subprocess
from collections import Counter

from build_pdls import OUT
from pdl_norm_sources import UPSTREAM, pdln_derive
from check_pdln_cpu import proxy
from check_pdl_diag_cpu import Capture, MAIN, nv
from check_pdl16_cpu import instructions


def main():
    parent = (UPSTREAM / 'norm_residual.cuh').read_text()
    records = {}; counts = []
    for arm,source in [('parent',parent),('early',pdln_derive(parent))]:
        code = proxy(source)
        name = 'muse_spark_norm_residual_kernel' if arm=='parent' else 'pdln_norm_kernel'
        expr = '&sglang::'+name+'<8192,256,true,true,false,true,false,true>'
        compiler = Capture()
        result = compiler.compile(code.encode(),[expr],[],tags=True,lineinfo=True)
        (OUT / (arm+'-device-proxy.cu')).write_bytes(code.encode())
        (OUT / (arm+'-compile.log')).write_bytes(result['log'].encode())
        assert result['rc']==0,result['log'][-2500:]
        kernels = nv.parse_ptxas(result['log']); assert len(kernels)==1
        assert not any(kernels[0][f] for f in ['stack_frame','spill_stores','spill_loads'])
        cubin = OUT/(arm+'.cubin');cubin.write_bytes(compiler.cubin)
        sass = subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(cubin)],text=True)
        (OUT/(arm+'.sass')).write_bytes(sass.encode())
        ins = instructions(sass)
        waits=[a for a,t in ins if 'ACQBULK' in t];triggers=[a for a,t in ins if 'PREEXIT' in t]
        loads=[a for a,t in ins if 'LDG.' in t];stores=[a for a,t in ins if 'STG.' in t]
        assert len(waits)==len(triggers)==1 and waits[0]<loads[0]
        assert (triggers[0]>stores[-1]) if arm=='parent' else (triggers[0]<waits[0])
        math=Counter(t.split()[0]for a,t in ins if t.startswith(('FFMA','FADD','FMUL','MUFU','F2F','F2FP','SHFL')))
        counts.append(math)
        records[arm]=dict(device_proxy_sha256=hashlib.sha256(code.encode()).hexdigest(),
            kernels=kernels,trigger_address=triggers[0],wait_address=waits[0],first_LDG=loads[0],
            last_STG=stores[-1],math_counts=dict(math))
    assert counts[0]==counts[1]
    assert records['parent']['kernels'][0]['registers']==records['early']['kernels'][0]['registers']
    result=dict(GPU_jobs=0,HasRouter=True,records=records,arithmetic_instruction_counts_match=True,
        limitation='Exact scalar body/vector storage/CTA reduction with CUDA intrinsic/type adapters. Actual SDK build and H200 execution remain unverified.')
    (OUT/'cpu-compile-summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({arm:{k:v for k,v in r.items()if k!='math_counts'}for arm,r in records.items()},indent=2))


if __name__=='__main__':main()
