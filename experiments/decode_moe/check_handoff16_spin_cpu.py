"""CPU compiler/ownership screen of the only new diagnostic device kernel."""

import hashlib
import json
import subprocess

from build_handoff16_diag import OUT,SPIN
from check_pdl_diag_cpu import Capture,MAIN,nv
from check_pdl16_cpu import instructions


def main():
    device='typedef unsigned long long size_t;\n'+SPIN[:SPIN.index('int d16_spin_start(')]
    compiler=Capture();r=compiler.compile(device.encode(),['&d16_spin_kernel'],[],tags=True,lineinfo=True)
    (OUT/'spin-device-proxy.cu').write_bytes(device.encode());(OUT/'spin-compile.log').write_bytes(r['log'].encode())
    assert r['rc']==0,r['log'][-2500:]
    kernels=nv.parse_ptxas(r['log']);assert len(kernels)==1
    assert not any(kernels[0][k]for k in ['stack_frame','spill_stores','spill_loads'])
    cubin=OUT/'spin.cubin';cubin.write_bytes(compiler.cubin)
    sass=subprocess.check_output([str(MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'),'-sass',str(cubin)],text=True)
    (OUT/'spin.sass').write_bytes(sass.encode());ins=instructions(sass)
    trigger=[a for a,t in ins if 'PREEXIT'in t];timer=[a for a,t in ins if 'GLOBALTIMER'in t]
    barriers=[a for a,t in ins if 'BAR.SYNC'in t];stores=[a for a,t in ins if 'STG.'in t]
    assert len(trigger)==1and timer and barriers and stores
    assert trigger[0]<min(timer)<max(barriers)<min(stores)
    for m in [1,16,32]:
        for k in [1024,4096]:
            writes=[t*k for t in range(256)if t<m]
            assert len(writes)==m and len(set(writes))==m
            assert all(0<=i<m*k and i%k==0for i in writes)
    result=dict(GPU_jobs=0,spin_source_sha256=hashlib.sha256(SPIN.encode()).hexdigest(),
        kernels=kernels,trigger_address=trigger[0],first_timer=min(timer),
        last_block_barrier=max(barriers),first_store=min(stores),
        ownership_cases=6,only_first_column_writers=True,
        limitation='Device-only CPU compiler screen; five-microsecond runtime, graph capture, visibility and real kernel overlap remain unverified.')
    (OUT/'spin-cpu.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
