"""Compile only namespaced device proxies; inspect resources and wait ordering."""

import hashlib
import json
import subprocess

from pdl16_sources import ROOT
from build_pdl32_handoff import OUT, scoped
from check_pdl_diag_cpu import Capture, MAIN
from check_pdl16_cpu import ordering
from check_pdlu_cpu import order
import nvrtc_screen_alternatives as nv


def main():
    results={}
    exe=MAIN/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    for key,namespace,anchor,end,symbol,instance in [
        ('o32','d32o_private','#define OD __device__ __forceinline__','using OLEncode=',
         'pdlo32_projection','32,128,8,1,1,1,1'),
        ('u32','d32u_private','#define D __device__ __forceinline__','using EncodeFn=',
         'pdlu32_gemm','32,64,2,2,2,true,2')]:
        folder='pdl16' if key=='o32' else 'pdlu'
        source=(OUT.parent/folder/(key+'-candidate.cu')).read_text(encoding='utf-8')
        device=source.split(end,1)[0]
        proxy=scoped(device,namespace,anchor)
        proxy=proxy.replace('#include <cuda.h>',
            'struct __align__(64) CUtensorMap { unsigned long long opaque[16]; };')
        proxy=proxy.replace('#include <cuda_runtime.h>','')
        nvr=Capture();r=nvr.compile(proxy.encode(),['&'+namespace+'::'+symbol+'<'+instance+'>'],[],tags=True,lineinfo=True)
        (OUT/(key+'-device-proxy.cu')).write_bytes(proxy.encode())
        (OUT/(key+'-compile.log')).write_text(r['log'],encoding='utf-8')
        assert r['rc']==0,r['log'][-2000:]
        r['kernels']=nv.parse_ptxas(r['log'])
        assert len(r['kernels'])==1 and r['kernels'][0]['registers']==(42 if key=='o32'else 88)
        assert all(r['kernels'][0][k]==0 for k in ('spill_stores','spill_loads','stack_frame'))
        cubin=OUT/(key+'.cubin');cubin.write_bytes(nvr.cubin)
        sass=subprocess.check_output([str(exe),'-sass',str(cubin)],text=True)
        (OUT/(key+'.sass')).write_bytes(sass.encode())
        r['ordering']=ordering(key,sass) if key=='o32' else order(key,sass)
        r.update(proxy_sha256=hashlib.sha256(proxy.encode()).hexdigest(),GPU_jobs=0,
                 limitation='CPU SM90 proxy; H200 actual resources/bytes/preparation/whole-layer benefit unverified.')
        results[key]=r
        print(key,r['kernels'],r['ordering'],flush=True)
    (OUT/'device-compile-summary.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
