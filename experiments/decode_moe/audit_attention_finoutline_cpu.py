"""Screen hot-code size and call resources without loading a GPU kernel."""

import hashlib
import json
import re
import subprocess
import sys

from build_attention_finoutline import ROOT, HERE, OUT, derive

sys.path.insert(0,str(HERE.parents[2]/'opus-fresh/experiments/opus_fresh'))
import nvrtc_screen_alternatives as nv


class Capture(nv.NVRTC):
    def output(self,program,kind):
        result=super().output(program,kind)
        if kind=='CUBIN':self.cubin=result
        return result


def blocks(sass):
    result={}
    for match in re.finditer(r'Function : (\S+)(.*?)(?=Function :|\Z)',sass,re.S):
        symbol,body=match.groups()
        ins=re.findall(r'/\*[0-9a-fA-F]+\*/\s+(.*?;)',body)
        result[symbol]=dict(instructions=len(ins),calls=[s for s in ins if 'CALL' in s],
            local_instructions=[s for s in ins if re.search(r'\b(?:LDL|STL)\b',s)])
    return result


def main():
    _,_,_,_,parent,changed=derive();exe=ROOT.parents[2]/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    assert (OUT/'candidate.cu').read_bytes()==changed.encode()
    results={}
    for label,source in [('parent',parent),('outlined',changed)]:
        device=source.split('template <int L>\nvoid launch(',1)[0]
        device=re.sub(r'#include <(torch/extension.h|ATen/cuda/CUDAContext.h|type_traits|cuda.h)>\n','',device)
        device='#include <math_constants.h>\n#define INFINITY CUDART_INF_F\nnamespace std { template<class T,T V> struct integral_constant { static constexpr T value=V; }; }\n'+device
        compiler=Capture();r=compiler.compile(device.encode(),['&attn_kernel_fp8pv<1,true,0>'],[],tags=True,lineinfo=True)
        (OUT/(label+'-device.cu')).write_bytes(device.encode());(OUT/(label+'.log')).write_bytes(r['log'].encode())
        assert r['rc']==0,r['log'][-3000:]
        binary=OUT/(label+'.cubin');binary.write_bytes(compiler.cubin)
        sass=subprocess.check_output([str(exe),'-sass',str(binary)],text=True)
        (OUT/(label+'.sass')).write_bytes(sass.encode())
        results[label]=dict(kernels=nv.parse_ptxas(r['log']),functions=blocks(sass),cubin_sha256=r['cubin_sha256'])
        print(label,json.dumps(results[label])[:2200],flush=True)
    entries={label:next((v for k,v in r['functions'].items()if 'attn_kernel_fp8pv'in k),None)for label,r in results.items()}
    assert all(entries.values())
    ratio=entries['outlined']['instructions']/entries['parent']['instructions']
    clean=all(k['stack_frame']==k['spill_stores']==k['spill_loads']==0 for k in results['outlined']['kernels'])and not any(f['local_instructions']for f in results['outlined']['functions'].values())
    record=dict(CUDA_sha256=hashlib.sha256(changed.encode()).hexdigest(),resources=results,
        compiled_kernel_instruction_ratio=ratio,clean_resource_screen=clean,
        count_scope='Includes embedded leaf subroutines; not dynamic instruction count or measured instruction-cache pressure.',
        decision='CPU reject'if not clean or ratio>=.95 else 'CPU screen passes; H200 numerical and timing evidence still required',
        GPU_jobs=0,admitted_for_GPU=False,Decode_checks_used=0)
    (OUT/'cpu-audit.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in record.items()if k!='resources'},indent=2))


if __name__=='__main__':main()
