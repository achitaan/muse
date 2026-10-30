"""One metadata-specialized address-family screen, no GPU submission.

New-path eligibility would require L1,Hq16,Hkv2,page_size1 and both caches
strictly below 2**32 bytes. All other cases must retain the original source.
No Python dispatch payload or GPU admission is constructed here.
"""
import hashlib
import json
import re
import subprocess

from build_attention_finoutline import ROOT, derive
from audit_attention_finoutline_cpu import Capture, blocks

OUT=ROOT/'results/decode-moe-plan-20261030/attention-page1-constants'
CONSTANTS={'off32':1,'lp':0,'P':1,'lrb':7,'row_bytes':128,'Hkv':2,'Hq':16}


def main():
    raw,_,_,_,parent,_=derive()
    OUT.mkdir(exist_ok=True)
    split='template <int L>\nvoid launch('
    device,host=parent.split(split,1)
    changed=device;substitutions={};macros=[]
    for field,value in CONSTANTS.items():
        token='OKC_PAGE1_'+field.upper()
        assert token not in parent
        changed,count=re.subn(r'\bp\.'+field+r'\b',token,changed)
        assert count>0
        substitutions[field]=count
        macros.append('#define '+token+' '+str(value)+'\n')
    inverse=changed
    for field in CONSTANTS:inverse=inverse.replace('OKC_PAGE1_'+field.upper(),'p.'+field)
    assert inverse==device
    candidate=''.join(macros)+changed+split+host
    (OUT/'candidate.cu').write_bytes(candidate.encode())
    resources={}
    exe=ROOT.parents[2]/'.tools/cuda-compiler-13.0/nvidia/cu13/bin/cuobjdump.exe'
    for label,source in [('parent',parent),('constant',candidate)]:
        code=source.split(split,1)[0]
        code=re.sub(r'#include <(torch/extension.h|ATen/cuda/CUDAContext.h|type_traits|cuda.h)>\n','',code)
        code='#include <math_constants.h>\n#define INFINITY CUDART_INF_F\nnamespace std { template<class T,T V> struct integral_constant { static constexpr T value=V; }; }\n'+code
        compiler=Capture();r=compiler.compile(code.encode(),['&attn_kernel_fp8pv<1,true,0>'],[],tags=True,lineinfo=True)
        (OUT/(label+'-device.cu')).write_bytes(code.encode());(OUT/(label+'.log')).write_bytes(r['log'].encode())
        assert r['rc']==0,r['log'][-3000:]
        binary=OUT/(label+'.cubin');binary.write_bytes(compiler.cubin)
        sass=subprocess.check_output([str(exe),'-sass',str(binary)],text=True)
        (OUT/(label+'.sass')).write_bytes(sass.encode())
        parsed=blocks(sass)
        entry=next(v for k,v in parsed.items()if 'attn_kernel_fp8pv'in k)
        from audit_attention_finoutline_cpu import nv
        resources[label]=dict(kernels=nv.parse_ptxas(r['log']),entry=entry,cubin_sha256=r['cubin_sha256'])
        print(label,json.dumps(resources[label])[:1000],flush=True)
    a,b=resources['parent'],resources['constant']
    ratio=b['entry']['instructions']/a['entry']['instructions']
    clean=not b['entry']['local_instructions']
    for x,y in zip(a['kernels'],b['kernels']):
        clean=clean and all(y[k]<=x[k]for k in ('registers','stack_frame','spill_stores','spill_loads'))
    # Kernel metadata constants,not input values,are specialized. Arbitrary
    # physical page order remains device-driven. Large-cache/non-P1 fallback
    # is mandatory if this source ever becomes an executable candidate.
    report=dict(parent_sha256=hashlib.sha256(raw).hexdigest(),CUDA_sha256=hashlib.sha256(candidate.encode()).hexdigest(),
        constants=CONSTANTS,substitutions=substitutions,exact_inverse=True,resources=resources,
        compiled_instruction_ratio=ratio,no_resource_regression=clean,
        scope='L1 address arithmetic only within guarded metadata family;arithmetic,all key coverage,masks,cache writes,barriers,splits and precision rules remain original.',
        decision='CPU reject'if not clean or ratio>=.95 else 'CPU screen passes;layout/dispatch/GPU numerical/latency evidence still required.',
        GPU_jobs=0,admitted_for_GPU=False,Decode_checks_used=0,
        limitation='Static footprint is not dynamic instruction count,cache-miss evidence or a latency prediction.')
    (OUT/'cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in report.items()if k!='resources'},indent=2))


if __name__=='__main__':main()
