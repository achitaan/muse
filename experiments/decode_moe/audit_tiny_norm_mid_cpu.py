"""Screen the adjacent MID configuration without extending GPU dispatch."""
import ast
import hashlib
import json
import sys

from build_tiny_norm_source import derive, OUT
from tinyroute_source import HERE, ROOT


def main():
    _,candidate,_,_=derive()
    raw=(ROOT/'best/moe_verified.py').read_bytes()
    tree=ast.parse(raw)
    values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
            if isinstance(n,ast.Assign)and getattr(n.targets[0],'id','')in ('CUDA_SRC','MID_DEFINES')}
    flags=list(values['MID_DEFINES'])+['-DOKC_DQ7=1','-DOKC_NS=3','-DOKC_NSF=3']
    assert '-DOKC_NW=8'in flags and '-DOKC_TINY=1'in flags and '-DOKC_WS=1'in flags
    out=OUT/'mid-cpu-screen';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(HERE.parents[2]/'opus-fresh/experiments/opus_fresh'))
    import nvrtc_screen_alternatives as nv
    result={}
    for prmt in (False,True):
        suffix=['-DOKC_PRMTSC=1']if prmt else []
        for label,source,namespace in [('parent',values['CUDA_SRC'],'okc'),('norm',candidate,'tiny_norm_whole')]:
            kernel='moe_fused'if label=='parent'else 'moe_tnorm_fused'
            r=nv.NVRTC().compile(source.split('#include <torch/extension.h>',1)[0].encode(),
                ['&'+namespace+'::'+kernel+'<false>'],flags+suffix,tags=True,lineinfo=False)
            key=label+('_PRMT'if prmt else '_LUT');(out/(key+'.log')).write_bytes(r['log'].encode())
            assert r['rc']==0,r['log'][-3000:]
            result[key]=dict(kernels=nv.parse_ptxas(r['log']),cubin_sha256=r['cubin_sha256'])
            print(key,json.dumps(result[key]),flush=True)
    clean=all(k['stack_frame']==k['spill_stores']==k['spill_loads']==0 for name,r in result.items()if name.startswith('norm')for k in r['kernels'])
    report=dict(candidate_CUDA_sha256=hashlib.sha256(candidate.encode()).hexdigest(),MID_defines=values['MID_DEFINES'],
        result=result,zero_stack_spills=clean,
        decision='CPU only;no M9..16 dispatch added. Wait for the tiny check,then independently review MID route/job ownership and performance before any extension.',
        GPU_calls=0,admitted_for_GPU=False)
    (out/'cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())


if __name__=='__main__':main()
