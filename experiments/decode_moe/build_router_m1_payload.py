"""Bind the one-token R1 prototype to checked joint-unit diagnostics and shape fallback."""
import ast
import hashlib
import json
from harness import append_harness
from screen_router_m1_cpu import ROOT,OUT,DECODER,derive
from build_router_sidecar_payload import DENSE,sources as previous_sources

RUNTIME_SHA='5a45430874a57d837f92d176f2d0aadda9db472da01e711242727c2b4627c4d4'
ADAPTER='''
class _M1Extension:
    def __init__(self, original, specialized):
        self.original = original
        self.specialized = specialized
    def r2_launch(self, *args):
        # The specialized host rejects all M!=1; choose the retained backend explicitly.
        return (self.specialized if args[4] == 1 else self.original).r2_launch(*args)
'''


def sources():
    parent,candidate=derive()
    parent_pack,_,cpp=previous_sources()
    assert parent_pack.startswith(parent)
    bridge=parent_pack[len(parent):]
    assert bridge.count('v18s_gemv<1,1,1,1,0,0,1024,0,4096,8192,2,16,8,1,70>')==1
    new_bridge=bridge.replace('v18s_gemv<1,1,1,1,0,0,1024,0,4096,8192,2,16,8,1,70>',
                              'm1_gemv<1,1,1,1,0,0,1024,1,4096,8192,2,16,8,1,70>')
    renamed=candidate.replace('v18s_gemv','m1_gemv')
    assert renamed.replace('m1_gemv','v18s_gemv')==candidate
    return parent_pack,renamed+new_bridge,cpp


def runtime():
    raw=(ROOT/'experiments/decode_moe/router_sidecar_runtime.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==RUNTIME_SHA
    text=raw.decode()
    old='resources[:4] != [64, 0, 32768, 16384]'
    new="resources[:4] != ([57, 0, 4096, 16384] if arm == 'candidate' else [64, 0, 32768, 16384])"
    assert text.count(old)==1;text=text.replace(old,new)
    old='for variant in range(3):'
    assert text.count(old)==1;text=text.replace(old,'for variant in range(5):')
    old='''            elif variant == 2:
                fresh[:, -1] = 128.03125'''
    new=old+'''
            elif variant == 3:
                fresh.zero_().neg_()
            elif variant == 4:
                fresh.mul_(1.e-30)'''
    assert text.count(old)==1;text=text.replace(old,new)
    old='    state = _rs_parent_prepare(weights)'
    new="    if len(modules) == 2:\n        modules['candidate'] = _M1Extension(modules['parent'], modules['candidate'])\n"+old
    assert text.count(old)==1;text=text.replace(old,new)
    text=text.replace('_rs_','_rm_').replace('RS_','RM_').replace('sidecar_gemv','m1_gemv')
    return ADAPTER+text


def main():
    raw=(ROOT/'best/verified.py').read_bytes();assert hashlib.sha256(raw).hexdigest()==DENSE
    screen=json.loads((OUT/'cpu-audit.json').read_text());assert screen['clean_resource_screen']
    parent,candidate,cpp=sources()
    for name,src in [('parent',parent),('candidate',candidate)]:
        (OUT/(name+'-packaged.cu')).write_bytes(src.encode())
    (OUT/'diagnostic.cpp').write_bytes(cpp.encode())
    (OUT/'runtime.py').write_bytes(runtime().encode())
    payload=append_harness(raw.decode())+'\n_rm_parent_prepare = prepare\n_rm_parent_kernel_fn = kernel_fn\n'
    payload+='RM_PARENT_CUDA = '+repr(parent)+'\nRM_CANDIDATE_CUDA = '+repr(candidate)+'\nRM_CPP = '+repr(cpp)+'\n'+runtime()
    result=payload.encode();ast.parse(result);assert len(result)<1_000_000 and result.startswith(raw.rstrip(b'\n'))
    path=ROOT/'experiments/decode_moe/dense_router_m1_diagnostic.py';path.write_bytes(result)
    record={'source_sha256':hashlib.sha256(result).hexdigest(),'bytes':len(result),'Dense_parent_sha256':DENSE,
        'Decode_parent_sha256':DECODER,'CUDA_parent_sha256':hashlib.sha256(parent.encode()).hexdigest(),
        'CUDA_candidate_sha256':hashlib.sha256(candidate.encode()).hexdigest(),
        'screened_candidate_CUDA_sha256':screen['candidate_CUDA_sha256'],
        'scope':'One-token specialization of retained129-CTA R1;all other M explicitly use retained module;public Dense unchanged',
        'private_marker':'m1_gemv','original_public_prefix_exact':True,
        'numerical_gate':'Twenty changing-input graph probes acrossM1,8,2,4;zero/normal/last-column spike/negative-zero/tiny values;byte-identical pooler/FP32 partials/router weights/IDs/counter;parent-repeat controls;nonM1 uses retained module.',
        'timing_gate':'M1 normal/spike,both orders,25 samples per arm;all four cost ratios<0.97;then unchanged controls within2%;raw observations retained.',
        'actual_resource_expectations':{'parent':[64,0,32768,16384,1],'candidate':[57,0,4096,16384,1]},
        'preparation_cutoffs_seconds':[268,274,278],'GPU_jobs':0,'admitted_for_GPU':False,
        'no_Dense_gain_claim':True,'no_Decode_transfer':True,'no_scored_submission':True,
        'next_gate':'Packaged-device,PDL/source,ABI and adapter audit;fresh quota/queue/spacing admission before any GPU job'}
    (OUT/'payload-binding.json').write_bytes((json.dumps(record,indent=2)+'\n').encode());print(json.dumps(record,indent=2))


if __name__=='__main__':main()
