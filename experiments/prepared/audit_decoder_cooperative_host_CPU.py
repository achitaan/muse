"""CPU-only host bridge/identity and guard fault-injection audit."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'STATE.json').exists())
sys.path.insert(0,str(ROOT/'scripts'))
from decoder_cooperative_evidence import PACKET,PARENT,TRIAL,validate_packet,full_result,bound_result,verify_trial


def freeze(path,data):
    assert not path.exists() or path.read_bytes()==data
    path.write_bytes(data)


def main():
    raw=(PACKET/'v3/submission.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==TRIAL
    compile(raw,'cooperative-frozen-candidate','exec')
    cpp=(PACKET/'v3/declaration.cpp').read_text()
    host=(ROOT/'experiments/prepared/decoder_cooperative_boundary_host.cu').read_text()
    declaration=cpp[cpp.index('int64_t cb_launch('):].rstrip().removesuffix(';')
    definition=host[host.index('int64_t cb_launch('):host.index(' {\n#define CB_ARGS')]
    assert declaration==definition
    assert len(declaration[declaration.index('(')+1:-1].split(','))==20
    calls=[]
    class Extension:
        def cb_launch(self,*args):
            assert len(args)==20
            assert all(isinstance(v,int) for v in args[:18])
            assert all(isinstance(v,float) for v in args[18:])
            calls.append(args)
            return 0
    fake_torch=SimpleNamespace(empty=torch.empty,empty_like=torch.empty_like,
        bfloat16=torch.bfloat16,float32=torch.float32,
        cuda=SimpleNamespace(current_stream=lambda *args:SimpleNamespace(cuda_stream=1234)))
    namespace=dict(torch=fake_torch,HIDDEN=8192,prepare=lambda weights:{})
    addon=(PACKET/'v3/addon.py').read_bytes()
    exec(compile(addon,'cooperative-addon-CPU-mock','exec'),namespace)
    extension=Extension()
    assert namespace['_cb_query'](extension,1,1,3)==0
    assert calls[-1][:4]==(1,1,3,1)
    # Explicit signature test avoids importing or monkey-patching Torch modules.
    assert calls[-1][4:18]==(0,)*14 and calls[-1][18:]==(1e-8,1e-5)
    plan=dict(fmt=1,wp=torch.zeros(16,dtype=torch.int32),esc=torch.zeros(1,dtype=torch.int32),t01=0,u01=0)
    norm={k:torch.zeros(8192,dtype=(torch.bfloat16 if k=='_cc_pre_weight' else torch.float32))
          for k in ('_cc_alpha','_cc_beta','_cc_pre_weight')}
    norm.update(post_eps=1e-8,pre_eps=1e-5)
    x=torch.zeros((1,1024),dtype=torch.bfloat16);residual=torch.zeros((1,8192),dtype=torch.float32)
    outputs=namespace['_cb_call'](extension,1,plan,x,residual,norm)
    assert [o.dtype for o in outputs]==[torch.bfloat16,torch.float32,torch.float32,torch.bfloat16]
    assert all(o.shape==(1,8192) for o in outputs)
    prepared={'cooperative_boundary':dict(plans={1:plan},ext=extension,norm=norm)}
    assert namespace['_cb_apply'](x,residual,prepared) is None
    identity=[]
    for name in ('TASK.md','reference.py','checks.py','baseline.py','AGENTS.md','nvfp4.py'):
        pinned=(ROOT/'muse_decode_layer'/name).read_bytes()
        fresh=(ROOT/'.tools/decoder-coop-fresh-task-20261029'/name).read_bytes()
        identity.append(dict(file=name,pinned_sha256=hashlib.sha256(pinned).hexdigest(),
            fresh_sha256=hashlib.sha256(fresh).hexdigest(),exact=pinned==fresh))
    assert all(r['exact'] for r in identity)
    freeze(PACKET/'evaluator-identity.json',(json.dumps(dict(problem='muse_decode_layer',files=identity),indent=2)+'\n').encode())
    report=dict(source_sha256=TRIAL,query_and_launch_arity=20,unsupported_layout_rejected=True,
        candidate_python_compiles=True,CPP_declaration_matches_host=True,CPU_mock_only=True,
        GPU_jobs=0,exact_CUDA_host_ABI_not_validated=True)
    freeze(PACKET/'host-CPU-admission.json',(json.dumps(report,indent=2)+'\n').encode())
    assert validate_packet()==raw
    full_result(bound_result(PARENT))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
