"""Source, preparation protocol and terminal-report retention filters."""

import ast
import base64
import hashlib
import json
import zlib

import torch

from build_norm_pair_diag import HERE, ROOT, OUT, DENSE_SHA
from audit_pdln_interface_cpu import extract


def main():
    raw=(HERE/'dense_norm_pair_diag.py').read_bytes();parent=(ROOT/'best/verified.py').read_bytes()
    binding=json.loads((OUT/'source-binding.json').read_text())
    assert hashlib.sha256(parent).hexdigest()==DENSE_SHA and raw.startswith(parent)
    assert hashlib.sha256(raw).hexdigest()==binding['source_sha256']
    pf={n.name:n for n in ast.parse(parent).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    cf={n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    assert ast.dump(pf['kernel_fn'],include_attributes=False)==ast.dump(cf['kernel_fn'],include_attributes=False)
    decode=ast.parse((HERE/'decoder_pdli.py').read_text())
    df={n.name:n for n in decode.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    for name in ['_PdliFns','_pdli_rc','_pdli_route','_pdli_equal','_pdli_probe']:
        assert ast.dump(df[name],include_attributes=False)==ast.dump(cf[name],include_attributes=False)
    source=HERE/'norm_pair_diag_runtime.py'
    cases=[]
    for early_us,expected in [(8.4,True),(8.6,False),(9.8,False)]:
        calls=[];seeds=[]
        def norm(stream,branch,p): return branch.clone(),stream.clone(),stream.clone()
        def route(xb,xr,state):
            return xb[:,:4096].clone(),torch.ones((xb.shape[0],8)),torch.zeros((xb.shape[0],8),dtype=torch.int32)
        def probe(fns,mutate,**kwargs):
            calls.append(kwargs['label'])
            for seed in range(3):
                mutate(seed);outputs=[fn()for fn in fns.values()]
                assert all(torch.equal(a,b)for a,b in zip(*outputs))
                if seed==1:assert torch.count_nonzero(outputs[0][0])==0
                seeds.append(seed)
            return True
        def timing(fns,dev,reps):
            assert reps==25
            for fn in fns.values():
                assert torch.count_nonzero(fn()[0])>0,'Adversarial zero inputs leaked into timing'
            return dict(parent=10.,early=early_us)
        env=dict(torch=torch,_okh_budget=lambda *a:True,_pdli_probe=probe,
            _np_original=norm,_pdls_boundary=lambda s,b,state:norm(s,b,state['pdls']),
            _pdli_route=route,_okh_time=timing,
            _okh_pick=lambda t,base,margin:'early'if t['early']<margin*t[base]else base)
        extract(source,{'_np_probe'},env)
        report=dict(norm_bytes=[],pairs=[],budget_stop=False)
        env['_np_probe']({'pdli':{'plans':{'r1':{}}}},1,torch.Generator().manual_seed(7),'cpu',0,report)
        assert calls==['NP_norm_M1_channel','NP_pair_M1','NP_norm_M1_scalar']
        assert seeds==[0,1,2]*3 and len(report['norm_bytes'])==2
        assert report['pairs'][0]['causal_gate']==expected
        assert [x['order']for x in report['pairs'][0]['orders']]==[['parent','early'],['early','parent']]
        cases.append(dict(early_us=early_us,gate=expected))
    # Terminal summary remains available after verbose earlier output and suffix.
    report['errors']=['x'*2000];payload=base64.b64encode(zlib.compress(json.dumps(report).encode(),9)).decode()
    tail=('p'*50000+'\nNP_SUMMARY_Z64 '+payload+'\nNP_DONE 250\n'+'q'*1500)[-16000:]
    recovered=next(x.split(' ',1)[1]for x in tail.splitlines()if x.startswith('NP_SUMMARY_Z64 '))
    assert json.loads(zlib.decompress(base64.b64decode(recovered)))==report and len(payload)<6000
    record=dict(GPU_jobs=0,source_sha256=binding['source_sha256'],Dense_prefix_and_timed_kernel_unchanged=True,
        exact_Decode_R2f_and_byte_probe_AST=True,protocol_cases=cases,
        changing_zero_outlier_probes=True,ordinary_inputs_restored_before_timing=True,
        terminal_report_retention_pass=True,
        limitation='CPU source and mocked protocol audit only; no H200 SDK link, resources, byte correctness or timing.')
    (OUT/'cpu-audit.json').write_bytes((json.dumps(record,indent=2)+'\n').encode())
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
