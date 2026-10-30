"""Verify measured-scope restriction, unchanged CUDA and conservative gates."""

import ast
import hashlib
import json

from build_pdls1 import HERE, OUT, UNIT, BASE_SHA
from audit_pdln_interface_cpu import extract


def main():
    raw=(HERE/'decoder_pdls1_draft.py').read_bytes();src=raw.decode();old=(HERE/'decoder_pdls_draft.py').read_text(encoding='utf-8')
    binding=json.loads((OUT/'source-binding.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==binding['candidate_sha256']
    parent=(HERE/'decoder_pdli.py').read_bytes();assert hashlib.sha256(parent).hexdigest()==BASE_SHA and raw.startswith(parent)
    funcs=lambda s:{n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    a,b=funcs(old),funcs(src)
    for name in ['_pdls_build','_pdls_layer','_pdli_route','_pdli_ab','_pdli_probe','_experts']:
        assert ast.dump(a[name],include_attributes=False)==ast.dump(b[name],include_attributes=False)
    # Kernel resource/query/source constants remain exactly the H200-tested leaf.
    def constants(s):
        return {n.targets[0].id:ast.literal_eval(n.value)for n in ast.parse(s).body
            if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ['_PDLS_HEADERS','_PDLS_QUERY','_PDLS_NOTICE','_PDLS_SOURCE_SHA']}
    assert constants(old)==constants(src)
    loop=next(n for n in ast.walk(b['_pdls_select'])if isinstance(n,ast.For)and isinstance(n.target,ast.Name)and n.target.id=='m')
    assert ast.literal_eval(loop.iter)==(1,)
    sentinel=dict(_pdls_layer=lambda *a:'private',_pdls_parent_kernel_fn=lambda *a:'parent')
    exec(compile(ast.Module(body=[b['kernel_fn']],type_ignores=[]),
                 'PDLS1-public-dispatch-CPU','exec'),sentinel)
    for batch,q,m,want in [(1,1,1,'private'),(1,8,8,'parent'),(8,1,8,'parent'),(1,1,8,'parent')]:
        state={'pdls':{'selected':{(1,True):4,(8,True):4}}}
        assert sentinel['kernel_fn']({'config':dict(batch=batch,q_len=q,tokens=m,window=2048)},state)==want
    gates=[]
    for values,expected in [([8.4,8.4],True),([8.4,8.6],False),([8.6,8.6],False),([9.8,9.8],False)]:
        times=iter(values);mutations=[]
        def timing(fns,dev,reps):assert reps==25;return dict(parent=10.,pdl=next(times))
        env=dict(_okh_budget=lambda *a:True,_pdli_probe=lambda *a,**kw:True,
            _okh_time=timing,_okh_pick=lambda t,base,margin:'pdl'if t['pdl']<margin*t[base]else base)
        exec(UNIT,env)
        result=env['_pdls1_unit_ab']('CPU-mock',{'parent':lambda:None,'pdl':lambda:None},mutations.append,'cpu',0)
        assert result==expected and mutations==[0]
        gates.append(dict(early_us=values,passes=expected))
    result=dict(GPU_jobs=0,candidate_sha256=binding['candidate_sha256'],bytes=len(raw),
        full_scored_parent_prefix=True,H200_leaf_source_and_ABI_unchanged=True,
        MoE_projection_and_whole_layer_guards_unchanged=True,M1_only_selection=True,
        unsupported_public_shape_sentinels=4,unit_gate_cases=gates,
        limitation='CPU integration/acceptance proof; source not run on H200, whole-layer gain and preparation unresolved.')
    (OUT/'cpu-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))


if __name__=='__main__':main()
