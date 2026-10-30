"""Audit exact provenance, dispatcher fallbacks and unchanged full-layer gates."""

import ast
import hashlib
import json
from types import SimpleNamespace

from pdl16_sources import HERE, BASE_SHA
from build_pdl32_handoff import OUT, HOST


def main():
    raw=(HERE/'decoder_pdl32_handoff_draft.py').read_bytes()
    parent=(HERE/'decoder_pdli.py').read_bytes()
    binding=json.loads((OUT/'source-binding.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest()==binding['source_sha256']
    assert hashlib.sha256(parent).hexdigest()==BASE_SHA and raw.startswith(parent)
    assert len(raw)<1_000_000
    app=raw[len(parent):].decode(); nodes=ast.parse(app).body
    cuda=(OUT/'isolated.cu').read_text(encoding='utf-8')
    remaining=cuda
    for name,namespace in [('pdl16/o32-candidate.cu','d32o_private'),('pdlu/u32-candidate.cu','d32u_private')]:
        original=(OUT.parent/name).read_text(encoding='utf-8')
        close='\n} // '+namespace+'\n'
        chunk,remaining=remaining.split(close,1)
        opening='namespace '+namespace+' {\n'
        assert chunk.count(opening)==1 and chunk.replace(opening,'',1)==original
    assert remaining==HOST
    assert '#include <pybind11/pybind11.h>' in (OUT/'isolated.cpp').read_text(encoding='utf-8')
    # No native-MoE/attention/router/norm source or flag rebinding in the suffix.
    assigns=[t.id for n in nodes if isinstance(n,ast.Assign)for t in n.targets if isinstance(t,ast.Name)]
    assert all(not x.startswith(('_native_moe','_native_attn','_fpr')) for x in assigns)
    assert '_d32_data(32,True,dev,gen)' in app and app.count('_pdli_ab(')==1
    assert "(data['k_cache'],data['v_cache'])" in app
    assert "if _pdli_ab('D32_M32_sliding'" in app
    assert app.index("state['d32_candidate'] = candidate")<app.index("state['d32_selected'] = True")
    # Execute only dispatch functions with sentinels, never the module/build thread.
    selected=[n for n in nodes if isinstance(n,ast.FunctionDef)and n.name in ('kernel_fn','_achita_dense_project')]
    bf16=object();env={'torch':SimpleNamespace(bfloat16=bf16),'_D32_EXT':object(),
        '_d32_parent_kernel_fn':lambda data,state:state,
        '_d32_parent_project':lambda x,layer,state,rows:('parent',layer),
        '_d32_leaf':lambda x,layer,state:('private',layer)}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<dispatch-only>','exec'),env)
    parent_state={'d32_selected':True,'d32_candidate':{'marker':'candidate'}}
    configs=[(32,32,1,2048,True),(32,32,1,0,False),(32,4,8,2048,False),
        (32,16,1,2048,False),(31,31,1,2048,False),(33,33,1,2048,False),
        (128,16,8,2048,False),(256,32,8,2048,False),(257,257,1,2048,False)]
    for m,b,q,w,private in configs:
        data={'config':dict(tokens=m,batch=b,q_len=q,window=w)}
        got=env['kernel_fn'](data,parent_state)
        assert got is (parent_state['d32_candidate'] if private else parent_state)
        disabled=dict(parent_state,d32_selected=False)
        assert env['kernel_fn'](data,disabled)is disabled
    class Tensor:
        def __init__(self,shape,dtype=bf16,contiguous=True,ptr=16):
            self.shape=shape;self.dtype=dtype;self.contiguous=contiguous;self.ptr=ptr
        def is_contiguous(self):return self.contiguous
        def data_ptr(self):return self.ptr
    plan={'d32_plan':object()}; projection_cases=0
    for layer,k in [('o',1024),('unpooler',4096)]:
        x=Tensor((32,k)); assert env['_achita_dense_project'](x,layer,plan)==('private',layer)
        for bad in [Tensor((31,k)),Tensor((33,k)),Tensor((32,k),object()),
                    Tensor((32,k),contiguous=False),Tensor((32,k),ptr=17)]:
            assert env['_achita_dense_project'](bad,layer,plan)==('parent',layer)
            projection_cases+=1
        assert env['_achita_dense_project'](x,layer,{})==('parent',layer)
        assert env['_achita_dense_project'](x,layer,plan,1024)==('parent',layer)
        projection_cases+=3
    assert env['_achita_dense_project'](Tensor((32,8192)),'qkv_gate',plan)==('parent','qkv_gate')
    o=json.loads((OUT.parent/'o32-resource-fixed/outcome.json').read_text(encoding='utf-8'))
    u=json.loads((OUT.parent/'handoff16-diag/outcome.json').read_text(encoding='utf-8'))
    assert o['projections'][0]['byte_equal'] and o['projections'][0]['handoff_gate']
    assert any(x['m']==32 and x['layer']=='unpooler'and x['byte_equal']and x['handoff_gate']for x in u['projection_results'])
    report=dict(source_sha256=binding['source_sha256'],checks_passed=True,
        exact_parent_prefix=True,namespace_inverse_exact=True,explicit_binding_header=True,
        dispatcher_cases=2*len(configs),projection_cases=projection_cases+1,
        original_shared_full_layer_and_cache_gate=True,synthetic_unit_qualification_only=True,
        GPU_jobs=0,H200_integration_unverified=True,admitted_for_GPU=False)
    (OUT/'cpu-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
