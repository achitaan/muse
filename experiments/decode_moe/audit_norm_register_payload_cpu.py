# Helper-local alternative adapted from frozen grouped-norm tooling.
"""Prevent parent/guard/interface drift in the unsubmitted norm payload."""

import ast
import contextlib
import hashlib
import io
import json
from types import SimpleNamespace

from norm_register_source import OUT, derive
from build_norm_register_payload import GATE_SHA
from tinyroute_source import HERE, ROOT, BASE_SHA
from harness import HARNESS_SRC


def main():
    parent=(ROOT/'best/moe_verified.py').read_bytes()
    gate=(HERE/'moe_tinyroute_bounded.py').read_bytes()
    raw=(HERE/'moe_norm_register.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest()==BASE_SHA
    assert hashlib.sha256(gate).hexdigest()==GATE_SHA
    binding=json.loads((OUT/'payload-binding.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(raw).hexdigest()==binding['source_sha256']
    assert raw.startswith(parent.rstrip(b'\n'))and len(raw)<1_000_000
    pt,gt,t=ast.parse(parent),ast.parse(gate),ast.parse(raw)
    assert [ast.dump(n)for n in t.body[:len(pt.body)]]==[ast.dump(n)for n in pt.body]
    def last(tree,name):
        return [n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name][-1]
    def renamed(node):
        return ast.dump(ast.parse(ast.unparse(node).replace('_trb_','_tnb_').replace('_tr_','_tn_').replace('tr8','tnorm').replace('TR8','TNORM')))
    names=['_tr_launch','_tr_probe_data','_tr_mutate','_trb_snapshot','_trb_relative','_trb_gate','_tr_probe','prepare','kernel_fn']
    for name in names:
        new=name.replace('_trb_','_tnb_').replace('_tr_','_tn_')
        assert renamed(last(gt,name))==ast.dump(ast.parse(ast.unparse(last(t,new)))),name
    for name in ('_okh_time','_okh_pick','_okh_budget','_okh_bg'):
        assert ast.dump(last(t,name))==ast.dump(last(ast.parse(HARNESS_SRC),name))
    values={n.targets[0].id:ast.literal_eval(n.value)for n in t.body
            if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)
            and n.targets[0].id in ('CUDA_SRC','CUDA_TN_SRC','CPP_TN_SRC','_TN_EXTS')}
    original,cuda,cpp,_=derive()
    assert values['CUDA_SRC']==original and values['CUDA_TN_SRC']==cuda and values['CPP_TN_SRC']==cpp
    assert values['_TN_EXTS']=={}and '-DOKC_ROUTE_WARP=1'not in raw.decode()
    loader=ast.unparse(last(t,'_tn_ext'))
    assert '_TNormExtension'in loader and '_TN_EXTS'in loader
    assert not any(isinstance(target,ast.Attribute)for n in ast.walk(last(t,'_tn_ext'))if isinstance(n,ast.Assign)for target in n.targets)
    adapter=next(n for n in t.body if isinstance(n,ast.ClassDef)and n.name=='_TNormExtension')
    for n in ast.walk(adapter):
        if isinstance(n,ast.Assign):
            assert all(isinstance(target,ast.Attribute)and isinstance(target.value,ast.Name)and target.value.id=='self'for target in n.targets)
    assert 'CUDA_TN_SRC'in loader and 'CPP_TN_SRC'in loader
    assert 'okc_moe_norm_register_v1_resources_'in loader
    assert loader.index('okc_tnorm_resources()')<loader.index('_TNormExtension(')
    assert 'metadata[3] >= 1'in loader and 'metadata[4:] == [214784, 288]'in loader
    assert '_tn_ext'not in ast.unparse(last(t,'_tn_launch'))
    # Execute the real loader guard with a simulated native API boundary.
    # The impossible-resource cases must fail before a new module is returned;
    # no mock kernel call is made. This tests the guard,not GPU attributes.
    guard_cases=0
    for metadata,valid in [([146,0,0,1,214784,288],True),
                           ([143,0,0,1,214784,288],True),
                           ([146,8,0,1,214784,288],False),
                           ([146,0,64,1,214784,288],False),
                           ([146,0,0,0,214784,288],False),
                           ([146,0,0,1,214912,288],False),
                           ([146,0,0,1,214784,320],False)]:
        calls={'query':0,'launch':0,'set_pdl':[],'build':0}
        def query():
            calls['query']+=1
            return metadata
        def run(*args):calls['launch']+=1
        def build(**kwargs):
            calls['build']+=1
            assert 'okc_tnorm_resources'in kwargs['functions']
            return SimpleNamespace(okc_tnorm_run=run,okc_tnorm_resources=query,
                                   okc_tnorm_set_pdl=lambda on:calls['set_pdl'].append(on))
        scope=dict(_TN_EXTS={},CUDA_TN_SRC=cuda,CPP_TN_SRC=cpp,DQ7=True,PDL=False,MAIN_NS=3,
                   torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda:(9,0))),load_inline=build)
        unit=ast.Module(body=[adapter,last(t,'_tn_ext')],type_ignores=[])
        exec(compile(unit,'<actual-norm-loader-guard>','exec'),scope)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                result=scope['_tn_ext'](('-DOKC_TINY=1','-DOKC_WS=1'))
            except RuntimeError:
                assert not valid
            else:
                assert valid
                assert scope['_tn_ext'](('-DOKC_TINY=1','-DOKC_WS=1'))is result
        assert calls['query']==calls['build']==1 and calls['launch']==0
        assert calls['set_pdl']==([0]if valid else [])
        guard_cases+=1
    report=dict(source_sha256=binding['source_sha256'],parent_prefix_and_all_definitions_exact=True,
        unchanged_gate_launch_mutation_and_prepare_functions=names,shared_harness_exact=True,
        private_CUDA_and_CPP_binding_exact=True,no_TR8_combination=True,no_private_lazy_build_in_timed_launch=True,
        no_loaded_module_attribute_assignment=True,
        GPU_jobs=0,admitted_for_GPU=False,
        limitation='Static source/interface evidence only; no H200 correctness,resource or speed claim.')
    report['actual_resource_query_before_private_module_return']=True
    report['mock_native_resource_guard_cases']=guard_cases
    (OUT/'payload-cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
