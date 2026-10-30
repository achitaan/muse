"""Prevent parent/guard/interface drift in the unsubmitted norm payload."""

import ast
import hashlib
import json

from build_tiny_norm_source import OUT, derive
from build_tiny_norm_whole import GATE_SHA
from tinyroute_source import HERE, ROOT, BASE_SHA
from harness import HARNESS_SRC


def main():
    parent=(ROOT/'best/moe_verified.py').read_bytes()
    gate=(HERE/'moe_tinyroute_bounded.py').read_bytes()
    raw=(HERE/'moe_tiny_norm_whole.py').read_bytes()
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
    assert 'okc_moe_tiny_norm_whole_v1_'in loader
    assert '_tn_ext'not in ast.unparse(last(t,'_tn_launch'))
    report=dict(source_sha256=binding['source_sha256'],parent_prefix_and_all_definitions_exact=True,
        unchanged_gate_launch_mutation_and_prepare_functions=names,shared_harness_exact=True,
        private_CUDA_and_CPP_binding_exact=True,no_TR8_combination=True,no_private_lazy_build_in_timed_launch=True,
        no_loaded_module_attribute_assignment=True,
        GPU_jobs=0,admitted_for_GPU=False,
        limitation='Static source/interface evidence only; no H200 correctness,resource or speed claim.')
    (OUT/'payload-cpu-audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
