"""Compare the new norm call against the pinned public SDK adapter on CPU."""

import ast
import hashlib
import json
from types import SimpleNamespace

import torch

from pdl_norm_sources import HERE, OUT


def extract(path, names, env):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(nodes) == len(names)
    # The selected functions have no decorators; do not evaluate SDK imports.
    assert not any(n.decorator_list for n in nodes)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'CPU-SDK-adapter-audit', 'exec'), env)


def main():
    class Module:
        def __init__(self): self.calls = []
        def run_std(self, *args): self.calls.append(args)
    module = Module(); configs = []
    def get_module(*args):
        configs.append(args)
        return module
    fake_torch = SimpleNamespace(bfloat16=torch.bfloat16, float32=torch.float32,
        Tensor=torch.Tensor, empty_like=torch.empty_like,
        cuda=SimpleNamespace(get_device_capability=lambda d: (9, 0)))
    public = dict(torch=fake_torch, Optional=object, Tuple=object)
    # Preserve the SDK's pure dispatch constants and functions exactly. Types
    # are real CPU tensors; the C++ invocation is recorded, never executed.
    pin = OUT.parent / 'pdl-chain/stock-norm-pinned/norm_residual.py'
    tree = ast.parse(pin.read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign):
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('_'):
                public[n.targets[0].id] = ast.literal_eval(n.value)
    # Postpone SDK type annotations, as the real module does.
    public.update(Optional=__import__('typing').Optional, Tuple=__import__('typing').Tuple,
                  _jit_module=get_module)
    extract(pin, {'_launch_config','_cluster_num_threads','_cluster_size_for_shape',
                  'cluster_size_for_num_tokens','_cluster_launch_config','fused_norm_residual'}, public)
    private = dict(torch=fake_torch, _PDLN_EXT=module)
    extract(HERE / 'pdl_norm_runtime.py', {'_pdln_boundary'}, private)
    pre = torch.ones(8192, dtype=torch.bfloat16)
    alpha = torch.ones(8192); beta = torch.ones(8192)
    p = dict(ready=True, pre=pre, alpha=alpha, beta=beta, post_eps=1e-8, pre_eps=1e-5)
    rows = []
    for m in [16, 32]:
        branch = torch.empty((m, 8192), dtype=torch.bfloat16)
        stream = torch.empty((m, 8192), dtype=torch.float32)
        original_out = public['fused_norm_residual'](branch, stream, p['post_eps'], p['pre_eps'],
            pre_weight=pre, alpha=alpha, beta=beta, want_router=False, pre_norm_bf16=False)
        new_out = private['_pdln_boundary'](stream, branch, {'pdln': p})
        old, new = module.calls[-2:]
        assert len(old) == len(new) == 12
        for i in [0, 1, 4, 5, 6, 7, 8]:
            assert old[i] is new[i]
        assert old[9:] == new[9:] == (1e-8, 1e-5, False)
        assert configs[-1] == (8192, 256, True, 1)
        for i in [2, 3]:
            assert old[i].shape == new[i].shape and old[i].dtype == new[i].dtype
            assert old[i].is_contiguous() and new[i].is_contiguous()
            assert old[i].data_ptr() != new[i].data_ptr()
        assert new_out[0] is new[2] and new_out[1] is new[3] and new_out[2] is None
        rows.append(dict(m=m, SDK_arguments_match=True, pre_norm_bf16=False,
                         router_output=None, config=list(configs[-1])))
    fallback = []
    noncontiguous = torch.empty((8192,32)).t()
    private['_pdln_boundary'](noncontiguous, branch, {'pdln':p,
        'attn_boundary':lambda *args,**kwargs:fallback.append((args,kwargs))})
    assert len(fallback) == 1 and fallback[0][0][0] is noncontiguous
    assert fallback[0][1] == dict(branch_dtype=torch.bfloat16,router_dtype=None)
    p['ready'] = False
    before = len(module.calls)
    try:
        private['_pdln_boundary'](stream, branch, {'pdln': p})
    except RuntimeError:
        pass
    else:
        raise AssertionError('Unadmitted path was callable')
    assert len(module.calls) == before
    order = []
    env = dict(_pdln_core_prepare=lambda w: order.append('core') or w,
               _pdln_thread=SimpleNamespace(join=lambda: order.append('norm_join')),
               _p16_thread=SimpleNamespace(join=lambda: order.append('projection_join')))
    extract(HERE / 'pdl_norm_runtime.py', {'_achita_layer_prepare'}, env)
    marker = object()
    assert env['_achita_layer_prepare'](marker) is marker
    assert order == ['core','norm_join','projection_join']
    record = dict(GPU_jobs=0, SDK_signature_and_dispatch_only=True, cases=rows,
                  unadmitted_call_rejected=True, unsupported_stride_calls_parent=True, preparation_join_order=order,
                  arithmetic_executed=False,
                  limitation='Records the pinned SDK adapter argument list and metadata on CPU. No CUDA/TVM link, graph replay or numerical validation.')
    (OUT / 'interface-audit.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
