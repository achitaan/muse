"""SDK ABI, router precision, immutable parent and dispatch screen on CPU."""

import ast
import hashlib
import json
from types import SimpleNamespace

import torch

from audit_pdln_interface_cpu import extract
from build_pdls import HERE, OUT
from pdl16_sources import BASE_SHA


def main():
    class Module:
        def __init__(self): self.calls = []
        def run_std(self, *args): self.calls.append(args)
        def run_std_router(self, *args): self.calls.append(args)
        def run_router(self, *args): self.calls.append(args)
    module = Module(); configs = []
    def get_module(*args):
        configs.append(args)
        return module
    fake = SimpleNamespace(bfloat16=torch.bfloat16, float32=torch.float32,
        Tensor=torch.Tensor, empty_like=torch.empty_like,
        cuda=SimpleNamespace(get_device_capability=lambda d: (9, 0)))
    pin = OUT.parent / 'pdl-chain/stock-norm-pinned/norm_residual.py'
    public = dict(torch=fake, Optional=__import__('typing').Optional,
                  Tuple=__import__('typing').Tuple, _jit_module=get_module)
    for n in ast.parse(pin.read_text()).body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith('_')):
            public[n.targets[0].id] = ast.literal_eval(n.value)
    extract(pin, {'_launch_config', '_cluster_num_threads', '_cluster_size_for_shape',
                  'cluster_size_for_num_tokens', '_cluster_launch_config', 'fused_norm_residual'}, public)
    private = dict(torch=fake, _PDLS_EXT=module)
    extract(HERE / 'pdls_runtime.py', {'_pdls_boundary'}, private)
    pre = torch.ones(8192, dtype=torch.bfloat16)
    alpha, beta = torch.ones(8192), torch.ones(8192)
    p = dict(ready=True, pre=pre, alpha=alpha, beta=beta, post_eps=1e-8, pre_eps=1e-5)
    cases = []
    for m in (1, 8):
        branch = torch.empty((m, 8192), dtype=torch.bfloat16)
        stream = torch.empty((m, 8192), dtype=torch.float32)
        public['fused_norm_residual'](branch, stream, p['post_eps'], p['pre_eps'],
            pre_weight=pre, alpha=alpha, beta=beta, want_router=True, pre_norm_bf16=False)
        outputs = private['_pdls_boundary'](stream, branch, {'pdls': p})
        old, new = module.calls[-2:]
        assert len(old) == len(new) == 12 and configs[-1] == (8192, 256, True, 1)
        for i in (0, 1, 5, 6, 7, 8): assert old[i] is new[i]
        assert old[9:] == new[9:] == (1e-8, 1e-5, False)
        for i in (2, 3, 4):
            assert old[i].shape == new[i].shape and old[i].dtype == new[i].dtype
            assert new[i].is_contiguous() and old[i].data_ptr() != new[i].data_ptr()
        assert outputs[2] is new[4] and outputs[2].dtype == torch.float32
        cases.append(dict(m=m, SDK_arguments_equal=True, FP32_router=True,
                          FP32_pre_norm_input=True, config=list(configs[-1])))
    fallback = []
    parent = lambda *args, **kw: fallback.append((args, kw))
    for m in (3, 16):
        private['_pdls_boundary'](torch.empty((m,8192)),
            torch.empty((m,8192), dtype=torch.bfloat16), {'pdls':p, 'attn_boundary':parent})
    private['_pdls_boundary'](torch.empty((8192,8)).t(), branch,
                             {'pdls':p, 'attn_boundary':parent})
    assert len(fallback) == 3
    assert all(kw == dict(branch_dtype=torch.bfloat16, router_dtype=torch.float32) for args,kw in fallback)
    p['ready'] = False
    try: private['_pdls_boundary'](stream, branch, {'pdls':p})
    except RuntimeError: pass
    else: raise AssertionError('Unadmitted private path callable')
    # Execute only the public routing decision with sentinel layer functions.
    dispatch = dict(_pdls_layer=lambda *a: 'private', _pdls_parent_kernel_fn=lambda *a: 'parent')
    extract(HERE / 'pdls_runtime.py', {'kernel_fn'}, dispatch)
    selected = {(1,True):4, (8,True):4, (8,False):4}
    for batch,qlen,m,window,expected in [(1,1,1,2048,'private'),(1,8,8,0,'private'),
            (8,1,8,2048,'parent'),(1,3,3,2048,'parent'),(16,1,16,2048,'parent'),
            (1,1,1,0,'parent'),(1,8,16,0,'parent')]:
        data={'config':dict(batch=batch,q_len=qlen,tokens=m,window=window)}
        assert dispatch['kernel_fn'](data,{'pdls':{'selected':selected}}) == expected
    parent = (HERE / 'decoder_pdli.py').read_bytes()
    raw = (HERE / 'decoder_pdls_draft.py').read_bytes()
    assert hashlib.sha256(parent).hexdigest() == BASE_SHA and raw.startswith(parent)
    pf = {n.name:n for n in ast.parse(parent).body if isinstance(n,ast.FunctionDef)}
    cf = {n.name:n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)}
    original = ast.get_source_segment(parent.decode(), pf['_pdli_layer'])
    clone = ast.get_source_segment(raw.decode(), cf['_pdls_layer'])
    inverse = clone.replace('def _pdls_layer(', 'def _pdli_layer(', 1).replace(
        "_pdls_boundary(data['stream'], attn_output, state)",
        "state['attn_boundary'](data['stream'], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)")
    assert inverse == original
    for name in ('_experts', '_pdli_route', '_pdli_project', '_pdli_ab', '_pdli_probe'):
        assert ast.dump(pf[name],include_attributes=False) == ast.dump(cf[name],include_attributes=False)
    binding = json.loads((OUT / 'candidate-binding.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == binding['candidate_sha256'] and len(raw)<1_000_000
    result = dict(GPU_jobs=0,candidate_sha256=binding['candidate_sha256'],bytes=len(raw),
        ABI_cases=cases, unsupported_shapes_stride_parent=True,
        unadmitted_call_rejected=True, public_dispatch_sentinels=7,
        exact_parent_prefix=True, layer_clone_inverse_exact=True,
        MoE_projection_router_and_strict_byte_performance_gates_unchanged=True,
        limitation='CPU metadata/source audit only; no CUDA build, bytes, overlap or timing validation.')
    (OUT / 'cpu-audit.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
