"""Source/interface checks and native-boundary mocks; never loads CUDA."""
import ast
import hashlib
import json
from types import SimpleNamespace

from attention_nc1_source import OUT, ROOT, PARENT_SHA, derive
from audit_attention_nc1_selector_cpu import Tensor, Recorder, retained_launch_scope


def functions(tree, name):
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]


def main():
    parent, _, cuda, cpp = derive()
    raw = (ROOT / 'experiments/decode_moe/attention_nc1.py').read_bytes()
    binding = json.loads((OUT / 'payload-binding.json').read_text())
    assert hashlib.sha256(parent).hexdigest() == PARENT_SHA
    assert hashlib.sha256(raw).hexdigest() == binding['source_sha256']
    assert raw.startswith(parent.rstrip(b'\n'))
    before, tree = ast.parse(parent), ast.parse(raw)
    for n in before.body:
        if isinstance(n, ast.FunctionDef):
            assert ast.dump(n) == ast.dump(functions(tree, n.name)[0])
    literal = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
               if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in ('CUDA_NC1_SRC', 'CPP_NC1_SRC')}
    assert literal == dict(CUDA_NC1_SRC=cuda, CPP_NC1_SRC=cpp)
    # Execute only whitelisted pure metadata functions/classes, never imports,
    # CUDA strings, module builders or the public kernel implementation.
    names = ('_nc1_family', '_nc1_run_family', '_nc1_resources_ok', '_nc1_reset_workspace')
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.ClassDef) and n.name in ('_NC1Direct', '_NC1Dispatch'))]
    pure = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<nc1-pure-boundary>', 'exec'), pure)
    good = [128,0,0,67584,256,2,89,0,0,58368,128,3]
    resource_cases = [(good, True), (good[:-1], False), (good+[1], False)]
    for index, bad in ((0,0),(0,256),(1,8),(2,16),(3,67585),(4,128),(5,0),
                       (7,8),(8,16),(9,58369),(10,256),(11,0)):
        changed = good.copy(); changed[index] = bad
        resource_cases.append((changed, False))
    for values, expected in resource_cases:
        assert pure['_nc1_resources_ok'](values) is expected
    class Buffer:
        def __init__(self): self.zeros = 0
        def zero_(self): self.zeros += 1
    owned, other, caller = [Buffer() for _ in range(3)], [Buffer()], Buffer()
    pure['_WS'] = {('cuda:0',1,None):tuple(owned),('cuda:1',1,None):tuple(other)}
    pure['_nc1_reset_workspace']('cuda:0')
    assert all(x.zeros == 1 for x in owned) and other[0].zeros == caller.zeros == 0
    scope = retained_launch_scope(parent)
    init = functions(tree, '_nc1_initialize')[0]
    groups_node = next(n.value for n in ast.walk(init) if isinstance(n, ast.Assign)
                       and any(isinstance(t, ast.Name) and t.id == 'groups' for t in n.targets))
    assert not any(isinstance(n, (ast.Call, ast.Name, ast.Attribute)) for n in ast.walk(groups_node))
    groups = eval(compile(ast.Expression(groups_node), '<fixed-fixture-configs>', 'eval'), {'__builtins__':{}})
    fixture_cases = 0
    sample_args = {}
    for expected, configs in groups.items():
        for B, ctx, sw, P, pad, hkv, scale, coherent in configs:
            data = dict(config=dict(batch=B,q_len=1,num_q_heads=8*hkv,num_kv_heads=hkv,
                                    head_dim=64,page_size=P,window=2048 if sw else 0,
                                    rope_theta=10000. if sw else 0.,sm_scale=.125,qk_scale_factor=5.4730064863875),
                        qkv=Tensor(),page_table=Tensor((B,(ctx+P)//P+pad)))
            data.update({k:Tensor() for k in ('seq_lens','cache_loc','k_cache','v_cache','k_scale','v_scale')})
            recorder = Recorder();scope['_launch'](data,_mod=recorder)
            assert pure['_nc1_run_family'](recorder.args) == expected, (expected, configs, recorder.args[13:])
            sample_args[expected] = recorder.args
            fixture_cases += 1
    # Reproduce the diagnosed H200 fixture failure under retained overrides.
    bad = dict(config=dict(batch=1,q_len=1,num_q_heads=16,num_kv_heads=2,head_dim=64,
                           page_size=1,window=0,rope_theta=0.,sm_scale=.125,qk_scale_factor=5.4730064863875),
               qkv=Tensor(),page_table=Tensor((1,64)))
    bad.update({k:Tensor() for k in ('seq_lens','cache_loc','k_cache','v_cache','k_scale','v_scale')})
    record = Recorder();scope['_launch'](bad,_mod=record)
    assert record.args[20] == 252 and record.args[22] == 8 and record.args[27] == 2
    assert pure['_nc1_run_family'](record.args) is None
    class Module:
        def run(self, *args): return 'parent'
        def run_nc1(self, *args): return 'candidate'
    module = Module();routes = 0
    for expected, args in sample_args.items():
        for enabled in ((), (expected,), ('another_family',)):
            state = pure['_NC1Dispatch'](module,module,enabled)
            assert state.run(*args) == ('candidate' if expected in enabled else 'parent')
            routes += 1
        for index, value in ((14,8),(22,2),(25,2),(27,2),(27,1<<16)):
            changed = list(args);changed[index] = value
            state = pure['_NC1Dispatch'](module,module,groups)
            assert state.run(*changed) == 'parent'
            try: pure['_NC1Direct'](module,expected).run(*changed)
            except RuntimeError: pass
            else: raise AssertionError('Strict diagnostic accepted wrong entry')
            routes += 1
    # The setup constructor receives only the device, never a caller's data.
    assert [a.arg for a in init.args.args] == ['dev']
    public = functions(tree, 'kernel_fn')[-1]
    call = next(n for n in ast.walk(public) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == '_nc1_initialize')
    assert len(call.args) == 1 and isinstance(call.args[0], ast.Name) and call.args[0].id == 'dev'
    dispatch = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_NC1Dispatch')
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('item','cpu','synchronize','nc1_resources')
                   for n in ast.walk(dispatch))
    assert '_nc1_build_native' not in ast.unparse(public)
    assert ast.unparse(public).index('is_current_stream_capturing') < ast.unparse(public).index('_nc1_initialize')
    assert 'torch.uint8' in ast.unparse(functions(tree, '_nc1_byte_probe')[0])
    # Exercise the real public wrapper: unsupported/L8 and first-capture calls
    # must retain the parent; eager setup receives the device only, once.
    public_cases = 0
    for supported, length, capturing in ((False,1,False),(True,8,False),
                                         (True,1,True),(True,1,False)):
        calls = []
        def initialize(dev):
            calls.append(dev)
            return 'prepared-dispatch'
        wrapper_scope = dict(_NC1_STATES={}, _supported=lambda cfg: supported,
                             _fits=lambda data: True, _nc1_parent_kernel_fn=lambda data:'parent',
                             _nc1_initialize=initialize,
                             _launch=lambda data, _mod: ('launch', _mod))
        class DeviceContext:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        wrapper_scope['torch'] = SimpleNamespace(cuda=SimpleNamespace(
            is_current_stream_capturing=lambda:capturing, device=lambda dev:DeviceContext()))
        exec(compile(ast.Module(body=[public], type_ignores=[]), '<public-nc1-wrapper>', 'exec'), wrapper_scope)
        data = dict(config=dict(q_len=length),qkv=Tensor())
        got = wrapper_scope['kernel_fn'](data)
        if supported and length == 1 and not capturing:
            assert got == ('launch','prepared-dispatch') and calls == ['cuda:0']
            assert wrapper_scope['kernel_fn'](data) == got and calls == ['cuda:0']
        else:
            assert got == 'parent' and not calls
        public_cases += 1
    # Original module attributes are never patched; only our own adapters have attributes assigned.
    for cls in ('_NC1Direct','_NC1Dispatch'):
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Attribute): assert isinstance(t.value, ast.Name) and t.value.id == 'self'
    report = dict(source_sha256=binding['source_sha256'], parent_prefix_and_functions_exact=True,
                  private_CPP_CUDA_binding_exact=True, resource_guard_cases=len(resource_cases),
                  actual_parent_plan_fixture_cases=fixture_cases, dispatcher_and_rejection_cases=routes,
                  public_wrapper_cases=public_cases,
                  synthetic_scratch_reset_device_scoped=True,
                  actual_retained_tuning_preserved=True, diagnosed_GPU_counterexample_reproduced=True,
                  calibration_receives_device_only=True, raw_byte_snapshots=True,
                  no_private_build_or_resource_query_in_dispatch=True, no_loaded_module_attribute_mutation=True,
                  GPU_calls=0, admitted_for_GPU=False,
                  limitation='Static and mocked integration evidence only;host compile,H200 resources,repeatability and performance pending.')
    (OUT / 'payload-cpu-audit.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
