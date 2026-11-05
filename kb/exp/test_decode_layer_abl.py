"""CPU-only contract tests; no torch, GPU, or extension import required."""
import ast
import hashlib
import math
from pathlib import Path
import types
import unittest
import contextlib
import io

SOURCE = Path(__file__).with_name('decode_layer_abl.py').read_bytes()
BASE_BYTES = 977700
BASE_SHA = '96757583753818779e0a10f8f0f5ff1ba1d50e9848588521c2e8dca44af038b3'


class Tensor:
    pass


def namespace():
    ns = {'prepare': object(), 'torch': types.SimpleNamespace(Tensor=Tensor)}
    exec(compile(SOURCE[BASE_BYTES:], '<diagnostic-only>', 'exec'), ns)
    return ns


class Contracts(unittest.TestCase):
    def test_exact_production_prefix_and_size(self):
        self.assertEqual(hashlib.sha256(SOURCE[:BASE_BYTES]).hexdigest(), BASE_SHA)
        self.assertLess(len(SOURCE), 1000000)
        ast.parse(SOURCE)
        base = ast.parse(SOURCE[:BASE_BYTES]).body
        complete = ast.parse(SOURCE).body
        self.assertEqual([ast.dump(n) for n in base], [ast.dump(n) for n in complete[:len(base)]])

    def test_score_weights_and_quantiles(self):
        n = namespace()
        gain = n['_da_gain']
        anchor = n['_DA_ANCHOR']
        self.assertEqual(gain({}), 1)
        self.assertEqual(gain({'d1S': -100}), 1)
        self.assertAlmostEqual(gain(anchor), 4/3)
        s = 46*anchor['d1S'] + 16*anchor['d1F']
        self.assertAlmostEqual(gain({'d1S': 10}), (s/(s-46*2.5))**(1/6))
        self.assertAlmostEqual(n['_da_quantile'](list(range(12)), .1), 1.1)
        shapes = n['_DA_SHAPES']
        self.assertEqual([s[0] for s in shapes[:8]],
                         ['d16S','d16F','d1S','d1F','v1S','v1F','d32S','d32F'])
        self.assertEqual(set(s[0] for s in shapes), set(anchor))

    def test_nested_capture_classification_and_output_slots(self):
        n = namespace()
        nodes = set()
        def launch():
            nodes.add(len(nodes)+1)
        class Mod:
            def __getattr__(self, key):
                return lambda *a, **kw: launch()
        class Tri:
            def __getitem__(self, grid):
                return lambda *a, **kw: launch()
        def fn(*a, **kw):
            launch()
            return Tensor()
        for name in ('dense','_decode_attention7','_pool_router_fp32','_gle_frt','_graft_boundary'):
            n[name] = fn
        for name in ('_sl_cluster_module','_gle_ext','_moe'):
            n[name] = lambda *a, **kw: Mod()
        tris = ('_boundary1','_boundary2','_sl_cluster_finish','_sl_tile_route',
                '_sl_graft_route','_graft_route','_route','_route_split','_gle_route')
        for name in tris:
            n[name] = Tri()
        n['_pool_router_fp32'] = lambda *a, **kw: n['_gle_frt'](*a, **kw)
        n['_DA_EXT'] = types.SimpleNamespace(da_nodes=lambda: list(nodes))
        ctx = n['_DA_CTX'] = dict(nodes={}, values={}, depth=0)
        before = {k:n[k] for k in (*tris,'dense','_decode_attention7','_pool_router_fp32',
                                   '_gle_frt','_graft_boundary','_sl_cluster_module','_gle_ext','_moe')}
        hooks = n['_da_hooks']()
        try:
            launch()  # qkv torch.mm, not intercepted
            qkv = Tensor()
            n['_decode_attention7'](None,None,qkv)
            n['dense'](None,None,16,None,'o')
            tensors = [Tensor() for _ in range(18)]
            n['_graft_boundary'](*tensors[:5],False,*tensors[6:8])
            n['_pool_router_fp32']()
            n['_sl_tile_route'][(1,2)](*tensors)
            n['_moe']().okc_run(*tensors)
            n['dense'](None,None,16,None,'un')
            n['_sl_cluster_finish'][(1,2)](*tensors)
            self.assertEqual(set(ctx['nodes']),set(n['_DA_COMPONENTS']))
            self.assertEqual(set.union(*ctx['nodes'].values()),nodes)
            self.assertEqual(sum(map(len,ctx['nodes'].values())),len(nodes))
            self.assertEqual(ctx['values']['qkv_proj'],[qkv])
            self.assertEqual(ctx['values']['o_boundary'],tensors[6:8])
            self.assertEqual(ctx['values']['moe'],[tensors[5]])
            self.assertEqual(ctx['values']['route'],[tensors[i] for i in (5,6,7,9)])
            self.assertEqual(ctx['values']['finish'],tensors[7:9])
            self.assertEqual(len(ctx['nodes']['pooler_router']),1)  # nested hook suppressed
        finally:
            n.update(hooks)
            n['_DA_CTX'] = None
        for key, value in before.items():
            self.assertIs(n[key],value)

    def test_abba_reset_flush_order_rounds_and_null(self):
        n = namespace()
        trace,clock = [],[0.0]
        class Event:
            def __init__(self, **kw):
                pass
            def record(self):
                trace.append('event')
                self.t = clock[0]
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return (other.t-self.t)/1000
        def replay(handle):
            trace.append(handle)
            clock[0] += 100 if handle==1 else 80
        n['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event),
                                          sum=lambda *a,**kw: trace.append('flush'))
        n['_DA_EXT'] = types.SimpleNamespace(da_replay=replay)
        reset = lambda: trace.append('reset')
        r = n['_da_pair'](1,2,reset,None,None,math.inf)
        self.assertEqual(r['n'],12)
        self.assertAlmostEqual(r['saved'],20)
        self.assertAlmostEqual(r['p10'],20)
        self.assertEqual((r['base'],r['arm']),(100,80))
        for i in range(0,len(trace),5):
            self.assertEqual(trace[i:i+3],['reset','flush','event'])
            self.assertEqual(trace[i+4],'event')
        launches = [x for x in trace if isinstance(x,int)]
        self.assertEqual(launches, [1,2,2,1,2,1,1,2]*7)
        self.assertEqual(n['_da_pair'](1,1,reset,None,None,math.inf)['saved'],0)
        with self.assertRaises(TimeoutError):
            n['_da_pair'](1,2,reset,None,None,-1)

    def test_exception_releases_capture_depth(self):
        n = namespace()
        n['_DA_CTX'] = dict(nodes={},values={},depth=0)
        n['_DA_EXT'] = types.SimpleNamespace(da_nodes=lambda: [])
        def fail():
            raise ValueError('test')
        with self.assertRaises(ValueError):
            n['_da_call']('attention',fail,(),{})
        self.assertEqual(n['_DA_CTX']['depth'],0)

    def test_validate_rejects_timed_route_drift(self):
        n = namespace()
        class Event:
            def __init__(self, **kw): pass
            def record(self): pass
            def synchronize(self): pass
            def elapsed_time(self, other): return .1
        n['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event),sum=lambda *a,**k: None)
        n['_DA_EXT'] = types.SimpleNamespace(da_replay=lambda handle: None)
        calls = []
        def validate():
            calls.append(1)
            if len(calls)==11:
                raise RuntimeError('route drift')
        with self.assertRaisesRegex(RuntimeError,'route drift'):
            n['_da_pair'](1,2,lambda: None,None,None,math.inf,validate=validate)
        self.assertEqual(len(calls),11)

    def test_output_indices_against_base_function_signatures(self):
        funcs = {f.name:f for f in ast.parse(SOURCE[:BASE_BYTES]).body if isinstance(f,ast.FunctionDef)}
        expectations = {
            '_boundary1': {7:'S1OUT',8:'HOUT',9:'ROUTER_OUT'},
            '_boundary2': {9:'SOUT',10:'HOUT'},
            '_sl_cluster_finish': {7:'SOUT',8:'HOUT'},
            '_route': {5:'IDS',6:'WTS',7:'XOUT',9:'LIDS'},
            '_route_split': {4:'IDS',5:'WTS',6:'XOUT',8:'LIDS'},
            '_sl_tile_route': {5:'IDS',6:'WTS',7:'XOUT',9:'LIDS'},
            '_sl_graft_route': {5:'IDS',6:'WTS',7:'XOUT',9:'LIDS'},
            '_graft_route': {5:'IDS',6:'WTS',7:'XOUT',9:'LIDS'},
            '_gle_route': {6:'IDS',7:'WTS',8:'XOUT',10:'LIDS',11:'RT',17:'S1'},
            '_graft_boundary': {5:'post',6:'s_out',7:'h_out'},
        }
        for name, indices in expectations.items():
            args = funcs[name].args.args
            for i, expected in indices.items():
                self.assertEqual(args[i].arg,expected,(name,i))

    def test_cuda_dependency_preservation_contract(self):
        src = namespace()['_DA_CUDA']
        for text in ('cudaGraphClone','cudaGraphNodeFindInClone','cudaGraphKernelNodeSetParams',
                     'cudaGraphGetEdges_v2','cudaStreamGetCaptureInfo_v3',
                     'griddepcontrol.wait','griddepcontrol.launch_dependents'):
            self.assertIn(text,src)
        for text in ('cudaGraphRemoveDependencies','cudaGraphDestroyNode','cudaGraphAddDependencies'):
            self.assertNotIn(text,src)

    def test_prepare_fallback_restores_production_globals(self):
        n = namespace()
        class Thread:
            def __init__(self, **kw): pass
            def start(self): pass
            def join(self, **kw): pass
        n['threading'] = types.SimpleNamespace(Thread=Thread)
        n['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
        globals_to_restore = ('_WS7','_CL7','_GUARDS','_GUARDED_IDS','bfp_SK2_PDL','_PF7','_GLE_W_CUR')
        saved = {k:object() for k in globals_to_restore}
        n.update(saved)
        n['_AF_KINDS'] = original_kinds = ('test-kind',)
        st = {'af_pick':{},'ma_ext':{}}
        def base(weights):
            self.assertEqual(n['_AF_KINDS'],())
            return st
        n['_DA_PARENT_PREPARE'] = base
        n['_GLE_EXT'] = n['_DE_EXT'] = None
        n['bfp__get_ext'] = lambda: types.SimpleNamespace(sk2_set_pdl=lambda flag: None)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            out = n['prepare']({'qkv_gate':types.SimpleNamespace(device='cuda')})
        self.assertIs(out,st)
        self.assertIs(n['_AF_KINDS'],original_kinds)
        for key,value in saved.items():
            self.assertIs(n[key],value)
        self.assertIn('DEC_ABL_ABORT',output.getvalue())
        self.assertIn('shapes=0/12',output.getvalue())


if __name__ == '__main__':
    unittest.main()
