"""CPU-only admission/source/layout tests; optional torch/reference math tests.

python3 kb/exp/test_decode_combo_cpu.py -v
DEC_COMBO_TASK_REF=/path/to/public/task python_with_torch ... -v
No import of the submission, CUDA compiler, or GPU is needed.
"""
import ast
from array import array
import importlib.util
import itertools
import math
import os
from pathlib import Path
import statistics
import sys
import types
import unittest

HERE = Path(__file__).resolve().parent
PATH = HERE / 'decode_combo.py'
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def namespace(names, **initial):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = {'_kp_math': math, '_kp_stats': statistics}
    ns.update(initial)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), 'exec'), ns)
    return ns


def constant(tree, name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


class GateTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace({'_dc_accept', '_dc_pick', '_dc_arms', '_dc_validate', '_dc_within'})

    def record(self, **kw):
        return dict(dict(rounds=16, abba=8, baab=8, ids_equal=True, within_tol=True,
                         ratio=.994, paired=.994, null=1., null_paired=1.), **kw)

    def test_three_pass_median_and_each_pass_boundaries(self):
        r = self.record
        accept = self.ns['_dc_accept']
        self.assertTrue(accept([r(paired=.991), r(paired=.995), r(paired=.999999)]))
        self.assertFalse(accept([r(paired=.994), r(paired=.994), r(paired=1.)]))
        self.assertFalse(accept([r(paired=.991), r(paired=.995001), r(paired=.996)]))
        # Raw medians remain diagnostic, paired ratios determine admission.
        self.assertTrue(accept([r(ratio=1.001)] * 3))
        self.assertTrue(accept([r(null=.995, null_paired=1.005)] * 3))

    def test_every_failure_and_nonfinite_is_closed(self):
        good, accept = self.record(), self.ns['_dc_accept']
        for count in (0, 1, 2, 4):
            self.assertFalse(accept([good] * count))
        for k, v in (('rounds', 15), ('abba', 7), ('baab', 7), ('ids_equal', False),
                     ('within_tol', False), ('null', .994999), ('null_paired', 1.005001),
                     ('paired', 0.), ('paired', -1.)):
            for i in range(3):
                rs = [good] * 3
                rs[i] = dict(good, **{k: v})
                self.assertFalse(accept(rs), (k, v, i))
        for k in ('ratio', 'paired', 'null', 'null_paired'):
            for v in (float('nan'), float('inf'), -float('inf')):
                self.assertFalse(accept([good, good, dict(good, **{k: v})]))

    def test_arms_and_best_admitted_only(self):
        self.assertEqual(self.ns['_dc_arms']((1, 1, True)), ('KV',))
        self.assertEqual(self.ns['_dc_arms']((1, 8, False)), ('POOL', 'KV+POOL'))
        r, pick = self.record, self.ns['_dc_pick']
        self.assertEqual(pick({'POOL': [r()] * 2}), 'BASE')
        self.assertEqual(pick({'POOL': [r()] * 3, 'KV+POOL': [r(paired=.991)] * 3}), 'KV+POOL')
        self.assertEqual(pick({'POOL': [r()] * 3, 'KV+POOL': [r(paired=1.)] * 3}), 'POOL')

    def test_reference_envelope_is_not_bit_equality_or_tolerance_floor(self):
        ns = self.ns
        ns['torch'] = types.SimpleNamespace(equal=lambda a, b: a == b)
        ns['_dc_metrics'] = lambda a, b: (abs(a - b), abs(a - b))
        refs = [({'hidden': .01, 'cache': 0.}, (1, 2)), ({'hidden': .02, 'cache': 0.}, (1, 2))]
        ref, tol = {'hidden': 0., 'cache': 0.}, {'hidden': .03, 'cache': .08}
        check = lambda cand, untouched=True: ns['_dc_validate'](refs, [cand], ref, tol, untouched)
        self.assertEqual(check(({'hidden': .019, 'cache': 0.}, (1, 2))), (True, True))
        # A more accurate result is allowed even if no incumbent equals it.
        self.assertEqual(check(({'hidden': .005, 'cache': 0.}, (1, 2))), (True, True))
        self.assertEqual(check(({'hidden': .020001, 'cache': 0.}, (1, 2))), (True, False))
        self.assertEqual(check(({'hidden': .01, 'cache': 1e-12}, (1, 2))), (True, False))
        self.assertEqual(check(({'hidden': .01, 'cache': 0.}, (1, 3))), (False, True))
        self.assertEqual(check(({'hidden': .01, 'cache': 0.}, (1, 2)), False), (True, False))
        self.assertFalse(check(({'hidden': float('nan'), 'cache': 0.}, (1, 2)))[1])
        refs[1][0]['hidden'] = .04
        self.assertFalse(check(({'hidden': .01, 'cache': 0.}, (1, 2)))[1])

    def test_dispatch_and_threadlocal_restoration(self):
        ns = namespace({'kernel_fn', '_kp_key'})
        ns['_KP_TLS'] = tls = types.SimpleNamespace(ext='saved', observe='saved_observer')
        ns['_kp_compatible'] = lambda data: data.get('compatible', True)
        calls = []
        def base(data, state):
            calls.append((tls.ext, tls.observe, state['_dp_active']))
            self.assertNotIn('_dp_active', st)
            if data.get('raise'):
                raise ValueError('test')
            return 'ok'
        ns['_KP_KERNEL'] = base
        data = {'config': dict(batch=1, q_len=8, window=2048)}
        st = {'combo_pick': {}, 'kvpre_ext': 'kv', '_kp_observe': 'observe'}
        for arm, expected in (('BASE', (None, 'observe', False)), ('KV', ('kv', 'observe', False)),
                              ('POOL', (None, 'observe', True)), ('KV+POOL', ('kv', 'observe', True))):
            st['combo_pick'][(1, 8, True)] = arm
            self.assertEqual(ns['kernel_fn'](data, st), 'ok')
            self.assertEqual(calls[-1], expected)
            self.assertEqual((tls.ext, tls.observe), ('saved', 'saved_observer'))
        ns['kernel_fn'](dict(data, compatible=False), st)
        self.assertEqual(calls[-1], (None, 'observe', False))
        with self.assertRaises(ValueError):
            ns['kernel_fn'](dict(data, **{'raise': True}), st)
        self.assertEqual((tls.ext, tls.observe), ('saved', 'saved_observer'))


def strides(shape):
    out, v = [], 1
    for n in reversed(shape):
        out.append(v)
        v *= n
    return tuple(reversed(out))


class Tensor:
    def __init__(self, data, shape, stride=None):
        self.data, self.shape = data, shape
        self.stride = strides(shape) if stride is None else stride

    def view(self, *shape):
        assert math.prod(shape) == len(self.data) and self.stride == strides(self.shape)
        return Tensor(self.data, shape)

    def permute(self, *axes):
        return Tensor(self.data, tuple(self.shape[a] for a in axes), tuple(self.stride[a] for a in axes))

    def contiguous(self):
        data = array('I', (self.data[sum(i * s for i, s in zip(ix, self.stride))]
                          for ix in itertools.product(*(range(n) for n in self.shape))))
        return Tensor(data, self.shape)


class SourceTests(unittest.TestCase):
    def test_both_kernel_transforms_match_measured_variants(self):
        ns = namespace({'_kp_source', '_kp_replace'})
        kv_tree = ast.parse((HERE / 'decode_kvpre.py').read_text())
        kvns = {}
        exec(compile(ast.Module(body=[n for n in kv_tree.body if isinstance(n, ast.FunctionDef)
                                     and n.name in ('_kp_source', '_kp_replace')], type_ignores=[]), 'kv', 'exec'), kvns)
        src = constant(TREE, 'ATT7_CUDA_SRC')
        self.assertEqual(src, constant(ast.parse((HERE.parent / 'v1/decode_v1.py').read_text()), 'ATT7_CUDA_SRC'))
        self.assertEqual(ns['_kp_source'](src), kvns['_kp_source'](src))
        names = {'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC', 'bfp_SK2_SRC',
                 'bfp_SK2_CONFIGS', 'bfp_SK2_PICK', 'bfp_SK2_K', 'bfp__sk2_geom', '_gle_source', '_dp_source'}
        ns = namespace(names)
        old, new = ns['_gle_source'](), ns['_dp_source']()
        old_copy = '''          for (int r = 0; r < 8; ++r)
            bulk_g2s(s0 + r * 2048, reinterpret_cast<const uint8_t*>(ep.w32 + (size_t)(rt * 8 + r) * 8192 + kb), 2048, full0 + 8 * slot);'''
        new_copy = '''          // [router stage j][128 CTAs][8 rows][512 FP32 columns].
          // One 16-KiB transaction, identical destination bytes and consumers.
          const size_t wp = ((size_t)((kb & 2047) >> 9) * 128 + cta) * 4096;
          bulk_g2s(s0, reinterpret_cast<const uint8_t*>(ep.w32 + wp), 16384, full0 + 8 * slot);'''
        self.assertEqual(old.count(old_copy), 1)
        self.assertEqual(new, old.replace(old_copy, new_copy))

    def test_every_incumbent_ast_node_is_preserved(self):
        original = (HERE / 'decode_kvpre.py').read_text()
        prefix = original[:original.index('# ---------------------------------------------------------------- MFIN: paired in-prepare')]
        nodes = ast.parse(prefix).body
        self.assertEqual([ast.dump(n) for n in TREE.body[:len(nodes)]], [ast.dump(n) for n in nodes])

    def test_pack_all_payloads_and_every_copy_range(self):
        ns = namespace({'_dp_pack', '_dp_unpack'})
        source = Tensor(array('I', range(256 * 8192)), (256, 8192))
        packed = ns['_dp_pack'](source)
        for j in range(4):
            for cta in range(128):
                rt, rq = divmod(cta, 4)
                for row in range(8):
                    at = (j * 128 + cta) * 4096 + row * 512
                    old = (rt * 8 + row) * 8192 + rq * 2048 + j * 512
                    self.assertEqual(packed.data[at:at + 512], source.data[old:old + 512])
        self.assertEqual(ns['_dp_unpack'](packed).data, source.data)

    def test_build_flags_size_forbidden_and_reproducible(self):
        ns = namespace({'_kvpre_load'})
        builds = 0
        for n in ast.walk(TREE):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.assertNotIn(n.func.id, ('load_inline', '_tvm_load_inline'))
                builds += n.func.id == '_kvpre_load'
        self.assertEqual(builds, 14)
        raw = ['-O3', '-std=c++17', '-gencode=arch=compute_90,code=sm_90', '--fmad=false']
        kw = ns['_kvpre_load'](lambda **kw: kw, extra_cflags=['-std=c++17'], extra_cuda_cflags=raw)
        self.assertEqual(kw['extra_cflags'], ['-std=c++20'])
        self.assertEqual(kw['extra_cuda_cflags'], ['-O3', '--fmad=false', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'])
        self.assertIn('-std=c++17', raw)
        self.assertLess(len(SOURCE.encode()), 1_000_000)
        for forbidden in ('subprocess', 'Popen', 'multiprocessing', 'os.system',
                          'cudaGraphInstantiate_v2', 'cudaGraphInstantiate_v3', 'DEC_KVPRE_ATTN', '_mf_ab'):
            self.assertNotIn(forbidden, SOURCE)
        spec = importlib.util.spec_from_file_location('builder', HERE / 'build_decode_combo.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.assemble(), SOURCE)

    def test_read512_outside_timing_and_input_distribution(self):
        ns = namespace({'_kp_block'})
        log = []
        class Event:
            def record(self): log.append('event')
            def synchronize(self): log.append('sync')
            def elapsed_time(self, other): return .1
        class Graph:
            def __init__(self, i): self.i = i
            def replay(self): log.append(self.i)
        ns.update(_kp_reset=lambda *a: log.append('reset'),
                  torch=types.SimpleNamespace(sum=lambda *a, **kw: log.append('read')))
        for order in ((0, 1, 1, 0), (1, 0, 0, 1)):
            log.clear()
            ns['_kp_block'](Graph(0), Graph(1), order, None, None, 'flush', 'sink', (Event(), Event()))
            expected = []
            for i in order:
                expected += ['reset', 'read', 'event', i, 'event', 'sync']
            self.assertEqual(log, expected)
        self.assertIn('seq = 80000', SOURCE)
        self.assertIn('basis = torch.Generator().manual_seed(5354)', SOURCE)
        self.assertIn('(512 << 20) // 4', SOURCE)
        self.assertIn('_DC_ROUNDS = 16', SOURCE)


TORCH = importlib.util.find_spec('torch') is not None


@unittest.skipUnless(TORCH, 'torch not installed; stdlib source/gate/layout checks still run')
class NumericTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        names = {n.name for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name.startswith('_dc_ref')}
        names |= {'_dc_metrics', '_dc_budget', '_dc_within', '_dc_validate'}
        cls.ns = namespace(names, torch=torch)
        cls.ns.update(torch=torch, F=torch.nn.functional, _kp_time=types.SimpleNamespace(monotonic=lambda: 0))
        torch.set_num_threads(4)

    def test_metrics_match_exact_judge_global_and_row_contract(self):
        t, ns = self.torch, self.ns
        b = t.tensor([[1., 2., 3.], [.0001, -.0001, 0.]])
        a = b.clone()
        a[0, 0] += .01
        a[1, 1] += .001
        whole, row = ns['_dc_metrics'](a, b)
        self.assertAlmostEqual(whole, float((a - b).norm() / b.norm()))
        rn = b.norm(dim=-1)
        self.assertAlmostEqual(row, float(((a - b).norm(dim=-1) / t.maximum(rn, .1 * rn.mean())).max()))
        self.assertEqual(ns['_dc_metrics'](a, b.double()), (math.inf, math.inf))
        a[0, 0] = float('nan')
        self.assertEqual(ns['_dc_metrics'](a, b), (math.inf, math.inf))

    def test_fp4_decode_exhaustive_nibbles_and_scale_blocks(self):
        t, ns = self.torch, self.ns
        packed = t.arange(256, dtype=t.uint8).view(8, 32)
        sf = t.tensor([[.5, 1., 2., 4.]] * 8).to(t.float8_e4m3fn)
        out = ns['_dc_ref_dequant'](packed, sf, t.tensor(1.5))
        grid = (0., .5, 1., 1.5, 2., 3., 4., 6.)
        want = []
        for row in packed.tolist():
            vals = []
            for byte in row:
                for code in (byte & 15, byte >> 4):
                    vals.append((-1 if code & 8 else 1) * grid[code & 7])
            want.append([v * (.5, 1., 2., 4.)[i // 16] * 1.5 for i, v in enumerate(vals)])
        self.assertTrue(t.equal(out, t.tensor(want)))

    def test_whole_pass_protocol_with_cpu_graph_stubs(self):
        t = self.torch
        ns = namespace({'_dc_pass', '_dc_arms', '_kp_summarize', '_dc_budget'})
        log, created = [], []
        class Graph:
            def __init__(self, name): self.name = name
            def replay(self): log.append(('sample', self.name))
        class Event:
            def __init__(self, **kw): pass
        fake_cuda = types.SimpleNamespace(Event=Event, synchronize=lambda: None)
        fake_torch = types.SimpleNamespace(cuda=fake_cuda, sum=lambda *a, **kw: None, bincount=t.bincount, uint8=t.uint8)
        data = dict(config={'q_len': 8}, k_cache=t.zeros(1, dtype=t.uint8),
                    v_cache=t.zeros(1, dtype=t.uint8), local_start=t.zeros(1, dtype=t.int32))
        ids = t.arange(8, dtype=t.int32).view(1, 8)
        def layer(data, st, key, name):
            created.append(name)
            return dict(graph=Graph(name), out=(None, None, ids))
        def block(a, b, order, *args):
            log.append(('block', b.name, order))
            ratio = 1. if b.name == 'BASE' else .991 if b.name == 'POOL' else .993
            return ([100., 100.], [100. * ratio, 100. * ratio])
        def validate(refs, samples, reference, tolerances, untouched):
            self.assertEqual(len(refs), 21)
            self.assertEqual(len(samples), 4)
            self.assertTrue(untouched)
            return True, True
        ns.update(torch=fake_torch, _kp_time=types.SimpleNamespace(monotonic=lambda: 0), _DC_ROUNDS=16,
                  _kp_inputs=lambda *a: data, _dc_layer=layer, _kp_slots=lambda d: (),
                  _kp_reset=lambda *a: None, _dc_ref_front=lambda *a: {'biased': t.zeros(1, 256)},
                  _dc_reference=lambda *a: ({}, {}, None), _dc_ref_routing=lambda *a: True,
                  _dc_snapshot=lambda *a: ({}, ids), _kp_untouched=lambda *a: True,
                  _kp_block=block, _dc_validate=validate, _check_guards=lambda: None)
        for turn in range(3):
            log.clear()
            created.clear()
            results = ns['_dc_pass']({}, {}, (1, 8, True), turn, None, None, 10)
            self.assertEqual(created, ['BASE', 'BASE', 'POOL', 'KV+POOL'])
            self.assertEqual(set(results), {'POOL', 'KV+POOL'})
            for name, record in results.items():
                self.assertAlmostEqual(record['paired'], .991 if name == 'POOL' else .993)
                self.assertEqual((record['rounds'], record['abba'], record['baab']), (16, 8, 8))
                self.assertEqual((record['null'], record['null_paired']), (1., 1.))
                self.assertTrue(record['within_tol'] and record['ids_equal'])
            blocks = [entry for entry in log if entry[0] == 'block']
            self.assertEqual(len(blocks), 48)
            for name in ('BASE', 'POOL', 'KV+POOL'):
                orders = [order for _, arm, order in blocks if arm == name]
                self.assertEqual(orders.count((0, 1, 1, 0)), 8)
                self.assertEqual(orders.count((1, 0, 0, 1)), 8)

    def test_reference_math_against_public_task_when_available(self):
        task = os.environ.get('DEC_COMBO_TASK_REF')
        if not task:
            self.skipTest('set DEC_COMBO_TASK_REF for full task/reference comparison')
        sys.path.insert(0, task)
        try:
            spec = importlib.util.spec_from_file_location('combo_public_reference', Path(task) / 'reference.py')
            official = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(official)
        finally:
            sys.path.pop(0)
        t, ns = self.torch, self.ns
        gen = t.Generator().manual_seed(41234)
        def matrix(n, k):
            return (t.randn((n, k), generator=gen) / math.sqrt(k)).to(t.bfloat16)
        def vector(n):
            return (t.randn(n, generator=gen) * .1).to(t.bfloat16)
        weights = dict(qkv_gate=matrix(2304, 8192), o=matrix(8192, 1024),
                       pooler=matrix(4096, 8192), router=matrix(256, 8192).float(),
                       router_bias=t.randn(256, generator=gen) * .05,
                       unpooler=matrix(1024, 4096),
                       attn_residual_gate=vector(8192), ffn_residual_gate=vector(8192),
                       pre_ffn_norm=vector(8192), pre_expert_norm=vector(4096),
                       post_expert_norm=vector(4096), post_ffn_norm=vector(8192), next_pre_attn_norm=vector(8192))
        # One physical expert, and one local slot/token, avoids GBs of test
        # allocations while exercising the full-sized 4096-wide reference.
        for name, rows in (('w13', 8192), ('w2', 4096)):
            weights[name] = t.randint(0, 256, (1, rows, 2048), generator=gen, dtype=t.uint8)
            weights[name + '_sf'] = t.ones((1, rows, 256)).to(t.float8_e4m3fn)
            weights[name + '_gs'] = t.tensor([.003])
        for length, sliding in ((1, False), (8, True)):
            data = official.generate_input(1, 5, 'sliding' if sliding else 'full', 812 + length,
                                           q_len=length, page=1, device='cpu')
            for dtype in (t.bfloat16, t.float8_e4m3fn):
                expected = official.ref_front(data, weights, dtype)
                got = ns['_dc_ref_front'](data, weights, dtype)
                for key in expected:
                    if isinstance(expected[key], t.Tensor):
                        # Small CPU GEMM/einsum batched-vs-unbatched reduction
                        # differences are allowed only at FP32 roundoff here.
                        self.assertTrue(t.allclose(got[key].float(), expected[key].float(), atol=2e-6, rtol=2e-6), (length, dtype, key))
                    else:
                        self.assertEqual(got[key], expected[key])
                ids = t.arange(8).view(1, 8).expand(length, -1).to(t.int32) + 32
                ids[:, 0] = 0
                actual = ns['_dc_ref_back'](got, weights, ids, math.inf)
                want = official.ref_back(expected, weights, ids)
                for a, b in zip(actual, want):
                    self.assertTrue(t.equal(a, b), (length, dtype, (a.float() - b.float()).abs().max()))
                self.assertEqual(ns['_dc_ref_routing'](ids, got['biased'], 0),
                                 official.check_routing(ids, got['biased'], 0) is None)
        for _ in range(20):
            biased = t.randn((8, 256), generator=gen)
            ids = biased.topk(8, dim=-1).indices.to(t.int32)
            self.assertTrue(ns['_dc_ref_routing'](ids, biased, 0))
            ids[0, 0] = ids[0, 1]
            self.assertFalse(ns['_dc_ref_routing'](ids, biased, 0))


if __name__ == '__main__':
    unittest.main()
