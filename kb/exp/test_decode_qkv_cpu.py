"""CPU-only structural/behavioral tests; no CUDA/torch import or compilation.

Expected configurations are from kb/muse_bf16_proj.py's tuned SK2 table.
Admission thresholds and timing rounds are the caller's explicit contract.
"""
import ast
from pathlib import Path
import statistics
import types
import unittest


SOURCE = Path(__file__).with_name('decode_qkv.py')
TEXT = SOURCE.read_text()
TREE = ast.parse(TEXT)


def load(names, **bindings):
    nodes = []
    for node in TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
    ns = dict(bindings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


class ConfigurationTests(unittest.TestCase):
    def test_existing_config_and_ring_geometry(self):
        ns = load({'_DQ_CFG', 'bfp_SK2_CONFIGS', 'bfp_SK2_K', 'bfp__sk2_geom'})
        cfg = ns['_DQ_CFG']
        dense = ast.parse(SOURCE.parents[1].joinpath('muse_bf16_proj.py').read_text())
        self.assertTrue(any(isinstance(n, ast.Tuple) and ast.literal_eval(n) == cfg
                            for n in ast.walk(dense)
                            if isinstance(n, ast.Tuple) and n.elts
                            and isinstance(n.elts[0], ast.Constant) and n.elts[0].value == cfg[0]))
        for T, stages in ((16, 5), (32, 4)):
            old = next(c for c in ns['bfp_SK2_CONFIGS'] if c[0] == f'qkv{T}_r{176 if T == 16 else 208}')
            self.assertEqual(old[1:8], cfg[1:8])  # exact same weight stream/CTA ownership
            self.assertEqual(old[9], cfg[9])
            self.assertEqual((old[8], cfg[8]), (1, 2))
            self.assertEqual(ns['bfp__sk2_geom'](old, T), (False, stages))
            self.assertEqual(ns['bfp__sk2_geom'](cfg, T), (False, stages))
        self.assertEqual(2304 // cfg[3], 96)

    def test_source_dispatch_and_unchanged_arithmetic(self):
        ns = load({'_DQ_CFG', 'bfp_SK2_K', 'bfp__sk2_geom', 'bfp_SK2_SRC',
                   'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC', '_dq_source'})
        src = ns['_dq_source']()
        self.assertNotIn('@@', src)
        self.assertIn('dq_launch_sk2<1, 16, 3, 1, 4, 2, 5, true, false, 2, 1, __nv_bfloat16>', src)
        self.assertIn('dq_launch_sk2<2, 32, 3, 1, 4, 2, 4, true, false, 2, 1, __nv_bfloat16>', src)
        self.assertIn('float acc[NACC][MT][NT][4]', src)
        base = ast.parse(SOURCE.parents[1].joinpath('v1/decode_v1.py').read_text())
        old = next(ast.literal_eval(n.value) for n in base.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'bfp_SK2_SRC' for t in n.targets))
        self.assertEqual(ns['bfp_SK2_SRC'], old)

    def test_all_inline_calls_use_cxx20_sm90a(self):
        ns = load({'_dq_host_flags', '_dq_nvcc_flags'})
        self.assertEqual(ns['_dq_host_flags'](['-O3', '-std=c++17']), ['-O3', '-std=c++20'])
        self.assertEqual(ns['_dq_nvcc_flags'](['-O3', '-std=c++17', '-gencode=arch=compute_90,code=sm_90']),
                         ['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'])
        count = 0
        for node in ast.walk(TREE):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id in ('load_inline', '_tvm_load_inline'):
                kws = {k.arg: k.value for k in node.keywords}
                for k, name in (('extra_cflags', '_dq_host_flags'), ('extra_cuda_cflags', '_dq_nvcc_flags')):
                    self.assertIsInstance(kws[k], ast.Call)
                    self.assertEqual(kws[k].func.id, name)
                count += 1
        self.assertEqual(count, 13)

    def test_source_limits_and_forbidden_operations(self):
        self.assertLess(SOURCE.stat().st_size, 1_000_000)
        for word in ('subprocess', 'Popen', 'multiprocessing', 'os.system'):
            self.assertNotIn(word, TEXT)
        import re
        self.assertIsNone(re.search(r'(?:cudaGraph|cuGraph)\w*_v[23]', TEXT))


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({'_dq_null_ok', '_dq_admit'})

    @staticmethod
    def result(ratio=.990, null=1.0):
        t = dict.fromkeys(('ratio', 'paired', 'abba', 'baab'), ratio)
        t['null'] = dict.fromkeys(('ratio', 'paired', 'abba', 'baab'), null)
        return t

    def test_two_pass_exact_null_and_both_orders_required(self):
        admit = self.ns['_dq_admit']
        self.assertTrue(admit(True, True, [self.result(), self.result()]))
        for valid, ids, ts in ((False, True, [self.result()] * 2),
                               (True, False, [self.result()] * 2),
                               (True, True, [self.result()]),
                               (True, True, [self.result(), self.result(.993)]),
                               (True, True, [self.result(), self.result(null=1.006)]),
                               (True, True, [self.result(), self.result(null=.994)])):
            self.assertFalse(admit(valid, ids, ts))
        for k in ('ratio', 'paired', 'abba', 'baab'):
            bad = self.result()
            bad[k] = .993
            self.assertFalse(admit(True, True, [self.result(), bad]))
            bad = self.result()
            bad['null'][k] = 1.006
            self.assertFalse(admit(True, True, [self.result(), bad]))
        self.assertTrue(admit(True, True, [self.result(.992, .995), self.result(.992, 1.005)]))
        self.assertFalse(admit(True, True, [self.result(float('nan')), self.result()]))

    def test_default_and_other_shapes_never_switch(self):
        calls = []
        def parent(data, st):
            calls.append(st)
            return st.get('_dq_candidate', False)
        ns = load({'kernel_fn'}, _DQ_KERNEL=parent, _dq_supported=lambda d, s: True)
        # Only execute the last kernel_fn wrapper, as load() does by definition order.
        fn = ns['kernel_fn']
        data = {'config': {'batch': 32, 'q_len': 1, 'window': 2048}}
        st = {'dq_pick': {}}
        self.assertFalse(fn(data, st))
        self.assertIs(calls[-1], st)
        st['dq_pick'][(32, 1, True)] = True
        self.assertTrue(fn(data, st))
        self.assertNotIn('_dq_candidate', st)
        data['config']['window'] = 0
        self.assertFalse(fn(data, st))
        data['config']['window'] = 2048
        ns['_dq_supported'] = lambda d, s: False
        self.assertFalse(fn(data, st))

    def test_read_flush_and_abba_samples(self):
        log, clock = [], [0.]
        class Event:
            def __init__(self, **_):
                pass
            def record(self):
                self.at = clock[0]
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return (other.at - self.at) / 1000
        class Graph:
            def __init__(self, name, us):
                self.name, self.us = name, us
            def replay(self):
                log.append(self.name)
                clock[0] += self.us
        def read_sum(value, *, dim, out):
            self.assertIs(value, flush)
            log.append('READ512')
        flush, sink = object(), object()
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event), sum=read_sum)
        ns = load({'_dq_pair'}, torch=fake_torch, _dq_time=types.SimpleNamespace(monotonic=lambda: 0),
                  _dq_stats=statistics)
        result = ns['_dq_pair']((Graph('A', 100.), Graph('B', 98.)), flush, sink,
                                lambda: log.append('restore'), blocks=12)
        self.assertEqual(log[:12], ['restore', 'READ512', 'A', 'restore', 'READ512', 'B',
                                    'restore', 'READ512', 'B', 'restore', 'READ512', 'A'])
        self.assertEqual(log[12:24:3], ['restore'] * 4)
        self.assertEqual(log[14:24:3], ['B', 'A', 'A', 'B'])
        self.assertEqual(log[-1], 'B')  # candidate's KV is checked before null replay
        self.assertEqual(log.count('A'), 24)
        self.assertEqual(log.count('B'), 24)
        self.assertEqual(log.count('READ512'), 48)
        self.assertEqual(result['ratio'], .98)
        self.assertEqual(result['paired'], .98)
        with self.assertRaises(TimeoutError):
            ns['_dq_pair']((Graph('A', 1), Graph('B', 1)), flush, sink, lambda: None, deadline=0)


if __name__ == '__main__':
    unittest.main()
