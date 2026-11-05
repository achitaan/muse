"""CPU-only contract tests, without importing torch or compiling CUDA.

Geometry expectations derive from the incumbent SK2 kernel: R=NWR*NT*8,
N/R CTAs, K/(WSK*UPW*64) stages, no row tail mask. Timing thresholds
and READ512/ABBA/BAAB requirements are the experiment's explicit contract.
"""
import ast
from pathlib import Path
import re
import statistics
import types
import unittest

SOURCE = Path(__file__).with_name('decode_qkvfill.py')
TEXT = SOURCE.read_text()
TREE = ast.parse(TEXT)
BASE = ast.parse(SOURCE.with_name('decode_qkv.py').read_text())


def load(names, tree=TREE, **bindings):
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name in names) or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    ns = dict(bindings)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.ns = load({'_DQ_CFGS', 'bfp_SK2_CONFIGS', 'bfp_SK2_K', 'bfp__sk2_geom'})

    def test_grid_divisibility_and_unchanged_k_arithmetic(self):
        ns = self.ns
        for (layer, T), cfg in ns['_DQ_CFGS'].items():
            name, wl, n, r, nt, nwr, wsk, upw, nacc, cl = cfg
            old_name = {('qkv', 16): 'qkv16_r176', ('qkv', 32): 'qkv32_r208',
                        ('un', 16): 'unpooler16h_r100', ('un', 32): 'unpooler32_r144'}[layer, T]
            old = next(c for c in ns['bfp_SK2_CONFIGS'] if c[0] == old_name)
            self.assertEqual(cfg[5:], old[5:])  # NWR, K warps/units, FP32 chains, cluster
            self.assertEqual(r, nwr * nt * 8)
            self.assertEqual(n % r, 0)
            self.assertEqual(ns['bfp_SK2_K'][wl] % (wsk * upw * 64), 0)
            self.assertGreaterEqual(n // r, 132)
            self.assertEqual(n // r, 144 if layer == 'qkv' else 512 if T == 16 else 256)
            self.assertEqual(ns['bfp__sk2_geom'](cfg, T), ns['bfp__sk2_geom'](old, T))
            # q/k/v/gate row boundaries in the 2304-row matrix (1024+128+128+1024).
            if layer == 'qkv':
                self.assertTrue(all(b % r == 0 for b in (1024, 1152, 1280, 2304)))

    def test_stream_is_exact_cover_and_escape_local_rows(self):
        for (_, T), cfg in self.ns['_DQ_CFGS'].items():
            _, wl, n, r, nt, nwr, wsk, upw, _, _ = cfg
            U = self.ns['bfp_SK2_K'][wl] // 64
            # These are the exact tile/unit formulas in bfp_sk2_stream.
            pairs = [(c * (r // 8) + (w // wsk) * nt + t,
                      s * wsk * upw + (w % wsk) * upw + u)
                     for s in range(U // (wsk * upw)) for c in range(n // r)
                     for w in range(nwr * wsk) for u in range(upw) for t in range(nt)]
            self.assertEqual(len(pairs), n // 8 * U)
            self.assertEqual(set(pairs), {(row, k) for row in range(n // 8) for k in range(U)})
            # Escape encoding/ownership must reconstruct every original row,
            # especially rows that moved across old 24/64-row CTA boundaries.
            for row in range(n):
                cta, local = row // r, row % r
                encoded = local << 16 | 8191
                self.assertEqual(cta * r + (encoded >> 16), row)
                self.assertLess(local, r)

    def test_source_unchanged_kernel_and_sm_limits(self):
        names = {'_DQ_CFGS', 'bfp_SK2_K', 'bfp__sk2_geom', 'bfp_SK2_SRC',
                 'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC', '_dq_source',
                 'bfp_sk2_stream', 'bfp_sk2_table'}
        ns = load(names)
        old = load({'bfp_SK2_SRC', 'bfp_sk2_stream', 'bfp_sk2_table'}, tree=BASE)
        self.assertEqual(ns['bfp_SK2_SRC'], old['bfp_SK2_SRC'])
        for name in ('bfp_sk2_stream', 'bfp_sk2_table'):
            nodes = [next(n for n in tr.body if isinstance(n, ast.FunctionDef) and n.name == name)
                     for tr in (TREE, BASE)]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]))
        src = ns['_dq_source']()
        self.assertNotIn('@@', src)
        self.assertIn('float acc[NACC][MT][NT][4]', src)
        for ci, ((_, T), cfg) in enumerate(ns['_DQ_CFGS'].items()):
            _, _, n, r, nt, nwr, wsk, upw, nacc, cl = cfg
            _, stages = ns['bfp__sk2_geom'](cfg, T)
            self.assertIn(f'cfg == {ci} && N == {n} && M == {T}', src)
            self.assertIn(f'dq_launch_sk2<{T // 16}, {T}, {nt}, {nwr}, {wsk}, {upw}, '
                          f'{stages}, true, false, {nacc}, {cl}, __nv_bfloat16>', src)
            stagebytes = (nwr*wsk*upw*nt*768 + wsk*upw*T*128 + 1023) // 1024 * 1024
            smem = 1024 + stages*stagebytes + 16*stages + 32 + min(stages, 4)*T*(r+1)*4 + 64*8 + 128
            self.assertLessEqual(smem, 232448)

    def test_flags_and_forbidden_operations(self):
        self.assertLess(SOURCE.stat().st_size, 1_000_000)
        self.assertIsNone(re.search(r'subprocess|Popen|multiprocessing|os\.system|(?:cudaGraph|cuGraph)\w*_v[23]', TEXT))
        count = 0
        for n in ast.walk(TREE):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('load_inline', '_tvm_load_inline'):
                kws = {k.arg: k.value for k in n.keywords}
                for k, expected in (('extra_cflags', '_dq_host_flags'), ('extra_cuda_cflags', '_dq_nvcc_flags')):
                    self.assertIsInstance(kws[k], ast.Call)
                    self.assertEqual(kws[k].func.id, expected)
                count += 1
        self.assertEqual(count, 13)
        for name in ('_dq_host_flags', '_dq_nvcc_flags'):
            a = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
            b = next(n for n in BASE.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(a), ast.dump(b))

    def test_pack_rebuilds_streams_tables_and_rejects_excess_smem(self):
        compressed, streams, tables = [], [], []
        def compress(w):
            compressed.append(w)
            return dict(packed=w, base4=17)
        def stream(*args):
            streams.append(args)
            return object()
        pmax = [64]
        def table(comp, n, r):
            tables.append((comp['packed'], n, r))
            return object(), pmax[0]
        ns = load({'_dq_pack', '_DQ_CFGS', 'bfp_SK2_K', 'bfp__sk2_geom'},
                  bfp_sk2_compress=compress, bfp_sk2_stream=stream, bfp_sk2_table=table)
        state = {'qkv': object(), 'unpool': object()}
        ns['_dq_pack'](state)
        self.assertEqual(compressed, [state['qkv'], state['unpool']])
        self.assertEqual([a[2] for a in streams], [16, 16, 16, 32])
        self.assertEqual([a[2] for a in tables], [16, 16, 16, 32])
        self.assertEqual(list(state['dq_streams']), list(ns['_DQ_CFGS']))
        self.assertEqual([v[0] for v in state['dq_streams'].values()], [0, 1, 2, 3])
        pmax[0] = 100_000
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ns['_dq_pack'](state)
        self.assertEqual(state['dq_streams'], {})


class GateTests(unittest.TestCase):
    @staticmethod
    def result(ratio=.994, null=1.):
        t = dict.fromkeys(('ratio', 'paired', 'abba', 'baab'), ratio)
        t['null'] = dict.fromkeys(('ratio', 'paired', 'abba', 'baab'), null)
        return t

    def test_three_pass_median_and_fail_closed_controls(self):
        admit = load({'_dq_admit', '_dq_null_ok'}, _dq_stats=statistics)['_dq_admit']
        good = [self.result(.994), self.result(.995), self.result(.999)]
        self.assertTrue(admit(True, True, good))
        self.assertFalse(admit(True, True, good[:2]))
        self.assertFalse(admit(False, True, good))
        self.assertFalse(admit(True, False, good))
        self.assertFalse(admit(True, True, [self.result(.996)] * 3))
        for metric in ('ratio', 'paired', 'abba', 'baab'):
            for bad_value in (1., 1.01, float('nan')):
                ts = [self.result() for _ in range(3)]
                ts[1][metric] = bad_value
                self.assertFalse(admit(True, True, ts))
            for bad_value in (.9949, 1.0051, float('nan')):
                ts = [self.result() for _ in range(3)]
                ts[1]['null'][metric] = bad_value
                self.assertFalse(admit(True, True, ts))
        self.assertTrue(admit(True, True, [self.result(.995, .995), self.result(.995, 1.005), self.result(.995)]))

    def test_dispatch_is_shape_and_layer_local(self):
        calls = []
        def parent(data, state):
            calls.append(state)
            return state.get('_dq_layers', ())
        ns = load({'kernel_fn'}, _DQ_KERNEL=parent, _dq_supported=lambda d, s, l: True)
        fn = ns['kernel_fn']
        data = {'config': dict(batch=32, q_len=1, window=2048)}
        state = {'dq_pick': {}}
        self.assertEqual(fn(data, state), ())
        self.assertIs(calls[-1], state)
        state['dq_pick'][(32, 1, True)] = ('qkv', 'un')
        self.assertEqual(fn(data, state), ('qkv', 'un'))
        self.assertNotIn('_dq_layers', state)
        ns['_dq_supported'] = lambda d, s, l: l == 'un'
        self.assertEqual(fn(data, state), ('un',))
        data['config']['window'] = 0
        self.assertEqual(fn(data, state), ())

    def test_read512_sixteen_alternating_rounds(self):
        log, clock = [], [0.]
        class Event:
            def __init__(self, **_): pass
            def record(self): self.at = clock[0]
            def synchronize(self): pass
            def elapsed_time(self, other): return (other.at - self.at) / 1000
        class Graph:
            def __init__(self, name, us): self.name, self.us = name, us
            def replay(self):
                log.append(self.name)
                clock[0] += self.us
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event),
                                     sum=lambda *a, **k: log.append('READ512'))
        pair = load({'_dq_pair'}, torch=fake, _dq_stats=statistics,
                    _dq_time=types.SimpleNamespace(monotonic=lambda: 0))['_dq_pair']
        result = pair((Graph('A', 100.), Graph('B', 99.)), object(), object(), lambda: log.append('restore'))
        self.assertEqual(log[2::3], list('ABBA' + 'BAAB') * 8)
        self.assertEqual(log[0::3], ['restore'] * 64)
        self.assertEqual(log[1::3], ['READ512'] * 64)
        self.assertEqual(log[-1], 'B')
        self.assertEqual(result['paired'], .99)
        with self.assertRaises(TimeoutError):
            pair((Graph('A', 1), Graph('B', 1)), None, None, lambda: None, deadline=0)

    def test_incremental_gate_validates_combined_path(self):
        configs = load({'_DQ_CFGS'})['_DQ_CFGS']
        captured, pairs = [], []
        class Value:
            def long(self): return self
            def __floordiv__(self, other): return self
            def flatten(self): return self
            def argmax(self): return 0
            def fill_(self, _): pass
        data = {'local_start': Value()}
        def capture(fn):
            out = fn()
            return types.SimpleNamespace(replay=lambda: None), out
        def parent(d, s):
            captured.append(s['_dq_layers'])
            return (Value(), Value(), Value())
        def pair(gs, f, s, r, **kw):
            pairs.append(kw['blocks'])
            return dict(self.result(1. if gs[0] is gs[1] else .994), base=100., short=99.4)
        ns = load({'_dq_shape', '_dq_admit', '_dq_null_ok'},
                  _DQ_CFGS=configs, _dq_inputs=lambda *a: (data, lambda: None),
                  _dq_supported=lambda *a: True, _dq_capture=capture,
                  _DQ_KERNEL=parent, _dq_time=types.SimpleNamespace(monotonic=lambda: 0),
                  _dq_stats=statistics, _dq_cache=lambda d: (), _dq_reset=lambda *a: None,
                  _dq_snapshot=lambda *a: (), _dq_same=lambda *a: (True,) * 7,
                  _dq_pair=pair, _DQ_SUMMARY=[], LOCAL_EXPERTS=32,
                  torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None),
                                              bincount=lambda *a, **kw: Value()))
        st = {'dq_pick': {}}
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            ns['_dq_shape'](st, 32, True, 'qkv', None, None, 1)
            ns['_dq_shape'](st, 32, True, 'un', None, None, 1)
        self.assertEqual(captured, [(), ('qkv',), ('qkv',), ('qkv', 'un')])
        self.assertEqual(st['dq_pick'], {(32, 1, True): ('qkv', 'un')})
        self.assertEqual(pairs, [16] * 12)  # three nulls + three comparisons per layer
        self.assertEqual(len(ns['_DQ_SUMMARY']), 6)


if __name__ == '__main__':
    unittest.main()
