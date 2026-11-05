"""CPU-only layout, generated-source and admission checks (stdlib unittest)."""
import ast
from array import array
import importlib.util
import itertools
import math
from pathlib import Path
import statistics
import types
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'decode_pooler.py'
TREE = ast.parse(SOURCE.read_text())


def namespace(names):
    nodes = []
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns


def strides(shape):
    out, v = [], 1
    for n in reversed(shape):
        out.append(v)
        v *= n
    return tuple(reversed(out))


class Tensor:
    """Small dense CPU adapter, storing arbitrary 32-bit payloads, not floats."""
    def __init__(self, data, shape, stride=None):
        self.data, self.shape = data, shape
        self.stride = strides(shape) if stride is None else stride

    def view(self, *shape):
        assert math.prod(shape) == len(self.data)
        assert self.stride == strides(self.shape)
        return Tensor(self.data, shape)

    def permute(self, *axes):
        return Tensor(self.data, tuple(self.shape[a] for a in axes), tuple(self.stride[a] for a in axes))

    def contiguous(self):
        data = array('I', (self.data[sum(i * s for i, s in zip(ix, self.stride))]
                          for ix in itertools.product(*(range(n) for n in self.shape))))
        return Tensor(data, self.shape)


class PoolerTests(unittest.TestCase):
    def test_full_router_payload_and_addresses(self):
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

    def test_generated_source_changes_only_weight_copy(self):
        names = {'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC', 'bfp_SK2_SRC',
                 'bfp_SK2_CONFIGS', 'bfp_SK2_PICK', 'bfp_SK2_K', 'bfp__sk2_geom',
                 '_gle_source', '_dp_source'}
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
        self.assertEqual(old.split('// ---------------- consumers')[1],
                         new.split('// ---------------- consumers')[1])
        self.assertNotIn('@@', new)

    def test_admission_boundaries_and_fail_closed(self):
        ns = namespace({'_dp_null_ok', '_dp_admit'})
        null = dict(ratio=1.0, paired=1.0, abba=.995, baab=1.005)
        arm = dict(ratio=.992, paired=.992, abba=.992, baab=.992, null=null)
        admit = ns['_dp_admit']
        self.assertTrue(admit(True, True, [arm, arm]))
        for exact, ids in ((False, True), (True, False), (False, False)):
            self.assertFalse(admit(exact, ids, [arm, arm]))
        for rounds in ([], [arm], [arm, arm, arm]):
            self.assertFalse(admit(True, True, rounds))
        for key in ('ratio', 'paired', 'abba', 'baab'):
            for value in (.992001, float('nan'), float('inf')):
                self.assertFalse(admit(True, True, [arm, dict(arm, **{key: value})]))
            for value in (.994999, 1.005001, float('nan')):
                self.assertFalse(admit(True, True, [arm, dict(arm, null=dict(null, **{key: value}))]))

    def test_explicit_build_flags_every_load_inline(self):
        calls = 0
        ns = namespace({'_dp_host_flags', '_dp_nvcc_flags'})
        for node in ast.walk(TREE):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                continue
            if node.func.id not in ('load_inline', '_tvm_load_inline'):
                continue
            calls += 1
            kw = {v.arg: v.value for v in node.keywords}
            for key, helper in (('extra_cflags', '_dp_host_flags'), ('extra_cuda_cflags', '_dp_nvcc_flags')):
                self.assertIsInstance(kw[key], ast.Call)
                self.assertEqual(kw[key].func.id, helper)
        self.assertEqual(calls, 13)
        self.assertEqual(ns['_dp_host_flags'](['-O3', '-std=c++17']), ['-O3', '-std=c++20'])
        self.assertEqual(ns['_dp_nvcc_flags'](['-O3', '-std=c++17', '-gencode=arch=compute_90,code=sm_90']),
                         ['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'])

    def test_production_default_and_admitted_dispatch_do_not_mutate_state(self):
        ns = namespace({'kernel_fn'})
        calls = []
        ns['_DP_KERNEL'] = lambda data, state: calls.append(state) or 'output'
        data = {'config': dict(batch=1, q_len=1, tokens=1, window=2048)}
        state = {'dp_pick': {}}
        self.assertEqual(ns['kernel_fn'](data, state), 'output')
        self.assertIs(calls[-1], state)
        state['dp_pick'][(1, 1, True)] = True
        self.assertEqual(ns['kernel_fn'](data, state), 'output')
        self.assertIsNot(calls[-1], state)
        self.assertTrue(calls[-1]['_dp_active'])
        self.assertNotIn('_dp_active', state)
        data['config']['q_len'] = 8
        ns['kernel_fn'](data, state)
        self.assertIs(calls[-1], state)

    def test_pair_uses_12_measured_abba_rounds_and_read_flush_outside_events(self):
        ns = namespace({'_dp_pair'})
        log, elapsed = [], [0.0]

        class Event:
            def __init__(self, **kw):
                pass

            def record(self):
                log.append('event')

            def synchronize(self):
                log.append('sync')

            def elapsed_time(self, other):
                return elapsed[0] / 1000

        class Graph:
            def __init__(self, index):
                self.index = index

            def replay(self):
                log.append(self.index)
                elapsed[0] = (100, 99.1)[self.index]

        ns.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event),
                  sum=lambda *a, **kw: log.append('read')),
                  _dp_time=types.SimpleNamespace(monotonic=lambda: 0), _dp_stats=statistics)
        result = ns['_dp_pair']((Graph(0), Graph(1)), 'flush', 'sink', lambda: log.append('reset'), 1)
        self.assertEqual(result['old_us'], 100)
        self.assertAlmostEqual(result['new_us'], 99.1)
        for key in ('ratio', 'paired', 'abba', 'baab'):
            self.assertAlmostEqual(result[key], .991)
        expected = []
        for r in range(14):
            for i in ((0, 1, 1, 0) if r % 2 == 0 else (1, 0, 0, 1)):
                expected += ['reset', 'read', 'event', i, 'event', 'sync']
        self.assertEqual(log, expected)

    def test_source_limit_and_no_process_or_graph_version_api(self):
        text = SOURCE.read_text()
        self.assertLess(len(text.encode()), 1_000_000)
        for forbidden in ('subprocess', 'Popen', 'multiprocessing', 'os.system', 'cudaGraphInstantiate_v2',
                          'cudaGraphInstantiate_v3', 'cudaGraphGetNodes_v2', 'cudaGraphGetNodes_v3'):
            self.assertNotIn(forbidden, text)

    def test_reproducible_assembly(self):
        spec = importlib.util.spec_from_file_location('builder', HERE / 'build_decode_pooler.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.assemble(), SOURCE.read_text())


if __name__ == '__main__':
    unittest.main()
