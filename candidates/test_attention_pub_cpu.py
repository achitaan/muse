"""CPU-only source/layout checks; no torch, CUDA toolkit, or process starts."""
import ast
import contextlib
import io
from pathlib import Path
import random
import struct
import unittest


SOURCE = Path(__file__).with_name('attention_pub.py')
TREE = ast.parse(SOURCE.read_text())


def source_env():
    env = {}
    names = {'CUDA_SRC', '_ATTN_PHASE_HELPERS', '_ATTN_PHASE_ANCHOR',
             '_PUB_OLD_EXCHANGE', '_PUB_NEW_EXCHANGE', '_PUB_CUDA'}
    functions = {'_ph_replace', '_ph_pair_source', '_pub_phase_source'}
    for node in TREE.body:
        wanted = (isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))
        wanted |= isinstance(node, ast.FunctionDef) and node.name in functions
        if wanted:
            exec(compile(ast.Module([node], type_ignores=[]), str(SOURCE), 'exec'), env)
    return env


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = source_env()

    def test_exactly_one_kernel_replacement(self):
        e = self.env
        self.assertEqual(e['CUDA_SRC'].count(e['_PUB_OLD_EXCHANGE']), 1)
        self.assertEqual(e['_PUB_CUDA'].count(e['_PUB_NEW_EXCHANGE']), 1)
        self.assertEqual(e['_PUB_CUDA'].replace(e['_PUB_NEW_EXCHANGE'], e['_PUB_OLD_EXCHANGE']),
                         e['CUDA_SRC'])
        self.assertEqual(e['_PUB_CUDA'].count('csync<WS, NT>();'),
                         e['CUDA_SRC'].count('csync<WS, NT>();'))

    def test_diagnostic_anchors(self):
        for name in ('CUDA_SRC', '_PUB_CUDA'):
            source = self.env['_pub_phase_source'](self.env[name])
            for slot in (22, 23, 24, 25):
                self.assertEqual(source.count('ph_stamp(p.dbg, %d);' % slot), 1)
            self.assertIn('ph_stamp(p.dbg, 4); ph_stamp(p.dbg, 5);', source)

    def test_bit_transport_and_addressing(self):
        rng = random.Random(20261104)
        shared = bytearray(3 * 8192)
        expected = {}
        addresses = set()
        for head in range(2):
            for lane in range(32):
                record = head * 32 + lane
                base = record * 80
                self.assertEqual(base % 16, 0)
                self.assertLessEqual(base + 72, len(shared))
                # Include signed zeros, infinities, and NaN payloads: transport
                # must be a bit-copy, not a floating-point conversion.
                bits = [rng.getrandbits(32) for _ in range(18)]
                if lane == 0:
                    bits[:6] = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0x7fa00001]
                old = struct.pack('<18I', *bits)  # m, l, 16 acc values
                expected[record] = old
                for t in range(0, 8, 2):
                    off = base + 8 * t
                    self.assertEqual(off % 16, 0)
                    shared[off:off + 16] = old[8 + 8 * t:24 + 8 * t]
                    for byte in range(off, off + 16):
                        self.assertNotIn(byte, addresses)
                        addresses.add(byte)
                shared[base + 64:base + 72] = old[:8]
        for record, old in expected.items():
            base = record * 80
            self.assertEqual(bytes(shared[base + 64:base + 72] + shared[base:base + 64]), old)

    def test_vector_transactions_conflict_free(self):
        for head in range(2):
            for start_lane in range(0, 32, 8):
                for t in range(0, 8, 2):
                    banks = [((head * 32 + lane) * 20 + 2 * t + element) % 32
                             for lane in range(start_lane, start_lane + 8)
                             for element in range(4)]
                    self.assertEqual(sorted(banks), list(range(32)))

    def test_no_process_start(self):
        forbidden = {'subprocess', 'multiprocessing', 'Popen', 'system', 'fork', 'posix_spawn'}
        for node in ast.walk(TREE):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = ([node.module] if isinstance(node, ast.ImportFrom)
                           else [a.name for a in node.names])
                self.assertFalse(any(m and m.split('.')[0] in forbidden for m in modules))
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', '')
                self.assertNotIn(name, forbidden)

    def test_load_inline_host_flags(self):
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in ('load_inline', '_zh_load_inline')]
        self.assertEqual(len(calls), 6)
        for call in calls:
            keywords = {k.arg: k.value for k in call.keywords}
            self.assertIn('extra_cflags', keywords)
            self.assertIn('-std=c++20', ast.literal_eval(keywords['extra_cflags']))
            self.assertIn('extra_cuda_cflags', keywords)
        for func in (n for n in ast.walk(TREE) if isinstance(n, ast.FunctionDef)):
            text = ast.unparse(func)
            if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id in ('load_inline', '_zh_load_inline') for n in ast.walk(func)):
                # Nested build functions inherit the exact flags from the outer
                # function. Those two outers are checked explicitly below.
                if func.name not in ('build', '_ph_builds'):
                    self.assertIn('-std=c++20', text)
                    self.assertIn('compute_90a,code=sm_90a', text)
        specs = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_ph_build_specs')
        self.assertIn('-std=c++20', ast.unparse(specs))
        self.assertIn('compute_90a,code=sm_90a', ast.unparse(specs))

    def test_two_pass_exact_gate(self):
        case = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_pub_case')
        text = ast.unparse(case)
        self.assertIn('for rep in (1, 2):', text)
        gate = next(n.value for n in ast.walk(case) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'keep' for t in n.targets)
                    and isinstance(n.value, ast.BoolOp))
        self.assertIsInstance(gate.op, ast.And)
        self.assertEqual([n.id for n in gate.values if isinstance(n, ast.Name)], ['keep', 'exact', 'cache'])
        self.assertEqual([ast.unparse(n) for n in gate.values if isinstance(n, ast.Compare)],
                         ['ratio <= 0.985', 'paired <= 0.985'])
        self.assertIn('ATTN_PUB_NULL', text)

    def test_diagnostics_are_deferred_until_all_gates(self):
        case = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_pub_case')
        self.assertNotIn('_pub_trace(', ast.unparse(case))
        prepare = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == 'prepare'][-1]
        gate_loop = next(n for n in ast.walk(prepare) if isinstance(n, ast.For)
                         and isinstance(n.iter, ast.Name) and n.iter.id == '_PUB_CASES')
        self.assertNotIn('_pub_diagnostics(', ast.unparse(gate_loop))
        calls = [n for n in ast.walk(prepare) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in ('_pub_case', '_pub_diagnostics')]
        lines = {n.func.id: n.lineno for n in calls}
        self.assertGreater(lines['_pub_diagnostics'], gate_loop.end_lineno)

    def test_diagnostic_analysis_failure_is_nonfatal(self):
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_pub_diagnostics')
        visited = []
        selected = {(16, 1, 2048)}

        def trace(*args):
            visited.append(args[-1])
            if len(visited) == 1:
                raise ValueError('fewer than eight complete cold phase replays')

        env = {'_pub_trace': trace, '_ATBudget': type('Budget', (Exception,), {}),
               '_PUB_SELECTED': selected}
        exec(compile(ast.Module([node], type_ignores=[]), str(SOURCE), 'exec'), env)
        jobs = [(None, None, 32, None, 'd1S'), (None, None, 256, None, 'd16S')]
        with contextlib.redirect_stdout(io.StringIO()) as log:
            env['_pub_diagnostics'](jobs, None, None, 100.)
        self.assertEqual(visited, ['d1S', 'd16S'])
        self.assertIn('diagnostic_nonfatal ValueError fewer than eight', log.getvalue())
        self.assertEqual(selected, {(16, 1, 2048)})


if __name__ == '__main__':
    unittest.main()
