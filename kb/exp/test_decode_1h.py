"""CPU-only invariance, launch-policy, dispatch and adversarial safety-gate tests."""
import ast
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def functions(source, names, namespace):
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<helpers>', 'exec'), namespace)


class OneHeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = (HERE / 'decode_1h.py').read_text()
        cls.tree = ast.parse(cls.source)
        cls.arm = (HERE / 'decode_1h_arm.py').read_text()
        cls.base = (HERE.parent / 'v1/decode_v1.py').read_text()

    def test_incumbent_ast_identical_after_lossless_dedup(self):
        # Golden = checked-in incumbent, including every CUDA string and compile flag.
        original = ast.parse(self.base)
        prefix = ast.parse(self.source.split('# One-head attention: byte-identical')[0])
        ns = {'_MA_ALT_SRCS': {}}
        functions(self.source, {'_d1h_cuda_lines'}, ns)
        for node in original.body:
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'MOE_CUDA_SRC':
                ns['MOE_CUDA_SRC'] = ast.literal_eval(node.value)
        rebuilt = []
        for node in prefix.body:
            if isinstance(node, ast.FunctionDef) and node.name == '_d1h_cuda_lines':
                continue
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "_MA_ALT_SRCS['moe_helper0']":
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<dedup>', 'exec'), ns)
                node.value = ast.Constant(value=ns['_MA_ALT_SRCS']['moe_helper0'])
            rebuilt.append(node)
        self.assertEqual(ast.dump(original), ast.dump(ast.Module(body=rebuilt, type_ignores=[])))

    def test_no_processes_codecs_or_timing_gate(self):
        self.assertLess(len(self.source.encode()), 1_000_000)
        for word in ('subprocess', 'Popen', 'multiprocessing', 'os.system(', 'b85decode',
                     'b64decode', 'zlib.decompress', 'lzma.decompress'):
            self.assertNotIn(word, self.source)
        for word in ('Event(', 'elapsed_time', 'perf_counter', 'monotonic', 'CUDAGraph'):
            self.assertNotIn(word, self.arm)
        self.assertTrue(self.source.startswith('# okc: problem=muse_decode_layer'))

    def test_cuda_board_lineage_and_flags(self):
        board = HERE.parent.parent / 'board_best/attention_kb1h2_scored_2.6536.py'
        if not board.exists():
            board = Path('/Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse/board_best/attention_kb1h2_scored_2.6536.py')
        def constants(source):
            return {ast.unparse(n.targets[0]): ast.literal_eval(n.value)
                    for n in ast.parse(source).body if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) in ('AH_CUDA', 'AH_CPP')}
        self.assertEqual(constants(self.source), constants(board.read_text()))
        base_fn = next(n for n in ast.parse(self.base).body if isinstance(n, ast.FunctionDef) and n.name == '_att7')
        arm_fn = next(n for n in ast.parse(self.arm).body if isinstance(n, ast.FunctionDef) and n.name == '_d1h_build')
        def flags(fn):
            call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'load_inline')
            return ast.dump(next(k.value for k in call.keywords if k.arg == 'extra_cuda_cflags'))
        self.assertEqual(flags(base_fn), flags(arm_fn))

    def test_gate_never_admits_partial_or_bad_results(self):
        ns = {}
        functions(self.arm, {'_d1h_admit'}, ns)
        admit = ns['_d1h_admit']
        # Public judge minimum tolerance .03 global / .15 per last-dimension row.
        errors = [(.03, .15)] * 4
        self.assertTrue(admit(errors, True, True, True))
        for args in ((False, True, True), (True, False, True), (True, True, False)):
            self.assertFalse(admit(errors, *args))
        self.assertFalse(admit(errors[:3], True, True, True))
        for bad in ((.030001, .01), (.01, .150001), (float('inf'), .01), (float('nan'), .01)):
            for i in range(4):
                changed = list(errors)
                changed[i] = bad
                self.assertFalse(admit(changed, True, True, True))

    def test_dispatch_and_prelaunch_fallback(self):
        calls = []
        class Tensor:
            def view(self, *shape):
                return self
        class Rejected(ValueError):
            pass
        def launch(data, splits, coop):
            calls.append((splits, coop))
            return Tensor()
        ns = {'_D1H_SPLITS': {16: 8, 32: 4}, '_d1h_compatible': lambda d, q: True,
              '_d1h_ah_launch': launch, 'QKV_ROWS': 2304, 'Q_DIM': 1024,
              '_AHLaunchRejected': Rejected, '_D1H_OLD_ATTN': lambda *a: 'incumbent'}
        functions(self.source, {'_d1h_key'}, ns)
        functions(self.arm, {'_decode_attention7'}, ns)
        fn = ns['_decode_attention7']
        for B, splits in ((16, 8), (32, 4)):
            cfg = dict(batch=B, q_len=8, tokens=B*8, window=2048)
            data = dict.fromkeys(('seq_lens', 'cache_loc', 'k_cache', 'v_cache', 'page_table', 'k_scale', 'v_scale'))
            data['config'] = cfg
            obs = {}
            state = dict(dec_1h={(B, 8, True)}, _dec_1h_observe=obs)
            self.assertIsInstance(fn(data, state, Tensor()), Tensor)
            self.assertEqual(calls[-1], (splits, 1))
            self.assertTrue(obs['used'])
            self.assertEqual(fn(data, {}, Tensor()), 'incumbent')
            ns['_d1h_compatible'] = lambda d, q: False
            self.assertEqual(fn(data, state, Tensor()), 'incumbent')
            self.assertFalse(obs['used'])
            ns['_d1h_compatible'] = lambda d, q: True
        ns['_d1h_ah_launch'] = lambda *args: (_ for _ in ()).throw(Rejected('metadata'))
        self.assertEqual(fn(data, state, Tensor()), 'incumbent')
        ns['_d1h_ah_launch'] = lambda *args: (_ for _ in ()).throw(RuntimeError('device fault'))
        with self.assertRaisesRegex(RuntimeError, 'device fault'):
            fn(data, state, Tensor())


if __name__ == '__main__':
    unittest.main()
