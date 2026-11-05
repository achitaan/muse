"""CPU-only control-flow tests; extract helpers without importing the CUDA build.

The acceptance oracle is the base file's existing judge scorer, not equality to
the base kernel. GPU numerical/cache validation remains an import-time check.
"""
import ast
import contextlib
import io
import math
from pathlib import Path
import types
import unittest
from unittest.mock import Mock


SOURCE = Path(__file__).with_name('attn_1hfix.py')
TREE = ast.parse(SOURCE.read_text())


class Cache:
    def view(self, dtype):
        return self

    def clone(self):
        return Cache()


class DefaultRouteTests(unittest.TestCase):
    def setUp(self):
        names = {'_AT_1HFIX', '_at_1hfix_shape', '_at_shape'}
        nodes = [node for node in TREE.body
                 if (isinstance(node, ast.FunctionDef) and node.name in names)
                 or (isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in names
                             for t in node.targets))]
        self.env = dict(
            torch=types.SimpleNamespace(uint8='uint8'), _at_math=math,
            _AT_SUMMARY={}, _AT_BASE=Mock(return_value='base-output'),
            _at_drain=Mock(), _at_pack=Mock(return_value=('ref', 0.08)),
            _at_reset_cache=Mock(), _ah_probe=Mock(return_value='candidate-output'),
            _at_score=Mock(return_value=(True, 0.04, 0.06, True)),
            _at_candidates=Mock(side_effect=AssertionError('default must not time')),
            _at_capture=Mock(side_effect=AssertionError('default must not capture')),
            _at_base_us=Mock(side_effect=AssertionError('default must not time')),
            _at_abba=Mock(side_effect=AssertionError('default must not time')),
        )
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), self.env)
        self.data = dict(config=dict(batch=16, q_len=8, window=2048),
                         k_cache=Cache(), v_cache=Cache())
        self.env['_at_data'] = Mock(return_value=self.data)

    def run_shape(self, batch=16):
        self.data['config']['batch'] = batch
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = self.env['_at_shape'](batch, 8, 2048, 'cuda', None, None, 123)
        return result, out.getvalue()

    def test_both_default_plans(self):
        for batch, splits in ((16, 8), (32, 4)):
            with self.subTest(batch=batch):
                result, log = self.run_shape(batch)
                self.assertEqual(result, (f'1h-s{splits}', '1h', dict(n_splits=splits, coop=1)))
                self.assertIn(f'ATTN_1HFIX v{batch}S 1h-s{splits}', log)
                self.assertIn('cache_ok=1 used=1', log)
        self.assertEqual(self.env['_ah_probe'].call_count, 2)

    def test_reference_and_cache_are_from_base_before_poisoned_probe(self):
        calls = []
        for name in ('_AT_BASE', '_at_drain', '_at_pack', '_at_reset_cache',
                     '_ah_probe', '_at_score'):
            self.env[name].side_effect = lambda *a, _name=name, **k: (
                calls.append(_name) or self.env[_name].return_value)
        self.run_shape()
        self.assertEqual(calls, ['_AT_BASE', '_at_drain', '_at_pack',
                                 '_at_reset_cache', '_ah_probe', '_at_score'])
        self.assertEqual(self.env['_at_score'].call_args.args[1:3],
                         ('candidate-output', ('ref', 0.08)))

    def test_judge_rejection_falls_back(self):
        # Existing scorer implements whole-relative and per-row judge bounds.
        self.env['_at_score'].return_value = (False, 0.09, 0.1, True)
        result, log = self.run_shape()
        self.assertIsNone(result)
        self.assertIn('used=0 reason=judge-global-or-row-bound-failed', log)

    def test_cache_mismatch_falls_back(self):
        self.env['_at_score'].return_value = (False, 0.04, 0.06, False)
        result, log = self.run_shape()
        self.assertIsNone(result)
        self.assertIn('cache_ok=0 used=0 reason=cache-bytes-mismatch', log)

    def test_nonfinite_falls_back(self):
        self.env['_at_score'].return_value = (False, float('nan'), float('nan'), True)
        result, log = self.run_shape()
        self.assertIsNone(result)
        self.assertIn('used=0 reason=nonfinite-output-or-error', log)

    def test_preflight_rejection_falls_back_without_scoring(self):
        self.env['_ah_probe'].return_value = None
        result, log = self.run_shape()
        self.assertIsNone(result)
        self.assertIn('used=0 reason=launch-preflight-rejected', log)
        self.env['_at_score'].assert_not_called()

    def test_device_fault_is_not_silently_recovered(self):
        self.env['_ah_probe'].side_effect = RuntimeError('device fault')
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'device fault'):
            self.env['_at_shape'](16, 8, 2048, 'cuda', None, None, 123)

    def test_other_shapes_still_enter_original_tuner(self):
        with self.assertRaisesRegex(AssertionError, 'default must not time'):
            self.env['_at_shape'](16, 1, 2048, 'cuda', None, None, 123)

    def test_original_functions_unchanged_except_default_dispatch(self):
        base = SOURCE.parents[2] / 'board_best' / 'attention_kb1h2_scored_2.6536.py'
        base_tree = ast.parse(base.read_text())
        original = {n.name: n for n in base_tree.body if isinstance(n, ast.FunctionDef)}
        updated = {n.name: n for n in TREE.body if isinstance(n, ast.FunctionDef)}
        for name, node in original.items():
            if name != '_at_shape':
                self.assertEqual(ast.dump(node), ast.dump(updated[name]), name)
        # Only the default lookup/branch was added after synthetic data creation.
        shape = updated['_at_shape']
        self.assertEqual(ast.dump(original['_at_shape']), ast.dump(ast.FunctionDef(
            name=shape.name, args=shape.args, body=shape.body[:2] + shape.body[4:],
            decorator_list=shape.decorator_list, returns=shape.returns,
            type_comment=shape.type_comment)))


if __name__ == '__main__':
    unittest.main()
