"""CPU regression tests for PVT d32 diagnostic timing, NOT GPU correctness.

Exercises the actual old/new selectors using CPU tensors and mocked launches,
scores and clocks. The old source/log establish the rejected comparison; real
CUDA outputs are unavailable locally. Direct BF16 differential tests use Torch
CPU arithmetic. Existing PVT fragment tests remain applicable: CUDA is unchanged.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
import math
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'candidates/attention_q1_pvt.py'
CANDIDATE = ROOT / 'candidates/attention_q1_pvt_d32probe.py'


def definitions(path):
    constants, funcs = {}, {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    constants[name.id] = node.value.value
        elif isinstance(node, ast.FunctionDef):
            funcs[node.name] = node
    return constants, funcs


def load(names, env, path=CANDIDATE):
    _, funcs = definitions(path)
    for name in names:
        exec(compile(ast.Module(body=[funcs[name]], type_ignores=[]), str(path), 'exec'), env)
    return env


class Graph:
    def __init__(self, out):
        self.out = out
    def replay(self):
        pass


def selector_case(*, old=False, full=False, candidate_error=None, base_error=None,
                  timings=((.995, .996), (.995, .996)), fail=None, cache_fail=False):
    """A launch/clock fixture; numerical metrics are explicit inputs, not GPU data."""
    base = torch.ones((32, 1, 1024), dtype=torch.bfloat16)
    pvt = base.clone()
    pvt.view(-1)[0] = 1.0078125  # exactly one next BF16 value above 1
    base_error = base_error or ((.003261800001, .004377100001) if full else (.0047349, .0065459))
    candidate_error = candidate_error or ((.003261800002, .004377100002) if full else (.0047351, .0065459))
    choice = ('q1-pvt', 'pvt', {})
    calls, scores = [], []
    comparisons = iter(timings)
    data = dict(k_cache=torch.zeros(8, dtype=torch.uint8), v_cache=torch.zeros(8, dtype=torch.uint8))
    def score(data, out, pack, cache_bytes, tag, deadline):
        scores.append(tag)
        is_base = 'base' in tag
        rel, row = base_error if is_base else candidate_error
        bad = fail is not None and fail in tag
        return (not bad or cache_fail), rel, row, not (bad and cache_fail)
    def abba(*args):
        calls.append('abba')
        ratio, paired = next(comparisons)
        return 10., 10. * ratio, ratio, paired
    env = dict(torch=torch, _AT_GATE=.988, _AT_MODULES={'pvt': 1},
               _AT_TIMES={}, _AT_SUMMARY={}, _at_check=lambda *_: None,
               _at_drain=lambda *_: None, _at_data=lambda *_: data,
               _at_candidates=lambda *_: [('base', 'base', {}), choice],
               _at_capture=lambda data, ch, _: (Graph(base if ch[1] == 'base' else pvt), base if ch[1] == 'base' else pvt),
               _at_base_us=lambda *_: 10., _at_pack=lambda *_: None,
               _at_reset_cache=lambda *_: None, _at_score=score,
               _ah_probe=lambda *_: pvt.clone(), _at_abba=abba)
    if old:
        load(('_at_gate', '_at_shape'), env, PARENT)
    else:
        load(('_at_gate', '_p32_diff', '_p32_measure', '_p32_shape', '_at_shape'), env)
    output = io.StringIO()
    with redirect_stdout(output):
        selected = env['_at_shape'](32, 1, 0 if full else 2048, None, None, None, None)
    return selected, calls, output.getvalue(), env, scores


class D32ProbeTests(unittest.TestCase):
    def test_reproduces_original_d32s_skip_before_timing(self):
        result, calls, text, _, _ = selector_case(old=True)
        # Source: d526452be7674a5f log lines 15-18. A 2e-7 reference-error
        # increase (4.2e-5 relative increase in the error) skips all timings.
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertIn('probe-numerical-reject', text)

    def test_reproduces_hidden_precision_d32f_rejection(self):
        b, c = .003261800001, .003261800002
        # Analytic: distinct binary floats can print identically at %.5g.
        self.assertEqual('%.5g' % b, '%.5g' % c)
        self.assertGreater(c, b)
        result, calls, text, _, _ = selector_case(old=True, full=True)
        self.assertIsNone(result)
        self.assertEqual(calls, [])
        self.assertIn('probe-numerical-reject', text)

    def test_d32s_and_d32f_valid_but_no_worse_failure_get_timed_not_adopted(self):
        for full in (False, True):
            result, calls, text, env, scores = selector_case(full=full)
            self.assertIsNone(result)
            self.assertEqual(calls, ['abba'])
            self.assertIn('diagnostic_only=1', text)
            self.assertIn('all_no_worse=0', text)
            self.assertIn('base-envelope', text)
            self.assertIn('max_bf16_ulp=1', text)
            self.assertEqual(sum('base-repeat' in s for s in scores), 3)
            self.assertEqual(sum('pvt-probe' in s for s in scores), 3)
            self.assertEqual(env['_AT_TIMES'][32, 1, 0 if full else 2048], (10., 10.))

    def test_diagnostic_speed_win_gets_independent_confirmation_but_never_published(self):
        result, calls, text, env, _ = selector_case(timings=((.95, .96), (.96, .97)))
        self.assertIsNone(result)
        self.assertEqual(calls, ['abba', 'abba'])
        self.assertIn('adopted=0', text)
        self.assertIn('confirm=', env['_AT_SUMMARY'][32, 1, 2048])
        self.assertEqual(env['_AT_TIMES'][32, 1, 2048], (10., 10.))

    def test_no_reference_error_epsilon_added(self):
        # Even a one-unit-in-last-float increase remains an adoption failure.
        result, calls, text, _, _ = selector_case(
            base_error=(.003, .004), candidate_error=(.0030000000000000005, .004),
            timings=((.9, .9), (.9, .9)))
        self.assertEqual(len(calls), 2)
        self.assertIsNone(result)
        self.assertIn('adopted=0', text)

    def test_hard_judge_or_cache_failure_never_gets_initial_timing(self):
        for stage in ('base', 'pvt-probe0', 'pvt-captured'):
            for cache_fail in (False, True):
                result, calls, _, _, _ = selector_case(fail=stage, cache_fail=cache_fail)
                self.assertIsNone(result)
                self.assertEqual(calls, [], (stage, cache_fail))

    def test_late_hard_failure_cannot_adopt(self):
        for stage, expected_calls in (('pvt-repeat', 1), ('pvt-preconfirm', 1), ('pvt-confirm', 2)):
            result, calls, text, env, _ = selector_case(
                candidate_error=(.0047349, .0065459), timings=((.95, .96), (.96, .97)), fail=stage)
            self.assertIsNone(result)
            self.assertEqual(len(calls), expected_calls)
            self.assertEqual(env['_AT_TIMES'][32, 1, 2048], (10., 10.))

    def test_original_both_metrics_both_passes_performance_gate(self):
        for search, confirm, expected in (
                ((.975, .976), (.977, .978), True),
                ((.981, .97), (.97, .97), False),
                ((.97, .981), (.97, .97), False),
                ((.97, .97), (.981, .97), False),
                ((.97, .97), (.97, .981), False),
                ((float('nan'), .97), (.97, .97), False)):
            result, _, _, _, _ = selector_case(
                candidate_error=(.0047349, .0065459), timings=(search, confirm))
            self.assertEqual(result is not None, expected, (search, confirm))

    def test_actual_bf16_differential_metric_handles_sign_zero_and_ulps(self):
        env = load(('_p32_diff',), {'torch': torch})
        base = torch.tensor([0., -0., 1., -1.], dtype=torch.bfloat16)
        value = base.clone()
        bits = value.view(torch.int16)
        bits[1] = 0  # +/-zero have zero ULP/value distance
        bits[2] += 1
        bits[3] += 1
        diff = env['_p32_diff'](value, base)
        # Expected values follow adjacent IEEE BF16 encodings, independently
        # of any norm comparison or PVT kernel.
        self.assertEqual(diff['changed'], 2)
        self.assertEqual(diff['max_ulp'], 1)
        self.assertEqual(diff['over_one_ulp'], 0)
        bits[2] += 1
        diff = env['_p32_diff'](value, base)
        self.assertEqual(diff['max_ulp'], 2)
        self.assertEqual(diff['over_one_ulp'], 1)

    def test_actual_cpu_judge_valid_one_bf16_ulp_still_fails_exact_no_worse(self):
        env = dict(torch=torch, _at_math=math, _AT_ROW_FLOOR=.1, _AT_ROW_FACTOR=5.,
                   _at_drain=lambda *_: None, _at_check=lambda *_: None)
        load(('_at_judge', '_at_score', '_p32_diff', '_p32_measure'), env)
        base = torch.ones((32, 1, 1024), dtype=torch.bfloat16)
        ref = torch.full_like(base, .99609375)  # exact adjacent BF16 value below 1
        value = base.clone()
        value.view(-1)[0] = 1.0078125  # exact adjacent BF16 value above 1
        data = dict(k_cache=torch.zeros(8, dtype=torch.uint8), v_cache=torch.zeros(8, dtype=torch.uint8))
        caches = tuple(data[k].clone() for k in ('k_cache', 'v_cache'))
        _, rel, row = env['_at_judge'](base, ref, .15)
        text = io.StringIO()
        with redirect_stdout(text):
            valid, no_worse, candidate_rel, _ = env['_p32_measure'](
                data, value, base.clone(), (ref, .15), caches, (rel, row), 'CPU-real-score', None)
        # Analytic: one element farther from the fixed reference strictly raises
        # its norm error, although it is only ONE BF16 ULP and satisfies the
        # unmodified judge bound. These are actual CPU scores, not supplied ones.
        self.assertTrue(valid)
        self.assertFalse(no_worse)
        self.assertGreater(candidate_rel, rel)
        self.assertIn('max_bf16_ulp=1 over_one_ulp=0', text.getvalue())

    def test_differential_exposes_a_head_permutation(self):
        env = load(('_p32_diff',), {'torch': torch})
        base = torch.arange(1, 17, dtype=torch.bfloat16).repeat_interleave(64).reshape(1, 1, 1024)
        diff = env['_p32_diff'](base.roll(64, -1), base)
        self.assertEqual(diff['changed'], 1024)
        self.assertGreater(diff['max_ulp'], 1)
        self.assertGreater(diff['over_one_ulp'], 0)

    def test_cuda_and_original_hard_admission_guards_are_identical(self):
        parent, oldf = definitions(PARENT)
        candidate, newf = definitions(CANDIDATE)
        for name in ('CUDA_SRC', 'CPP_SRC', 'AH_CUDA', 'AH_CPP', 'ZH_WST_CUDA', 'ZH_VFG_CUDA'):
            self.assertEqual(parent[name], candidate[name], name)
        for name in ('_pvt_source', '_pvt_build', '_at_judge', '_at_score', '_at_reset_cache',
                     '_at_reference', '_at_abba', '_at_gate', '_at_key', 'kernel_fn'):
            self.assertEqual(ast.dump(oldf[name]), ast.dump(newf[name]), name)
        oldenv = load(('_pvt_source',), dict(parent), PARENT)
        newenv = load(('_pvt_source',), dict(candidate))
        a, b = oldenv['_pvt_source'](parent['CUDA_SRC']), newenv['_pvt_source'](candidate['CUDA_SRC'])
        self.assertEqual(a, b)
        print('Unchanged PVT CUDA SHA256:', hashlib.sha256(a.encode()).hexdigest())

    def test_no_closed_d1_d16_retries(self):
        env = load(('_at_candidates',), {'_AT_MODULES': {'pvt': 1, '1h': 1}})
        experimental = []
        for batch in (1, 16, 32):
            for length in (1, 8):
                for window in (0, 2048):
                    if len(env['_at_candidates'](batch, length, window, None)) > 1:
                        experimental.append((batch, length, window))
        self.assertEqual(set(experimental), {(32, 1, 2048), (32, 1, 0), (32, 8, 2048)})


if __name__ == '__main__':
    unittest.main(verbosity=2)
