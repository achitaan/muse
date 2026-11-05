"""CPU-only structural/policy checks; does not import Torch or run a GPU build."""
import ast
import contextlib
import io
import itertools
from pathlib import Path
import re
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'candidates/attention_q8gate.py'
TEXT = SOURCE.read_text()
TREE = ast.parse(TEXT)
ENV = {}
for node in TREE.body:
    if isinstance(node, ast.Assign) and any(
            isinstance(x, ast.Name) and x.id in {'CUDA_SRC', 'AH_CUDA'} for x in node.targets):
        exec(compile(ast.Module([node], []), str(SOURCE), 'exec'), ENV)
    if isinstance(node, ast.FunctionDef) and node.name in {
            '_ph_replace', '_q8g_source', '_q8g_case', '_q8g_diagnostics'}:
        exec(compile(ast.Module([node], []), str(SOURCE), 'exec'), ENV)


class Tensor:
    def view(self, _):
        return self

    def clone(self):
        return self


class Q8GateTests(unittest.TestCase):
    def test_incumbent_prefix_unchanged(self):
        base = (ROOT / 'candidates/attention_pub.py').read_text()
        prefix = base[:base.index('# Publication experiment below owns prepare();')]
        self.assertTrue(TEXT.startswith(prefix))

    def test_forbidden_and_flags(self):
        self.assertFalse(re.search(r'\b(subprocess|Popen|multiprocessing)\b|os\.system\s*\(', TEXT))
        self.assertIn("extra_cflags=['-std=c++20'], extra_cuda_cflags=flags", TEXT)
        fn = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_q8g_builds')
        body = ast.get_source_segment(TEXT, fn)
        self.assertIn('-gencode=arch=compute_90a,code=sm_90a', body)
        self.assertIn('-std=c++20', body)

    def test_generated_source_and_arithmetic_order(self):
        for key, heads, group in [('CUDA_SRC', 16, 'pair'), ('AH_CUDA', 8, 'head')]:
            original = ENV[key]
            candidate = ENV['_q8g_source'](original, key == 'AH_CUDA')
            self.assertEqual(candidate.count('float sg[8];'), 1)
            a = candidate.index('// Gate staging')
            b = candidate.index('      } else {', a)
            body = candidate[a:b]
            self.assertLess(body.index('atom.acq_rel'), body.index('const uint4 graw'))
            self.assertLess(body.index('sigm(gt.x)'), body.index('ld.acquire'))
            self.assertLess(body.index('ld.acquire'), body.index('sum_copies'))
            self.assertEqual(body.count('csync<WS, NT>();'), 1)
            self.assertIn('R <= NT / 8', candidate)
            self.assertIn('(p.window && p.nc == 1', candidate)
            self.assertIn('(%d * %s + hh) * 64 + part * 8' % (heads, group), body)
            # The scale, SS accumulation, shuffle order and rsqrt are verbatim.
            norm_begin = '    const float inv = lsum > 0.f ? __fdividef(vscale, lsum) : 0.f;'
            norm_end = '    const float rn = rsqrtf(ss * (1.f / 64.f) + 1e-5f);'
            norm = body[body.index(norm_begin):body.index(norm_end) + len(norm_end)]
            normalized_original = '\n'.join(line.lstrip() for line in original.splitlines())
            self.assertIn('\n'.join(line.lstrip() for line in norm.splitlines()), normalized_original)
            store = '__floats2bfloat162_rn(y[2 * e] * rn * s0, y[2 * e + 1] * rn * s1)'
            self.assertIn(store, original)
            self.assertIn(store, body)
            # Everything before the cooperative epilogue remains byte-identical.
            anchor = '    asm volatile("" : "+r"(nact), "+r"(R));\n'
            self.assertEqual(original[:original.index(anchor)], candidate[:candidate.index(anchor)])
            with self.assertRaises(ValueError):
                ENV['_q8g_source'](original.replace(anchor, ''))

    def test_single_wave_row_ownership(self):
        for heads, nt, nact in [(16, 256, 8), (8, 128, 4), (8, 128, 5)]:
            rows = 8 * heads
            r = (rows + nact - 1) // nact
            self.assertLessEqual(r, nt // 8)
            ownership = []
            for split in range(nact):
                lo, hi = split * r, min(rows, (split + 1) * r)
                old = [(rr, tid & 7) for tid in range(nt)
                       for rr in range(lo + (tid >> 3), hi, nt // 8)]
                new = [(lo + (tid >> 3), tid & 7) for tid in range(nt)
                       if lo + (tid >> 3) < hi]
                self.assertEqual(old, new)
                ownership.extend(new)
            self.assertEqual(sorted(ownership), list(itertools.product(range(rows), range(8))))

    def test_counter_protocol_all_arrival_orders(self):
        for sizes in [(64, 64, 11), (512, 512, 512, 512, 7)]:
            for order in itertools.permutations(range(len(sizes))):
                count, leaders = 0, []
                for i in order:
                    old = count
                    count += sizes[i]
                    if old + sizes[i] == sum(sizes):
                        leaders.append(i)
                self.assertEqual(leaders, [order[-1]])
                self.assertEqual(count, sum(sizes))

    def policy(self, ratios=((.98, .98), (.98, .98)), exact=True,
               null=(1., 1.), null_exact=True, budget_second=False):
        calls = []
        times = iter([(10., 10., *null)] + [(10., 9.8, *r) for r in ratios])
        checks = iter([(null_exact, True), (exact, True), (exact, True)])

        class Budget(Exception):
            pass

        def abba(*_):
            calls.append('time')
            if budget_second and len(calls) == 3:
                raise Budget()
            return next(times)

        env = dict(ENV, _Q8G_MODS={'pair': object()}, _Q8G_SELECTED={},
                   _at_data=lambda *_: {'k_cache': Tensor(), 'v_cache': Tensor()},
                   _ph_route=lambda _: ('pair', {'coop': 1, 'nc': 1}, 128),
                   _ph_capture=lambda *_: (object(), Tensor()),
                   _at_abba=abba, _q8g_exact=lambda *_: next(checks),
                   torch=SimpleNamespace(uint8=object()))
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_q8g_case')
        exec(compile(ast.Module([node], []), str(SOURCE), 'exec'), env)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            try:
                env['_q8g_case'](16, 8, 2048, None, None, None, 0)
            except Budget:
                pass
        return bool(env['_Q8G_SELECTED']), calls, output.getvalue()

    def test_two_pass_policy(self):
        admitted, calls, output = self.policy()
        self.assertTrue(admitted)
        self.assertEqual(len(calls), 3)
        self.assertIn('pass=2', output)
        self.assertEqual(output.count('admitted=1'), 1)
        for kw in [dict(ratios=((.986, .98), (.98, .98))),
                   dict(ratios=((.98, .98), (.98, .986))),
                   dict(exact=False), dict(null=(1.02, 1.)),
                   dict(null_exact=False), dict(budget_second=True)]:
            self.assertFalse(self.policy(**kw)[0], kw)

    def test_diagnostics_after_both_gates(self):
        fn = next(n for n in reversed(TREE.body) if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
        body = ast.get_source_segment(TEXT, fn)
        self.assertLess(body.index('_q8g_case('), body.index('_q8g_diagnostics('))
        self.assertIn('except Exception as exc:', ast.get_source_segment(TEXT, next(
            n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == '_q8g_diagnostics')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
