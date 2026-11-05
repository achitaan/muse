"""CPU-only gate-branch source, admission, and synchronization regressions."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import random
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from test_decode_gate_diag import BASE, definitions


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import build_decode_gate
        cls.source = build_decode_gate.build(ROOT)
        cls.tree = ast.parse(cls.source)
        names = {'bfp_SK2_SRC', 'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC', 'bfp_SK2_CONFIGS',
                 'bfp_SK2_PICK', 'bfp_SK2_K', 'bfp__sk2_name', 'bfp__sk2_tokr', 'bfp__sk2_geom',
                 '_de_source', 'ATT7_CUDA_SRC'}
        cls.ns = definitions(BASE, names)
        definitions(cls.tree, {'_gg_original_sk_source', '_gg_replace', '_gg_sk_source', '_gg_att_source'}, cls.ns)

    def test_standalone_readable_source(self):
        self.assertLess(len(self.source.encode()), 1_000_000)
        self.assertTrue(self.source.startswith('# okc: problem=muse_decode_layer'))
        for bad in ('b64decode(', 'zlib.decompress(', 'lzma.decompress('):
            self.assertNotIn(bad, self.source)

    def test_sk_waits_output_offsets_and_completion(self):
        for narrow in (False, True):
            src = self.ns['_gg_sk_source'](narrow)
            self.assertIn('if (gate_branch) asm volatile("griddepcontrol.launch_dependents;', src)
            self.assertEqual(src.count('if (pdl && !gate_branch) asm volatile("griddepcontrol.wait;'), 2)
            self.assertIn('out + (size_t)tt * N + gate_offset + cta * R + row', src)
            self.assertIn('if (!gate_branch && gate_flags)', src)
            self.assertIn('named_bar(1, NC);\n    if (tid == 0)', src)
            self.assertIn('atom.acq_rel.gpu.global.add.s32', src)
            self.assertIn('g_gate_flags, g_gate_branch, g_gate_offset));', src)
            self.assertIn('g_gate_flags, g_gate_branch, g_gate_offset);', src)
            # Trigger at entry, not merely at the consumer's normal late trigger:
            # PDL releases attention only after ALL 42 gate CTAs have started.
            self.assertLess(src.index('if (gate_branch) asm volatile'), src.index('const int dgv'))
            self.assertIn('flags.numel()?flags.data_ptr<int>():nullptr', src)

    def test_attention_only_delays_waits_not_arithmetic(self):
        old, new = self.ns['ATT7_CUDA_SRC'], self.ns['_gg_att_source']()
        self.assertIn('gate_qkv_wait(p);   // prefix release, not gate completion', new)
        self.assertEqual(new.count('pdl_wait();   // gate branch completion'), 4)
        # Every instruction in the original CUDA attention remains, except
        # the one relocated wait and the two reset statements. The expected
        # arithmetic comes from the checked-in incumbent (not a guessed output).
        for line in old.splitlines():
            if any(op in line for op in ('mma_async', 'floats2bfloat162', 'const float rn',
                                        'const float s0 = sigm', 'ss +=', 'sum_copies<', 'pr8(y[')):
                self.assertIn(line, new)
        for line in new.splitlines():
            if 'p.qkv +' in line and 'graw' in line:
                previous = new.splitlines()[new.splitlines().index(line) - 1]
                self.assertIn('pdl_wait();', previous)
        self.assertIn('p.ws_cnt[32767] = 0;\n        p.ws_cnt[24576 + bp] = ep + 1;', new)
        self.assertIn('if (tid == 0) { p.ws_cnt[bp] = 0; p.ws_cnt[32767] = 0; }', new)
        self.assertIn('p.ws_cnt[bp] = 0; p.ws_cnt[32767] = 0;\n        asm volatile("st.release', new)
        self.assertIn('p.ws_cnt[16384 + bp] = 0; p.ws_cnt[32767] = 0;', new)

    def test_speculative_miss_retries_before_ready_wait(self):
        src = self.ns['_gg_att_source']()
        guard = 'if (seq_real_ != seq) return false;   // before any qkv/async work'
        self.assertEqual(src.count(guard), 1)
        self.assertLess(src.index(guard), src.index('gate_qkv_wait(p);'))
        self.assertLess(src.index(guard), src.index('const int ntiles = ONE'))
        # Empty real splits still return without consuming (or needing) ready.
        kernel = src[src.index('attn_kernel(const __grid_constant__ Params p)'):]
        self.assertLess(kernel.index('if (start >= end)'), kernel.index('if (segment<L, FLAT'))

    def test_exact_weight_row_and_output_partition(self):
        # bfp_sk2_stream assigns three 8-row tiles to each CTA. Repacking
        # [stage][0:54] and [stage][54:96] must preserve ALL rows and all K units.
        rng = random.Random(5353)
        for _ in range(10000):
            row, unit, lane = rng.randrange(2304), rng.randrange(128), rng.randrange(16)
            oldcta, localrow = divmod(row, 24)
            branch = oldcta >= 54
            localcta = oldcta - (54 if branch else 0)
            offset = 1296 if branch else 0
            self.assertEqual(offset + localcta * 24 + localrow, row)
            # No change to the row's warp, stage, K-unit or BF16 words.
            self.assertEqual((localrow // 8, unit, lane), (row % 24 // 8, unit, lane))
        self.assertEqual(54 + 42, 96)

    def test_counter_protocol_random_schedules(self):
        # Exhaust the ordering argument by randomizing independent events.
        # Attention queries cannot start before all 54 prefix releases. The
        # reset is legal only after every active split has merged, hence every
        # split must have passed the ready wait before the count returns to 0.
        rng = random.Random(32767)
        for _ in range(500):
            nsplit = rng.choice((32, 33, 252))
            count, read, merged = 0, set(), set()
            pending = [('prefix', i) for i in range(54)] + [('read', i) for i in range(nsplit)]
            while pending:
                eligible = [i for i, (kind, _) in enumerate(pending) if kind == 'prefix' or count == 54]
                k = rng.choice(eligible)
                kind, idx = pending.pop(k)
                if kind == 'prefix':
                    count += 1
                else:
                    self.assertEqual(count, 54)
                    read.add(idx)
            order = list(read)
            rng.shuffle(order)
            for split in order:
                merged.add(split)
                if len(merged) == nsplit:
                    self.assertEqual(len(read), nsplit)
                    count = 0
            self.assertEqual(count, 0)

    def test_strict_admission_and_byte_identity(self):
        src = (ROOT / 'kb/exp/gate_overlap_impl.py').read_text()
        self.assertIn("x.view(torch.uint8), y.view(torch.uint8)", src)
        self.assertIn("t[k] <= .992 for k in ('ratio', 'paired', 'abba', 'baab')", src)
        self.assertIn('len(timings) == 2', src)
        self.assertIn('valid &= all(checks)', src)
        self.assertIn("st['gate_pick'] = {}", src)
        self.assertIn("'-gencode=arch=compute_90a,code=sm_90a'", src)
        self.assertIn("_glue_bind_context(data['attn_input'])", src)
        self.assertIn('GATE_ARM shape={label}', src)
        self.assertIn('old_us=', src)

    def test_admission_boundaries_null_and_two_passes(self):
        ns = definitions(self.tree, {'_gg_null_ok', '_gg_admit'})
        good = dict(ratio=.991, paired=.991, abba=.991, baab=.991,
                    null=dict(ratio=1., paired=1., abba=1., baab=1.))
        self.assertTrue(ns['_gg_admit'](True, [good, good]))
        self.assertFalse(ns['_gg_admit'](False, [good, good]))
        self.assertFalse(ns['_gg_admit'](True, [good]))
        for k in ('ratio', 'paired', 'abba', 'baab'):
            for bad in (.99201, float('nan'), float('inf')):
                self.assertFalse(ns['_gg_admit'](True, [good, dict(good, **{k: bad})]))
            self.assertTrue(ns['_gg_admit'](True, [good, dict(good, **{k: .992})]))
            for bad in (.99499, 1.00501, float('nan')):
                arm = dict(good, null=dict(good['null'], **{k: bad}))
                self.assertFalse(ns['_gg_admit'](True, [arm, good]))
            for boundary in (.995, 1.005):
                arm = dict(good, null=dict(good['null'], **{k: boundary}))
                self.assertTrue(ns['_gg_admit'](True, [arm, good]))

    def test_every_inline_build_has_pinned_host_and_nvcc_flags(self):
        ns = definitions(self.tree, {'_gate_host_flags', '_gate_nvcc_flags'})
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id in ('load_inline', '_tvm_load_inline')]
        self.assertEqual(len(calls), 14)
        for call in calls:
            kw = {k.arg: k.value for k in call.keywords}
            for key, name in (('extra_cflags', '_gate_host_flags'), ('extra_cuda_cflags', '_gate_nvcc_flags')):
                self.assertIn(key, kw, call.lineno)
                self.assertIsInstance(kw[key], ast.Call)
                self.assertEqual(kw[key].func.id, name)
        old = ['-O3', '-std=c++17', '-gencode=arch=compute_90,code=sm_90', '--use_fast_math', '--fmad=false', '-DFOO=1']
        self.assertEqual(ns['_gate_nvcc_flags'](old), ['-O3', '--use_fast_math', '--fmad=false', '-DFOO=1',
                         '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'])
        self.assertEqual(ns['_gate_host_flags'](['-O3', '-std=c++17']), ['-O3', '-std=c++20'])

    def test_isolated_prefix_never_publishes_ready_counter(self):
        node = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == '_gg_prefix_only')
        src = ast.get_source_segment(self.source, node)
        self.assertIn("ext.gate_setup(state['bfp']['dummy'][:0], 0, 0)", src)
        self.assertNotIn('_workspace7', src)
        src = (ROOT / 'kb/exp/gate_overlap_impl.py').read_text()
        self.assertIn('_gg_ceiling(data, st, label, flush, sink, deadline)', src)
        self.assertLess(src.index('_gg_ceiling(data, st, label, flush, sink, deadline)\n'),
                        src.index("if 'att' not in _GG_EXT:"))

    def test_flags_source_rewrite_handles_trailing_comma(self):
        from gate_build_flags import pin_inline_builds
        source = '# okc: problem=muse_decode_layer\n"""doc"""\n' + '''
def build(load_inline, _tvm_load_inline):
    load_inline('a', cpp_sources='x', extra_cuda_cflags=['-O3'],)
    _tvm_load_inline('b', cuda_sources='x')
'''
        ns, calls = {}, []
        exec(pin_inline_builds(source), ns)
        ns['build'](lambda *a, **kw: calls.append(kw), lambda *a, **kw: calls.append(kw))
        for kw in calls:
            self.assertEqual(kw['extra_cflags'], ['-O3', '-std=c++20'])
            self.assertEqual(kw['extra_cuda_cflags'], ['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'])

    def test_shape_harness_null_order_and_rejection(self):
        class Ids:
            def long(self): return self
            def __floordiv__(self, other): return self
            def flatten(self): return self
        class Count:
            def __getitem__(self, item): return self if isinstance(item, slice) else 0
            def zero_(self): pass
        class Graph:
            def replay(self): pass
        class Local:
            def fill_(self, value): pass
        for null_ratio, arm_ratio, exact in ((1., .991, True), (.994, .991, True), (1., .993, True), (1., .991, False)):
            order, summary = [], []
            data = dict(local_start=Local())
            def capture(fn):
                return Graph(), (None, None, Ids())
            def pair(graphs, *args, **kwargs):
                null = graphs[0] is graphs[1]
                order.append('null' if null else 'arm')
                self.assertEqual(kwargs['blocks'], 12)
                r = null_ratio if null else arm_ratio
                return dict(base=100., short=100*r, ratio=r, paired=r, abba=r, baab=r)
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None),
                      bincount=lambda *args, **kwargs: types.SimpleNamespace(argmax=lambda: 0))
            ns = dict(torch=torch, _gg_time=types.SimpleNamespace(monotonic=lambda: 0), _GG_EXT={'sk': 1, 'att': 1},
                      _GG_SUMMARY=summary, _GG_FLAG_SLOT=32767, LOCAL_EXPERTS=32,
                      _dg_inputs=lambda *args: (data, lambda: None), _gg_supported=lambda *args: True,
                      _gg_ceiling=lambda *args: None, _workspace7=lambda *args: (None, None, Count()),
                      _dg_capture=capture, _dg_cache=lambda *args: None, _dg_reset=lambda *args: None,
                      _gg_snapshot=lambda *args: None, _gg_same=lambda *args: (exact,)*7, _dg_pair=pair)
            definitions(self.tree, {'_gg_null_ok', '_gg_admit', '_gg_shape'}, ns)
            state = dict(qkv=types.SimpleNamespace(device='cuda'), gate_pick={})
            with contextlib.redirect_stdout(io.StringIO()):
                ns['_gg_shape'](state, 1, 1, True, None, None, 1)
            self.assertEqual(order, ['null', 'arm', 'arm', 'null'])
            self.assertEqual(bool(state['gate_pick']), null_ratio == 1. and arm_ratio < .992 and exact)
            self.assertEqual(len(summary), 3)
            self.assertTrue(summary[0].startswith('GATE_ARM shape=d1S pass=1 old_us='))


if __name__ == '__main__':
    unittest.main()
