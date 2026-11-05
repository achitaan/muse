"""CPU-only policy, error-envelope and single-call protocol regression tests."""
import ast
import contextlib
import io
import math
from pathlib import Path
import types
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = (HERE / 'decode_lockin.py').read_text()
TREE = ast.parse(SOURCE)
BASE = ast.parse((HERE / 'decode_combo.py').read_text())


def namespace(names, **initial):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names
             or isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
    ns = {'_kp_math': math, **initial}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'lockin', 'exec'), ns)
    return ns


def functions(tree):
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


class LockinTests(unittest.TestCase):
    def test_exact_defaults_and_dispatch_tls_restoration(self):
        ns = namespace({'_DL_DEFAULTS', '_KP_KEYS', '_kp_key', 'kernel_fn'})
        self.assertEqual(ns['_DL_DEFAULTS'], {(1, 1, False): 'KV', (1, 8, True): 'POOL'})
        ns['_KP_TLS'] = tls = types.SimpleNamespace(ext='saved', observe='saved_obs')
        ns['_kp_compatible'] = lambda d: d.get('compatible', True)
        calls = []
        def call(data, state):
            calls.append((tls.ext, tls.observe, state['_dp_active']))
            if data.get('raise'):
                raise ValueError('test')
            return 'ok'
        ns['_KP_KERNEL'] = call
        # Even a malformed pick cannot enable KV at d1S or POOL at v1F.
        picks = {(1, 1, False): 'KV', (1, 8, True): 'POOL',
                 (1, 1, True): 'KV', (1, 8, False): 'POOL', (2, 1, False): 'KV'}
        st = dict(lockin_pick=picks, kvpre_ext='kv', _kp_observe='obs')
        for key, expected in (((1, 1, False), ('kv', 'obs', False)),
                              ((1, 8, True), (None, 'obs', True)),
                              ((1, 1, True), (None, 'obs', False)),
                              ((1, 8, False), (None, 'obs', False)),
                              ((2, 1, False), (None, 'obs', False))):
            b, l, s = key
            data = dict(config=dict(batch=b, q_len=l, window=2048 if s else 0))
            self.assertEqual(ns['kernel_fn'](data, st), 'ok')
            self.assertEqual(calls[-1], expected)
            self.assertEqual((tls.ext, tls.observe), ('saved', 'saved_obs'))
        data = dict(config=dict(batch=1, q_len=1, window=0))
        ns['kernel_fn'](dict(data, compatible=False), st)
        self.assertEqual(calls[-1], (None, 'obs', False))
        with self.assertRaises(ValueError):
            ns['kernel_fn'](dict(data, **{'raise': True}), st)
        self.assertEqual((tls.ext, tls.observe), ('saved', 'saved_obs'))
        self.assertNotIn('_dp_active', st)

    def test_strict_single_incumbent_error_and_judge_checks(self):
        ns = namespace({'_dl_validate', '_dc_within'})
        ns['torch'] = types.SimpleNamespace(equal=lambda a, b: a == b)
        ns['_dc_metrics'] = lambda a, b: (abs(a - b), abs(a - b))
        reference = {k: 0. for k in ('hidden', 'stream', 'update', 'k_new', 'v_new')}
        tolerances = dict.fromkeys(reference, .03)
        inc = (dict.fromkeys(reference, .01), (1, 2, 3))
        def check(vals=None, ids=inc[1], untouched=True, routing=True):
            new = (dict(inc[0], **(vals or {})), ids)
            return ns['_dl_validate'](inc, new, reference, tolerances, untouched, routing)
        # The documented contract: improved/identical outputs accepted, not bit-equality.
        self.assertTrue(check()[3])
        self.assertTrue(check({'hidden': .009})[3])
        for key in reference:
            self.assertIn('worse_' + key, check({key: .01000000001})[4])
            self.assertFalse(check({key: math.nan})[3])
            self.assertFalse(check({key: math.inf})[3])
            self.assertIn('new_judge_' + key, check({key: .031})[4])
        inc[0]['k_new'] = 0.
        self.assertFalse(check({'k_new': 1e-20})[3])
        self.assertIn('expert_ids', check(ids=(1, 3, 2))[4])
        self.assertIn('untouched_cache', check(untouched=False)[4])
        self.assertIn('incumbent_routing', check(routing=False)[4])
        inc[0]['hidden'] = .04
        self.assertIn('inc_judge_hidden', check({'hidden': .02})[4])

    def test_worst_row_bound_is_also_strict(self):
        ns = namespace({'_dl_validate', '_dc_within'}, torch=types.SimpleNamespace(equal=lambda a, b: True))
        ns['_dc_metrics'] = lambda a, b: a
        inc = ({'hidden': (.01, .1)}, ())
        reference, tol = {'hidden': None}, {'hidden': .03}
        check = lambda err: ns['_dl_validate'](inc, ({'hidden': err}, ()), reference, tol, True, True)
        self.assertTrue(check((.009, .099))[3])
        self.assertIn('worse_hidden', check((.009, .10000001))[4])
        self.assertIn('new_judge_hidden', check((.009, .151))[4])

    def test_single_layer_call_no_warmup_and_component_proof(self):
        ns = namespace({'_dl_layer'})
        calls = []
        ns['torch'] = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
        ns['_kp_private'] = dict
        def kernel(data, st):
            calls.append(st['lockin_pick'])
            st['_kp_observe'].update(qkv=None, attn=None, used='KV' in st['lockin_pick'].values(),
                                     pool_used='POOL' in st['lockin_pick'].values())
            return 'out'
        ns['kernel_fn'] = kernel
        key = (1, 1, False)
        self.assertEqual(ns['_dl_layer']({}, {}, key, 'BASE'), 'out')
        self.assertEqual(ns['_dl_layer']({}, {}, key, 'KV'), 'out')
        self.assertEqual(calls, [{}, {key: 'KV'}])
        ns['kernel_fn'] = lambda d, s: None
        with self.assertRaisesRegex(RuntimeError, 'native component'):
            ns['_dl_layer']({}, {}, key, 'KV')

    def test_per_shape_failure_does_not_disable_other_default_and_cleanup(self):
        ns = namespace({'_dl_safety', '_DL_DEFAULTS'})
        log = []
        ext = types.SimpleNamespace(sk2_set_pdl=lambda x: log.append(('pdl', x)))
        ma = types.SimpleNamespace(okc_set_pdl=lambda x: log.append(('ma', x)))
        saved = ({'ws': 1}, {'cl': 1}, ['guards'], {'ids'}, 17, 'pf', 'gle')
        keys = ('_WS7', '_CL7', '_GUARDS', '_GUARDED_IDS', 'bfp_SK2_PDL', '_PF7', '_GLE_W_CUR')
        ns.update(zip(keys, saved))
        ns.update(torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None)),
                  bfp__get_ext=lambda: ext, _GLE_EXT=ext, _DE_EXT=None, _DP_EXT=ext,
                  _check_guards=lambda: None)
        ns['_dl_check'] = lambda st, w, key, arm, d: (True, {'hidden': (0., 0.)}, {'hidden': (0., 0.)}, True, 'ok')
        for arm, expected in (('KV', {(1, 8, True): 'POOL'}), ('POOL', {(1, 1, False): 'KV'})):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                picks = ns['_dl_safety']({'ma_ext': {1: ma}, '_dl_unavailable': {arm: 'failed'}}, {}, 10)
            self.assertEqual(picks, expected)
            self.assertIn('DEC_LOCKIN shape=d1F arm=KV ids_equal=', out.getvalue())
            self.assertIn('DEC_LOCKIN shape=v1S arm=POOL ids_equal=', out.getvalue())
            self.assertIn('why=RuntimeError:failed', out.getvalue())
            self.assertTrue(all(ns[k] is v for k, v in zip(keys, saved)))
        self.assertIn(('pdl', 17), log)
        self.assertIn(('ma', 0), log)

    def test_measured_cuda_build_flags_reference_and_incumbent_unchanged(self):
        old, new = functions(BASE), functions(TREE)
        preserve = {'_kvpre_load', '_kp_source', '_kp_replace', '_kp_start_build', '_dp_source',
                    '_dp_build', '_dp_pack', '_dp_unpack', '_gle_frt', '_kp_inputs', '_kp_untouched',
                    '_dc_metrics', '_dc_within'}
        preserve |= {n for n in old if n.startswith('_dc_ref')}
        for name in preserve:
            self.assertEqual(ast.dump(old[name]), ast.dump(new[name]), name)
        marker = '# DEC_COMBO: independent KV/POOL candidates, native incumbent by default.'
        prefix = (HERE / 'decode_combo.py').read_text().split(marker)[0]
        self.assertTrue(SOURCE.startswith(prefix))
        self.assertLess(len(SOURCE.encode()), 1_000_000)
        for name in ('_dc_gate', '_dc_accept', '_dc_pass', '_dc_pick', '_kp_block', '_kp_capture', '_kp_summarize'):
            self.assertNotIn(name, new)
        suffix = SOURCE[len(prefix):]
        for text in ('enable_timing', '512 << 20', 'elapsed_time', 'DEC_COMBO', '.replay('):
            self.assertNotIn(text, suffix)
        for text in ('subprocess', 'Popen', 'multiprocessing', 'os.system', 'os.fork', 'os.spawn',
                     'cudaGraphInstantiate_v2', 'cudaGraphInstantiate_v3'):
            self.assertNotIn(text, SOURCE)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'CPU torch unavailable')
class ProtocolTests(unittest.TestCase):
    def test_single_synthetic_input_two_calls_reset_and_reference_ids(self):
        ns = namespace({'_dl_check', '_dc_budget'})
        t = torch
        data = dict(k_cache=t.zeros(2, dtype=t.uint8), v_cache=t.zeros(2, dtype=t.uint8),
                    local_start=t.zeros(1, dtype=t.int32), stream=t.ones(1))
        biased = t.zeros(1, 256)
        biased[:, 64:72] = 1.
        ids = t.arange(64, 72).view(1, 8)
        log = []
        def layer(data, st, key, arm):
            log.append(('call', arm, int(data['local_start'])))
            return arm
        def reference(data, weights, front, chosen_ids, deadline):
            self.assertTrue(t.equal(chosen_ids, ids))
            self.assertEqual(front['local_start'], 64)
            log.append('reference')
            return {}, {}, None
        ns.update(torch=t, _kp_time=types.SimpleNamespace(monotonic=lambda: 0),
                  _kp_inputs=lambda *a: log.append('input') or data,
                  _kp_slots=lambda d: (), _kp_reset=lambda *a: log.append('reset'),
                  _dc_ref_front=lambda *a: {'biased': biased}, _dc_ref_coeff=lambda *a: (1., 1.),
                  _dl_layer=layer, _dl_snapshot=lambda *a: ({}, ids),
                  _kp_untouched=lambda *a: log.append('untouched') or True,
                  _dc_ref_routing=lambda *a: True, _dc_reference=reference,
                  _dl_validate=lambda *a: (True, {}, {}, True, 'ok'), _check_guards=lambda: log.append('guards'))
        weights = dict(attn_residual_gate=None, ffn_residual_gate=None)
        self.assertTrue(ns['_dl_check']({}, weights, (1, 1, False), 'KV', 10)[3])
        self.assertEqual(log, ['input', 'reset', ('call', 'BASE', 64), 'untouched', 'reference',
                               'reset', ('call', 'KV', 64), 'untouched', 'guards'])

    def test_real_metrics_zero_cache_baseline_and_hidden_dtype(self):
        ns = namespace({'_dl_validate', '_dc_metrics', '_dc_within'}, torch=torch)
        t = torch
        # Analytic zero-error incumbent: no nonzero candidate error is allowed.
        vals = dict(hidden=t.ones(1, 8, dtype=t.bfloat16), stream=t.ones(1, 8),
                    update=t.ones(1, 8), k_new=t.ones(1, 1, 2, 64).to(t.float8_e4m3fn),
                    v_new=t.ones(1, 1, 2, 64).to(t.float8_e4m3fn))
        ids = t.arange(8).view(1, 8)
        inc = (vals, ids)
        tol = dict.fromkeys(vals, .03)
        self.assertTrue(ns['_dl_validate'](inc, inc, vals, tol, True, True)[3])
        new = {k: v.clone() for k, v in vals.items()}
        new['k_new'] = (new['k_new'].float() * 1.125).to(t.float8_e4m3fn)
        result = ns['_dl_validate'](inc, (new, ids), vals, tol, True, True)
        self.assertFalse(result[3])
        self.assertIn('worse_k_new', result[4])


if __name__ == '__main__':
    unittest.main()
