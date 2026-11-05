"""CPU-only structural, indexing, and timing-harness regressions. No torch/GPU."""
import ast
import hashlib
import importlib.util
from pathlib import Path
import random
import statistics
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
BASE = ast.parse((ROOT / 'kb/v1/decode_v1.py').read_text())
IMPL = ast.parse((ROOT / 'kb/exp/gate_diag_impl.py').read_text())


def definitions(tree, names, ns=None):
    ns = {} if ns is None else ns
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            exec(compile(ast.Module(body=[n], type_ignores=[]), '<selected>', 'exec'), ns)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            exec(compile(ast.Module(body=[n], type_ignores=[]), '<selected>', 'exec'), ns)
    return ns


class DiagTests(unittest.TestCase):
    def test_compaction_preserves_every_kernel_literal(self):
        spec = importlib.util.spec_from_file_location('build', ROOT / 'scripts/build_decode_gate_diag.py')
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        original = (ROOT / 'kb/v1/decode_v1.py').read_text()
        compact = mod.compact(original)
        # Expected source is the unchanged checked-in decode_v1, not a generated
        # numerical golden result. Compare the full AST, including all strings.
        self.assertEqual(ast.dump(BASE), ast.dump(ast.parse(compact)))
        built = mod.build(ROOT)
        self.assertLess(len(built.encode()), 1_000_000)
        self.assertTrue(built.startswith('# okc: problem=muse_decode_layer'))
        for bad in ('b64decode(', 'zlib.decompress(', 'lzma.decompress('):
            self.assertNotIn(bad, built)

    def test_small_source_changes_only_launch_grid(self):
        names = {'bfp_SK2_SRC', 'bfp_CUDA_SRC', 'bfp_MBAR_HELPERS', 'bfp_MAP_SRC', 'bfp_SK2_CONFIGS',
                 'bfp_SK2_PICK', 'bfp_SK2_K', 'bfp__sk2_name', 'bfp__sk2_tokr', 'bfp__sk2_geom', '_de_source'}
        ns = definitions(BASE, names)
        definitions(IMPL, {'_dg_sk_source'}, ns)
        for narrow in (False, True):
            src = ns['_dg_sk_source'](narrow)
            self.assertEqual(src.count('lc.gridDim = dim3(Ws.size(1) * CL)'), 1)
            self.assertEqual(src.count('kern<<<Ws.size(1),'), 1)
            self.assertIn('const size_t wstride = (size_t)(gridDim.x / CL) * WBY', src)
            self.assertIn('out + (size_t)tt * N + cta * R + row', src)
            self.assertNotIn('@@', src)
        self.assertIn('cr[lane*CP+(e.x>>16)]', ns['_dg_sk_source'](True))

    def test_compact_stage_major_prefix_is_exact(self):
        # Source bfp_sk2_stream: tile=c*3+nt, unit=stage*8+wk*2+uu.
        # This independent address model checks arbitrary sentinel weight words
        # across all stages/warps/tiles, including the mixed qkv/gate CTA.
        rng = random.Random(5353)
        for _ in range(10000):
            stage, cta = rng.randrange(16), rng.randrange(54)
            wk, uu, nt, byte = rng.randrange(4), rng.randrange(2), rng.randrange(3), rng.randrange(768)
            intra = (((wk * 2 + uu) * 3 + nt) * 768 + byte)
            wby = 4 * 2 * 3 * 768
            old = (stage * 96 + cta) * wby + intra
            new = (stage * 54 + cta) * wby + intra
            old_stage, rest = divmod(old, 96 * wby)
            old_cta, old_intra = divmod(rest, wby)
            new_stage, rest = divmod(new, 54 * wby)
            new_cta, new_intra = divmod(rest, wby)
            self.assertEqual((old_stage, old_cta, old_intra), (new_stage, new_cta, new_intra))
        self.assertEqual(54 * 24, 1296)
        self.assertEqual(2304 - 1296, 1008)

    def test_single_request_context_matches_speculative_width(self):
        from build_decode_gate_diag import build
        src = build(ROOT)
        node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == '_dg_inputs')
        inputs = ast.get_source_segment(src, node)
        self.assertIn('H, per = st["qkv"].shape[1], (80000 if B == 1 else 86000) + L', inputs)
        self.assertIn('torch.full((1,), 80000', inputs)

    def test_rq_case_ids_preserved(self):
        chen = next(n for n in BASE.body if isinstance(n, ast.FunctionDef) and n.name == '_make_chen')
        tree = ast.Module(body=chen.body, type_ignores=[])
        ns = definitions(tree, {'RQCU', 'RQV', 'RQMS', 'RQI', 'RQPROD', '_rqwg', '_rqss', '_rqsrc'})
        sources = []
        def load_inline(*args, **kwargs):
            sources.append(kwargs['cuda_sources'])
            return object()
        env = dict(_CHEN=ns, _DG_EXT={}, _dg_hash=hashlib, load_inline=load_inline)
        definitions(IMPL, {'_dg_build_rq'}, env)
        ns['RQFLAGS'] = []
        env['_dg_build_rq']({'chen_rq': {('qkv', 128): {'i': 1}, ('qkv', 256): {'i': 6}}})
        src = sources[0]
        import re
        self.assertEqual(re.findall(r'case (\d+): return rq_go', src), ['1', '6'])
        self.assertIn('out.size(1) < W.size(0) * 128', src)
        prod = [ns['RQV'][ns['RQI'][nm]] for nm in ns['RQPROD']]
        for i in (1, 6):
            expected = ', '.join(str(a) for a in prod[i][1:8] + (ns['RQMS'].get(prod[i][0], 1),))
            self.assertIn(f'case {i}: return rq_go<{expected}>', src)

    def test_rq_escape_prefix_reindex(self):
        # Original escape address ((split*NB+nb)*8+warp)*cap+entry.
        # Prefix-repack must change NB in the physical layout, not truncate it.
        ks, oldnb, newnb, cap = 7, 18, 10, 32
        old = [(s, nb, w, e) for s in range(ks) for nb in range(oldnb) for w in range(8) for e in range(cap)]
        new = [old[((s * oldnb + nb) * 8 + w) * cap + e]
               for s in range(ks) for nb in range(newnb) for w in range(8) for e in range(cap)]
        for s in range(ks):
            for nb in range(newnb):
                for w in range(8):
                    for e in (0, 31):
                        self.assertEqual(new[((s * newnb + nb) * 8 + w) * cap + e], (s, nb, w, e))

    def test_abba_read_flush_outside_events(self):
        clock, trace = [0.0], []
        class Event:
            def __init__(self, **kwargs):
                self.t = None
            def record(self):
                self.t = clock[0]
                trace.append('event')
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return (other.t - self.t) / 1000
        class Graph:
            def __init__(self, name, us):
                self.name, self.us = name, us
            def replay(self):
                trace.append(self.name)
                clock[0] += self.us
        def read_flush(*args, **kwargs):
            trace.append('read')
            clock[0] += 100
        def restore():
            trace.append('restore')
            clock[0] += 10
        torch = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event), sum=read_flush)
        env = dict(torch=torch, _dg_stats=statistics)
        definitions(IMPL, {'_dg_pair'}, env)
        result = env['_dg_pair']((Graph('A', 100), Graph('B', 96)), None, None, restore, blocks=12)
        self.assertAlmostEqual(result['save'], 4)
        self.assertAlmostEqual(result['ratio'], .96)
        self.assertAlmostEqual(result['paired'], .96)
        self.assertAlmostEqual(result['abba'], .96)
        self.assertAlmostEqual(result['baab'], .96)
        self.assertEqual([v for v in trace if v in ('A', 'B')][:8], list('ABBABAAB'))
        for i, v in enumerate(trace):
            if v == 'read':
                self.assertEqual(trace[i - 1], 'restore')
                self.assertEqual(trace[i + 1], 'event')

    def test_reservation_dma_release_and_watchdog(self):
        src = (ROOT / 'kb/exp/gate_diag_impl.py').read_text()
        self.assertIn('if ((int)blockIdx.x < cap) return;', src)
        self.assertIn('while (atomicAdd(state, 0) != nsm)', src)
        self.assertIn('5000000000ull', src)
        self.assertIn('state[1:2].copy_(stop, non_blocking=True)', src)
        self.assertIn('pin_memory=True', src)
        self.assertIn('actual != nsm', src)
        self.assertIn('reserve.synchronize()', src)

    def test_support_build_pins_wgmma_architecture(self):
        # Regression for real job 3cbe4bc4: parallel builders share the process
        # environment, so TORCH_CUDA_ARCH_LIST cannot pin this extension's ISA.
        calls = []
        ns = definitions(BASE, {'ATT7_CUDA_SRC', 'ATT7_CPP_SRC'})
        ns.update(_DG_EXT={}, _dg_hash=hashlib,
                  load_inline=lambda *args, **kwargs: calls.append(kwargs))
        definitions(IMPL, {'_DG_SUPPORT', '_dg_build_support'}, ns)
        ns['_dg_build_support']()
        self.assertIn('-gencode=arch=compute_90a,code=sm_90a', calls[0]['extra_cuda_cflags'])
        self.assertIn('-std=c++20', calls[0]['extra_cuda_cflags'])
        self.assertIn('-std=c++20', calls[0]['extra_cflags'])
        self.assertIn('attn_kernel(const __grid_constant__ Params p) {\n  if (p.dbg', calls[0]['cuda_sources'])
        self.assertNotIn('raise RuntimeError(\'; \'.join(errors))',
                         (ROOT / 'kb/exp/gate_diag_impl.py').read_text())


if __name__ == '__main__':
    unittest.main()
