"""Offline tests for RQ-Q1. No CUDA compilation/execution or timing is claimed.

The oracle is the incumbent's explicit eight-chunk FP32 reduction and shared
half-word permutation, plus an independent dense RMSNorm/RoPE calculation.
The actual new C++ helper runs under a host shim for BF16, half, and shuffles.
Host FP contraction is off; Hopper rsqrt/FMA/codegen are NOT emulated.
"""
from __future__ import annotations

import ast
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'board_best/attention_kb1h2_scored_2.6536.py'
CANDIDATE = ROOT / 'candidates/attention_q1_regquery.py'


def definitions(path):
    tree = ast.parse(path.read_text())
    constants, functions = {}, {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    constants[name.id] = node.value.value
        elif isinstance(node, ast.FunctionDef):
            functions[node.name] = node
    return constants, functions


def load_functions(names, env):
    _, functions = definitions(CANDIDATE)
    for name in names:
        exec(compile(ast.Module(body=[functions[name]], type_ignores=[]), str(CANDIDATE), 'exec'), env)
    return env


def bf16(x):
    words = np.asarray(x, np.float32).view(np.uint32)
    return ((words + 0x7fff + ((words >> 16) & 1)) >> 16).astype(np.uint16)


def floats(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def xor_reduce(values, offsets):
    result = values.copy()
    for offset in offsets:
        result = result + result[..., np.arange(result.shape[-1]) ^ offset]
    return result


def incumbent_query(bits, cs, sn, rope, scale):
    """Literal q1 norm (lines 1439-1485), followed by its staging/read mapping."""
    x = floats(bits).reshape(16, 8, 8)
    chunk_ss = np.zeros((16, 8), np.float32)
    for dim in range(8):
        chunk_ss += x[..., dim] * x[..., dim]
    ss = xor_reduce(chunk_ss, (1, 2, 4))
    r = np.float32(1) / np.sqrt(ss * np.float32(1 / 64) + np.float32(1e-5))
    x = (x * r[..., None]).reshape(16, 64)
    if rope:
        even, odd = x[:, ::2].copy(), x[:, 1::2].copy()
        x[:, ::2] = even * cs - odd * sn
        x[:, 1::2] = even * sn + odd * cs
    half = (x * scale).astype(np.float16).view(np.uint16)
    # The expected packing is the existing shared (0,2,1,3) permutation followed
    # by the actual qa[l0.x,l0.y,l0.z,l0.w,...] reload, NOT the new helper code.
    stage = half.reshape(16, 8, 4, 2)[:, :, [0, 2, 1, 3], :].copy()
    result = np.empty((4, 32, 4, 2), np.uint32)
    for warp in range(4):
        for lane in range(32):
            head, quarter = (warp & 1) * 8 + (lane >> 2), lane & 3
            raw = stage[head, 2 * quarter:2 * quarter + 2].reshape(8, 2)
            words = raw[:, 0].astype(np.uint32) | (raw[:, 1].astype(np.uint32) << 16)
            result[warp, lane] = words.reshape(4, 2)
    return result, x * scale, chunk_ss


HOST_SHIM = r'''
#include <cstdint>
#include <cstring>
#include <cmath>
#define __device__
#define __forceinline__ inline
struct alignas(16) uint4 { uint32_t x, y, z, w; };
struct __nv_bfloat162 { uint16_t x, y; };
struct float2 { float x, y; };
static float b2f(uint16_t b) {
  uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f;
}
static float2 __bfloat1622float2(__nv_bfloat162 b) { return {b2f(b.x), b2f(b.y)}; }
static uint32_t pack_h2(float x, float y) {
  _Float16 a = x, b = y; uint16_t aa, bb;
  std::memcpy(&aa, &a, 2); std::memcpy(&bb, &b, 2);
  return uint32_t(aa) | (uint32_t(bb) << 16);
}
static float rsqrtf(float x) { return 1.f / std::sqrt(x); }
static float chunks[32];
static int lane;
static bool collect;
static float quad_sum(float v) {
  if (collect) { chunks[lane] = v; return v; }
  const float a = chunks[lane] + chunks[lane ^ 1];
  const float b = chunks[lane ^ 2] + chunks[lane ^ 3];
  return a + b;
}
'''


HOST_RUN = r'''
extern "C" void run(const uint16_t* q, const float* cs, const float* sn,
                    int rope, float scale, uint32_t* out, float* local_ss) {
  for (int warp = 0; warp < 4; ++warp) {
    // Two passes simulate simultaneous warp shuffles, using the actual helper:
    // pass 0 captures its pre-shuffle scalar; pass 1 sees every lane's scalar.
    for (int pass = 0; pass < 2; ++pass) {
      collect = pass == 0;
      for (lane = 0; lane < 32; ++lane) {
        const int h = (warp & 1) * 8 + (lane >> 2), dim = (lane & 3) * 16;
        uint4 raw0, raw1;
        std::memcpy(&raw0, q + h * 64 + dim, 16);
        std::memcpy(&raw1, q + h * 64 + dim + 8, 16);
        uint32_t qa[4][2];
        rq_query_fragments(raw0, raw1, lane & 3, scale, cs, sn, rope != 0, qa);
        if (pass == 1) std::memcpy(out + (warp * 32 + lane) * 8, qa, 32);
      }
    }
    std::memcpy(local_ss + warp * 32, chunks, 32 * sizeof(float));
  }
}
'''


class RegisterQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.constants, cls.functions = definitions(CANDIDATE)
        cls.base_constants, cls.base_functions = definitions(BASE)
        cls.env = load_functions(('_rq_source',), dict(cls.constants))
        cls.cuda = cls.env['_rq_source'](cls.constants['CUDA_SRC'])
        compiler = shutil.which('clang++')
        if compiler is None:
            raise unittest.SkipTest('clang++ not available: actual C++ helper cannot run')
        cls.tmp = tempfile.TemporaryDirectory(prefix='.rq-test-', dir=ROOT)
        cls.addClassCleanup(cls.tmp.cleanup)
        cpp = Path(cls.tmp.name) / 'helper.cpp'
        lib = Path(cls.tmp.name) / 'helper.dylib'
        cpp.write_text(HOST_SHIM + cls.constants['RQ_QUERY'] + HOST_RUN)
        subprocess.run([compiler, '-std=c++20', '-O2', '-ffp-contract=off', '-shared',
                        '-fPIC', str(cpp), '-o', str(lib)], check=True, capture_output=True)
        cls.lib = ctypes.CDLL(str(lib))
        cls.lib.run.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_float] + [ctypes.c_void_p] * 2
        cls.lib.run.restype = None

    def run_helper(self, bits, cs, sn, rope, scale):
        out = np.empty((4, 32, 4, 2), np.uint32)
        local_ss = np.empty((4, 32), np.float32)
        self.lib.run(bits.ctypes.data, cs.ctypes.data, sn.ctypes.data, int(rope),
                     float(scale), out.ctypes.data, local_ss.ctypes.data)
        return out, local_ss

    def test_actual_cpp_helper_bitwise_fragments(self):
        rng = np.random.default_rng(0xA77152)
        raw = [np.zeros((16, 64)), np.ones((16, 64)),
               np.tile(np.arange(-32, 32), (16, 1))]
        raw += [rng.normal(size=(16, 64)) * 2 ** rng.uniform(-12, 12) for _ in range(60)]
        for index, query in enumerate(raw):
            bits = bf16(query)
            position = (0, 1, 2048, 80000, 1000000)[index % 5]
            angle = position * 500000.0 ** (-np.arange(32) / 32)
            cs, sn = np.cos(angle).astype(np.float32), np.sin(angle).astype(np.float32)
            for rope in (False, True):
                for scale in (np.float32(0.987), np.float32(0.987 / 64)):
                    expected, _, chunk_ss = incumbent_query(bits, cs, sn, rope, scale)
                    actual, incoming = self.run_helper(bits, cs, sn, rope, scale)
                    # Source: incumbent norm/stage/reload above. Exactly 258048
                    # packed words across 252 cases, including distinct heads.
                    np.testing.assert_array_equal(actual, expected)
                    for warp in range(4):
                        for lane in range(32):
                            h, p = (warp & 1) * 8 + (lane >> 2), 2 * (lane & 3)
                            self.assertEqual(incoming[warp, lane], chunk_ss[h, p] + chunk_ss[h, p + 1])

    def test_fragment_values_against_dense_norm_rope(self):
        rng = np.random.default_rng(130)
        bits = bf16(rng.normal(size=(16, 64)) * 2)
        angles = np.arange(32, dtype=np.float64) / 11
        cs, sn = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        scale = np.float32(0.987)
        packed, _ = self.run_helper(bits, cs, sn, True, scale)
        decoded = np.empty((4, 8, 64), np.float32)
        for warp in range(4):
            for lane in range(32):
                hh, dim = lane >> 2, (lane & 3) * 16
                # Independent interpretation of existing qa fragment coordinates.
                for k, offset in enumerate((0, 2, 8, 10)):
                    for column, plus in enumerate((0, 4)):
                        word = packed[warp, lane, k, column]
                        pair = np.array([word & 65535, word >> 16], np.uint16).view(np.float16)
                        decoded[warp, hh, dim + offset + plus:dim + offset + plus + 2] = pair
        q = floats(bits).astype(np.float64)
        q /= np.sqrt(np.mean(q * q, axis=-1, keepdims=True) + 1e-5)
        even, odd = q[:, ::2].copy(), q[:, 1::2].copy()
        q[:, ::2] = even * cs - odd * sn
        q[:, 1::2] = even * sn + odd * cs
        q *= scale
        for warp in range(4):
            expected = q[(warp & 1) * 8:(warp & 1) * 8 + 8]
            # Source: dense RMSNorm + orthogonal 2D RoPE rotations. Error is
            # f16 rounding; the tolerance is not the judge's larger FP8 allowance.
            self.assertLess(np.linalg.norm(decoded[warp] - expected) / np.linalg.norm(expected), 0.0003)

    def test_zero_and_constant_query_known_solutions(self):
        cs, sn = np.ones(32, np.float32), np.zeros(32, np.float32)
        out, _ = self.run_helper(bf16(np.zeros((16, 64))), cs, sn, True, np.float32(1))
        # Analytic: zero normalized input stays zero under RoPE and half packing.
        np.testing.assert_array_equal(out, 0)
        out, _ = self.run_helper(bf16(np.ones((16, 64))), cs, sn, False, np.float32(1))
        # Analytic: 1 / sqrt(1 + eps) rounds to exactly 1 in f16.
        np.testing.assert_array_equal(out, np.uint32(0x3c003c00))

    def test_addresses_cover_every_query_dimension_twice(self):
        seen = np.zeros((16, 64), np.int32)
        for warp in range(4):
            for lane in range(32):
                h, dim = (warp & 1) * 8 + (lane >> 2), (lane & 3) * 16
                for offset in (0, 8):
                    byte = 2 * (h * 64 + dim + offset)
                    self.assertEqual(byte % 16, 0)
                    self.assertLessEqual(byte + 16, 16 * 64 * 2)
                    seen[h, dim + offset:dim + offset + 8] += 1
        # Two independent key-half warps need identical Q; no cross-warp reads.
        np.testing.assert_array_equal(seen, 2)

    def test_original_cuda_strings_and_dispatch_guards_unchanged(self):
        for name in ('CUDA_SRC', 'CPP_SRC', 'AH_CUDA', 'AH_CPP', 'ZH_WST_CUDA',
                     'ZH_VFG_CUDA', 'ZH_WST_CPP', 'ZH_VFG_CPP'):
            self.assertEqual(self.constants[name], self.base_constants[name], name)
        for name in ('_at_key', '_at_score', '_at_reset_cache', '_at_reference',
                     '_at_abba', '_launch', '_vj_kernel_fn', 'kernel_fn'):
            self.assertEqual(ast.dump(self.functions[name]), ast.dump(self.base_functions[name]), name)

    def test_unchanged_qk_pv_traversal_cache_and_merge(self):
        original = self.constants['CUDA_SRC']
        start = '  for (int i = 0; i < ntiles; ++i) {\n    if constexpr (WS) {'
        # Every QK/PV instruction, mask, tile issue, and the complete inter-CTA
        # protocol is byte-identical after the register-query handoff.
        self.assertEqual(original[original.index(start):], self.cuda[self.cuda.index(start):])
        begin, end = '  if (has_new) {\n#pragma unroll\n    for (int ps', '  csync<WS, NT>();\n  if (!WS && has_new) {'
        cache_code = original[original.index(begin):original.index(end)]
        self.assertIn(cache_code, self.cuda)
        self.assertIn('constexpr bool QREG = L == 1 && !FLAT;', self.cuda)
        self.assertIn('if (has_new) csync<WS, NT>();', self.cuda)
        self.assertLess(self.cuda.index('  pdl_wait();   // qkv'), self.cuda.index('    qr_raw[0] ='))
        self.assertLess(self.cuda.index('S.sn[e] = sv;'), self.cuda.index('rq_query_fragments(qr_raw'))
        between = self.cuda[self.cuda.index('S.sn[e] = sv;'):self.cuda.index('rq_query_fragments(qr_raw')]
        self.assertIn('csync<WS, NT>();', between)

    def test_source_anchors_fail_closed(self):
        original = self.constants['CUDA_SRC']
        anchor = '  constexpr bool QDIR = L == 8 && !WS;\n'
        for bad in (original.replace(anchor, ''), original + anchor):
            with self.assertRaisesRegex(ValueError, 'anchor mismatch'):
                self.env['_rq_source'](bad)

    def test_search_only_new_route_and_prior_reproduced_arm(self):
        env = load_functions(('_at_candidates', '_at_gate'), {'_AT_MODULES': {'rq': 1, '1h': 1}, '_AT_GATE': .988})
        novel, prior = [], []
        for batch in (1, 16, 32):
            for length in (1, 8):
                for window in (0, 2048):
                    choices = env['_at_candidates'](batch, length, window, None)
                    for choice in choices[1:]:
                        if choice[1] == 'rq':
                            novel.append((batch, length, window))
                            self.assertEqual(choice[2], {})
                            self.assertEqual(env['_at_gate'](choice), .98)
                        else:
                            prior.append((batch, length, window, choice[2]))
        self.assertEqual(set(novel), {(1, 1, 0), (1, 1, 2048), (16, 1, 2048), (32, 1, 2048)})
        self.assertEqual(prior, [(32, 8, 2048, {'n_splits': 4, 'coop': 1})])

    def test_both_timing_metrics_in_both_passes_required(self):
        class Tensor:
            def view(self, _): return self
            def clone(self): return self
        class Graph:
            def replay(self): pass
        choice = ('q1-regquery', 'rq', {})
        def check(search, confirm):
            comparisons = iter((search, confirm))
            env = dict(
                torch=SimpleNamespace(uint8=None), _AT_GATE=.988,
                _AT_TIMES={}, _AT_SUMMARY={}, _at_check=lambda _: None,
                _at_data=lambda *_: {'k_cache': Tensor(), 'v_cache': Tensor()},
                _at_candidates=lambda *_: [('base', 'base', {}), choice],
                _at_capture=lambda *_: (Graph(), object()), _at_base_us=lambda *_: 10.,
                _at_pack=lambda *_: None, _at_reset_cache=lambda *_: None,
                _at_score=lambda *_: (True, .001, .002, True), _ah_probe=lambda *_: object(),
                _at_abba=lambda *_: (10., 9., *next(comparisons)), _at_drain=lambda _: None)
            load_functions(('_at_gate', '_at_shape'), env)
            return env['_at_shape'](1, 1, 2048, None, None, None, float('inf'))
        self.assertEqual(check((.975, .976), (.978, .979)), choice)
        for search, confirm in (((.981, .97), (.97, .97)), ((.97, .981), (.97, .97)),
                                ((.97, .97), (.981, .97)), ((.97, .97), (.97, .981))):
            self.assertIsNone(check(search, confirm))


if __name__ == '__main__':
    unittest.main(verbosity=2)
