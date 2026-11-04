"""CPU fragment/softmax tests for the Q1 PV-axis transpose (no GPU required).

Run with Python + NumPy. This models the documented PTX m16n8k16 and
ldmatrix.x4.trans.b16 fragment layouts, not their GPU execution/timing.
The independent oracle is dense P @ V and the stable softmax identity.
"""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import shutil
import subprocess
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidates/attention_q1_pvt.py"
BASE = ROOT / "board_best/attention_kb1h2_scored_2.6536.py"


def definitions(path: Path) -> tuple[ast.Module, dict]:
    tree = ast.parse(path.read_text())
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for name in node.targets:
                if isinstance(name, ast.Name):
                    env[name.id] = node.value.value
        if isinstance(node, ast.FunctionDef) and node.name == "_pvt_source":
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return tree, env


def fp8_decode(byte: int) -> float:
    """Exact finite E4M3FN interpretation; 0x7f/0xff are NaN."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        return float("nan")
    x = mantissa * 2.0**-9 if exponent == 0 else (1 + mantissa / 8) * 2.0**(exponent - 7)
    return -x if byte & 128 else x


def prmt(a: int, b: int, selector: int) -> int:
    words = a.to_bytes(4, "little") + b.to_bytes(4, "little")
    return int.from_bytes(bytes(words[(selector >> (4 * i)) & 7] for i in range(4)), "little")


def fp8_half_pair(word: int, high: bool = False) -> np.ndarray:
    word >>= 16 if high else 0
    return np.array([fp8_decode(word & 255), fp8_decode((word >> 8) & 255)], dtype=np.float32)


def sw_offset(key: int, head: int, chunk: int) -> int:
    return key * 128 + ((4 * head + chunk) ^ (key & 7)) * 16


def stage(values: np.ndarray) -> bytearray:
    """The unchanged kernel cp.async layout: [64 keys, 2 heads, 64 dims]."""
    memory = bytearray(8192)
    for key in range(64):
        for head in range(2):
            for chunk in range(4):
                off = sw_offset(key, head, chunk)
                memory[off:off + 16] = values[key, head, 16 * chunk:16 * chunk + 16].tobytes()
    return memory


def ldsm_trans(memory: bytearray, addresses: list[int]) -> np.ndarray:
    """PTX trans.b16: lane(g,tig) gets two rows of column g per matrix.

    Source lane 8*m+r supplies row r's address of matrix m. Each register
    has b16 elements at rows 2*tig and 2*tig+1 of that column, low first.
    """
    result = np.empty((32, 4), dtype=np.uint32)
    for lane in range(32):
        g, tig = lane >> 2, lane & 3
        for matrix in range(4):
            a0 = addresses[8 * matrix + 2 * tig] + 2 * g
            a1 = addresses[8 * matrix + 2 * tig + 1] + 2 * g
            assert a0 % 2 == a1 % 2 == 0 and 0 <= a0 <= 8190 and 0 <= a1 <= 8190
            result[lane, matrix] = int.from_bytes(memory[a0:a0 + 2] + memory[a1:a1 + 2], "little")
    return result


def pvt_a(memory: bytearray, kb: int, head: int, du: int) -> tuple[np.ndarray, np.ndarray]:
    # These address expressions and 0x3120 are the pvt_mma2 source expressions.
    addresses = [sw_offset(kb + 8 * ((lane >> 3) & 1) + (lane & 7),
                           head, 2 * du + (lane >> 4)) for lane in range(32)]
    assert all(address % 16 == 0 for address in addresses)
    words = ldsm_trans(memory, addresses)
    operands = []
    for tile in range(2):
        a = np.empty((32, 4, 2), dtype=np.float32)
        for lane in range(32):
            r0 = prmt(int(words[lane, 2 * tile]), 0, 0x3120)
            r1 = prmt(int(words[lane, 2 * tile + 1]), 0, 0x3120)
            a[lane] = [fp8_half_pair(r0), fp8_half_pair(r0, True),
                       fp8_half_pair(r1), fp8_half_pair(r1, True)]
        operands.append(a)
    return tuple(operands)


def mma(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Independent PTX m16n8k16.row.col.f32.f16.f16.f32 fragment model."""
    left, right = np.empty((16, 16), dtype=np.float32), np.empty((16, 8), dtype=np.float32)
    for lane in range(32):
        g, tig = lane >> 2, lane & 3
        for fragment, row, k in ((0, g, 2 * tig), (1, g + 8, 2 * tig),
                                 (2, g, 8 + 2 * tig), (3, g + 8, 8 + 2 * tig)):
            left[row, k:k + 2] = a[lane, fragment]
        right[2 * tig:2 * tig + 2, g] = b[lane, 0]
        right[8 + 2 * tig:10 + 2 * tig, g] = b[lane, 1]
    product = np.zeros((16, 8), dtype=np.float32)
    for lane in range(32):
        g, tig = lane >> 2, lane & 3
        product[g, 2 * tig:2 * tig + 2] = c[lane, :2]
        product[g + 8, 2 * tig:2 * tig + 2] = c[lane, 2:]
    # Fixed sequential FP32 K order, common to both layouts. This does NOT
    # assert Hopper's internal MMA summation tree or instruction scheduling.
    for k in range(16):
        product += left[:, k:k + 1] * right[k:k + 1, :]
    out = np.empty_like(c)
    for lane in range(32):
        g, tig = lane >> 2, lane & 3
        out[lane, :2] = product[g, 2 * tig:2 * tig + 2]
        out[lane, 2:] = product[g + 8, 2 * tig:2 * tig + 2]
    return out


def probabilities_b(probabilities: np.ndarray, ks: int) -> np.ndarray:
    b = np.empty((32, 2, 2), dtype=np.float32)
    p = probabilities.astype(np.float16).astype(np.float32)
    for lane in range(32):
        g, tig = lane >> 2, lane & 3
        b[lane, 0] = p[g, 16 * ks + 2 * tig:16 * ks + 2 * tig + 2]
        b[lane, 1] = p[g, 16 * ks + 8 + 2 * tig:16 * ks + 10 + 2 * tig]
    return b


def pvt_step(o: np.ndarray, memory: bytearray, head: int, half: int, p: np.ndarray) -> np.ndarray:
    out = o.copy()
    for ks in range(2):
        b = probabilities_b(p, ks)
        for du in range(2):
            aa = pvt_a(memory, 32 * half + 16 * ks, head, du)
            for index, a in enumerate(aa):
                tile = 2 * du + index
                out[tile] = mma(a, b, out[tile])
    return out


def baseline_step(memory: bytearray, head: int, half: int, p: np.ndarray) -> np.ndarray:
    """Literal old load_v_frags + padded-query mma_f16, then old output map."""
    out = np.zeros((8, 32, 4), dtype=np.float32)
    for ks in range(2):
        a = np.zeros((32, 4, 2), dtype=np.float32)
        b = np.empty((8, 32, 2, 2), dtype=np.float32)
        for lane in range(32):
            g, tig = lane >> 2, lane & 3
            ph = p.astype(np.float16).astype(np.float32)
            a[lane, 0] = ph[g, 16 * ks + 2 * tig:16 * ks + 2 * tig + 2]
            a[lane, 2] = ph[g, 16 * ks + 8 + 2 * tig:16 * ks + 10 + 2 * tig]
            vv = []
            for delta in (0, 1, 8, 9):
                key = 32 * half + 16 * ks + 2 * tig + delta
                off = sw_offset(key, head, g >> 1) + (g & 1) * 8
                vv.append(memory[off:off + 8])
            for fragment, pair in enumerate(((0, 1), (2, 3))):
                for word in range(2):
                    lo = int.from_bytes(vv[pair[0]][4 * word:4 * word + 4], "little")
                    hi = int.from_bytes(vv[pair[1]][4 * word:4 * word + 4], "little")
                    for index, selector in enumerate((0x5140, 0x7362)):
                        packed = prmt(lo, hi, selector)
                        tile = 4 * word + 2 * index
                        b[tile, lane, fragment] = fp8_half_pair(packed)
                        b[tile + 1, lane, fragment] = fp8_half_pair(packed, True)
        for tile in range(8):
            out[tile] = mma(a, b[tile], out[tile])
    result = np.empty((8, 64), dtype=np.float32)
    for lane in range(32):
        g, tig = lane >> 2, lane & 3
        for tile in range(8):
            for col in range(2):
                result[g, 16 * tig + 8 * col + tile] = out[tile, lane, col]
    return result


def dense_from_native(o: np.ndarray) -> np.ndarray:
    result = np.empty((8, 64), dtype=np.float32)
    for tile in range(4):
        for lane in range(32):
            g, tig = lane >> 2, lane & 3
            for rowbit in range(2):
                for colbit in range(2):
                    result[2 * tig + colbit, 16 * tile + 2 * g + rowbit] = o[tile, lane, 2 * rowbit + colbit]
    return result


def rescale_native(o: np.ndarray, factors: np.ndarray) -> np.ndarray:
    out = o.copy()
    for lane in range(32):
        tig = lane & 3
        out[:, lane, (0, 2)] *= factors[2 * tig]
        out[:, lane, (1, 3)] *= factors[2 * tig + 1]
    return out


def emit_native(o: np.ndarray, factors: np.ndarray) -> np.ndarray:
    result = np.full((8, 64), np.nan, dtype=np.float32)
    counts = np.zeros((8, 64), dtype=np.int32)
    for tile in range(4):
        for lane in range(32):
            g, tig = lane >> 2, lane & 3
            odd = g & 1
            head, dim = 2 * tig + odd, 16 * tile + 4 * (g >> 1)
            own, peer = o[tile, lane], o[tile, lane ^ 4]
            # Literal pvt_pack4 + head-dependent fixed-reference scale.
            if odd:
                packed = np.array([peer[1], peer[3], own[1], own[3]])
            else:
                packed = np.array([own[0], own[2], peer[0], peer[2]])
            result[head, dim:dim + 4] = packed * factors[head]
            counts[head, dim:dim + 4] += 1
    assert np.all(counts == 1), "Every partial coordinate must be written exactly once"
    return result


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree, cls.env = definitions(CANDIDATE)
        cls.base_tree, base_env = definitions(BASE)
        cls.base_source = base_env["CUDA_SRC"]
        cls.source = cls.env["_pvt_source"](cls.env["CUDA_SRC"])

    def test_original_and_q8_paths_unchanged(self):
        self.assertEqual(self.env["CUDA_SRC"], self.base_source)
        # Independent string regions: q8 mainloop, Q1 QK+softmax and all host
        # launch/protocol/finalize code are retained byte-for-byte.
        for begin, end in (("// ---- L == 8 on sm_90:", "// ---- merge:"),
                           ("// Q A-fragments", "l_r[0] = l_r[0] * alpha + ls;"),
                           ("// (last-CTA finalizes):", 'void run(torch::Tensor qkv')):
            self.assertEqual(self.source.split(begin, 1)[1].split(end, 1)[0],
                             self.base_source.split(begin, 1)[1].split(end, 1)[0])

    def test_source_generation_is_fail_closed(self):
        with self.assertRaises(ValueError):
            self.env["_pvt_source"](self.base_source.replace("load_v_frags(V,", "renamed(V,"))

    def test_new_candidate_policy_and_runtime_dispatch(self):
        names = {node.name: node for node in self.tree.body if isinstance(node, ast.FunctionDef)}
        env = {"_AT_MODULES": {"pvt": object(), "1h": object()}, "_AT_GATE": 0.988}
        for name in ("_at_candidates", "_at_gate"):
            exec(compile(ast.Module(body=[names[name]], type_ignores=[]), str(CANDIDATE), "exec"), env)
        for batch in (1, 16, 32):
            for length in (1, 8):
                for window in (0, 2048):
                    choices = env["_at_candidates"](batch, length, window, None)
                    self.assertEqual(choices[0], ("base", "base", {}))
                    self.assertEqual(any(c[1] == "pvt" for c in choices), length == 1)
                    self.assertEqual(any(c[1] == "1h" for c in choices), (batch, length, window) == (32, 8, 2048))
        self.assertEqual(env["_at_gate"](("q1-pvt", "pvt", {})), 0.98)
        # The actual timed entrypoint and metadata guard are exactly incumbent.
        for name in ("_at_key", "kernel_fn"):
            old = [n for n in self.base_tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
            new = [n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
            self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(new, include_attributes=False))

    def test_helper_cpp_syntax(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("No host C++ compiler; NVRTC/ptxas are also unavailable locally")
        helpers = self.source.split("// Q1 PV: two", 1)[1].split("template <int L, bool FLAT", 1)[0]
        helpers = "// Q1 PV: two" + helpers
        preamble = """
        #include <cstdint>
        #define __device__
        #define __forceinline__ inline
        struct float4 { float x,y,z,w; };
        constexpr int ROWB = 128;
        uint32_t smem_u32(const void*);
        uint32_t prmt(uint32_t,uint32_t,uint32_t);
        uint32_t cvt2(uint32_t); uint32_t cvt2h(uint32_t);
        float __shfl_xor_sync(uint32_t,float,int);
        float4 make_float4(float,float,float,float);
        void ldsm_x4_t(uint32_t,uint32_t&,uint32_t&,uint32_t&,uint32_t&);
        void mma_f16(float*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
        """
        # The syntax check is only of the new helper C++, not CUDA/PTX codegen.
        result = subprocess.run([compiler, "-std=c++17", "-x", "c++", "-fsyntax-only", "-"],
                                input=preamble + helpers, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_gpu_admission_requires_both_metrics_and_confirmation(self):
        names = {node.name: node for node in self.tree.body if isinstance(node, ast.FunctionDef)}
        choice = ("q1-pvt", "pvt", {})

        class Graph:
            def replay(self):
                pass

        for search, confirm, expected in (
                ((0.979, 0.979), (0.979, 0.979), True),
                ((0.981, 0.979), (0.979, 0.979), False),
                ((0.979, 0.981), (0.979, 0.979), False),
                ((0.979, 0.979), (0.981, 0.979), False),
                ((0.979, 0.979), (0.979, 0.981), False)):
            timings = iter((search, confirm))

            def abba(*args):
                ratio, paired = next(timings)
                return 10.0, 10.0 * ratio, ratio, paired

            env = {
                "_AT_GATE": 0.988, "_AT_SUMMARY": {}, "_AT_TIMES": {},
                "_at_data": lambda *args: {"k_cache": None, "v_cache": None},
                "_at_candidates": lambda *args: [("base", "base", {}), choice],
                "_at_capture": lambda *args: (Graph(), object()),
                "_at_base_us": lambda *args: 10.0,
                "_at_pack": lambda *args: (None, None),
                "_at_reset_cache": lambda *args: None,
                "_at_score": lambda *args: (True, 0.01, 0.02, True),
                "_at_check": lambda *args: None,
                "_at_drain": lambda *args: None,
                "_ah_probe": lambda *args: object(),
                "_at_abba": abba,
            }

            class Cache:
                def view(self, *args):
                    return self

                def clone(self):
                    return self

            env["_at_data"] = lambda *args: {"k_cache": Cache(), "v_cache": Cache()}
            env["torch"] = type("TorchStub", (), {"uint8": "uint8"})
            for name in ("_at_gate", "_at_shape"):
                exec(compile(ast.Module(body=[names[name]], type_ignores=[]), str(CANDIDATE), "exec"), env)
            with redirect_stdout(io.StringIO()):
                result = env["_at_shape"](32, 1, 2048, None, None, None, None)
            self.assertEqual(result == choice, expected, (search, confirm))


class FragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rng = np.random.default_rng(0xA771)
        cls.bytes = cls.rng.integers(0, 254, (64, 2, 64), dtype=np.uint8)
        cls.bytes[cls.bytes == 127] = 126
        cls.memory = stage(cls.bytes)
        cls.values = np.array([fp8_decode(i) for i in range(256)], dtype=np.float32)[cls.bytes]

    def test_all_fp8_values_are_exact_half(self):
        values = np.array([fp8_decode(i) for i in range(256) if i not in (127, 255)], dtype=np.float32)
        np.testing.assert_array_equal(values.astype(np.float16).astype(np.float32), values)

    def test_transposed_a_fragments_all_coordinates(self):
        for head in range(2):
            for half in range(2):
                for ks in range(2):
                    for du in range(2):
                        operands = pvt_a(self.memory, 32 * half + 16 * ks, head, du)
                        for index, a in enumerate(operands):
                            tile = 2 * du + index
                            for lane in range(32):
                                g, tig = lane >> 2, lane & 3
                                for fragment in range(4):
                                    key = 32 * half + 16 * ks + 2 * tig + 8 * (fragment >> 1)
                                    dim = 16 * tile + 2 * g + (fragment & 1)
                                    # Oracle: the exact original V[key,head,dim],
                                    # not another swizzled-fragment implementation.
                                    np.testing.assert_array_equal(a[lane, fragment], self.values[key:key + 2, head, dim])

    def test_pv_matches_dense_and_original_layout(self):
        for head in range(2):
            for half in range(2):
                for kind in ("uniform", "onehot", "random"):
                    p = self.rng.random((8, 32), dtype=np.float32)
                    if kind == "uniform":
                        p.fill(1 / 32)
                    elif kind == "onehot":
                        p.fill(0)
                        p[np.arange(8), np.arange(8) * 4] = 1
                    p /= p.sum(1, keepdims=True)
                    out = pvt_step(np.zeros((4, 32, 4), dtype=np.float32), self.memory, head, half, p)
                    got = dense_from_native(out)
                    old = baseline_step(self.memory, head, half, p)
                    # Oracle: direct dense attention with the SAME allowed FP16
                    # probability and exact FP8 value operands. FP64 sums bound
                    # the FP32 arithmetic error independently of both layouts.
                    expected = (p.astype(np.float16).astype(np.float64)
                                @ self.values[32 * half:32 * half + 32, head].astype(np.float64))
                    np.testing.assert_allclose(got, expected, rtol=2e-5, atol=1e-5)
                    np.testing.assert_array_equal(got, old)

    def test_rescale_exchange_and_final_pack(self):
        o0, o1 = (self.rng.standard_normal((4, 32, 4)).astype(np.float32) for _ in range(2))
        f0, f1 = (self.rng.random(8, dtype=np.float32) for _ in range(2))
        merged = rescale_native(o0, f0) + rescale_native(o1, f1)
        # Oracle: per-head rescale of dense partial matrices.
        expected = dense_from_native(o0) * f0[:, None] + dense_from_native(o1) * f1[:, None]
        np.testing.assert_array_equal(dense_from_native(merged), expected)
        f = self.rng.random(8, dtype=np.float32)
        np.testing.assert_array_equal(emit_native(merged, f), expected * f[:, None])

    def test_online_softmax_masks_halves_splits_and_scales(self):
        # Include empty second warps, key-tile edges, the sliding window, and
        # a non-aligned nearby length. All heads have distinct running maxima.
        rng = np.random.default_rng(0xC0FFEE)
        for count in (1, 17, 31, 32, 33, 63, 64, 65, 129, 2048, 2055):
            scores = rng.standard_normal((8, count)).astype(np.float32) * 2
            scores += np.arange(8, dtype=np.float32)[:, None] / 3
            vals = self.values[np.arange(count) % 64, 0]
            sum_o, sum_l = np.zeros((8, 64), dtype=np.float32), np.zeros(8, dtype=np.float32)
            for split_start in range(0, count, 256):
                end = min(count, split_start + 256)
                oo, mm, ll = [], [], []
                for half in range(2):
                    o = np.zeros((4, 32, 4), dtype=np.float32)
                    m, l = np.full(8, -np.inf, dtype=np.float32), np.zeros(8, dtype=np.float32)
                    for tile0 in range(split_start, end, 64):
                        lo, hi = tile0 + 32 * half, min(tile0 + 32 * half + 32, end)
                        if lo >= hi:
                            continue
                        sc = np.full((8, 32), -np.inf, dtype=np.float32)
                        sc[:, :hi - lo] = scores[:, lo:hi]
                        mn = np.maximum(m, sc.max(1))
                        alpha = np.exp2(m - mn)
                        p = np.exp2(sc - mn[:, None])
                        o = rescale_native(o, alpha)
                        o = pvt_step(o, self.memory, 0, half, p)
                        l = l * alpha + p.sum(1)
                        m = mn
                    oo.append(o); mm.append(m); ll.append(l)
                m = np.maximum(mm[0], mm[1])
                f0, f1 = np.exp2(mm[0] - m), np.exp2(mm[1] - m)
                merged = rescale_native(oo[0], f0) + rescale_native(oo[1], f1)
                l = ll[0] * f0 + ll[1] * f1
                f = np.exp2(m)
                sum_o += emit_native(merged, f)
                sum_l += l * f
            p = np.exp2(scores.astype(np.float64) - scores.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            expected = p @ vals.astype(np.float64)
            for scale in (1.0, 1 / 64):
                got = sum_o / sum_l[:, None] * scale
                # Independent stable-softmax oracle, allowing just the kernel's
                # unchanged FP16-P rounding plus FP32 summation error.
                error = np.linalg.norm(got - expected * scale) / np.linalg.norm(expected * scale)
                self.assertLess(error, 0.001, (count, scale, error))


if __name__ == "__main__":
    print("Candidate SHA256", hashlib.sha256(CANDIDATE.read_bytes()).hexdigest())
    print("No GPU execution or NVRTC/ptxas compilation is implied by these tests.")
    unittest.main(verbosity=2)
