"""CPU-only invariants; no torch import, CUDA build, or candidate import."""
import ast
import copy
import math
from pathlib import Path
import random
import statistics
import struct
import threading
import unittest

PATH = Path(__file__).with_name("decode_pub.py")
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)


def literal(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def functions(*names, **env):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), "exec"), env)
    return env


def record(**kw):
    value = dict(rounds=16, abba=8, baab=8, equiv=True, ratio=.994, paired=.994,
                 null=1.0, null_paired=1.0)
    value.update(kw)
    return value


class PublicationTests(unittest.TestCase):
    def test_source_identity_and_default_off(self):
        incumbent = literal("ATT7_CUDA_SRC")
        old, new = literal("_DP_OLD"), literal("_DP_NEW")
        env = functions("_dp_source", _DP_OLD=old, _DP_NEW=new)
        changed = env["_dp_source"](incumbent)
        header = "#ifndef OKC_DEC_PUB\n#define OKC_DEC_PUB 0\n#endif\n"
        self.assertTrue(changed.startswith(header))
        conditional = "#if OKC_DEC_PUB\n" + new + "#else\n" + old + "#endif\n"
        # Source oracle: decode_v1's unchanged TU is recovered exactly when off.
        self.assertEqual(changed[len(header):].replace(conditional, old), incumbent)
        self.assertEqual(incumbent.count(old), 1)
        self.assertIn("if (L == 1 && !CONT && !(OKC_NOXCH1 && !M_CL))", incumbent)
        with self.assertRaises(RuntimeError):
            env["_dp_source"](incumbent + old)
        for fragment in ("csync<WS, NT>();", "l_r[0] = l_r[0] * f0 + l1 * f1;",
                         "const float mm = fmaxf(m_r[0], m1);"):
            self.assertEqual(old.count(fragment), new.count(fragment))
        self.assertIn("uint8_t k[Cfg<L, F>::STAGES * STAGE_BYTES]", incumbent)
        self.assertIn("extern __shared__ __align__(1024) uint8_t smem_raw[];", incumbent)

    def test_record_transfer_all_lanes(self):
        rng = random.Random(42)
        for lane in range(64):
            acc = [rng.uniform(-30, 30) for _ in range(16)]
            m, l = rng.uniform(-5, 5), rng.uniform(.1, 5)
            old = struct.pack("18f", m, l, *acc)
            new = bytearray(80)
            struct.pack_into("2f", new, 64, m, l)
            for t in range(0, 16, 4):
                self.assertEqual((lane * 80 + t * 4) % 16, 0)
                struct.pack_into("4f", new, t * 4, *acc[t:t + 4])
            # Oracle: the original scalar exchange's exact FP32 payload.
            self.assertEqual(old[:8], new[64:72])
            self.assertEqual(old[8:], new[:64])
        self.assertEqual(64 * 80, 5120)
        self.assertLessEqual(64 * 80, 8192)

    def test_gate_three_passes_and_boundaries(self):
        accept = functions("_dp_accept", _dp_math=math, _dp_stats=statistics)["_dp_accept"]
        self.assertTrue(accept([record() for _ in range(3)]))
        self.assertFalse(accept([record(), record()]))
        self.assertTrue(accept([record(paired=x, ratio=x) for x in (.992, .995, .999)]))
        for delta in (dict(paired=1), dict(ratio=1), dict(null=.99499), dict(null_paired=1.00501),
                      dict(rounds=15), dict(abba=7), dict(baab=7), dict(equiv=False), dict(paired=float("nan"))):
            self.assertFalse(accept([record(), record(), record(**delta)]), delta)
        self.assertFalse(accept([record(paired=.99501) for _ in range(3)]))
        self.assertTrue(accept([record(null=.995, null_paired=1.005) for _ in range(3)]))

    def test_no_tolerance_floor(self):
        env = functions("_dp_equivalent", _dp_math=math,
                        _dp_bits=lambda a, b: a == b,
                        _dp_error=lambda a, b: (abs(a - b), abs(a - b)))
        ref = ({"attn": 10., "k_new": 1., "v_new": 2.}, 7)
        spread = {k: (0., 0.) for k in ref[0]}
        self.assertEqual(env["_dp_equivalent"](ref, [ref], spread), (True, True))
        candidate = copy.deepcopy(ref)
        candidate[0]["attn"] += .001
        self.assertFalse(env["_dp_equivalent"](candidate, [ref], spread)[0])
        spread["attn"] = (.002, .002)
        self.assertFalse(env["_dp_equivalent"](candidate, [ref], spread)[0])
        repeat = copy.deepcopy(ref)
        repeat[0]["attn"] += .002
        refs = [ref, repeat]
        self.assertEqual(env["_dp_equivalent"](candidate, refs, spread), (True, False))
        candidate[0]["k_new"] += .001
        spread["k_new"] = (1., 1.)
        self.assertFalse(env["_dp_equivalent"](candidate, [ref], spread)[0])
        self.assertFalse(env["_dp_equivalent"]((ref[0], 8), [ref], spread)[0])

    def test_build_flags_all_calls_wrapped(self):
        env = functions("_dp_load")
        result = env["_dp_load"](lambda *a, **kw: kw, extra_cflags=["-std=c++17", "-O2"],
                                  extra_cuda_cflags=["-std=c++17", "-arch=sm_90", "--use_fast_math"])
        self.assertEqual(result["extra_cflags"], ["-O2", "-std=c++20"])
        self.assertEqual(result["extra_cuda_cflags"],
                         ["--use_fast_math", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"])
        calls = [n for n in ast.walk(TREE) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id in ("load_inline", "_tvm_load_inline")]
        self.assertEqual(calls, [])
        self.assertEqual(literal("_DP_KEYS"), ((1, 1, True), (16, 1, True), (32, 1, True)))
        self.assertEqual(literal("_DP_ROUNDS"), 16)

    def test_default_dispatch_and_exception_restore(self):
        tls = threading.local()
        env = functions("kernel_fn", _DP_TLS=tls, _DP_KERNEL=lambda d, s: tls.ext,
                        _dp_key=lambda c: c, _dp_compatible=lambda d: True)
        fn = env["kernel_fn"]
        self.assertIsNone(fn({"config": 1}, {}))
        self.assertEqual(fn({"config": 1}, {"dec_pub_pick": {1}, "dec_pub_ext": 7}), 7)
        self.assertIsNone(tls.ext)
        tls.ext = 13
        def fail(*args):
            raise RuntimeError("intentional")
        env["_DP_KERNEL"] = fail
        with self.assertRaises(RuntimeError):
            fn({"config": 1}, {})
        self.assertEqual(tls.ext, 13)

    def test_file_and_forbidden_apis(self):
        self.assertLess(PATH.stat().st_size, 1_000_000)
        for word in ("subprocess", "Popen", "multiprocessing", "os.system"):
            self.assertNotIn(word, SOURCE)


if __name__ == "__main__":
    unittest.main()
