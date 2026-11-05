"""CPU-only gate/source contract tests; no torch, CUDA, or module import needed.

Expected admission thresholds come from the KVPRE task specification. Source
invariants compare to the unmodified kb/v1/decode_v1.py, not a generated golden.
These tests do not establish GPU numerical correctness or timing.
"""
import ast
import math
from pathlib import Path
import statistics
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "exp/decode_kvpre.py"
SOURCE = PATH.read_text()
TREE = ast.parse(SOURCE)
BASE_TREE = ast.parse((ROOT / "v1/decode_v1.py").read_text())


def constant(tree, name):
    return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(v, ast.Name) and v.id == name for v in n.targets))


NAMES = {"_kp_source", "_kp_replace", "_kp_accept", "_kp_summarize", "_kvpre_load", "_kp_equivalent"}
NS = {"_kp_math": math, "_kp_stats": statistics}
exec(compile(ast.Module(body=[n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in NAMES],
                        type_ignores=[]), str(PATH), "exec"), NS)


class GateTest(unittest.TestCase):
    def record(self, **updates):
        return dict(rounds=12, abba=6, baab=6, equiv=True, ratio=0.991,
                    paired=0.991, null=1.0, null_paired=1.0, **updates)

    def test_two_pass_success_and_threshold_boundaries(self):
        r = self.record()
        self.assertTrue(NS["_kp_accept"]([r, r]))
        r.update(ratio=0.992, paired=0.992, null=0.995, null_paired=1.005)
        self.assertTrue(NS["_kp_accept"]([r, r]))

    def test_incomplete_wrong_or_noisy_trials_fail_closed(self):
        good = self.record()
        for records in ([], [good], [good, good, good]):
            self.assertFalse(NS["_kp_accept"](records))
        for key, value in (("rounds", 11), ("abba", 5), ("baab", 5), ("equiv", False),
                           ("ratio", 0.992001), ("paired", 0.992001), ("ratio", -1.0),
                           ("paired", 0.0), ("null", 0.994999), ("null", 1.005001),
                           ("null_paired", 0.994999), ("null_paired", 1.005001)):
            bad = dict(good, **{key: value})
            with self.subTest(key=key, value=value):
                self.assertFalse(NS["_kp_accept"]([good, bad]))
                self.assertFalse(NS["_kp_accept"]([bad, good]))
        for key in ("ratio", "paired", "null", "null_paired"):
            for value in (float("nan"), float("inf"), -float("inf")):
                self.assertFalse(NS["_kp_accept"]([good, dict(good, **{key: value})]))

    def test_medians_and_paired_ratios_are_both_required(self):
        # Closed-form synthetic event samples: every B is exactly 0.99*A.
        blocks = [([100.0, 102.0], [99.0, 100.98]), ([200.0, 202.0], [198.0, 199.98])]
        old, new, ratio, paired = NS["_kp_summarize"](blocks)
        self.assertEqual(old, 151.0)
        self.assertAlmostEqual(new, 149.49)
        self.assertAlmostEqual(ratio, 0.99)
        self.assertAlmostEqual(paired, 0.99)

    def test_repeatability_is_not_a_fixed_tolerance(self):
        # Test only the decision logic with scalar stand-ins, not torch numerics.
        NS.update(torch=SimpleNamespace(equal=lambda a, b: a == b),
                  _kp_bits=lambda a, b: a == b,
                  _kp_error=lambda a, b: (abs(a - b), abs(a - b)))
        refs = [({"hidden": 1.0, "k_new": 0.0}, (1, 2)),
                ({"hidden": 1.125, "k_new": 0.0}, (2, 1))]
        spread = {"hidden": (0.125, 0.125), "k_new": (0.0, 0.0)}
        check = NS["_kp_equivalent"]
        self.assertEqual(check(refs[0], refs, spread), (True, True))
        self.assertEqual(check(refs[1], refs, spread), (True, False))
        self.assertFalse(check(({"hidden": 1.126, "k_new": 0.0}, (1, 2)), refs, spread)[0])
        self.assertFalse(check(({"hidden": 1.0, "k_new": 1e-12}, (1, 2)), refs, spread)[0])
        self.assertFalse(check(({"hidden": 1.0, "k_new": 0.0}, (1, 3)), refs, spread)[0])
        self.assertFalse(check(({"hidden": float("nan"), "k_new": 0.0}, (1, 2)), refs, spread)[0])


class SourceTest(unittest.TestCase):
    def test_incumbent_attention_is_byte_identical_and_file_fits(self):
        self.assertEqual(constant(TREE, "ATT7_CUDA_SRC"), constant(BASE_TREE, "ATT7_CUDA_SRC"))
        self.assertEqual(constant(TREE, "ATT7_CPP_SRC"), constant(BASE_TREE, "ATT7_CPP_SRC"))
        self.assertLess(PATH.stat().st_size, 1_000_000)

    def test_all_load_inline_calls_use_cpp20_sm90a_wrapper(self):
        wrapped = 0
        for n in ast.walk(TREE):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                self.assertNotIn(n.func.id, ("load_inline", "_tvm_load_inline"))
                wrapped += n.func.id == "_kvpre_load"
        self.assertEqual(wrapped, 13)  # 12 inherited builds and one candidate
        raw = ["-O3", "-std=c++17", "-gencode=arch=compute_90,code=sm_90", "--fmad=false"]
        flags = NS["_kvpre_load"](lambda **kw: kw, extra_cflags=["-O2", "-std=c++17"], extra_cuda_cflags=raw)
        self.assertEqual(flags["extra_cflags"], ["-O2", "-std=c++20"])
        self.assertEqual(flags["extra_cuda_cflags"],
                         ["-O3", "--fmad=false", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"])
        self.assertIn("-std=c++17", raw)  # wrapper must not mutate inherited lists

    def test_early_copy_order_and_group_counts(self):
        base = constant(TREE, "ATT7_CUDA_SRC")
        src = NS["_kp_source"](base)
        early = src.index("if constexpr (OKC_KVPRE && !WS && !FLAT")
        wait = src.index("  pdl_wait();   // qkv", early)
        piece = src[early:wait]
        self.assertLess(piece.index("cp_wait<0>()"), piece.index("csync<WS, NT>()"))
        self.assertLess(piece.index("seq_real_ != seq"), piece.index("issue_kv(0, false)"))
        self.assertNotIn("p.qkv", piece)
        self.assertNotIn("cp_commit();", piece)
        self.assertEqual(src.count("cp_commit();"), base.count("cp_commit();"))
        self.assertIn("if (!kvpre) issue_one();", src)
        self.assertIn("!(kvpre && k == 0)", src)
        self.assertIn("#define OKC_KVPRE 0", src)
        self.assertEqual(src.count("cp_wait<IDXF ? 1 : 0>();"), 1)
        self.assertEqual(src.count("cp_wait<STAGES - 1>();"), base.count("cp_wait<STAGES - 1>();"))
        # The contiguous ONE-copy block is moved verbatim, never duplicated.
        a = base.index("      one = true;\n") + len("      one = true;\n")
        b = base.index("#pragma unroll\n      for (int k = 0; k < STAGES - 1; ++k) cp_commit();", a)
        self.assertEqual(src.count(base[a:b]), 1)
        with self.assertRaises(RuntimeError):
            NS["_kp_source"](base + base)

    def test_prologue_completion_model(self):
        # cp.async wait_group N completes every group except the newest N.
        # Model the exact non-WS prologue sequence (no device execution claimed).
        for stages in (3, 4, 6):
            for early in (False, True):
                pending, done = [["idx"]], set()
                def wait(n):
                    while len(pending) > n:
                        done.update(pending.pop(0))
                if early:
                    wait(0)
                    self.assertIn("idx", done)
                pending.append(["q", "new"] + (["kv0"] if early else []))
                wait(1)
                self.assertIn("idx", done)
                for k in range(stages - 1):
                    pending.append([f"idx{k + stages - 1}"] + ([] if early and k == 0 else [f"kv{k}"]))
                wait(stages - 1)
                self.assertTrue({"q", "new"} <= done)
                wait(stages - 2)
                self.assertIn("kv0", done)

    def test_row_partition_has_no_new_row_read_before_wait(self):
        # Exhaust the tile-tail cases, including no cached rows and L=8 straddle.
        for length in (1, 8):
            for seq in range(0, 130):
                for start in range(0, seq + length, 16):
                    end = min(start + 64, seq + length)
                    cached = {r for r in range(64) if start + r < min(end, seq)}
                    new = {r for r in range(64) if seq <= start + r < end}
                    zero = {r for r in range(64) if start + r >= end}
                    self.assertFalse(cached & new)
                    self.assertFalse(cached & zero)
                    self.assertFalse(new & zero)
                    self.assertEqual(cached | new | zero, set(range(64)))


if __name__ == "__main__":
    unittest.main()
