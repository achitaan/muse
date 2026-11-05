"""Dependency-free structural/admission checks; CUDA correctness needs the judge."""
import ast
import math
from pathlib import Path
import unittest


FILE = Path(__file__).with_name("decode_attnfull.py")
TREE = ast.parse(FILE.read_text())


def functions(*names, **ns):
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(FILE), "exec"), ns)
    return ns


def literal(name):
    for node in TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class AttentionFullTests(unittest.TestCase):
    def test_generated_source_and_unchanged_flat_producer(self):
        src = literal("ATT7_CUDA_SRC")
        ns = functions("_kp_replace", "_kp_source")
        got = ns["_kp_source"](src)
        self.assertEqual(got.count("if (k < ntiles) issue_kv(k, false);"), 1)
        self.assertIn("if (k < PRE && k < ntiles && !kvpre) issue_kv(k, false);", got)
        self.assertIn("if constexpr (OKC_KVPRE && !WS && !FLAT && !Cfg<L, FLAT>::S2)", got)
        a, b = "    if constexpr (ROLE == 1) {", "  if constexpr (ROLE != 1) {"
        self.assertEqual(src[src.index(a):src.index(b)], got[got.index(a):got.index(b)])
        self.assertIn("std::vector<int64_t> afull_resources()", got)
        # Original ring depth and barrier count remain unchanged.
        self.assertIn("#define OKC_ST1F 3", got)
        self.assertIn("#define OKC_STN 3", got)
        self.assertEqual(src.count("cp_commit();"), got.count("cp_commit();"))

    def test_every_tile_gathered_once(self):
        # Model the changed gather dispatch, including the ONE and short cases.
        for one in (False, True):
            for ntiles in range(1, 17):
                if one and ntiles != 1:
                    continue
                pre = list(range(min(2, ntiles)))
                # Post-wait prologue skips all pre-issued stages, the unchanged
                # main loop starts its refills at tile STAGES-1.
                post = list(range(2, ntiles))
                self.assertEqual(pre + post, list(range(ntiles)))

    def test_gate_boundaries(self):
        accept = functions("_kp_accept", _kp_math=math)["_kp_accept"]
        record = dict(rounds=12, abba=6, baab=6, equiv=True,
                      ratio=0.992, paired=0.992, null=0.995, null_paired=1.005)
        self.assertTrue(accept([record, record]))
        self.assertFalse(accept([record]))
        for key, bad in (("rounds", 11), ("abba", 5), ("baab", 5), ("equiv", False),
                         ("ratio", 0.99201), ("paired", 0.99201), ("null", 0.99499),
                         ("null_paired", 1.00501), ("ratio", float("nan"))):
            self.assertFalse(accept([record, dict(record, **{key: bad})]), key)

    def test_only_full_b1_candidate_and_pdl_plans(self):
        self.assertEqual(literal("_KP_KEYS"), ((1, 1, False), (1, 8, False)))
        ns = functions("_plan7", "_pdl", _num_sms7=lambda dev: 132,
                       PDL=True, PDL_MAX_T=128, OKC_F2=0, OKC_S2=0,
                       OKC_FLAT1B=0, OKC_PLAIN8=1)
        for batch in (16, 32):
            self.assertTrue(ns["_pdl"](batch))
            self.assertEqual(ns["_plan7"](batch, 1, 1, 0, 86000, 1, None)["flat"], 1)
        for qlen in (1, 8):
            self.assertEqual(ns["_plan7"](1, 1, qlen, 0, 86000, 1, None)["flat"], 0)

    def test_compiler_flags_and_size(self):
        load = functions("_kvpre_load")["_kvpre_load"]
        got = load(lambda **kw: kw, extra_cflags=["-O3", "-std=c++17"],
                   extra_cuda_cflags=["-O3", "-std=c++17", "-arch=sm_90"])
        self.assertIn("-std=c++20", got["extra_cflags"])
        self.assertIn("-std=c++20", got["extra_cuda_cflags"])
        self.assertIn("-gencode=arch=compute_90a,code=sm_90a", got["extra_cuda_cflags"])
        self.assertNotIn("-std=c++17", got["extra_cuda_cflags"])
        self.assertLess(FILE.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
