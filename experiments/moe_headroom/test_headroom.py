"""CPU/static diagnostic gates only; does not import Torch or compile CUDA."""
from __future__ import annotations

import ast
import contextlib
import difflib
import io
import math
from pathlib import Path
import random
import re
import struct
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "board_best/moe_m3_scored_2.3797.py"
CAND = HERE / "moe_headroom.py"


def addon_namespace():
    old, new = ast.parse(BASE.read_text()), ast.parse(CAND.read_text())
    addon = ast.Module(body=new.body[len(old.body):], type_ignores=[])
    ns = {"prepare": lambda weights: weights, "__name__": "headroom_offline"}
    exec(compile(addon, str(CAND), "exec"), ns)
    return ns, addon


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def split(total, c, G, ulps=0):
    # Mirror the CUDA small-domain reciprocal + exact product correction.
    reciprocal = f32(1.0 / G)
    bits = struct.unpack("I", struct.pack("f", reciprocal))[0] + ulps
    reciprocal = struct.unpack("f", struct.pack("I", bits))[0]
    n = total * c
    q = int(f32(f32(n) * reciprocal))
    if q * G > n:
        q -= 1
    elif (q + 1) * G <= n:
        q += 1
    assert q == n // G, (total, c, G, ulps)
    return q


def plan(experts, c, G):
    D = len(experts)
    result = []
    for pb, shift in ((0, 9), (1, 8)):
        total = D << shift
        for v in range(split(total, c, G), split(total, c + 1, G)):
            result.append((pb, experts[v >> shift], v & ((1 << shift) - 1)))
    return result


def ring(n, seed):
    rng = random.Random(seed)
    pending = list(range(min(3, n)))
    ready = {}
    issued = set(pending)
    phases = [0] * 3
    for k in range(n):
        while k not in ready:
            completed = pending.pop(rng.randrange(len(pending)))
            phases[completed % 3] ^= 1
            ready[completed] = completed
        assert phases[k % 3] != ((k // 3) & 1)
        assert ready.pop(k) == k
        # All consumers' line samples finish before the CTA barrier releases
        # this slot to the issuer. There is no global or expert-ready flag.
        nxt = k + 3
        if nxt < n:
            assert nxt not in issued
            assert all(p % 3 != nxt % 3 for p in pending)
            pending.append(nxt)
            issued.add(nxt)
    assert not pending and not ready and issued == set(range(n))


class HeadroomTests(unittest.TestCase):
    def test_exact_prefix_and_added_lines_only(self):
        base, cand = BASE.read_bytes(), CAND.read_bytes()
        self.assertTrue(cand.startswith(base))
        self.assertEqual(cand[:len(base)], base)
        changes = list(difflib.unified_diff(BASE.read_text().splitlines(), CAND.read_text().splitlines()))
        self.assertFalse(any(l.startswith("-") and not l.startswith("---") for l in changes))
        self.assertTrue(any(l.startswith("+") and not l.startswith("+++") for l in changes))
        # Preserve inherited base whitespace; newly appended lines are clean.
        self.assertTrue(all(l.rstrip() == l for l in cand[len(base):].decode().splitlines()))

    def test_production_source_and_kernel_fn_not_modified(self):
        ns, addon = addon_namespace()
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name == "kernel_fn" for n in addon.body))
        for n in ast.walk(addon):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                self.assertNotIn(n.id, {"CUDA_SRC", "CPP_SRC", "kernel_fn", "_MOE_SEL", "MAIN_DEFINES", "TINY_DEFINES", "MID_DEFINES"})
        self.assertIn("mbarrier.arrive.expect_tx", ns["_MH_CUDA"])
        self.assertIn("complete_tx::bytes", ns["_MH_CUDA"])
        self.assertEqual(ns["_MH_CUDA"].count("__global__"), 1)
        self.assertEqual(ns["_MH_UNIT"], 36 * 1024)

    def test_byte_counts_and_duplicate_jobs(self):
        ns, _ = addon_namespace()
        counts = [3, 9] + [0] * 30
        mid = ns["_mh_stats"](counts, ("-DOKC_TINY=1", "-DOKC_SPLIT=1"))
        main = ns["_mh_stats"](counts, ("-DOKC_WGS=1",))
        # Exact known NVFP4 sizes: codes 24 MiB + E4M3 scales 3 MiB/expert.
        self.assertEqual(mid["bulk_bytes"], 54 * 2**20)
        self.assertEqual(mid["aux_bytes"], 8192 + 16)
        self.assertEqual((mid["D"], mid["R"], mid["jobs"], mid["dup_jobs"]), (2, 12, 3, 1))
        self.assertEqual(main["dup_jobs"], 0)
        self.assertEqual(ns["_mh_stats"]([0] * 32, ())["bytes"], 0)
        wide = ns["_mh_stats"]([129] + [0] * 31, ("-DOKC_WGS=1",))
        self.assertEqual((wide["jobs"], wide["dup_jobs"], wide["job_bulk_bytes"]), (3, 2, 81 * 2**20))

    def test_extension_declarations_and_exports_match(self):
        ns, addon = addon_namespace()
        for name in ("mh_configure", "mh_run"):
            signatures = []
            for text in (ns["_MH_CPP"], ns["_MH_CUDA"]):
                match = re.search(r"(std::vector<int64_t>|void)\s+" + name + r"\s*\((.*?)\)", text, re.S)
                self.assertIsNotNone(match)
                signatures.append(re.sub(r"\s+", "", match[0]))
            self.assertEqual(*signatures)
        builder = next(n for n in addon.body if isinstance(n, ast.FunctionDef) and n.name == "_mh_ext")
        load = next(n for n in ast.walk(builder) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "load_inline")
        exports = next(kw.value for kw in load.keywords if kw.arg == "functions")
        self.assertEqual(ast.literal_eval(exports), ["mh_configure", "mh_run"])

    def test_same_physical_weight_layouts(self):
        ns, _ = addon_namespace()
        state = {"w": list(range(14))}
        counts = [1, 9] + [0] * 30
        wgs = ns["_mh_sources"](state, ("-DOKC_WGS=1",), counts)
        self.assertEqual(wgs, ([12, 13, 12, 13, 2, 5, 7], 0, True, "wgs"))
        tiny = ns["_mh_sources"](state, ("-DOKC_TINY=1", "-DOKC_L2H=0"), counts)
        self.assertEqual(tiny, ([10, 11, 10, 11, 2, 5, 7], 0, False, "unit"))
        fallback = ns["_mh_sources"](state, (), counts)
        self.assertEqual(fallback, ([10, 11, 8, 9, 2, 5, 7], 2, True, "unit+fat"))

    def test_reciprocal_range_correction(self):
        for D in range(33):
            for G in (1, 3, 17, 132, 160, 256):
                for total in (D * 512, D * 256):
                    for c in range(G + 1):
                        for error in (-2, 0, 2):
                            split(total, c, G, error)

    def test_every_packed_chunk_read_once(self):
        for D in range(33):
            experts = list(range(32 - D, 32))
            expected = {(pb, e, k) for e in experts for pb, count in ((0, 512), (1, 256)) for k in range(count)}
            for G in (1, 3, 17, 132, 256):
                seen = set()
                total = 0
                for c in range(G):
                    for pb, e, k in plan(experts, c, G):
                        self.assertNotIn((pb, e, k), seen)
                        seen.add((pb, e, k))
                        capacity = 32 * (256 if pb else 512) * 36864
                        offset = (e * (256 if pb else 512) + k) * 36864
                        self.assertGreaterEqual(offset, 0)
                        self.assertLessEqual(offset + 36864, capacity)
                        total += 36864
                self.assertEqual(seen, expected)
                self.assertEqual(total, D * 27 * 2**20)

    def test_ring_reuse_and_line_samples(self):
        for seed in range(100):
            ring(seed * 3, seed)
        for n in (0, 1, 2, 3, 4, 65, 256, 1024, 24576):
            ring(n, n)
        lines = [line for tid in range(128) for line in range(tid, 36864 // 128, 128)]
        self.assertEqual(sorted(lines), list(range(288)))
        addresses = [line * 128 + (line & 31) * 4 for line in lines]
        self.assertLess(max(addresses) + 3, 36864)
        for warp in range(4):
            for iteration in range(3):
                active = [warp * 32 + lane + iteration * 128 for lane in range(32)
                          if warp * 32 + lane + iteration * 128 < 288]
                banks = [((line * 128 + (line & 31) * 4) // 4) % 32 for line in active]
                self.assertEqual(len(banks), len(set(banks)))

    def test_balanced_abba_and_deadline(self):
        ns, _ = addon_namespace()
        order = []
        def time_one(which):
            order.append(which)
            return 10.0 if which == "inc" else 8.0
        result = ns["_mh_abba"](time_one, "inc", "stream", 100, rounds=8, clock=lambda: 0)
        self.assertEqual(order, ["inc", "stream", "stream", "inc"] * 8)
        self.assertEqual(result, {"inc_us": 10., "stream_us": 8., "ratio": 1.25, "gap_us": 2., "quads": 8})
        with self.assertRaises(TimeoutError):
            ns["_mh_abba"](time_one, "inc", "stream", 0, clock=lambda: 1)

    def test_read_flush_is_outside_event_interval(self):
        _, addon = addon_namespace()
        measure = next(n for n in addon.body if isinstance(n, ast.FunctionDef) and n.name == "_mh_measure")
        timer = next(n for n in ast.walk(measure) if isinstance(n, ast.FunctionDef) and n.name == "time_one")
        text = ast.unparse(timer)
        self.assertLess(text.index("flush.amax()"), text.index("start.record()"))
        self.assertLess(text.index("start.record()"), text.index("graph.replay()"))
        self.assertLess(text.index("graph.replay()"), text.index("end.record()"))
        self.assertNotIn("zero_", text)
        self.assertNotIn("fill_", text)
        self.assertIn("_moe_routes(M, gen, dev)", ast.unparse(measure))

    def test_budget_skips_before_compilation(self):
        ns, _ = addon_namespace()
        ns["_mh_time"] = SimpleNamespace(monotonic=lambda: 200.)
        ns["_mh_ext"] = lambda: self.fail("compiled despite expired setup budget")
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            ns["_mh_measure"](None, 0.)
        self.assertIn("reason=prepare_budget", text.getvalue())

    def test_prepare_catches_diagnostic_and_logging_failures(self):
        ns, _ = addon_namespace()
        state = object()
        ns["_mh_original_prepare"] = lambda _: state
        def fail(*args):
            raise RuntimeError("diagnostic failure")
        ns["_mh_measure"] = fail
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(ns["prepare"]({}), state)
        ns["print"] = fail
        self.assertIs(ns["prepare"]({}), state)

    def test_weak_summaries_survive_tail_and_board_uses_six_loads(self):
        ns, _ = addon_namespace()
        records = [{"case": c, "D": 12, "R": 24, "inc_us": 100., "stream_us": 90.,
                    "ratio": 100/90, "gap_us": 10., "bytes": 12 * 27 * 2**20 + 8 * 12 + 8192,
                    "job_bulk_bytes": 12 * 27 * 2**20} for c in ("decode16", "decode32")]
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            ns["_mh_report"](records)
        tail = text.getvalue()[-4000:]
        self.assertIn("MOE_HEADROOM_WEAK case=decode16", tail)
        self.assertIn("MOE_HEADROOM_WEAK case=decode32", tail)
        value = float(re.search(r"weak_only_ideal_gain_pct=([\d.]+)", tail)[1])
        self.assertAlmostEqual(value, 100*((100/90)**(2/6)-1), places=3)

    def test_complete_histograms_fit_log_tail_budget(self):
        ns, addon = addon_namespace()
        measure = next(n for n in addon.body if isinstance(n, ast.FunctionDef) and n.name == "_mh_measure")
        line = next(n.args[0] for n in ast.walk(measure)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"
                    and n.args and isinstance(n.args[0], ast.JoinedStr)
                    and any(isinstance(v, ast.Constant) and str(v.value).startswith("MOE_HEADROOM case=") for v in n.args[0].values))
        records, lines = [], []
        for case, M, draws in ns["_MH_CASES"]:
            # Constructed all-local top-8 loads distributed uniformly across
            # 32 experts, with deliberately long timing/layout/flag fields.
            counts = [M // 4] * 32
            stats = ns["_mh_stats"](counts, ())
            times = {"inc_us": 999.999, "stream_us": 888.888, "ratio": 1.125, "gap_us": 111.111, "quads": 8}
            b = stats["bytes"]
            stream_bw = b / (times["stream_us"] * 1e6)
            layout, hint, flag = "unit+fat", True, "above_HBM_peak"
            for draw in range(draws):
                lines.append(eval(compile(ast.Expression(line), "<format>", "eval"), ns, locals()))
                records.append({"case": case, **stats, **times})
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            ns["_mh_report"](records)
        self.assertEqual(len(lines), 28)
        # Reserve 2500 bytes for base selector/build warnings and the header.
        self.assertLess(len("\n".join(lines)) + len(text.getvalue()) + 2500, 16000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
