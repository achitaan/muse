"""Offline gates: no Torch, CUDA toolkit, GPU, submission, or network needed."""
from __future__ import annotations

import ast
from collections import Counter
import contextlib
import difflib
import io
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "board_best/moe_m3_scored_2.3797.py"
CAND = HERE / "moe_stall.py"
BASE_TREE = ast.parse(BASE.read_text())
CAND_TREE = ast.parse(CAND.read_text())


def base_values():
    result = {}
    for n in BASE_TREE.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id in ("CUDA_SRC", "CPP_SRC", "MAIN_DEFINES", "MID_DEFINES", "MAIN_NS", "DQ7", "PDL", "HIDDEN"):
                    result[t.id] = ast.literal_eval(n.value)
    return result


def addon_namespace(prebuild=False, builder=None):
    addon = ast.Module(body=CAND_TREE.body[len(BASE_TREE.body):], type_ignores=[])
    ns = {"prepare": lambda w: w, "_moe_sel_original_prepare": lambda w: w,
          "_MOE_SEL": {}, "load_inline": builder or (lambda *a, **k: None),
          "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: prebuild)), **base_values()}
    exec(compile(addon, str(CAND), "exec"), ns)
    return ns, addon


def synthetic_slab(ns, grid=32, nwarps=18, shift=0):
    idx = ns["_MS_INDEX"]
    slab = [[[[0] * len(idx) for _ in range(3)] for _ in range(nwarps)] for _ in range(grid)]
    for cta in range(grid):
        if cta & 15 != shift & 15:
            continue
        for warp in range(nwarps):
            phases = slab[cta][warp]
            for pb in (0, 1):
                row = phases[pb]
                row[idx["begin_ns"]] = 1000 + 1000 * pb
                row[idx["end_ns"]] = 2000 + 1000 * pb
                for name, value in (("weight_wait", 100), ("activation_wait", 100),
                                    ("dqmma_inclusive", 500), ("deferred", 50),
                                    ("dependency_wait", 50), ("partial_empty_wait", 50),
                                    ("epilogue", 100), ("retire_wait", 30)):
                    row[idx[name]] = value
            final = phases[2]
            for name, value in (("begin_ns", 3000), ("end_ns", 4000), ("entry_ns", 500),
                                ("exit_ns", 4000), ("entry_clock", 600), ("exit_clock", 4100),
                                ("finalizer_wait", 200), ("smid", cta)):
                final[idx[name]] = value
    return slab


class StallTests(unittest.TestCase):
    def test_byte_identical_production_prefix(self):
        base, cand = BASE.read_bytes(), CAND.read_bytes()
        self.assertTrue(cand.startswith(base), "production prefix bytes changed (including CRLF)")
        self.assertEqual(cand[:len(base)], base)
        diff = list(difflib.unified_diff(BASE.read_text().splitlines(), CAND.read_text().splitlines()))
        self.assertFalse(any(x.startswith("-") and not x.startswith("---") for x in diff))
        self.assertTrue(all(x.rstrip() == x for x in cand[len(base):].decode().splitlines()))
        self.assertEqual([ast.dump(n, include_attributes=False) for n in CAND_TREE.body[:len(BASE_TREE.body)]],
                         [ast.dump(n, include_attributes=False) for n in BASE_TREE.body])

    def test_no_production_functions_or_selection_overridden(self):
        _, addon = addon_namespace()
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name in ("kernel_fn", "_ext", "_moe_select", "_moe_routes", "_make_scratch") for n in addon.body))
        protected = {"CUDA_SRC", "CPP_SRC", "_EXTS", "_MOE_SEL", "MAIN_DEFINES", "MID_DEFINES", "TINY_DEFINES", "TINY2_DEFINES"}
        for n in ast.walk(addon):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                self.assertNotIn(n.id, protected)

    def test_source_generation_and_anchor_drift_fail_closed(self):
        ns, _ = addon_namespace()
        src = ns["_ms_source"](ns["CUDA_SRC"])
        self.assertIn("MS_TOCK(msempty, pb, MS_EMPTY)", src)
        self.assertIn("MS_TOCK(msweight, PB, MS_WEIGHT)", src)
        self.assertIn("MS_TOCK(msmath, PB, MS_DQMMA)", src)
        self.assertIn("MS_TOCK(msfinal, 2, MS_FINAL_WAIT)", src)
        self.assertIn("MS_TOCK_ALIAS(msdep, PB, MS_DEPENDENCY, MS_TRANSITION, i == 0)", src)
        self.assertIn("PB && seq == S.wgr[1] - S.wgr[0]", src)
        with self.assertRaises(ValueError):
            ns["_ms_source"](ns["CUDA_SRC"].replace("  const int seq0 = seq;", "  int seq0 = seq;"))
        with self.assertRaises(ValueError):
            ns["_ms_replace"]("xx", "x", "y")

    def test_ring_synchronization_and_arithmetic_not_changed(self):
        ns, _ = addon_namespace()
        old = ns["CUDA_SRC"].split("}  // namespace okc")[0]
        new = ns["_ms_source"](ns["CUDA_SRC"]).split("}  // namespace okc")[0]
        # Every incumbent synchronization, atomic publication, TMA instruction,
        # and MMA instruction is retained, in order, with the same arguments.
        calls = r"\b(?:mbar_wait|mbar_arrive1|mbar_init|wgs_wait|CTA_SYNC|WS_JOIN|__syncwarp|__threadfence|atomicAdd|dq4|mma_bf16|wg_wait|wg_commit|wg_fence)(?:<[^>]*>)?\([^;]*?\);"
        self.assertEqual(re.findall(calls, old), re.findall(calls, new))
        for prefix in ("wgmma.", "cp.async.bulk.", "red.release.", "red.global.", "atom.acq_rel."):
            pattern = r'"(' + re.escape(prefix) + r'[^"\n]+)"'
            self.assertEqual(re.findall(pattern, old), re.findall(pattern, new))
        # No shared-memory enlargement: WGS is already close to the H200 limit.
        def smem(s):
            return s[s.index("struct Smem {"):s.index("// floor(a * c / G)")].split("// Only lane 0")[0].strip()
        self.assertEqual(smem(old), smem(new))

    def test_field_layout_and_single_writer(self):
        ns, _ = addon_namespace()
        enum = re.search(r"enum MsField \{(.*?)\};", ns["_MS_HELPERS"], re.S)[1]
        fields = re.findall(r"\bMS_[A-Z_]+\b", enum)
        self.assertEqual(len(fields), len(ns["_MS_FIELDS"]))
        self.assertEqual(len(fields), 24)
        self.assertEqual(fields[ns["_MS_INDEX"]["weight_issue"]], "MS_ISSUE")
        self.assertEqual(fields[ns["_MS_INDEX"]["entry_clock"]], "MS_ENTRY_CLOCK")
        for grid in (16, 32, 132, 160):
            for nw in (9, 18):
                addresses = set()
                for shift in range(0, 16, 2):
                    selected = [c for c in range(grid) if c & 15 == shift]
                    self.assertTrue(selected)
                    for c in selected:
                        for w in range(nw):
                            for pb in range(3):
                                for m in range(24):
                                    addr = ((c * nw + w) * 3 + pb) * 24 + m
                                    self.assertNotIn(addr, addresses)
                                    self.assertLess(addr, grid * nw * 3 * 24)
                                    addresses.add(addr)
                # Each of the eight disjoint shift draws covers half the CTAs.
                self.assertLessEqual(len(addresses), grid * nw * 3 * 24)
        self.assertIn("threadIdx.x & 31", ns["_MS_HELPERS"])
        self.assertIn("clock64()", ns["_MS_HELPERS"])
        self.assertIn("%%globaltimer", ns["_MS_HELPERS"])
        self.assertIn("for (int i = 0; i < MS_NF * MS_NP; i += 4)", ns["_MS_HELPERS"])

    def test_cpp20_counter_helpers_execute_on_cpu(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("no host C++ compiler")
        ns, _ = addon_namespace()
        helpers = ns["_MS_HELPERS"]
        # Substitute only hardware clock/SM reads. Compile and execute the real
        # addressing, sampling, reset, accumulation, stamps, and alias macros.
        helpers = helpers.replace('asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");', 't = ++fake_clock;')
        helpers = helpers.replace('asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));', 'sm = blockIdx.x;')
        harness = r'''
#include <cassert>
#include <cstddef>
#include <vector>
#define __device__
#define __forceinline__ inline
struct Index { unsigned x; } threadIdx, blockIdx;
constexpr int NTL = 576;
unsigned long long fake_clock = 1000;
unsigned long long clock64() { return fake_clock * 2; }
struct alignas(16) ulonglong4 { unsigned long long x, y, z, w; };
ulonglong4 make_ulonglong4(unsigned long long a, unsigned long long b, unsigned long long c, unsigned long long d) { return {a,b,c,d}; }
struct Params { unsigned long long* stall; int stall_shift; };
'''
        harness += helpers
        harness += r'''
int main() {
  constexpr int SIZE = 32 * 18 * 3 * 24;
  std::vector<unsigned long long> buffer(SIZE + 8, 99);
  Params p{buffer.data() + 4, 2};
  for (int c = 0; c < 32; ++c) for (int w = 0; w < 18; ++w) {
    blockIdx.x = c; threadIdx.x = w * 32;
    ms_init(p);
    if (ms_on(p)) {
      assert(ms_row(p, 0)[MS_WEIGHT] == 0);
      ms_add(p, 0, MS_WEIGHT, 5);
      ms_add(p, 0, MS_WEIGHT, 7);
      assert(ms_row(p, 0)[MS_WEIGHT] == 12);
      assert(ms_row(p, 0)[MS_EVENTS] == 2);
      MS_TICK(start);
      MS_TOCK_ALIAS(start, 1, MS_DEPENDENCY, MS_TRANSITION, true);
      assert(ms_row(p, 1)[MS_DEPENDENCY] == ms_row(p, 1)[MS_TRANSITION]);
      ms_stamp(p, 0, MS_BEGIN_NS); ms_stamp(p, 0, MS_END_NS); ms_exit(p);
      assert(ms_row(p, 2)[MS_EXIT_NS] > ms_row(p, 2)[MS_ENTRY_NS]);
      assert(ms_row(p, 2)[MS_EXIT_CLOCK] > ms_row(p, 2)[MS_ENTRY_CLOCK]);
      ms_init(p); // second graph replay: accumulation must NOT leak
      assert(ms_row(p, 0)[MS_WEIGHT] == 0);
      assert(ms_row(p, 0)[MS_EVENTS] == 0);
    } else assert(ms_row(p, 0)[MS_WEIGHT] == 99);
  }
  for (int i = 0; i < 4; ++i) assert(buffer[i] == 99 && buffer[SIZE + 4 + i] == 99);
  blockIdx.x = 2; threadIdx.x = 1; assert(!ms_on(p));
  threadIdx.x = 0; p.stall = nullptr; assert(!ms_on(p)); ms_init(p); ms_exit(p);
}
'''
        with tempfile.TemporaryDirectory(prefix="cpu_counter_", dir=HERE) as temp:
            source, binary = Path(temp) / "counter.cpp", Path(temp) / "counter"
            source.write_text(harness)
            result = subprocess.run([compiler, "-std=c++20", "-O2", "-Wall", "-Wextra", str(source), "-o", str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_export_signatures_match_and_cpp20_both_builds(self):
        ns, _ = addon_namespace()
        src = ns["_ms_source"](ns["CUDA_SRC"])
        for name, typ in (("ms_run", "void"), ("ms_resources", r"std::vector<int64_t>")):
            def args(text):
                m = re.search(typ + r"\s+" + name + r"\s*\((.*?)\)", text, re.S)
                self.assertIsNotNone(m)
                return [re.sub(r"\s+\w+\s*$", "", x.strip()).replace(" ", "") for x in m[1].split(",") if x.strip()]
            self.assertEqual(args(ns["_MS_CPP"]), args(src))
        captured = []
        ns["_ms_torch_load_inline"] = lambda *a, **kw: captured.append(kw) or object()
        ns["torch"] = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (9, 0)))
        ns.update(_EXTS={}, ALT_SRCS={})
        ext = next(n for n in BASE_TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_ext")
        exec(compile(ast.Module(body=[ext], type_ignores=[]), "<base _ext>", "exec"), ns)
        def builder(*a, **kw):
            captured.append(kw)
            return SimpleNamespace(okc_set_pdl=lambda x: None)
        ns["_ms_torch_load_inline"] = builder
        ns["_ext"](ns["MAIN_DEFINES"] + ("-DOKC_PRMTSC=1",))
        with contextlib.redirect_stdout(io.StringIO()):
            ns["_ms_ext"](ns["MID_DEFINES"] + ("-DOKC_PRMTSC=1",))
        self.assertEqual(len(captured), 2)
        for call in captured:
            for key in ("extra_cflags", "extra_cuda_cflags"):
                self.assertEqual([f for f in call[key] if f.startswith(("-std=", "--std="))], ["-std=c++20"])
            self.assertIn("-gencode=arch=compute_90a,code=sm_90a", call["extra_cuda_cflags"])
        self.assertEqual(captured[0]["cuda_sources"], ns["CUDA_SRC"])
        self.assertIn("void ms_run(", captured[1]["cuda_sources"])
        self.assertEqual(ns["_ms_cxx20_flags"](["-O3", "--std=c++17", "-std", "c++14"]), ["-O3", "-std=c++20"])

    def test_explicit_vs_implicit_mid_producer(self):
        ns, _ = addon_namespace()
        src = ns["_ms_source"](ns["CUDA_SRC"])
        skinny = src[src.index("__device__ void skinny_ring_ws("):src.index("__device__ void ws_helper(")]
        self.assertIn("MS_TOCK_ALIAS(msbar, PB, MS_BARRIER, MS_IMPLICIT_EMPTY, warp == 0)", skinny)
        self.assertNotIn("MS_EMPTY)", skinny)
        self.assertEqual(ns["_ms_role"](ns["MID_DEFINES"], 0), "consumer_refill")
        self.assertEqual(ns["_ms_role"](ns["MID_DEFINES"], 8), "epilogue_helper")
        self.assertEqual(ns["_ms_role"](ns["MAIN_DEFINES"], 16), "weight_producer")
        self.assertEqual(ns["_ms_role"](ns["MAIN_DEFINES"], 17), "activation_producer")

    def test_balanced_timing_and_deadline(self):
        ns, _ = addon_namespace()
        order = []
        def timing(arm):
            order.append(arm)
            return {"inc": 100., "off": 105., "on": 120.}[arm]
        result = ns["_ms_balanced"](timing, "inc", "off", "on", 100, clock=lambda: 0)
        self.assertEqual(order, ["inc", "off", "on", "on", "off", "inc"] * 4)
        self.assertEqual(result["on_ratio"], 1.2)
        self.assertEqual(result["off_ratio"], 1.05)
        self.assertAlmostEqual(result["counter_ratio"], 120/105)
        with self.assertRaises(TimeoutError):
            ns["_ms_balanced"](timing, "inc", "off", "on", 100, clock=lambda: 100)
        with self.assertRaises(ValueError):
            ns["_ms_balanced"](lambda arm: math.nan, "inc", "off", "on", 100, clock=lambda: 0)

    def test_read_flush_and_fresh_inputs_outside_events(self):
        _, addon = addon_namespace()
        measure = next(n for n in addon.body if isinstance(n, ast.FunctionDef) and n.name == "_ms_measure")
        timer = next(n for n in ast.walk(measure) if isinstance(n, ast.FunctionDef) and n.name == "time_one")
        text = ast.unparse(timer)
        self.assertLess(text.index("flush.amax()"), text.index("start.record()"))
        self.assertLess(text.index("start.record()"), text.index("graph.replay()"))
        self.assertLess(text.index("graph.replay()"), text.index("end.record()"))
        self.assertNotIn("fill_", text)
        self.assertNotIn("zero_", text)
        text = ast.unparse(measure)
        self.assertIn("torch.ones(128 << 20", text)
        self.assertIn("_moe_routes(bs, gen, dev)", text)
        self.assertIn("inc_state = dict(state, scratch=inc_scratch)", text)
        self.assertIn("slab = trace.cpu().tolist()", text)
        self.assertEqual(addon_namespace()[0]["_MS_CASES"], ((32, 6), (256, 6), (16, 6)))

    def test_decode_and_manufactured_accounting(self):
        ns, _ = addon_namespace()
        slab = synthetic_slab(ns)
        records = ns["_ms_decode"](slab, 32, 0, ns["MAIN_DEFINES"], 0)
        self.assertEqual(len(records), 36)
        self.assertEqual(Counter(r["role"] for r in records), {"consumer": 32, "weight_producer": 2, "activation_producer": 2})
        row = records[0]
        self.assertEqual(row["phase"][0]["dqmma"], 450)
        self.assertEqual(row["phase"][0]["epilogue_total"], 150)
        p = ns["_ms_partition"](row)
        self.assertEqual(p["total_ns"], 3000)
        self.assertAlmostEqual(p["producer_starved"], 200/3000)
        self.assertAlmostEqual(p["dependency_bound"], 400/3000)
        self.assertAlmostEqual(p["consumer_bound"], 1400/3000)
        self.assertAlmostEqual(p["tail"], 1000/3000)
        self.assertAlmostEqual(sum(p[k] for k in ("producer_starved", "consumer_bound", "dependency_bound", "tail")), 1.)
        slab[0][0][2][ns["_MS_INDEX"]["exit_ns"]] = 0
        with self.assertRaises(ValueError):
            ns["_ms_decode"](slab, 32, 0, ns["MAIN_DEFINES"], 0)

    def test_overflow_clipped_but_reported_and_no_overlap_double_count(self):
        ns, _ = addon_namespace()
        row = ns["_ms_decode"](synthetic_slab(ns), 32, 0, ns["MAIN_DEFINES"], 0)[0]
        row["phase"][0]["weight_wait"] = 1200
        row["phase"][0]["empty_wait"] = 9000000  # producer metric MUST NOT enter partition
        p = ns["_ms_partition"](row)
        self.assertEqual(p["overflow_ns"], 400)
        self.assertAlmostEqual(p["consumer_bound"], 700/3000)
        self.assertAlmostEqual(sum(p[k] for k in ("producer_starved", "consumer_bound", "dependency_bound", "tail")), 1.)

    def test_prepare_catches_diagnostic_and_logging_exceptions(self):
        ns, _ = addon_namespace()
        state = object()
        ns["_ms_original_prepare"] = lambda weights: state
        def fail(*a, **kw):
            raise RuntimeError("diagnostic error")
        ns["_ms_measure"] = fail
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            self.assertIs(ns["prepare"]({}), state)
        self.assertIn("reason=diagnostic_error", text.getvalue())
        ns["print"] = fail
        self.assertIs(ns["prepare"]({}), state)

    def test_budget_skip_before_torch_or_compile_access(self):
        ns, _ = addon_namespace()
        ns["_ms_time"] = SimpleNamespace(monotonic=lambda: 256.)
        ns["_ms_ext"] = lambda *a: self.fail("compiled after setup budget expired")
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            ns["_ms_measure"](None, 0.)
        self.assertIn("reason=incumbent_used_budget", text.getvalue())
        self.assertIn("remaining_s=14.000", text.getvalue())

    def test_prepare_bypasses_selector_and_clears_stale_choices(self):
        ns, _ = addon_namespace()
        self.assertIs(ns["_ms_original_prepare"], ns["_moe_sel_original_prepare"])
        order, state = [], object()
        ns["_MOE_SEL"][32] = ("wrong-build",)
        def base(weights):
            self.assertEqual(ns["_MOE_SEL"], {})
            order.append("default_prepare")
            return state
        def census(s, start):
            self.assertIs(s, state)
            order.append("census")
        ns["_ms_original_prepare"], ns["_ms_measure"] = base, census
        ns["_moe_select"] = lambda *a: self.fail("selector must not run")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertIs(ns["prepare"]({}), state)
        self.assertEqual(order, ["default_prepare", "census"])

    def test_import_prebuild_is_background_and_single_owner(self):
        calls, entered, release = [], threading.Event(), threading.Event()
        module = object()
        def builder(*a, **kw):
            calls.append((threading.current_thread(), kw))
            entered.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test failed to release build")
            return module
        with contextlib.redirect_stdout(io.StringIO()):
            ns, _ = addon_namespace(prebuild=True, builder=builder)
            defines = ns["MAIN_DEFINES"] + ("-DOKC_PRMTSC=1",)
            job = ns["_MS_BUILDS"][defines]
            try:
                self.assertTrue(entered.wait(timeout=5))
                self.assertFalse(job["done"].is_set(), "import must not wait for compile")
                self.assertIs(ns["_ms_start_build"](defines), job)
                self.assertNotEqual(calls[0][0], threading.current_thread())
                self.assertTrue(calls[0][0].daemon)
                self.assertEqual(len(ns["_MS_BUILDS"]), 1, "do not prebuild optional MID")
            finally:
                release.set()
                self.assertTrue(job["done"].wait(timeout=5))
        self.assertIsNone(job["error"])
        self.assertIs(ns["_MS_EXTS"][defines], module)
        self.assertEqual(len(calls), 1)
        self.assertIn("void ms_run(", calls[0][1]["cuda_sources"])
        self.assertIn("-std=c++20", calls[0][1]["extra_cuda_cflags"])

    def test_prebuild_exception_is_published_without_retry(self):
        def fail(*a, **kw):
            raise RuntimeError("compile failed\nsecond line")
        with contextlib.redirect_stdout(io.StringIO()):
            ns, _ = addon_namespace(prebuild=True, builder=fail)
            defines = ns["MAIN_DEFINES"] + ("-DOKC_PRMTSC=1",)
            job = ns["_MS_BUILDS"][defines]
            self.assertTrue(job["done"].wait(timeout=5))
            self.assertNotIn("\n", job["error"])
            ns["_ms_time"] = SimpleNamespace(monotonic=lambda: 185.)
            self.assertIsNone(ns["_ms_wait_build"](defines, 270., 32))
        self.assertIn("RuntimeError: compile failed", job["error"])

    def test_build_wait_is_bounded_and_mid_admission_is_optional(self):
        ns, _ = addon_namespace()
        ns["_ms_time"] = SimpleNamespace(monotonic=lambda: 185.)
        defines = ns["MAIN_DEFINES"] + ("-DOKC_PRMTSC=1",)
        timeouts = []
        ns["_MS_BUILDS"][defines] = {
            "done": SimpleNamespace(wait=lambda timeout: timeouts.append(timeout) or False), "error": None}
        text = io.StringIO()
        with contextlib.redirect_stdout(text):
            self.assertIsNone(ns["_ms_wait_build"](defines, 270., 32))
        self.assertEqual(timeouts, [55.])
        self.assertIn("reason=background_build_timeout", text.getvalue())
        ns["_ms_time"] = SimpleNamespace(monotonic=lambda: 220.)
        ns["_ms_start_build"] = lambda *a: self.fail("optional MID started with <60s")
        with contextlib.redirect_stdout(text):
            self.assertIsNone(ns["_ms_wait_build"](ns["MID_DEFINES"], 270., 16))
        self.assertIn("reason=optional_mid_compile_budget", text.getvalue())
        self.assertIn("remaining_s=50.000", text.getvalue())

    def test_original_185_second_skip_now_waits_for_prebuild(self):
        ns, _ = addon_namespace()
        ns["_ms_time"] = SimpleNamespace(monotonic=lambda: 185.103)
        ns["torch"].cuda.get_device_capability = lambda dev: (9, 0)
        ns["_moe_default"] = lambda bs, sfx: ns["MID_DEFINES" if bs == 16 else "MAIN_DEFINES"] + sfx
        waits = []
        ns["_ms_wait_build"] = lambda defines, deadline, bs: waits.append(bs)
        state = {"w": [SimpleNamespace(device="cuda")], "cnt": 32, "sfx": ("-DOKC_PRMTSC=1",)}
        with contextlib.redirect_stdout(io.StringIO()):
            ns["_ms_measure"](state, 0.)
        self.assertEqual(waits, [32])

    def test_priority_order_and_no_blocking_build_or_process_launcher(self):
        ns, addon = addon_namespace()
        self.assertEqual(ns["_MS_CASES"], ((32, 6), (256, 6), (16, 6)))
        measure = next(n for n in addon.body if isinstance(n, ast.FunctionDef) and n.name == "_ms_measure")
        text = ast.unparse(measure)
        self.assertNotIn("_ms_ext(", text)
        self.assertLess(text.index("_ms_wait_build(defines, deadline, 32)"), text.index("for bs, draws in cases:"))
        self.assertGreater(text.index("_ms_wait_build(defines, deadline, bs)"), text.index("for bs, draws in cases:"))
        # The standalone CHECK file, not the offline host-C++ test harness.
        self.assertNotRegex(CAND.read_text(), r"\b(subprocess|Popen|multiprocessing|ProcessPoolExecutor)\b")
        self.assertLess(ns["_MS_SOFT_LIMIT"], 290)
        self.assertGreaterEqual(ns["_MS_REPORT_RESERVE"], 10)

    def test_census_control_flow_finishes_wgs_before_optional_mid(self):
        # Exercise the real measurement loop, graph/timer flow and budget
        # admission with fake GPU operations. Numerical gates have separate
        # manufactured tests; this is not a substitute for an H200 run.
        class Tensor:
            device = "cuda"
            def __getitem__(self, key): return self
            def __ge__(self, other): return self
            def __lt__(self, other): return self
            def __and__(self, other): return self
            def __sub__(self, other): return self
            def __truediv__(self, other): return 0.0
            def to(self, *a): return self
            def long(self): return self
            def float(self): return self
            def norm(self): return self
            def clamp_min(self, *a): return self
            def amax(self): return self
            def cpu(self): return self
            def tolist(self): return [1] * 32
        tensor = lambda *a, **kw: Tensor()
        for optional in (True, False):
            with self.subTest(optional=optional):
                ns, _ = addon_namespace()
                now, done, builds, reports = [150.], [], [], []
                ns["_ms_time"] = SimpleNamespace(monotonic=lambda: now[0])
                balanced = ns["_ms_balanced"]
                ns["_ms_balanced"] = lambda *a: balanced(*a, clock=lambda: now[0])
                ns["torch"] = SimpleNamespace(
                    cuda=SimpleNamespace(get_device_capability=lambda dev: (9, 0),
                        synchronize=lambda *a: None,
                        CUDAGraph=lambda: SimpleNamespace(replay=lambda: None),
                        graph=lambda g: contextlib.nullcontext(),
                        Event=lambda **kw: SimpleNamespace(record=lambda: None,
                            synchronize=lambda: None, elapsed_time=lambda other: 0.1)),
                    ones=tensor, empty=tensor, empty_like=tensor, randn=tensor,
                    bincount=tensor, float32="fp32", bfloat16="bf16", int64="int64",
                    Generator=lambda **kw: SimpleNamespace(manual_seed=lambda seed: None))
                sfx = ("-DOKC_PRMTSC=1",)
                main, mid = ns["MAIN_DEFINES"] + sfx, ns["MID_DEFINES"] + sfx
                ns["_moe_default"] = lambda bs, s: mid if bs == 16 else main
                ns["_moe_routes"] = lambda *a: (Tensor(), Tensor())
                ns["_make_scratch"] = lambda dev: []
                ns["kernel_fn"] = lambda *a: Tensor()
                module = SimpleNamespace(ms_resources=lambda: [576], ms_run=lambda *a: None)
                def start(defines):
                    if defines not in ns["_MS_BUILDS"]:
                        builds.append((defines, list(done)))
                        ns["_MS_EXTS"][defines] = module
                        ns["_MS_BUILDS"][defines] = {"done": SimpleNamespace(wait=lambda **kw: True), "error": None}
                    return ns["_MS_BUILDS"][defines]
                ns["_ms_start_build"] = start
                start(main)  # import-time prebuild
                def decode(slab, bs, draw, defines, shift):
                    done.append((bs, draw))
                    if not optional and (bs, draw) == (256, 5):
                        now[0] = 220.  # still room to report, but not to build MID
                    return []
                ns["_ms_decode"] = decode
                ns["_ms_report"] = lambda rows, timings: reports.extend(timings)
                state = {"w": [Tensor()], "cnt": 32, "lo": 0, "grid": 132, "sfx": sfx}
                text = io.StringIO()
                with contextlib.redirect_stdout(text):
                    ns["_ms_measure"](state, 0.)
                wgs = [(bs, d) for bs in (32, 256) for d in range(6)]
                expected = wgs + ([(16, d) for d in range(6)] if optional else [])
                self.assertEqual(done, expected)
                self.assertEqual([(t["bs"], t["draw"]) for t in reports], expected)
                if optional:
                    self.assertEqual(builds, [(main, []), (mid, wgs)])
                else:
                    self.assertEqual(builds, [(main, [])])
                    self.assertIn("reason=optional_mid_compile_budget", text.getvalue())

    def test_report_units_skew_and_overhead_are_explicit(self):
        ns, _ = addon_namespace()
        self.assertEqual(ns["_ms_percentile"]([0, 10], .9), 9)
        self.assertEqual(ns["_ms_percentile"]([4], .9), 4)
        rows = ns["_ms_decode"](synthetic_slab(ns), 32, 0, ns["MAIN_DEFINES"], 0)
        times = [{"bs": 32, "draw": 0, "inc_us": 100., "off_us": 105., "on_us": 120.,
                  "off_ratio": 1.05, "on_ratio": 1.2, "counter_ratio": 120/105}]
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            ns["_ms_report"](rows, times)
        text = log.getvalue()
        self.assertIn("MOE_STALL bs=32 phase=A role=consumer metric=dqmma median=0.450 p90=0.450 unit=us", text)
        self.assertIn("metric=elapsed_clock", text)
        self.assertIn("unit=cycles", text)
        self.assertIn("metric=consumer_end_skew", text)
        self.assertIn("metric=sampled_cta_end_lag", text)
        self.assertIn("MOE_STALL_SUMMARY bs=32", text)
        self.assertIn("total_overhead_pct=20.00", text)
        self.assertIn("producer_waits_overlap_NOT_added", text)
        self.assertTrue(text.rstrip().endswith("approximate_NOT_causal_utilization"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
