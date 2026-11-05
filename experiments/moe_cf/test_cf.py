"""CPU-only structural/control-flow gates; not a substitute for CUDA execution."""
from __future__ import annotations

import ast
import contextlib
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
BASE = HERE.parents[1] / "board_best/moe_m3_scored_2.3797.py"
CAND = HERE / "moe_cf.py"
BT, CT = ast.parse(BASE.read_text()), ast.parse(CAND.read_text())


def namespace(prebuild=False, builder=None):
    ns = {"torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: prebuild)),
          "load_inline": builder or (lambda **kw: None), "_moe_sel_original_prepare": lambda w: w}
    for n in BT.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    try:
                        ns[t.id] = ast.literal_eval(n.value)
                    except (ValueError, TypeError):
                        pass
    addon = ast.Module(body=CT.body[len(BT.body):], type_ignores=[])
    exec(compile(addon, str(CAND), "exec"), ns)
    return ns, addon


def sections(src):
    return {
        "wgs": src[src.index("__device__ __forceinline__ void wgs_seg("):src.index("__device__ __forceinline__ void wgs_phase(")],
        "mid": src[src.index("__device__ void skinny_ring_ws("):src.index("__device__ __forceinline__ void ws_issue(", src.index("__device__ void skinny_ring_ws("))],
        "epi": src[src.index("__device__ __forceinline__ void wgs_epi_tile("):src.index("__device__ __forceinline__ void wgs_flush(")],
        "helper": src[src.index("__device__ void ws_helper("):src.index("template <bool BIG, bool PB>", src.index("__device__ void ws_helper("))],
    }


class CfTests(unittest.TestCase):
    def test_verbatim_prefix_and_ast(self):
        self.assertTrue(CAND.read_bytes().startswith(BASE.read_bytes()))
        self.assertEqual([ast.dump(n) for n in BT.body], [ast.dump(n) for n in CT.body[:len(BT.body)]])
        _, addon = namespace()
        self.assertFalse(any(isinstance(n, ast.FunctionDef) and n.name in ("kernel_fn", "_ext", "_moe_routes", "_make_scratch") for n in addon.body))
        protected = {"CUDA_SRC", "CPP_SRC", "MAIN_DEFINES", "MID_DEFINES", "_EXTS", "_MOE_SEL"}
        self.assertFalse(any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and n.id in protected for n in ast.walk(addon)))

    def test_source_anchors_fail_closed(self):
        ns, _ = namespace()
        for v in ns["_CF_VARIANTS"]:
            src = ns["_cf_source"](ns["CUDA_SRC"], v)
            self.assertIn("uint32_t* cf_sink;", src)
            with self.assertRaises(ValueError):
                ns["_cf_source"](ns["CUDA_SRC"].replace("  unsigned long long* dbg;", "  int* dbg;"), v)
        with self.assertRaises(ValueError):
            ns["_cf_source"](ns["CUDA_SRC"], "oops")
        with self.assertRaises(ValueError):
            ns["_cf_replace"]("xx", "x", "y")

    def test_untouched_ring_protocol_routing_and_smem(self):
        ns, _ = namespace()
        original = ns["CUDA_SRC"]
        old = sections(original)
        pattern = r"\b(?:mbar_wait|mbar_arrive1|CTA_SYNC|__syncwarp|__threadfence|atomicAdd|ld_acquire|WS_JOIN|wgs_wait|ws_issue)(?:<[^>]*>)?\([^;]*?\);"
        for v in ns["_CF_VARIANTS"]:
            src = ns["_cf_source"](original, v)
            new = sections(src)
            for section in ("wgs", "mid", "helper"):
                def protocol(s):
                    return [x for x in re.findall(pattern, s) if "atomicAdd(&ss_cur()" not in x]
                self.assertEqual(protocol(old[section]), protocol(new[section]), (v, section))
            for begin, end in (("struct Smem {", "// floor(a * c / G)"),
                               ("__device__ void wgs_wproducer(", "// Work left from the previous segment")):
                a = original[original.index(begin):original.index(end)]
                b = src[src.index(begin):src.index(end)]
                if begin == "struct Smem {":
                    b = b[:b.index("// Observable sink")].rstrip() + "\n"
                    self.assertEqual(a.rstrip(), b.rstrip())
                else:
                    self.assertEqual(a, b)
            # All publication and split-unit protocols remain literally intact.
            a = original[original.index("__device__ __forceinline__ void wgs_flush("):original.index("// one segment")]
            b = src[src.index("__device__ __forceinline__ void wgs_flush("):src.index("// one segment")]
            self.assertEqual(a, b)

    def test_variant_semantics(self):
        ns, _ = namespace()
        for variant in ns["_CF_VARIANTS"]:
            src = ns["_cf_source"](ns["CUDA_SRC"], variant)
            sec = sections(src)
            nomma = variant in ("NOMMA", "STREAM")
            nodq = variant in ("NODQ", "STREAM")
            noepi = variant in ("NOEPI", "STREAM")
            for key in ("mid", "wgs"):
                self.assertEqual("dq4(" not in sec[key], nodq, (variant, key))
                self.assertEqual("mma_bf16(" not in sec[key] and "WGSMMA<" not in sec[key], nomma, (variant, key))
            self.assertEqual("wg_wait<" not in sec["wgs"], nomma)
            self.assertEqual("st_keep_" not in sec["epi"], noepi)
            self.assertEqual("achita_marlin_out2" not in sec["helper"], noepi)
            self.assertEqual("return;  // CF: flags" in src, variant in ("NOFIN", "STREAM"))
            if nomma:
                self.assertIn("cf_save(p, cf_acc);", sec["mid"])
                self.assertIn("cf_save(p, cf_acc);", sec["wgs"])
            if variant == "NOMMA":
                self.assertIn("cf_acc = cf_mix(cf_acc, af[", sec["wgs"])
                self.assertIn("cf_acc = cf_mix(cf_acc, bmul2(", sec["mid"])
            if variant == "STREAM":
                self.assertIn("cf_mix(cf_acc, q, scg.x, scg.y", sec["wgs"])
                self.assertIn("cf_mix(cf_acc, ww[st]", sec["mid"])
                self.assertNotIn("bmul2", sec["wgs"])
                self.assertNotIn("SCDK(", sec["mid"])
            if variant == "NODQ":
                self.assertIn("cf_dummy(q, scg.x, scg.y)", sec["wgs"])
                self.assertIn("cf_dummy(ww[st], sc.x ^ sc.z, sc.y ^ sc.w)", sec["mid"])

    def test_cpp_export_signatures_match(self):
        ns, _ = namespace()
        for v in ns["_CF_VARIANTS"]:
            src = ns["_cf_source"](ns["CUDA_SRC"], v)
            for fn in ("okc_run", "okc_set_pdl"):
                def args(s):
                    m = re.search(r"void " + fn + r"\((.*?)\)", s, re.S)
                    return [re.sub(r"\s+\w+\s*$", "", x.strip()).replace(" ", "") for x in m[1].split(",")]
                self.assertEqual(args(ns["_CF_CPP"]), args(src))

    def test_cuda_preprocessor_all_variants_and_builds(self):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("clang++ unavailable")
        ns, _ = namespace()
        with tempfile.TemporaryDirectory(dir=HERE, prefix="cf_preprocess_") as temp:
            root = Path(temp)
            for path in ("cuda_bf16.h", "cuda_runtime.h", "torch/extension.h", "ATen/cuda/CUDAContext.h"):
                f = root / path
                f.parent.mkdir(parents=True, exist_ok=True)
                f.touch()
            for variant in ns["_CF_VARIANTS"]:
                f = root / (variant + ".cpp")
                f.write_text(ns["_cf_source"](ns["CUDA_SRC"], variant))
                for build in ("MAIN_DEFINES", "MID_DEFINES"):
                    cmd = [compiler, "-E", "-P", "-std=c++20", "-I", str(root), *ns[build],
                           "-DOKC_PRMTSC=1", "-DOKC_DQ7=1", "-DOKC_NS=3", "-DOKC_NSF=3", str(f)]
                    result = subprocess.run(cmd, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, (variant, build, result.stderr))
                    # Preprocessed active branch must be balanced even though
                    # actual CUDA types/PTX cannot be compiled on this machine.
                    text = re.sub(r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/', "", result.stdout, flags=re.S)
                    for left, right in (("{", "}"), ("(", ")"), ("[", "]")):
                        level = 0
                        for ch in text:
                            level += (ch == left) - (ch == right)
                            self.assertGreaterEqual(level, 0, (variant, build, left))
                        self.assertEqual(level, 0, (variant, build, left))

    def test_abba_12_rounds_pairing_and_null(self):
        ns, _ = namespace()
        order = []
        result = ns["_cf_abba"](lambda g: order.append(g) or {"A": 100., "B": 70.}[g], "A", "B", 100, clock=lambda: 0)
        self.assertEqual(order, ["A", "B", "B", "A"] * 12)
        self.assertEqual(result, {"base_us": 100., "var_us": 70., "saved_us": 30., "paired_ratio": .7})
        result = ns["_cf_abba"](lambda g: 100., "A", "B", 100, clock=lambda: 0)
        self.assertEqual(result["paired_ratio"], 1.)
        with self.assertRaises(TimeoutError):
            ns["_cf_abba"](lambda g: self.fail("timed after deadline"), "A", "B", 100, clock=lambda: 100)
        for bad in (math.nan, math.inf, 0., -1.):
            with self.assertRaises(ValueError):
                ns["_cf_abba"](lambda g: bad, "A", "B", 100, clock=lambda: 0)

    def test_drift_cancels_in_paired_abba(self):
        ns, _ = namespace()
        t = iter(range(48))
        result = ns["_cf_abba"](lambda g: 100. + next(t) + (0 if g == "A" else -20.), "A", "B", 1, clock=lambda: 0)
        self.assertEqual(result["saved_us"], 20.)

    def test_background_build_single_owner_and_flags(self):
        ns, _ = namespace()
        entered, release, calls = threading.Event(), threading.Event(), []
        def builder(**kw):
            calls.append((threading.current_thread(), kw))
            entered.set()
            self.assertTrue(release.wait(timeout=5))
            return SimpleNamespace(okc_set_pdl=lambda value: None)
        ns["_cf_native_load_inline"] = builder
        defines = ns["MAIN_DEFINES"] + ("-DOKC_PRMTSC=1",)
        with contextlib.redirect_stdout(io.StringIO()):
            job = ns["_cf_start_build"](defines, "NODQ")
            try:
                self.assertTrue(entered.wait(timeout=5))
                self.assertIs(job, ns["_cf_start_build"](defines, "NODQ"))
                self.assertFalse(job["done"].is_set())
            finally:
                release.set()
                self.assertTrue(job["done"].wait(timeout=5))
        self.assertIsNone(job["error"])
        self.assertEqual(len(calls), 1)
        thread, kw = calls[0]
        self.assertNotEqual(thread, threading.current_thread())
        self.assertTrue(thread.daemon)
        for key in ("extra_cflags", "extra_cuda_cflags"):
            self.assertEqual([f for f in kw[key] if f.startswith("-std=")], ["-std=c++20"])
        self.assertIn("-gencode=arch=compute_90a,code=sm_90a", kw["extra_cuda_cflags"])

    def test_import_prebuild_prioritizes_main(self):
        ns, _ = namespace()
        pending, calls = [], []
        class Done:
            def wait(self):
                calls.append("wait")
        def start(defines, variant):
            calls.append(("MAIN" if "-DOKC_WGS=1" in defines else "MID", variant))
            return {"done": Done()}
        ns["_cf_start_build"] = start
        ns["_cf_threading"] = SimpleNamespace(Thread=lambda **kw: SimpleNamespace(start=lambda: pending.append(kw["target"])))
        ns["_cf_prebuild"]()
        self.assertEqual(calls, [("MAIN", v) for v in ns["_CF_VARIANTS"]])
        pending[0]()
        self.assertEqual(calls[5:10], ["wait"] * 5)
        self.assertEqual(calls[10:], [("MID", v) for v in ns["_CF_VARIANTS"]])

    def test_failure_containment_and_selector_bypass(self):
        ns, _ = namespace()
        state, order = object(), []
        def original(weights):
            self.assertEqual(ns["_MOE_SEL"], {})
            order.append("base")
            return state
        def diagnostic(s, start):
            self.assertIs(s, state)
            order.append("diag")
            raise RuntimeError("bad diagnostic")
        ns["_MOE_SEL"][32] = "wrong"
        ns["_cf_original_prepare"], ns["_cf_measure"] = original, diagnostic
        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertIs(ns["prepare"]({}), state)
        self.assertEqual(order, ["base", "diag"])
        self.assertIn("reason=diagnostic_error", log.getvalue())

    def test_budget_skips_before_gpu_access(self):
        ns, _ = namespace()
        ns["_cf_time"] = SimpleNamespace(monotonic=lambda: 269.)
        with contextlib.redirect_stdout(io.StringIO()) as log:
            ns["_cf_measure"](None, 0.)
        self.assertIn("reason=incumbent_used_budget", log.getvalue())
        self.assertLess(ns["_CF_LIMIT"], 290)

    def test_read_flush_and_same_routes_static(self):
        ns, addon = namespace()
        measure = next(n for n in addon.body if isinstance(n, ast.FunctionDef) and n.name == "_cf_measure")
        timer = next(n for n in ast.walk(measure) if isinstance(n, ast.FunctionDef) and n.name == "time_one")
        text = ast.unparse(timer)
        self.assertLess(text.index("flush.amax()"), text.index("a.record()"))
        self.assertLess(text.index("a.record()"), text.index("graph.replay()"))
        self.assertLess(text.index("graph.replay()"), text.index("b.record()"))
        text = ast.unparse(measure)
        self.assertEqual(text.count("_moe_routes(bs, gen, dev)"), 1)
        self.assertIn("range(6)", text)
        self.assertIn("torch.ones(128 << 20", text)
        self.assertIn("scratches[v]", text)
        self.assertIn("_ext(defines)", text)
        self.assertNotIn("_cf_native_load_inline(", text)
        self.assertEqual(ns["_CF_CASES"], (32, 16, 256))

    def test_summary_known_subcosts_and_partial_draw_counts(self):
        ns, _ = namespace()
        rows = []
        for v, saving in (("BASE", 0), ("NODQ", 1), ("NOMMA", 20), ("NOEPI", 3), ("NOFIN", 2), ("STREAM", 30)):
            rows.append(dict(bs=32, variant=v, base_us=100., var_us=100.-saving,
                             saved_us=float(saving), paired_ratio=(100.-saving)/100., null_ratio=1.))
        with contextlib.redirect_stdout(io.StringIO()) as log:
            ns["_cf_report"](rows)
        text = log.getvalue()
        self.assertIn("MOE_CF bs=32 variant=NOMMA base_us=100.000000 var_us=80.000000 saved_us=20.000000", text)
        self.assertIn("draws=1/6 rounds=12", text)
        self.assertIn("mma_minus_dq_us=19.000000", text)
        self.assertIn("interpretation=mma_issue_wait_candidate", text)
        self.assertIn("nonadditive_counterfactuals=1", text)

    def test_full_measure_control_flow_with_fake_gpu(self):
        ns, _ = namespace()
        calls, draws, rows = [], [], []
        class Tensor:
            device = "cuda"
            def __init__(self, shape=(32, 4096)): self.shape = shape
            def __getitem__(self, key): return self
            def __ge__(self, other): return self
            def __lt__(self, other): return self
            def __and__(self, other): return self
            def __sub__(self, other): return self
            def __truediv__(self, other): return 0.
            def to(self, *a): return self
            def long(self): return self
            def float(self): return self
            def norm(self): return self
            def clamp_min(self, *a): return self
            def amax(self): return self
            def cpu(self): return self
            def tolist(self): return [1] * 32
        tensor = lambda shape=None, **kw: Tensor(shape)
        ns["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(get_device_capability=lambda dev: (9, 0), synchronize=lambda *a: None,
                CUDAGraph=lambda: SimpleNamespace(replay=lambda: None), graph=lambda g: contextlib.nullcontext(),
                Event=lambda **kw: SimpleNamespace(record=lambda: None, synchronize=lambda: None, elapsed_time=lambda b: .1)),
            ones=tensor, empty=tensor, empty_like=lambda x: Tensor(x.shape), randn=tensor, bincount=tensor,
            float32="float", int32="int", bfloat16="bf16", Generator=lambda **kw: SimpleNamespace(manual_seed=lambda s: None))
        # manual_seed normally returns the generator itself.
        ns["torch"].Generator = lambda **kw: SimpleNamespace(manual_seed=lambda s: object())
        ns["_cf_time"] = SimpleNamespace(monotonic=lambda: 1.)
        ns["_moe_default"] = lambda bs, sfx: ns["MID_DEFINES" if bs == 16 else "MAIN_DEFINES"] + sfx
        ns["_make_scratch"] = lambda dev: [object()]
        def route(bs, gen, dev):
            ids, tw = Tensor(), Tensor()
            draws.append((bs, ids, tw))
            return ids, tw
        ns["_moe_routes"] = route
        def module(v):
            return SimpleNamespace(okc_run=lambda *args: calls.append((v, args)))
        ns["_ext"] = lambda defines: module("BASE")
        ns["_cf_start_build"] = lambda defines, v: {"done": SimpleNamespace(wait=lambda **kw: True), "error": None, "module": module(v)}
        ns["kernel_fn"] = lambda data, state: Tensor(data["hidden_states"].shape)
        ns["_cf_report"] = lambda rr: rows.extend(rr)
        state = {"w": [Tensor()], "lo": 0, "cnt": 32, "grid": 132, "sfx": ("-DOKC_PRMTSC=1",), "scratch": [object()]}
        with contextlib.redirect_stdout(io.StringIO()):
            ns["_cf_measure"](state, 0.)
        self.assertEqual([d[0] for d in draws], [bs for bs in (32, 16, 256) for _ in range(6)])
        self.assertEqual(len(rows), 18 * 6)
        self.assertTrue(all(r["paired_ratio"] == 1. for r in rows))
        for bs, ids, tw in draws:
            same = [(v, args) for v, args in calls if args[3] is ids]
            self.assertEqual(len(same), 13)  # 6 eager, 7 captures incl BASE null
            self.assertTrue(all(args[4] is tw and args[2].shape[0] == bs for _, args in same))
            scratch_by_arm = {}
            for v, args in same:
                scratch_by_arm.setdefault(v, args[1])
                self.assertIs(scratch_by_arm[v], args[1])
                self.assertIsNot(args[1], state["scratch"])
            self.assertEqual(len({id(s) for s in scratch_by_arm.values()}), 6)

    def test_no_forbidden_processes_or_graph_apis(self):
        self.assertNotRegex(CAND.read_text(), r"\b(subprocess|Popen|multiprocessing|ProcessPoolExecutor)\b")
        self.assertNotRegex(CAND.read_text(), r"\b(?:cuda|cu)Graph\w*_v[23]\b")


if __name__ == "__main__":
    unittest.main(verbosity=2)
