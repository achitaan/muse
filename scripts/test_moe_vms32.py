"""CPU/static gate tests; CUDA compilation, latency and numerics need the judge.

Run: python3 scripts/test_moe_vms32.py
"""
import ast
import contextlib
import io
import math
import types
import unittest
from pathlib import Path

import test_moe_vmsgate as shared

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "candidates/moe_vms32.py").read_text()
TREE = ast.parse(SOURCE)
# Reuse the exact prior byte-permutation and fake-event harness on THIS source.
shared.SOURCE, shared.TREE = SOURCE, TREE
namespace = shared.namespace


def rows(count=24, ratio=0.98):
    # These synthetic scalar rows encode the user's admission specification,
    # not any measured CUDA performance or floating-point result.
    return [dict(numeric=True, null=1.0, ratio=ratio) for _ in range(count)]


class GateTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()

    def test_main_and_independent_confirmation_counts(self):
        admit = self.ns["_vmsgate_admit"]
        self.assertEqual(self.ns["VMSGATE_DRAWS"], 24)
        self.assertEqual(self.ns["VMSGATE_CONFIRM"], 8)
        self.assertTrue(admit(rows(), 24, 0.990))
        self.assertFalse(admit(rows(23), 24, 0.990))
        self.assertTrue(admit(rows(8, 0.994), 8, 0.995))
        self.assertFalse(admit(rows(8, 0.996), 8, 0.995))
        self.assertFalse(admit(rows(7), 8, 0.995))

    def test_geomean_and_median_are_both_required(self):
        admit = self.ns["_vmsgate_admit"]
        self.assertFalse(admit(rows(ratio=0.992), 24, 0.990))
        r = rows()
        # Analytically sqrt(0.9 * 1.0) < .990, but the median is 1.0.
        for row in r[:13]:
            row["ratio"] = 1.0
        for row in r[13:]:
            row["ratio"] = 0.9
        self.assertLess(self.ns["_vmsgate_gmean"]([v["ratio"] for v in r]), 0.990)
        self.assertFalse(admit(r, 24, 0.990))

    def test_every_null_must_pass_no_cancelling(self):
        admit = self.ns["_vmsgate_admit"]
        for value in (0.994999, 1.005001, math.nan, math.inf):
            r = rows()
            r[7]["null"] = value
            self.assertFalse(admit(r, 24, 0.990))
        for value in (0.995, 1.005):
            r = rows()
            r[7]["null"] = value
            self.assertTrue(admit(r, 24, 0.990))
        r = rows()
        r[0]["null"], r[1]["null"] = 0.99, 1.01
        self.assertFalse(admit(r, 24, 0.990))

    def test_invalid_draw_or_time_veto(self):
        for field, values in (("numeric", (False,)), ("ratio", (0.0, -1.0, math.nan, math.inf))):
            for value in values:
                r = rows()
                r[-1][field] = value
                self.assertFalse(self.ns["_vmsgate_admit"](r, 24, 0.990))

    def test_numeric_exact_requested_bounds(self):
        numeric = self.ns["_vmsgate_numeric"]
        # Expected inequalities: user 1.10x incumbent L2; TASK global .015,
        # row .075, with finite values and the kernel's sticky error flag zero.
        self.assertTrue(numeric(0.011, 0.075, 0.01, 0.07, 0))
        self.assertFalse(numeric(0.011000001, 0.075, 0.01, 0.07, 0))
        self.assertTrue(numeric(0.015, 0.075, 0.015, 0.075, 0))
        self.assertFalse(numeric(0.01500001, 0.01, 0.015, 0.01, 0))
        self.assertFalse(numeric(0.001, 0.07500001, 0.001, 0.01, 0))
        self.assertFalse(numeric(0.001, 0.01, 0.001, 0.01, 1))
        self.assertTrue(numeric(0, 0, 0, 0, 0))
        self.assertFalse(numeric(1e-12, 0, 0, 0, 0))
        for index in range(4):
            for value in (math.nan, math.inf, -1.0):
                args = [0.001, 0.01, 0.001, 0.01, 0]
                args[index] = value
                self.assertFalse(numeric(*args))

    def test_correct_uses_same_reference_for_both_outputs(self):
        seen, ns = [], self.ns
        def error(out, ref):
            seen.append((out, ref))
            return {"vms": (0.0027, 0.006), "base": (0.0025, 0.005)}[out]
        ns["_vmsgate_error"] = error
        result = ns["_vmsgate_correct"]("reference", "base", "vms", {"err": types.SimpleNamespace(item=lambda: 0)})
        self.assertEqual(seen, [("vms", "reference"), ("base", "reference")])
        self.assertTrue(result["numeric"])
        self.assertEqual(result["err_vms"], 0.0027)
        self.assertEqual(result["err_inc"], 0.0025)


class StaticTests(unittest.TestCase):
    def test_cuda_and_cpp_are_byte_identical_to_parent(self):
        parent = ast.parse((ROOT / "candidates/moe_vmsgate.py").read_text())
        def strings(tree):
            return {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                    and n.targets[0].id in ("CUDA_SRC", "CPP_SRC", "VMS_SRC", "VMS_CPP")}
        self.assertEqual(strings(TREE), strings(parent))

    def test_only_sm90a_cpp20_builds_and_no_process_starts(self):
        builds = [n for n in ast.walk(TREE) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Name) and n.func.id == "load_inline"]
        self.assertEqual(len(builds), 3)
        for build in builds:
            keys = {k.arg: k.value for k in build.keywords}
            for key in ("extra_cflags", "extra_cuda_cflags"):
                constants = [n.value for n in ast.walk(keys[key]) if isinstance(n, ast.Constant)]
                self.assertIn("-std=c++20", constants)
            self.assertIn("-gencode=arch=compute_90a,code=sm_90a",
                          [n.value for n in ast.walk(keys["extra_cuda_cflags"]) if isinstance(n, ast.Constant)])
        for forbidden in ("subprocess", "Popen", "multiprocessing", "os.system", "os.fork", "posix_spawn"):
            self.assertNotIn(forbidden, SOURCE)

    def test_no_exactness_fit_or_bs16_build(self):
        suffix = SOURCE[SOURCE.index("# === MOE_VMS32:"):]
        for removed in ("_vmsgate_bits", "_vmsgate_fit", "VMSGATE_TRAIN", "VMSGATE_SHAPES"):
            self.assertNotIn(removed, suffix)
        self.assertIn("pool.submit(_vmsgate_ext, 32)", suffix)
        self.assertIn('"MOE_VMS32_PICK', suffix)
        self.assertIn('"MOE_VMS32 phase=', suffix)
        self.assertIn("1700000 + 1009 * 32", suffix)
        self.assertIn("1100000 + 1009 * 32", suffix)

    def test_bs16_never_dispatches_vms_and_bs32_output_is_fresh(self):
        ns, trace = namespace(), []
        kernel = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn"][-1]
        exec(compile(ast.Module(body=[kernel], type_ignores=[]), "actual_wrapper", "exec"), ns)
        class Tensor:
            device = "cuda"
            def __init__(self, shape): self.shape = shape
            def contiguous(self): return self
            def to(self, _): return self
            def data_ptr(self): return 16
        ns.update(torch=types.SimpleNamespace(empty=lambda shape, **_: Tensor(shape), bfloat16="bf16", int32="i32", float32="f32"),
                  HIDDEN=4096, _bind_context=lambda _: None,
                  _vmsgate_parent_kernel=lambda data, state: "incumbent",
                  _vmsgate_call=lambda ext, vs, data, out: trace.append(out))
        gate = dict(adopt={16: True, 32: True}, vs=dict(lo=0, nloc=32), ext={32: None})
        def data(m, lo=0):
            return dict(hidden_states=Tensor((m, 4096)), topk_ids=Tensor((m, 8)), topk_weights=Tensor((m, 8)),
                        config=dict(local_start=lo, local_count=32))
        for m in (1, 4, 8, 16, 31, 33, 128, 256):
            self.assertEqual(ns["kernel_fn"](data(m), dict(vmsgate=gate)), "incumbent")
        self.assertEqual(ns["kernel_fn"](data(32, 1), dict(vmsgate=gate)), "incumbent")
        self.assertEqual(ns["kernel_fn"](data(32), {}), "incumbent")
        a = ns["kernel_fn"](data(32), dict(vmsgate=gate))
        b = ns["kernel_fn"](data(32), dict(vmsgate=gate))
        self.assertIsNot(a, b)
        self.assertEqual(a.shape, (32, 4096))
        self.assertEqual(len(trace), 2)


class PrepareTests(unittest.TestCase):
    def run_prepare(self, main_ratio=0.989, confirm_ratio=0.994, stress=True, flag=0, timeout=False):
        ns, calls = namespace(), []
        prepare = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "prepare"][-1]
        exec(compile(ast.Module(body=[prepare], type_ignores=[]), "actual_prepare", "exec"), ns)
        marker = types.SimpleNamespace(device="cuda", item=lambda: flag)
        ext = types.SimpleNamespace(vms_attr=lambda *args: [1, 100, 0, 0, 1024])
        backend = types.SimpleNamespace(allow_tf32=True)
        ns.update(torch=types.SimpleNamespace(
            cuda=types.SimpleNamespace(get_device_capability=lambda _: (9, 0)),
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=backend)),
            empty=lambda *a, **k: marker, zeros=lambda *a, **k: marker,
            bfloat16="bf16", float32="f32", int32="i32"), HIDDEN=4096,
            _vmsgate_parent_prepare=lambda _: dict(sfx=["-DOKC_PRMTSC=1"], w=[0] * 12, lo=0, cnt=32, grid=132),
            _vmsgate_ext=lambda m: ext, _vmsgate_repack=lambda *args: marker,
            _vmsgate_stress=lambda *args: stress)
        def draws(state, weights, extension, vs, flush, cache, t0, result, count, seed, phase):
            calls.append((count, seed, phase))
            if timeout:
                raise TimeoutError("synthetic cooperative deadline")
            result.extend(rows(count, main_ratio if phase == "main" else confirm_ratio))
        ns["_vmsgate_draws"] = draws
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            state = ns["prepare"]({"w13": marker})
        self.assertTrue(backend.allow_tf32)  # restored after success AND fallback
        self.assertIn("MOE_VMS32_PICK bs=32", output.getvalue())
        return state["vmsgate"], calls

    def test_actual_prepare_admits_only_after_fresh_confirmation(self):
        gate, calls = self.run_prepare()
        self.assertEqual(calls, [(24, 1132288, "main"), (8, 1732288, "confirm")])
        self.assertEqual(gate["adopt"], {32: True})
        self.assertIsNotNone(gate["vs"])
        self.assertEqual(set(gate["ext"]), {32})

    def test_actual_prepare_vetoes_and_releases_failed_arm(self):
        for kwargs, count in ((dict(main_ratio=0.991), 1), (dict(confirm_ratio=0.996), 2),
                              (dict(stress=False), 1), (dict(flag=1), 2), (dict(timeout=True), 1)):
            gate, calls = self.run_prepare(**kwargs)
            self.assertEqual(len(calls), count, kwargs)
            self.assertFalse(any(gate["adopt"].values()), kwargs)
            self.assertIsNone(gate["vs"], kwargs)
            self.assertEqual(gate["ext"], {}, kwargs)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    for cls in (GateTests, StaticTests, PrepareTests, shared.RepackTests):
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(cls))
    for name in ("test_deadline_uses_monotonic_clock", "test_actual_timing_loop_pairs_null_and_candidate",
                 "test_capture_replayed_before_reading_outputs"):
        suite.addTest(shared.TimingTests(name))
    for name in ("test_source_ast_and_signatures", "test_runtime_no_data_sync_or_histogram_policy"):
        suite.addTest(shared.StaticTests(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
