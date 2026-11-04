"""Dependency-free CPU/static preflight. This does NOT compile or run CUDA.

python3 scripts/test_moe_vmsgate.py
Optional source-integrity check:
python3 scripts/test_moe_vmsgate.py --reference /read-only/path/moe_vms5.py
"""
import ast
import contextlib
import hashlib
import io
import math
import random
import statistics
import sys
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "candidates/moe_vmsgate.py").read_text()
TREE = ast.parse(SOURCE)
REFERENCE = None
if "--reference" in sys.argv:
    index = sys.argv.index("--reference")
    REFERENCE = Path(sys.argv[index + 1])
    del sys.argv[index:index + 2]


def namespace():
    """Load the actual gate functions without importing torch or the incumbent."""
    result = dict(math=math, statistics=statistics, time=time)
    keep = []
    for node in TREE.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("VMSGATE_") or name in ("VMS_SRC", "VMS_CPP"):
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name.startswith("_vmsgate_"):
            keep.append(node)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "actual_candidate_gate", "exec"), result)
    return result


def good_rows(ns):
    # Synthetic times exercise the specified gate policy, not a GPU speed claim.
    return [dict(draw=i, D=i % 16 + 1, R_max=2, routes=24, numeric=True, exact=True, repeat=True, base_repeat=True, stress_ok=True,
                 err=0.0, row_err=0.0, ratio=0.98, null_ratio=1.0, inc_us=100.0, vms_us=98.0,
                 passes=[dict(ratio=0.98, null=1.0, inc_us=100.0, vms_us=98.0) for _ in range(ns["VMSGATE_PASSES"])])
            for i in range(ns["VMSGATE_DRAWS"])]


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.ns = namespace()

    def test_routing_histogram_edges(self):
        # Expected counts are direct enumeration of local expert ids [7, 10).
        hist = self.ns["_vmsgate_histogram"]
        self.assertEqual(hist([6, 7, 7, 8, 10, 255], 7, 3), dict(D=2, R_max=2, routes=3))
        self.assertEqual(hist([32, 255]), dict(D=0, R_max=0, routes=0))
        self.assertEqual(hist([0] * 256), dict(D=1, R_max=256, routes=256))
        self.assertEqual(hist(list(range(32)) * 8), dict(D=32, R_max=8, routes=256))

    def test_strict_gate_and_every_veto(self):
        admit = self.ns["_vmsgate_admit"]
        rows = good_rows(self.ns)
        self.assertTrue(admit(rows, True))
        self.assertFalse(admit(rows, False))
        self.assertFalse(admit(rows[:-1], True))
        for key in ("numeric", "exact", "repeat"):
            rows = good_rows(self.ns)
            rows[23][key] = False
            self.assertFalse(admit(rows, True), key)
        rows = good_rows(self.ns)
        rows[0]["passes"].pop()
        self.assertFalse(admit(rows, True))
        for invalid in (0.994999, 1.005001, float("nan"), float("inf")):
            rows = good_rows(self.ns)
            rows[7]["passes"][1]["null"] = invalid
            self.assertFalse(admit(rows, True), invalid)
        for boundary in (0.995, 1.005):
            rows = good_rows(self.ns)
            rows[7]["passes"][1]["null"] = boundary
            self.assertTrue(admit(rows, True))

    def test_null_drift_cannot_cancel_across_passes_or_draws(self):
        rows = good_rows(self.ns)
        rows[0]["passes"][0]["null"] = 0.99
        rows[0]["passes"][1]["null"] = 1.01
        rows[0]["null_ratio"] = 1.0
        self.assertFalse(self.ns["_vmsgate_admit"](rows, True))

    def test_both_split_passes_and_elapsed_ratios_required(self):
        for split in (slice(0, 16), slice(16, 32)):
            for ps in (0, 1):
                rows = good_rows(self.ns)
                for row in rows[split]:
                    row["passes"][ps]["ratio"] = 1.001
                self.assertFalse(self.ns["_vmsgate_admit"](rows, True))
                rows = good_rows(self.ns)
                for row in rows[split]:
                    row["passes"][ps]["vms_us"] = 100.0
                self.assertFalse(self.ns["_vmsgate_admit"](rows, True))

    def test_fit_uses_training_only(self):
        # Analytic optimum: select D<=8, the eight 0.9 training draws; including
        # any 1.1 draw strictly worsens the geometric-mean objective.
        train = [dict(D=i + 1, R_max=1, routes=1, ratio=0.9 if i < 8 else 1.1) for i in range(16)]
        test = [dict(D=i + 1, R_max=1, routes=1, ratio=1.2 if i < 8 else 0.8) for i in range(16)]
        fit = self.ns["_vmsgate_fit"](train + test)
        self.assertEqual(fit["policy"], ("D", 8))
        self.assertAlmostEqual(fit["train"], math.sqrt(0.9))
        self.assertAlmostEqual(fit["heldout"], math.sqrt(1.2))
        for row in train + test:
            row["ratio"] = 1.01
        self.assertIsNone(self.ns["_vmsgate_fit"](train + test)["policy"])

    def test_print_tail_contains_all_draws(self):
        rows = good_rows(self.ns)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            for bs in (16, 32):
                self.ns["_vmsgate_log"](bs, rows, False)
        log = output.getvalue()
        self.assertEqual(log.count(" draw="), 64)
        self.assertLess(len(log.encode()), 14000)
        self.assertIn("applied=0", log)
        self.assertEqual(log[-16000:].count(" draw="), 64)
        print(f"tail-test: {len(log.encode())} bytes for 64 draws + summaries")


class TimingTests(unittest.TestCase):
    def test_numeric_gate_exact_bounds_and_sticky_error(self):
        ns = namespace()
        ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None))
        ns["_vmsgate_bits"] = lambda a, b: a.value == b.value

        class Output:
            value = 123

            def clone(self):
                return Output()

        graph = types.SimpleNamespace(replay=lambda: None)
        for error, row_error, flag, expected in ((0.015, 0.075, 0, True), (0.015001, 0.0, 0, False),
                                                  (0.0, 0.075001, 0, False), (0.0, 0.0, 1, False),
                                                  (float("inf"), 0.0, 0, False)):
            ns["_vmsgate_error"] = lambda a, b: (error, row_error)
            vs = {"err": types.SimpleNamespace(item=lambda: flag)}
            actual = ns["_vmsgate_correct"]({}, None, Output(), graph, Output(), vs)
            # The bounds are from TASK.md's 1.5% global / 5x row requirement.
            self.assertEqual(actual["numeric"], expected)
            self.assertTrue(actual["exact"])

    def test_deadline_uses_monotonic_clock(self):
        ns = namespace()
        ns["time"] = types.SimpleNamespace(monotonic=lambda: 1000.0)
        ns["_vmsgate_deadline"](1000.0 - ns["VMSGATE_GATE_END"] + 0.001)
        with self.assertRaises(TimeoutError):
            ns["_vmsgate_deadline"](1000.0 - ns["VMSGATE_GATE_END"])

    def test_actual_timing_loop_pairs_null_and_candidate(self):
        ns = namespace()
        trace, clock = [], [0.0]

        class Event:
            def __init__(self, enable_timing):
                self.value = None

            def record(self):
                self.value = clock[0]

            def elapsed_time(self, other):
                return (other.value - self.value) / 1000.0

        class Graph:
            def __init__(self, name, duration):
                self.name, self.duration = name, duration

            def replay(self):
                trace.append(self.name)
                clock[0] += self.duration

        class Flush:
            def sum(self):
                trace.append("read_flush")
                clock[0] += 10000.0  # must be excluded from recorded kernel time

        ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(Event=Event, synchronize=lambda: None))
        ns["_vmsgate_deadline"] = lambda _: None
        row = dict(base=Graph("base", 100.0), null=Graph("null", 100.0), vms=Graph("vms", 90.0))
        result = ns["_vmsgate_measure"](row, Flush(), 0.0)
        # Source of expected values: deterministic fake clock with 100/90 us
        # replays, not a numerical/performance claim about the CUDA code.
        self.assertEqual(result, dict(ratio=0.9, null=1.0, inc_us=100.0, vms_us=90.0))
        rounds = ns["VMSGATE_ROUNDS"]
        self.assertEqual(trace.count("read_flush"), rounds * 16)
        graphs = trace[1::2]
        self.assertEqual(graphs[:8], ["base", "vms", "vms", "base", "vms", "base", "base", "vms"])
        self.assertEqual(graphs[8:16], ["base", "null", "null", "base", "null", "base", "base", "null"])
        self.assertEqual(graphs[16:24], ["base", "null", "null", "base", "null", "base", "base", "null"])

    def test_capture_replayed_before_reading_outputs(self):
        ns, trace = namespace(), []

        class Graph:
            def replay(self):
                trace.append("replay")

        class Stream:
            def wait_stream(self, _):
                pass

        ns["torch"] = types.SimpleNamespace(cuda=types.SimpleNamespace(
            Stream=Stream, current_stream=Stream, stream=lambda _: contextlib.nullcontext(),
            graph=lambda _: contextlib.nullcontext(), synchronize=lambda: trace.append("sync"), CUDAGraph=Graph))
        ns["_vmsgate_graph"](lambda: trace.append("call"))
        self.assertEqual(trace[-2:], ["replay", "sync"])


class LazyTensor:
    """Only reshape/permute/slice/assign, lazily tracking original byte positions.

    Runs the actual torch repack function without torch/numpy. No FP arithmetic
    or CUDA semantics are emulated; each source byte's value is its own address.
    """
    device = "cpu"

    def __init__(self, shape, get=None):
        self.shape = tuple(shape)
        self.writes = []
        self.get = get or (lambda coord: self.flat(coord, self.shape))

    @staticmethod
    def flat(coord, shape):
        value = 0
        for index, size in zip(coord, shape):
            value = value * size + index
        return value

    @staticmethod
    def unflat(value, shape):
        out = []
        for size in reversed(shape):
            out.append(value % size)
            value //= size
        return tuple(reversed(out))

    def numel(self):
        return math.prod(self.shape)

    def reshape(self, *shape):
        assert math.prod(shape) == self.numel()
        return LazyTensor(shape, lambda coord: self.get(self.unflat(self.flat(coord, shape), self.shape)))

    view = reshape

    def permute(self, *order):
        def get(coord):
            old = [0] * len(order)
            for axis, previous in enumerate(order):
                old[previous] = coord[axis]
            return self.get(tuple(old))
        return LazyTensor(tuple(self.shape[i] for i in order), get)

    def __getitem__(self, key):
        shape = tuple(len(range(*item.indices(size))) for item, size in zip(key, self.shape) if isinstance(item, slice))

        def get(coord):
            result, next_axis = [], 0
            for item, size in zip(key, self.shape):
                if isinstance(item, slice):
                    start, _, step = item.indices(size)
                    result.append(start + coord[next_axis] * step)
                    next_axis += 1
                else:
                    result.append(item)
            return self.get(tuple(result))
        return LazyTensor(shape, get)

    def __setitem__(self, key, tensor):
        target_shape = self[key].shape
        assert tensor.shape == target_shape
        self.writes.append((key, tensor))

    def at(self, *coord):
        for key, tensor in self.writes:
            source, matches = [], True
            for value, item, size in zip(coord, key, self.shape):
                if isinstance(item, slice):
                    start, stop, step = item.indices(size)
                    if not start <= value < stop or (value - start) % step:
                        matches = False
                        break
                    source.append((value - start) // step)
                elif value != item:
                    matches = False
                    break
            if matches:
                return tensor.get(tuple(source))
        raise AssertionError(f"unwritten output byte {coord}")


class RepackTests(unittest.TestCase):
    def test_actual_repack_byte_mapping(self):
        ns = namespace()
        ns["torch"] = types.SimpleNamespace(empty=lambda shape, **_: LazyTensor(shape), uint8="uint8")
        experts, tiles = 2, 3
        source = LazyTensor((experts * tiles * 8, 36864))
        result = ns["_vmsgate_repack"](source, tiles)
        rng = random.Random(20261104)
        coords = [(e, tile, ks, offset) for e in range(experts) for tile in range(tiles) for ks in range(8)
                  for offset in (0, 15, 16, 4095, 4096, 32767, 32768, 32783, 33279, 33280, 36863)]
        coords += [(rng.randrange(experts), rng.randrange(tiles), rng.randrange(8), rng.randrange(36864)) for _ in range(4096)]
        for e, tile, ks, offset in coords:
            # Expected mapping is derived from the documented VMS slot layout:
            # codes: 8 blocks x 4096 B; scales: 8 blocks x 512 B. Each incumbent
            # block stores its eight K slices consecutively, codes then scales.
            if offset < 32768:
                block, within = divmod(offset, 4096)
                source_offset = ks * 4096 + within
            else:
                block, within = divmod(offset - 32768, 512)
                source_offset = 32768 + ks * 512 + within
            expected = ((e * tiles + tile) * 8 + block) * 36864 + source_offset
            self.assertEqual(result.at(e, tile, ks, offset), expected)
        print(f"repack-test: {len(coords)} original-byte addresses, codes/scales/tile/expert boundaries")


class StaticTests(unittest.TestCase):
    def test_shape_only_dispatch_and_fresh_output(self):
        ns = namespace()
        kernel = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn"][-1]
        exec(compile(ast.Module(body=[kernel], type_ignores=[]), "actual_timed_wrapper", "exec"), ns)
        trace = []

        class Tensor:
            def __init__(self, shape):
                self.shape = shape
                self.device = "cuda"

            def contiguous(self):
                return self

            def to(self, _):
                return self

            def data_ptr(self):
                return 16

        ns["torch"] = types.SimpleNamespace(empty=lambda shape, **_: Tensor(shape), bfloat16="bf16", int32="i32", float32="f32")
        ns["HIDDEN"] = 4096
        ns["_bind_context"] = lambda _: None
        ns["_vmsgate_parent_kernel"] = lambda data, state: "incumbent"
        ns["_vmsgate_call"] = lambda ext, vs, data, out: trace.append((data, out))
        gate = dict(adopt={16: True, 32: True}, vs=dict(lo=0, nloc=32), ext={16: None, 32: None})

        def data(m, lo=0):
            return dict(hidden_states=Tensor((m, 4096)), topk_ids=Tensor((m, 8)), topk_weights=Tensor((m, 8)),
                        config=dict(local_start=lo, local_count=32))

        for m in (1, 4, 8, 33, 128, 256, 257):
            self.assertEqual(ns["kernel_fn"](data(m), dict(vmsgate=gate)), "incumbent")
        self.assertEqual(ns["kernel_fn"](data(16, 1), dict(vmsgate=gate)), "incumbent")
        self.assertEqual(ns["kernel_fn"](data(16), {}), "incumbent")
        first = ns["kernel_fn"](data(16), dict(vmsgate=gate))
        second = ns["kernel_fn"](data(16), dict(vmsgate=gate))
        self.assertIsNot(first, second)
        self.assertEqual(len(trace), 2)
        self.assertEqual(first.shape, (16, 4096))

    def test_source_ast_and_signatures(self):
        self.assertTrue(SOURCE.startswith("# okc: problem=muse_moe_w4a16\n"))
        self.assertNotIn("\r", SOURCE)
        namespace()["_vmsgate_static_check"]()
        ns = namespace()
        ns["VMS_CPP"] = ns["VMS_CPP"].replace("torch::Tensor out", "float out")
        with self.assertRaises(RuntimeError):
            ns["_vmsgate_static_check"]()

    def test_cpp20_all_builds_and_only_control_instantiated(self):
        builds = [node for node in ast.walk(TREE) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "load_inline"]
        self.assertEqual(len(builds), 3)  # incumbent, disabled BW diagnostic, VMS
        for node in builds:
            keys = {k.arg: k.value for k in node.keywords}
            for key in ("extra_cflags", "extra_cuda_cflags"):
                self.assertIn("-std=c++20", [n.value for n in ast.walk(keys[key]) if isinstance(n, ast.Constant)])
        src = namespace()["VMS_SRC"]
        start, stop = src.index("Kern pick("), src.index("static void ensure_smem")
        self.assertEqual(src[start:stop].count("kern_of<"), 1)
        self.assertIn("kern_of<PB, 0, CTL>()", src[start:stop])

    def test_runtime_no_data_sync_or_histogram_policy(self):
        kernel = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn"][-1]
        call = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_vmsgate_call")
        for node in (kernel, call):
            names = [n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)]
            self.assertFalse(set(names) & {"item", "cpu", "tolist", "synchronize", "equal", "sum", "bincount"})
            text = ast.unparse(node)
            self.assertNotIn("_vmsgate_fit", text)
            self.assertNotIn("_vmsgate_histogram", text)
            self.assertNotIn("_vmsgate_ref", text)
        self.assertIn("torch.empty", ast.unparse(kernel))

    @unittest.skipUnless(REFERENCE is not None, "provide --reference for source provenance/diff verification")
    def test_incumbent_and_control_source_integrity(self):
        from build_moe_vmsgate import build
        original = REFERENCE.read_text()
        self.assertEqual(SOURCE, build(original, SOURCE))
        original_tree = ast.parse(original)
        src = namespace()["VMS_SRC"]
        original_src = next(ast.literal_eval(n.value) for n in original_tree.body if isinstance(n, ast.Assign)
                            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "VMS_SRC")
        # Check every byte of the numerical device code, not only spot-picked
        # math helpers: two host dispatch-table ranges and a comment may differ.
        def arithmetic(text):
            start, stop = text.index("constexpr int NVAR"), text.index("__host__ __device__ constexpr bool var_fits")
            text = text[:start] + text[stop:]
            start, stop = text.index("template <bool PB>\nKern pick("), text.index("static void ensure_smem")
            text = text[:start] + text[stop:]
            return text.replace(" (iteration 5, control-only build)", " (iteration 5)")
        self.assertEqual(arithmetic(src), arithmetic(original_src))
        print("source-integrity: incumbent unchanged except C++20 build flags/header; VMS arithmetic byte-identical")
        print("candidate-sha256:", hashlib.sha256(SOURCE.encode()).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
