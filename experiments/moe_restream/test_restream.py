"""CPU/static TQ64 gates. No Torch, CUDA compilation or GPU performance claims."""
import ast
import importlib.util
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import unittest
import math
from types import SimpleNamespace
import json
import contextlib
import io

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = (HERE / "restream.cu").read_text()
TREE = ast.parse((ROOT / "board_best/moe_m3_scored_2.3797.py").read_text())
OLD = next(ast.literal_eval(n.value) for n in TREE.body if isinstance(n, ast.Assign)
           and any(getattr(t, "id", "") == "CUDA_SRC" for t in n.targets))


def function(text, name):
    m = re.search(r"^(?:__device__|constexpr)[^\n]*\b" + re.escape(name) + r"\([^\n]*\)\s*\{", text, re.M)
    assert m, name
    a = text.index("{", m.start()); depth = 1; b = a + 1
    while depth:
        depth += (text[b] == "{") - (text[b] == "}"); b += 1
    return text[m.start():b]


def metadata_header(pp):
    a = pp.index("struct Smem {"); b = pp.index("\n};", a) + 3
    result = [pp[a:b]]
    a = pp.index("constexpr unsigned long long wgs_pack8(")
    b = pp.index("__device__ __forceinline__ int rs_shift(", a)
    result.append(pp[a:b])
    for name in ("rs_shift", "rs_rows", "rs_job", "rs_count", "rs_ntbase", "rs_r16", "rs_done_index",
                 "wgs_split", "wgs_prefix", "wgs_stage_at", "wgs_range", "wgs_ranges", "wgs_cta_of"):
        result.append(function(pp, name))
    model = json.loads((HERE / "MODEL.json").read_text())
    histograms = [r["hist"] for M in ("128", "256") for r in model[M]["draws"]]
    result.append("const int actual_counts[8][32] = {" + ",".join("{" + ",".join(map(str, row)) + "}" for row in histograms) + "};")
    return "\n".join(result)


def host_namespace():
    tree = ast.parse((HERE / "host.py").read_text())
    ns = {"_moe_sel_original_prepare": lambda w: dict(w), "kernel_fn": lambda data, state: "base",
          "load_inline": lambda *a, **k: SimpleNamespace(okc_set_pdl=lambda value: None),
          "_mh_time": SimpleNamespace(monotonic=lambda: 0.), "_MOE_SEL": {},
          "_moe_default": lambda M, sfx: (("-DOKC_WGS=1",) if sfx else ()) + tuple(sfx)}
    exec(compile(tree, "<host>", "exec"), ns)
    return ns, tree


def pipeline(seed):
    rng = random.Random(seed)
    tags, drains = [], set()
    for _ in range(8):
        c = rng.choice((1, 64, 65, 74, 128, 129, 168, 192, 225, 256))
        tags += [c] * rng.randint(1, 32)
        drains.add(len(tags) - 1)
    n = len(tags)
    free = [[4] * 3, [4] * 2]
    ready = [[-1] * 3, [-1] * 2]
    nxt, inflight = [0, 0], {}
    groups = [[0, 0, -1] for _ in range(4)]  # stage, microstep, held x stage
    for _ in range(n * 50 + 100):
        actions = []
        for p, slots in enumerate((3, 2)):
            if nxt[p] < n and free[p][nxt[p] % slots] == 4:
                actions.append(("issue", p, nxt[p]))
        actions += [("copy", p, st) for p, st in inflight]
        for wg, (st, step, prev) in enumerate(groups):
            if st < n and (step or (ready[0][st % 3] == st and ready[1][st % 2] == st)):
                actions.append(("wg", wg, st))
        if not actions:
            assert all(st == n for st, _, _ in groups) and not inflight
            assert all(v == 4 for row in free for v in row)
            return
        kind, who, st = rng.choice(actions)
        if kind == "issue":
            slots = 3 if who == 0 else 2
            free[who][st % slots] = 0
            R = 1 if tags[st] > 128 else 2 if tags[st] > 64 else 4
            inflight[who, st] = R * 4608 if who == 0 else ((tags[st] + 7) // 8) * 8 * 256
            nxt[who] += 1
        elif kind == "copy":
            size = inflight.pop((who, st))
            assert 0 < size <= (18432 if who == 0 else 65536)
            ready[who][st % (3 if who == 0 else 2)] = st
        else:
            wg = who; _, step, prev = groups[wg]
            R = 1 if tags[st] > 128 else 2 if tags[st] > 64 else 4
            c = min(64, max(0, tags[st] - (wg // R) * 64))
            if step == 0:
                if not c:
                    assert prev == -1
                    free[0][st % 3] += 1; free[1][st % 2] += 1
                    groups[wg] = [st + 1, 0, -1]
                else:
                    # Both the raw row group and this WG's x slab lie inside
                    # the bytes actually copied, not just inside allocation.
                    assert ((wg & (R - 1)) + 1) * 4608 <= R * 4608
                    assert (wg // R) * 64 * 256 + ((c + 7) // 8) * 8 * 256 <= ((tags[st] + 7) // 8) * 8 * 256
                    groups[wg][1] = 1
            elif step == 1:
                if prev >= 0:
                    assert ready[1][prev % 2] == prev
                    free[1][prev % 2] += 1
                groups[wg][2] = st
                groups[wg][1] = 2
            elif step == 2:
                assert ready[0][st % 3] == st
                free[0][st % 3] += 1
                groups[wg][1] = 3
            else:
                if st in drains:
                    assert ready[1][st % 2] == st
                    free[1][st % 2] += 1
                    groups[wg][2] = -1
                groups[wg][0] += 1; groups[wg][1] = 0
    raise AssertionError("pipeline event budget exhausted")


def flat(coords, shape):
    n = 0
    for c, s in zip(coords, shape): n = n * s + c
    return n


def coords(n, shape):
    result = []
    for s in reversed(shape): result.append(n % s); n //= s
    return list(reversed(result))


class LazyTensor:
    def __init__(self, shape, get, dtype="u8"):
        self.shape, self.get, self.dtype = tuple(shape), get, dtype
    def numel(self): return math.prod(self.shape)
    def view(self, *shape):
        if shape == ("u8",):
            if self.dtype == "u8": return self
            return LazyTensor(self.shape[:-1] + (self.shape[-1] * 4,), lambda n: (self.get(n // 4) >> (8 * (n % 4))) & 255)
        shape = tuple(self.numel() if s == -1 else s for s in shape)
        assert math.prod(shape) == self.numel()
        return LazyTensor(shape, self.get, self.dtype)
    def permute(self, *axes):
        shape = tuple(self.shape[a] for a in axes)
        def get(n):
            out = coords(n, shape); old = [0] * len(shape)
            for a, v in zip(axes, out): old[a] = v
            return self.get(flat(old, self.shape))
        return LazyTensor(shape, get, self.dtype)
    def contiguous(self): return self
    def index_select(self, axis, rows):
        assert axis == 0
        shape = (rows.numel(),) + self.shape[1:]
        def get(n):
            c = coords(n, shape); c[0] = rows.values[c[0]]
            return self.get(flat(c, self.shape))
        return LazyTensor(shape, get, self.dtype)


def lazy_cat(tensors, dim):
    a, b = tensors; assert dim == 3
    shape = a.shape[:3] + (a.shape[3] + b.shape[3],)
    def get(n):
        c = coords(n, shape)
        if c[3] < a.shape[3]: return a.get(flat(c, a.shape))
        c[3] -= a.shape[3]; return b.get(flat(c, b.shape))
    return LazyTensor(shape, get)


class RestreamTests(unittest.TestCase):
    def test_async_active_empty_token_wg_pipeline(self):
        for seed in range(500): pipeline(seed)

    def test_native_indices_output_coverage_and_smem(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("no native C++ compiler")
        text = re.sub(r"^#include.*$", "", SRC, flags=re.M)
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            directory = Path(directory)
            pp = subprocess.run([compiler, "-E", "-P", "-x", "c++", "-DOKC_WGS=1", "-DOKC_NW=16",
                                 "-DOKC_NS=3", "-DOKC_NSF=3", "-DOKC_PRMTSC=1", "-"],
                                input=text, text=True, capture_output=True, check=True).stdout
            (directory / "metadata.hpp").write_text(metadata_header(pp))
            binary = directory / "model"
            compiled = subprocess.run([compiler, "-O2", "-std=c++20", "-I", str(directory),
                                       str(HERE / "test_native.cpp"), "-o", str(binary)], text=True, capture_output=True)
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            run = subprocess.run([str(binary)], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("PASS outputs=", run.stdout)
            print(run.stdout.strip(), flush=True)

    def test_read_once_weight_piece_coverage(self):
        for count in range(1, 257):
            R = 1 if count > 128 else 2 if count > 64 else 4
            for pb, normal_units in ((0, 32), (1, 16)):
                seen = set()
                for u in range(normal_units * 4 // R):
                    r64 = u * R
                    for st in range(32):
                        for q in range(r64 & 3, (r64 & 3) + R):
                            key = (r64 >> 2, st, q)
                            self.assertNotIn(key, seen)
                            seen.add(key)
                self.assertEqual(len(seen), normal_units * 32 * 4)
                self.assertEqual(len(seen) * 4608, (9 if pb else 18) * 2**20)

    def test_pack_permutations_and_qword_roundtrip(self):
        host = ast.parse((HERE / "host.py").read_text())
        pack = next(n for n in host.body if isinstance(n, ast.FunctionDef) and n.name == "_rs_pack")
        perms = [ast.literal_eval(ast.Tuple(elts=n.args, ctx=ast.Load())) for n in ast.walk(pack)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "permute"]
        self.assertCountEqual(perms, [(0, 1, 2, 5, 3, 4), (0, 3, 1, 4, 2, 6, 5), (0, 5, 1, 6, 2, 4, 7, 3)])
        node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_stored_pos")
        ns = {}; exec(compile(ast.Module(body=[node], type_ignores=[]), "<positions>", "exec"), ns)
        pos = ns["_stored_pos"]()
        rng = random.Random(20261106)
        for _ in range(10000):
            codes = [[rng.randrange(16) for _ in range(2)] for _ in range(4)]
            q = sum(((codes[r][e] >> bit) & 1) << pos[r, e, bit] for r in range(4) for e in range(2) for bit in range(4))
            rot = lambda n: ((q << n) | (q >> (32 - n))) & 0xffffffff
            decoded = [q & 0x81c081c0, rot(10) & 0x81c081c0, rot(13) & 0x81c081c0,
                       (((q & 0x001b001b) * 0x1084) & 0xffffffff) & 0x81c081c0]
            for r in range(4):
                for e in range(2):
                    got = sum(((decoded[r] >> (b + 16 * e)) & 1) << bit for bit, b in enumerate((6, 7, 8, 15)))
                    self.assertEqual(got, codes[r][e])
        # Each q64 piece's code/scales boundary exactly matches decoder loads.
        self.assertEqual(((1 * 4 + 3) * 32 + 31) * 16 + 3 * 4 + 4, 4096)
        self.assertEqual(4096 + ((1 * 4 + 3) * 8 + 7) * 8 + 7 + 1, 4608)

    def test_global_activation_layouts(self):
        for count in (1, 8, 33, 64, 65, 74, 128, 129, 168, 192, 225, 256):
            ntt = (count + 7) // 8
            for tk in range(count):
                for col in range(4096):
                    # Gather: each thread loads one contiguous uint4 of x.
                    tid, within = divmod(col, 8)
                    gathered = ((((tid >> 4) * ntt + (tk >> 3)) * 16 + (tid & 15)) * 8 + (tk & 7)) * 8 + within
                    # GEMM1 epilogue writes the same layout directly to xB.
                    epi = ((((col >> 7) * ntt + (tk >> 3)) * 16 + ((col & 127) >> 3)) * 64 + (tk & 7) * 8) + (col & 7)
                    self.assertEqual(gathered, epi)
                    self.assertLess(epi, ntt * 8 * 4096)
            R = 1 if count > 128 else 2 if count > 64 else 4
            for wg in range(4):
                c = min(64, max(0, count - (wg // R) * 64))
                if c:
                    end = (wg // R) * 64 * 128 * 2 + ((c + 7) // 8) * 8 * 128 * 2
                    self.assertLessEqual(end, ntt * 8 * 128 * 2)

    def test_actual_packer_with_lazy_cpu_tensors(self):
        host = ast.parse((HERE / "host.py").read_text())
        pack = next(n for n in host.body if isinstance(n, ast.FunctionDef) and n.name == "_rs_pack")
        tag = lambda n: ((n * 0x9e3779b1) ^ 0x12345678) & 0xffffffff
        ns = {"torch": SimpleNamespace(uint8="u8", cat=lazy_cat),
              "_pack_codes_nat": lambda codes, rows: LazyTensor((16 * 256 * 32,), tag, "i32")}
        exec(compile(ast.Module(body=[pack], type_ignores=[]), "<packer>", "exec"), ns)
        rows = SimpleNamespace(values=list(reversed(range(256))), numel=lambda: 256)
        sf = LazyTensor((256, 256), lambda n: ((n // 256) * 17 + (n % 256) * 7) & 255)
        packed = ns["_rs_pack"](None, sf, rows)
        self.assertEqual(packed.numel(), 256 * 4096 * 9 // 16)
        rng = random.Random(20261107)
        for _ in range(20000):
            st, q, lg, wp, lane, js, byte = rng.randrange(32), rng.randrange(4), rng.randrange(2), rng.randrange(4), rng.randrange(32), rng.randrange(4), rng.randrange(4)
            rb, ks, g, t = q * 4 + wp, st * 8 + lg * 4 + js, lane >> 2, lane & 3
            old = flat((rb, ks >> 5, (ks >> 2) & 7, g, t, ks & 3), (16, 8, 8, 8, 4, 4))
            offset = (st * 4 + q) * 4608 + ((lg * 4 + wp) * 32 + lane) * 16 + js * 4 + byte
            self.assertEqual(packed.get(offset), (tag(old) >> (byte * 8)) & 255)
            h = rng.randrange(2)
            offset = (st * 4 + q) * 4608 + 4096 + ((lg * 4 + wp) * 8 + g) * 8 + js * 2 + h
            physical_row = rows.values[(q * 4 + wp) * 16 + h * 8 + g]
            self.assertEqual(packed.get(offset), (physical_row * 17 + ks * 7) & 255)

    def test_arithmetic_precision_and_readiness_policy_unchanged(self):
        for name in ("dq4", "bmul2", "bmul2l", "bmul2h", "achita_marlin_out2", "wg_desc"):
            self.assertEqual(function(SRC, name), function(OLD, name))
        a, b = OLD.index("template <int NTT> struct WGSMMA;"), OLD.index("// two producer warps:")
        self.assertIn(OLD[a:b], SRC)
        self.assertIn("constexpr int WGS_NT = FT_T / 8", SRC)
        self.assertIn("JMAX * 16 >= 32 * 16", SRC)
        self.assertIn("j * 16 + (st >> 1)", SRC)
        self.assertNotIn("multicast", SRC[SRC.index("void wgs_wproducer"):SRC.index("// phase A's x gather")])
        self.assertEqual(SRC.count("griddepcontrol.wait"), OLD.count("griddepcontrol.wait"))

    def test_emitted_candidate_and_fallback_scope(self):
        candidate = (HERE / "moe_restream.py").read_bytes()
        baseline = (HERE.parent / "moe_headroom/moe_headroom.py").read_bytes()
        self.assertTrue(candidate.startswith(baseline))
        tree = ast.parse(candidate.decode())
        cuda = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                    and any(getattr(t, "id", "") == "_RS_CUDA" for t in n.targets))
        self.assertEqual(cuda, SRC)
        host = (HERE / "host.py").read_text()
        self.assertIn('state["_rs_selected"].add(256)', host)
        self.assertNotIn('.add(128)', host)
        self.assertIn('med <= .97', host)
        self.assertIn('clean_med <= 1.01', host)
        self.assertIn('state["_rs_selected"].clear()', host)
        self.assertIn('_rs_base_prepare = _moe_sel_original_prepare', host)
        self.assertNotIn('budget_or_nonPRMT', host)

    def test_shape_dispatch_and_exception_fallback(self):
        ns, _ = host_namespace()
        ns["_rs_run"] = lambda data, state: "tq"
        for M in (4, 16, 32, 128, 256, 257):
            data = {"hidden_states": SimpleNamespace(shape=(M, 4096))}
            self.assertEqual(ns["kernel_fn"](data, {"_rs_selected": {256}}), "tq" if M == 256 else "base")
        def fail(weights, state, start):
            state["_rs_selected"].add(256)
            state["_rs_w"] = "private"
            raise RuntimeError("gate failure")
        ns["_rs_gate"], ns["print"] = fail, lambda *a, **k: None
        ns["_rs_post_select"] = lambda state, start: None
        state = ns["prepare"]({"sentinel": 1})
        self.assertEqual(state, {"sentinel": 1, "_rs_selected": set()})

    def test_prior_log_proves_budget_not_nonPRMT(self):
        # b3fe8be00fcb4666 logged BOTH MAIN and FALLBACK at M256. Execute the
        # real incumbent candidate generator: without sfx that is impossible.
        values = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if getattr(target, "id", "") in ("MAIN_DEFINES", "MAIN_FALLBACK_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES"):
                        values[target.id] = ast.literal_eval(node.value)
        functions = [node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in ("_main_defines", "_moe_valid")]
        exec(compile(ast.Module(body=functions, type_ignores=[]), "<incumbent candidates>", "exec"), values)
        lut = values["_moe_valid"](256, ())
        prmt = values["_moe_valid"](256, ("-DOKC_PRMTSC=1",))
        self.assertEqual(lut, [values["MAIN_FALLBACK_DEFINES"]])
        self.assertEqual(len(prmt), 2)
        self.assertIn("-DOKC_WGS=1", prmt[0])
        self.assertEqual(prmt[1], values["MAIN_FALLBACK_DEFINES"] + ("-DOKC_PRMTSC=1",))
        # With sfx proven nonempty, the old disjunction can only mean >170s.
        self.assertFalse(not ("-DOKC_PRMTSC=1",))

    def test_distinct_admission_reasons_and_remaining_budget(self):
        ns, _ = host_namespace()
        ns["_mh_time"] = SimpleNamespace(monotonic=lambda: 171.)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertTrue(ns["_rs_admit"]({"sfx": ("-DOKC_PRMTSC=1",)}, 0.))
        self.assertIn("layout=wgs scale_decode=prmt prmt=1", log.getvalue())
        self.assertIn("remaining_s=129.00 gate_remaining_s=114.00", log.getvalue())
        self.assertNotIn("SKIP", log.getvalue())
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertFalse(ns["_rs_admit"]({"sfx": ()}, 0.))
        self.assertIn("reason=nonPRMT", log.getvalue())
        self.assertIn("layout=unit+fat scale_decode=lut", log.getvalue())
        self.assertNotIn("insufficient_compile_budget", log.getvalue())
        ns["_mh_time"] = SimpleNamespace(monotonic=lambda: 226.)
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            self.assertFalse(ns["_rs_admit"]({"sfx": ("-DOKC_PRMTSC=1",)}, 0.))
        self.assertIn("reason=insufficient_compile_budget", log.getvalue())
        self.assertIn("remaining_s=74.00 gate_remaining_s=59.00", log.getvalue())

    def test_candidate_runs_before_optional_selector(self):
        ns, _ = host_namespace()
        events, clock = [], {"now": 0.}
        ns["_mh_time"] = SimpleNamespace(monotonic=lambda: clock["now"])
        ns["print"] = lambda *a, **k: None
        def core(weights):
            events.append("core")
            clock["now"] = 129.3  # synthetic reproduction: +41.7 selector => old >170s skip
            return {"sfx": ("-DOKC_PRMTSC=1",)}
        def gate(weights, state, start):
            events.append("TQ_gate")
            self.assertTrue(ns["_rs_admit"](state, start))
            state["_rs_selected"].add(256)
            clock["now"] = 230.
        def post(state, start):
            events.append("optional_cached_selector")
            self.assertEqual(state["_rs_selected"], {256})
            self.assertNotIn(256, ns["_MOE_SEL"])
        ns["_rs_base_prepare"], ns["_rs_gate"], ns["_rs_post_select"] = core, gate, post
        state = ns["prepare"]({})
        self.assertEqual(events, ["core", "TQ_gate", "optional_cached_selector"])
        self.assertEqual(state["_rs_selected"], {256})

    def test_cached_selector_is_bounded_and_cannot_change_M256(self):
        ns, tree = host_namespace()
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_rs_post_select")
        text = ast.unparse(node)
        self.assertIn("for M in (16, 32)", text)
        self.assertIn("range(3)", text)
        self.assertIn("range(9)", text)
        self.assertIn("_rs_is_cached(d)", text)
        self.assertNotIn("load_inline", text)
        self.assertNotIn("_moe_select(", text)
        ns.update(DQ7=True, _EXTS={("-DOKC_WGS=1", "-DOKC_DQ7=1"): object()})
        self.assertTrue(ns["_rs_is_cached"](("-DOKC_WGS=1",)))
        self.assertFalse(ns["_rs_is_cached"](("-DOKC_CPF=0",)))
        ns["_mh_time"] = SimpleNamespace(monotonic=lambda: 291.)
        with contextlib.redirect_stdout(io.StringIO()):
            ns["_rs_post_select"](None, 0.)  # exits before Tensor access/compilation

    def test_shared_cpp20_policy_for_real_incumbent_and_TQ_build_calls(self):
        ns, _ = host_namespace()
        calls = []
        def builder(*args, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(okc_set_pdl=lambda value: None)
        ns["_rs_torch_load_inline"] = builder
        values = {}
        for node in TREE.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if getattr(target, "id", "") in ("MAIN_DEFINES", "MAIN_NS", "CPP_SRC", "CUDA_SRC", "DQ7", "PDL"):
                        values[target.id] = ast.literal_eval(node.value)
        ns.update(values, torch=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (9, 0))),
                  _EXTS={}, ALT_SRCS={}, _RS_CUDA=SRC)
        ext = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_ext")
        exec(compile(ast.Module(body=[ext], type_ignores=[]), "<incumbent _ext>", "exec"), ns)
        with contextlib.redirect_stdout(io.StringIO()):
            ns["_ext"](ns["MAIN_DEFINES"] + ("-DOKC_PRMTSC=1",))
            ns["_rs_ext"](("-DOKC_PRMTSC=1",))
        self.assertEqual(len(calls), 2)
        for kwargs in calls:
            for key in ("extra_cflags", "extra_cuda_cflags"):
                self.assertEqual([f for f in kwargs[key] if f.startswith(("-std=", "--std="))], ["-std=c++20"])
        self.assertEqual(calls[0]["cuda_sources"], OLD)
        self.assertEqual(calls[1]["cuda_sources"], SRC)
        mixed = ["-O3", "-std=c++17", "--std=c++14", "-std", "c++11", "--std", "c++17", "-lineinfo"]
        self.assertEqual(ns["_rs_cxx20_flags"](mixed), ["-O3", "-lineinfo", "-std=c++20"])

    def test_actual_histogram_model(self):
        import json
        model = json.loads((HERE / "MODEL.json").read_text())
        self.assertEqual([r["dup_jobs"] for r in model["256"]["draws"]], [4, 1, 3, 0])
        self.assertEqual([r["bytes_saved"] for r in model["256"]["draws"]], [108 * 2**20, 27 * 2**20, 81 * 2**20, 0])
        self.assertGreaterEqual(model["256"]["normal_stage_penalty_sensitivity_pct"]["2"], 3)
        self.assertFalse(model["128"]["target"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
