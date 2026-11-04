"""CPU/static gates only: no Torch, CUDA runtime, GPU timing, or submissions.

Expected values come from independent counted-barrier predicates, direct dot
products, and the unmodified board source. These do not prove GPU correctness.
"""
from __future__ import annotations

import ast
import difflib
import itertools
from math import comb
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "board_best/moe_m3_scored_2.3797.py"
CAND = HERE / "moe_joinfull.py"


def source(path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in {"CUDA_SRC", "MAIN_DEFINES", "TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES", "MAIN_FALLBACK_DEFINES"}:
                values[name] = ast.literal_eval(node.value)
    return tree, values


class CountedBarrier:
    """Abstract mbarrier: each phase needs all arrivals AND all expected bytes."""

    def __init__(self, arrivals):
        self.arrivals = arrivals
        self.pending = arrivals
        self.tx = 0
        self.phase = 0

    def advance(self):
        assert self.pending >= 0 and self.tx >= 0
        if self.pending == 0 and self.tx == 0:
            self.phase += 1
            self.pending = self.arrivals

    def arrive(self, nbytes):
        self.pending -= 1
        self.tx += nbytes
        self.advance()

    def complete(self, nbytes):
        self.tx -= nbytes
        self.advance()


def interleave(a, b):
    if not a:
        yield b
    elif not b:
        yield a
    else:
        yield from ((a[0],) + x for x in interleave(a[1:], b))
        yield from ((b[0],) + x for x in interleave(a, b[1:]))


def check_one_barrier(arrivals=2):
    checked = 0
    for nw, nx in itertools.product(range(1, 5), repeat=2):
        # Deliberately unequal byte counts; completion may be partial. A copy's
        # arrive.expect_tx always precedes that copy's complete_tx events.
        wa = (nw * 128,) + (-128,) * nw
        xa = (nx * 512,) + (-512,) * nx
        w = tuple(("w", v) for v in wa)
        x = tuple(("x", v) for v in xa)
        for trace in interleave(w, x):
            full = CountedBarrier(arrivals)
            issued = {"w": False, "x": False}
            outstanding = {"w": 0, "x": 0}
            for who, nbytes in trace:
                if nbytes > 0:
                    issued[who] = True
                    outstanding[who] += nbytes
                    full.arrive(nbytes)
                else:
                    outstanding[who] += nbytes
                    full.complete(-nbytes)
                # Independent oracle: a stage is ready iff both producers
                # issued and both copies finished, regardless of event order.
                ready = all(issued.values()) and not any(outstanding.values())
                assert (full.phase == 1) == ready, (trace, who, nbytes)
            assert full.phase == 1 and full.pending == arrivals and full.tx == 0
            checked += 1
    return checked


def ring_trace(seed, nstage, slots=3, joined=True, mutation=None):
    """Four asynchronous consumer WGs, two producers, independently retiring x.

    Each WG represents four warps' empty arrivals. Weight release is after the
    last group; previous x release is after the next stage's first group, or at
    a segment drain. Each TMA transfer completes in separately scheduled parts.
    The two empty-barrier sets are NOT combined.
    """
    rng = random.Random(seed)
    nwg = 4
    # Random segment boundaries cover short/split units and phase transitions.
    drains = {s for s in range(nstage) if rng.randrange(5) == 0} | {nstage - 1}
    nbytes = [[36864] * nstage, [rng.randint(1, 8) * 4096 for _ in range(nstage)]]
    full = [[CountedBarrier(1) for _ in range(slots)] for _ in range(2)]
    if joined:
        full[0] = [CountedBarrier(1 if mutation == "one_arrival" else 2) for _ in range(slots)]
        if mutation != "x_old_barrier":
            full[1] = full[0]
    empty = [[nwg] * slots for _ in range(2)]
    written = [[-1] * slots for _ in range(2)]
    next_producer = [0, 0]
    inflight = {}
    # [next stage, microstep (enter/first-group/last-group/drain), retained x]
    consumers = [[0, 0, -1] for _ in range(nwg)]
    sums = [0.0] * nwg
    payload = [[(rng.randint(-8, 8), rng.randint(-8, 8)) for _ in range(nstage)] for _ in range(nwg)]

    for _ in range(80 * nstage + 100):
        actions = []
        for p in range(2):
            s = next_producer[p]
            if s < nstage and empty[p][s % slots] == nwg:
                actions.append(("issue", p, s))
        actions += [("copy", p, s) for p, s in inflight]
        for g, (s, step, prev) in enumerate(consumers):
            if s >= nstage:
                continue
            if step or all(full[p][s % slots].phase > s // slots for p in range(2)):
                actions.append(("consume", g, s))
        if not actions:
            assert all(s == nstage for s, _, _ in consumers), (seed, consumers, next_producer, "deadlock")
            assert not inflight and next_producer == [nstage, nstage]
            assert all(v == nwg for row in empty for v in row)
            # Analytic, schedule-independent dot product of integer-valued
            # operands. These products/sums are exactly representable in FP32.
            assert sums == [float(sum(w * x for w, x in row)) for row in payload]
            return
        kind, who, s = rng.choice(actions)
        sl = s % slots
        if kind == "issue":
            empty[who][sl] = 0
            full[who][sl].arrive(nbytes[who][s])
            # Variable partial completions ensure neither arrival nor one
            # complete_tx event alone can incorrectly make a stage visible.
            parts = rng.randint(1, 4)
            a, b = divmod(nbytes[who][s], parts)
            inflight[who, s] = [a + (i < b) for i in range(parts)]
            next_producer[who] += 1
        elif kind == "copy":
            parts = inflight[who, s]
            size = parts.pop()
            if not parts:
                written[who][sl] = s
                del inflight[who, s]
            full[who][sl].complete(size)
        else:
            g = who
            _, step, prev = consumers[g]
            if step == 0:
                # Prevent parity ABA or reading a half-complete stage.
                assert all(full[p][sl].phase == s // slots + 1 for p in range(2))
                assert [written[p][sl] for p in range(2)] == [s, s], (seed, s, written)
                consumers[g][1] = 1
            elif step == 1:
                if prev >= 0:
                    assert written[1][prev % slots] == prev
                    empty[1][prev % slots] += 1
                consumers[g][2] = s
                consumers[g][1] = 2
            elif step == 2:
                assert written[0][sl] == written[1][sl] == s
                w, x = payload[g][s]
                sums[g] += float(w * x)
                empty[0][sl] += 1
                consumers[g][1] = 3
            else:
                if s in drains:
                    assert written[1][sl] == s
                    empty[1][sl] += 1
                    consumers[g][2] = -1
                consumers[g][0] += 1
                consumers[g][1] = 0
    raise AssertionError((seed, "event budget exhausted"))


def stage_sequence(pb, g0, g1, consumer=False):
    """Independently transcribe the producer loop and segmented consumer loop."""
    ue = ((g1 - 1) >> 4) << 4
    sp = ue if pb == 0 and g1 & 15 and ue > g0 else g0
    if not consumer:
        gs = sp
        stages = []
        for _ in range(g1 - g0):
            if gs == g1:
                gs = g0
            stages.append(gs)
            gs += 1
        return stages
    gs, ge = sp, g1
    stages = []
    while True:
        if gs >= ge:
            if ge == g1 and sp > g0:
                gs, ge = g0, sp
            else:
                return stages
        s0 = gs & 15
        s1 = min(16, s0 + ge - gs)
        stages.extend(range(gs, gs + s1 - s0))
        gs += s1 - s0


class JoinFullTests(unittest.TestCase):
    def test_all_single_phase_interleavings(self):
        # Choose positions of each producer's ordered arrival/completions.
        expected = sum(comb(nw + nx + 2, nw + 1) for nw, nx in itertools.product(range(1, 5), repeat=2))
        self.assertEqual(check_one_barrier(), expected)

    def test_wrong_initial_count_is_detected(self):
        with self.assertRaises(AssertionError):
            check_one_barrier(arrivals=1)

    def test_ring_interleavings_and_reuse(self):
        ring_trace(0, 0, joined=False)
        ring_trace(0, 0, joined=True)
        for seed in range(600):
            n = 1 + seed % 65
            ring_trace(seed, n, joined=False)
            ring_trace(seed, n, joined=True)

    def test_ring_mutations_are_detected(self):
        for mutation in ("one_arrival", "x_old_barrier"):
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                ring_trace(0, 9, mutation=mutation)

    def test_producer_and_consumer_stage_orders(self):
        rng = random.Random(20261104)
        ranges = [(0, 0), (0, 1), (0, 15), (0, 16), (1, 17), (17, 18), (3, 35)]
        ranges += [(g0 := rng.randrange(88 * 512), g0 + rng.randrange(1001)) for _ in range(1000)]
        for pb, (g0, g1) in itertools.product((0, 1), ranges):
            producer = stage_sequence(pb, g0, g1)
            consumer = stage_sequence(pb, g0, g1, consumer=True)
            self.assertEqual(producer, consumer)
            # Oracle: reorder is a permutation, not a skip/duplicate. This also
            # covers zero-work ranges and arbitrary phase-A trailing splits.
            self.assertEqual(sorted(producer), list(range(g0, g1)))

    def test_host_ast_unchanged_except_extension_name(self):
        base, _ = source(BASE)
        cand, _ = source(CAND)
        for tree in (base, cand):
            for n in tree.body:
                if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "CUDA_SRC" for t in n.targets):
                    n.value = ast.Constant("CUDA source separately checked")
            for n in ast.walk(tree):
                if isinstance(n, ast.Constant) and isinstance(n.value, str):
                    n.value = n.value.replace("v216x_pair1dq8_joinfull_", "v216x_pair1dq8_")
        self.assertEqual(ast.dump(base, include_attributes=False), ast.dump(cand, include_attributes=False))

    def test_cuda_patch_is_synchronization_only(self):
        a, b = source(BASE)[1]["CUDA_SRC"], source(CAND)[1]["CUDA_SRC"]
        changed = list(difflib.unified_diff(a.splitlines(), b.splitlines(), n=0))
        # Five distinct sites plus constants: no packing, dequantization, MMA,
        # FP32/BF16 rounding, epilogue, global readiness or launch changes.
        allowed_removed = {
            "      uint64_t* bar = S.bars + BX_F + xs;",
            "    for (int k = 0; k < BX_E + WGS_NX; ++k) mbar_init(S.bars + k, k < BW_E ? 1 : k < BX_F ? NW : k < BX_E ? 1 : NW);",
        }
        removed = {l[1:] for l in changed if l.startswith("-") and not l.startswith("---")}
        self.assertEqual(removed, allowed_removed)
        added_code = [l[1:].split("//")[0].strip() for l in changed if l.startswith("+") and not l.startswith("+++")]
        added_code = [l for l in added_code if l]
        # difflib can align the identical neighboring #endif directives in
        # either position; require the exact added-line multiset instead.
        self.assertCountEqual(added_code, [
            "#ifndef OKC_JOINFULL",
            "#define OKC_JOINFULL 1",
            "#endif",
            'static_assert(!OKC_JOINFULL || WGS_NS == WGS_NX, "JOINFULL requires identical weight/x slot cycles");',
            "constexpr int WGS_FULL_ARRIVALS = OKC_JOINFULL ? 2 : 1;",
            "constexpr int WGS_XFULL = OKC_JOINFULL ? BW_F : BX_F;",
            "uint64_t* bar = S.bars + WGS_XFULL + xs;",
            "#if !OKC_JOINFULL",
            "#endif",
            "if (nj == 0) {",
            "S.wtot = S.wtotB = 0;",
            "S.wgr[0] = S.wgr[1] = S.wgr[2] = S.wgr[3] = 0;",
            "return;",
            "}",
            "for (int k = 0; k < BX_E + WGS_NX; ++k)",
            "mbar_init(S.bars + k, k < BW_E ? WGS_FULL_ARRIVALS : k < BX_F ? NW : k < BX_E ? 1 : NW);",
        ])
        self.assertEqual(b.count("WGS_XFULL + xs"), 1)
        self.assertIn("WGS_FULL_ARRIVALS = OKC_JOINFULL ? 2 : 1", b)
        self.assertIn("static_assert(!OKC_JOINFULL || WGS_NS == WGS_NX", b)
        self.assertEqual(a.count("mbar_arrive1(S.bars + BW_E"), b.count("mbar_arrive1(S.bars + BW_E"))
        self.assertEqual(a.count("mbar_arrive1(S.bars + BX_E"), b.count("mbar_arrive1(S.bars + BX_E"))

    def test_non_wgs_preprocessing_identical(self):
        compiler = shutil.which("clang++") or shutil.which("c++")
        if not compiler:
            self.skipTest("no C++ preprocessor")
        _, a = source(BASE)
        _, b = source(CAND)
        # Empty SDK/Torch stubs are only for preprocessing: this is NOT CUDA
        # compilation. Real preprocessor pruning verifies inactive changes.
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            tmp = Path(tmp)
            (tmp / "torch").mkdir()
            for name in ("cuda_bf16.h", "cuda_runtime.h", "torch/extension.h", "ATen/cuda/CUDAContext.h", "c10/cuda/CUDAGuard.h"):
                path = tmp / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            for key in ("TINY_DEFINES", "TINY2_DEFINES", "MID_DEFINES", "MAIN_FALLBACK_DEFINES"):
                for prmt in (0, 1):
                    outputs = []
                    for values in (a, b):
                        text = values["CUDA_SRC"]
                        # Source uses these solely in host-only launcher code.
                        text = re.sub(r"^#include [<\"](?:torch|ATen|c10)/.*$", "", text, flags=re.M)
                        proc = subprocess.run([compiler, "-E", "-P", "-x", "c++", "-I", str(tmp), "-DOKC_NS=3", "-DOKC_NSF=3", f"-DOKC_PRMTSC={prmt}", *values[key], "-"], input=text, text=True, capture_output=True, check=True)
                        outputs.append(proc.stdout)
                    self.assertEqual(*outputs, (key, prmt))


if __name__ == "__main__":
    unittest.main(verbosity=2)
