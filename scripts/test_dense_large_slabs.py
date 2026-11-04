"""CPU-only checks for the optional rs128 H2 token-slab geometry.

Run: .venv-large/bin/python scripts/test_dense_large_slabs.py
Never imports the full submission: its unchanged nb factory builds CUDA at import.
Only the CW factory and the new functions are evaluated from the parsed AST.
"""
import ast
import copy
from contextlib import contextmanager, redirect_stdout
import io
import itertools
import math
from pathlib import Path
import random
import statistics
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
FILE = ROOT / "candidates/dense_large_slabs.py"
BASE = ROOT / "board_best/dense_kbv2_scored_1.5886.py"
TREE = ast.parse(FILE.read_text())


def isolated():
    """Run actual packing/source/policy helpers, not CUDA build entry points."""
    scope = {"_PRUNE": False, "_TABLE": {}, "_DEFAULT": "chenwainuo"}
    factory = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_make_chenwainuo")
    exec(compile(ast.Module(body=[factory], type_ignores=[]), str(FILE), "exec"), scope)
    scope["_K"] = {"chenwainuo": scope["_make_chenwainuo"]()}
    scope.update(_V6_N={"qkv_gate": 2304, "o": 8192, "pooler": 4096, "unpooler": 8192},
                 _V6_K={"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096},
                 _dt_math=math, _dt_stats=statistics, _dt_time=time, _dt_torch=torch)
    helpers = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
               and n.name in ("_dt_check", "_DTBudget", "_dt_inputs", "_dt_errors", "_dt_equiv")]
    exec(compile(ast.Module(body=helpers, type_ignores=[]), str(FILE), "exec"), scope)
    # References saved by the real file immediately above the new experiment.
    scope["prepare"] = lambda w: {"raw": w, "old_state": True}
    scope["kernel_fn"] = lambda d, s: d["x"] @ s["raw"][d["config"]["layer"]][:d["config"]["n"]].t()
    start = next(i for i, n in enumerate(TREE.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "_LS_SPEC" for t in n.targets))
    exec(compile(ast.Module(body=TREE.body[start:], type_ignores=[]), str(FILE), "exec"), scope)
    return scope


S = isolated()


def grid(m, n, kc, ks):
    """Independent ownership model of the *rendered* C++ slab indexing."""
    slabs = m // 128
    for split in range(ks):
        s0, s1 = split * kc // ks, (split + 1) * kc // ks
        for bx in range((n // 64) * slabs):
            row64, slab = divmod(bx, slabs)
            nb, hh = divmod(row64, 2)
            yield dict(bx=bx, split=split, cb=128 * nb + 64 * hh,
                       tok0=128 * slab, k0=64 * s0, k1=64 * s1,
                       zero0=128 * slab + split * 128 // ks,
                       zero1=128 * slab + (split + 1) * 128 // ks)


def prmt(a, b, selector):
    x = (int(b) << 32) | int(a)
    return sum(((x >> (8 * ((selector >> (4 * j)) & 7))) & 255) << (8 * j) for j in range(4))


class Geometry(unittest.TestCase):
    def test_all_production_cells_and_zero_ownership(self):
        for m, splits in S["_LS_SPLITS"].items():
            for layer, ks in splits.items():
                n, k = S["_LS_SHAPES"][layer]
                writes = np.zeros((m, n // 64), dtype=np.int16)
                zeros = np.zeros_like(writes)
                ranges = {}
                entries = list(grid(m, n, k // 64, ks))
                self.assertLessEqual(len(entries), 2 * 132)
                for e in entries:
                    col = e["cb"] // 64
                    writes[e["tok0"]:e["tok0"] + 128, col] += 1
                    zeros[e["zero0"]:e["zero1"], col] += 1
                    ranges.setdefault((e["tok0"], col), []).append((e["k0"], e["k1"]))
                    self.assertLess(e["bx"] + 2 * (n // 128) * (m // 128), 4 * (n // 128) * (m // 128))
                self.assertTrue(np.all(writes == ks), (m, layer))
                self.assertTrue(np.all(zeros == 1), (m, layer))
                for intervals in ranges.values():
                    self.assertEqual(intervals[0][0], 0)
                    self.assertEqual(intervals[-1][1], k)
                    self.assertTrue(all(a[1] == b[0] for a, b in zip(intervals, intervals[1:])))

    def test_sibling_slabs_reuse_exact_same_weights(self):
        entries = list(grid(256, 256, 15, 3))
        for a, b in zip(entries[::2], entries[1::2]):
            self.assertEqual((a["cb"], a["k0"], a["k1"]), (b["cb"], b["k0"], b["k1"]))
            self.assertEqual((a["tok0"], b["tok0"]), (0, 128))

    def test_swizzled_epilogue_addresses(self):
        # The inherited stmatrix produces one 64-column x 128-token slab,
        # SW128, with every 16-byte destination cell addressed exactly once.
        cells = []
        for w, lane, pp in itertools.product(range(4), range(32), range(8)):
            j, rr = lane >> 3, lane & 7
            i0 = 2 * pp
            i, ch = i0 + (j >> 1), 2 * w + (j & 1)
            tok = 8 * i + rr
            cells.append((tok, ch ^ rr))
        self.assertEqual(len(set(cells)), 128 * 8)
        self.assertEqual(set(cells), set(itertools.product(range(128), range(8))))

    def test_register_and_shared_budget(self):
        self.assertEqual(128 * 216 + 128 * 40, 32768)
        smem = 1024 + 4 * (128 * 128 + 6144) + 24 * 4 + 16
        self.assertEqual(smem, 91248)
        self.assertLess(2 * smem, 233472)


class PackingAndNumerics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # K=960 has 15 K64 stages, exercising non-even KS=2/4/7 partitions.
        gen = torch.Generator().manual_seed(5151)
        cls.w = (torch.randn((256, 960), generator=gen) / math.sqrt(960)).to(torch.bfloat16)
        cls.api = S["_K"]["chenwainuo"]
        cls.pack = cls.api["large_pack"](cls.w)

    def test_real_packer_roundtrip(self):
        self.assertIsNotNone(self.pack)
        self.assertTrue(torch.equal(self.api["large_unpack"](*self.pack, *self.w.shape),
                                    self.w.view(torch.int16).int() & 65535))

    def test_every_bf16_bit_pattern_roundtrip(self):
        # Encoding must be lossless even if uncommon exponents use many escapes.
        bits = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(128, 512)
        w = bits.view(torch.bfloat16)
        pk = self.api["large_pack"](w)
        self.assertIsNotNone(pk)
        self.assertTrue(torch.equal(self.api["large_unpack"](*pk, 128, 512), bits.int() & 65535))

    def test_independent_register_decode_and_escape_lists(self):
        lay, tb, esc6 = self.pack
        raw = lay.numpy()
        want = self.w.view(torch.int16).int().numpy() & 65535
        for ks in (1, 2, 3, 4, 7):
            arr = self.api["large_esc"](esc6, 2, 15, ks, self.w.device)[0].numpy().view(np.uint32)
            cap = (len(arr) - 32) // (ks * 2 * 8)
            for split in range(ks):
                s0, s1 = split * 15 // ks, (split + 1) * 15 // ks
                for nb, hh, warp, lane in itertools.product(range(2), range(2), range(4), range(32)):
                    patches = arr[((split * 2 + nb) * 8 + 4 * hh + warp) * cap:][:cap]
                    patches = [int(v) for v in patches if v != 0xFFFFFFFF and ((int(v) >> 12) & 31) == lane]
                    for stage in range(s0, s1):
                        planes = raw[nb, stage, 4 * hh + warp].reshape(3, 32, 16)
                        q = [planes[p, lane].copy().view(np.uint32) for p in range(3)]
                        regs = []
                        for sub in range(4):
                            c = int(q[2][sub])
                            rot4 = ((c >> 4) | ((c << 28) & 0xFFFFFFFF))
                            lohi = (c & 0x87878787) + tb
                            hihi = (rot4 & 0x87878787) + tb
                            lows = q[sub // 2]
                            off = 2 * (sub % 2)
                            regs.extend((prmt(lows[off], lohi, 0x5140), prmt(lows[off], lohi, 0x7362),
                                         prmt(lows[off + 1], hihi, 0x5140), prmt(lows[off + 1], hihi, 0x7362)))
                        for entry in patches:
                            if entry >> 17 == stage - s0:
                                reg, half, xr = (entry >> 8) & 15, (entry >> 7) & 1, entry & 127
                                regs[reg] ^= xr << (8 + 16 * half)
                        for step, kh, half, e in itertools.product(range(4), range(2), range(2), range(2)):
                            # PTX m16n8k16 A-register layout: [kh][half][pair].
                            reg = 4 * step + 2 * kh + half
                            bits = (regs[reg] >> (16 * e)) & 65535
                            row = 128 * nb + 64 * hh + 16 * warp + 8 * half + lane // 4
                            col = 64 * stage + 16 * step + 8 * kh + 2 * (lane % 4) + e
                            self.assertEqual(bits, want[row, col], (ks, split, row, col))

    def test_output_math_and_partial_rounding(self):
        torch.set_num_threads(4)
        gen = torch.Generator().manual_seed(1234)
        w = self.w
        for m, ks, dist in itertools.product((128, 256), (1, 2, 3, 4, 7), ("normal", "outliers", "probe")):
            x = torch.randn((m, 960), generator=gen)
            if dist == "outliers":
                x[:, ::97] *= 30
            if dist == "probe":
                x.zero_()
                x[torch.arange(m), (torch.arange(m) * 31) % 960] = 32
            x = x.to(torch.bfloat16)
            ref = x.double() @ w.double().t()
            got = torch.full((m, 256), float("nan"), dtype=torch.bfloat16)
            # Model the inherited self-zero then BF16 TMA-add semantics, using
            # FP32 matrix products for every K partition. Shuffle split order.
            entries = list(grid(m, 256, 15, ks))
            for e in entries:
                got[e["zero0"]:e["zero1"], e["cb"]:e["cb"] + 64] = 0
            random.Random(ks + m).shuffle(entries)
            for e in entries:
                sl = np.s_[e["tok0"]:e["tok0"] + 128, e["cb"]:e["cb"] + 64]
                partial = (x[e["tok0"]:e["tok0"] + 128, e["k0"]:e["k1"]].float() @
                           w[e["cb"]:e["cb"] + 64, e["k0"]:e["k1"]].float().t()).to(torch.bfloat16)
                got[sl] = (got[sl].float() + partial.float()).to(torch.bfloat16)
            err = float((got.double() - ref).norm() / ref.norm())
            row = float(((got.double() - ref).norm(dim=1) / ref.norm(dim=1)).max())
            self.assertLess(err, 8e-3, (m, ks, dist))
            self.assertLess(row, 4e-2, (m, ks, dist))
            if dist == "probe":
                self.assertTrue(torch.equal(got, ref.to(torch.bfloat16)))


class SourceAndPolicy(unittest.TestCase):
    def test_base_unchanged_except_helper_exports(self):
        base = ast.parse(BASE.read_text())
        prefix = copy.deepcopy(TREE.body[:len(base.body)])
        factory = next(n for n in prefix if isinstance(n, ast.FunctionDef) and n.name == "_make_chenwainuo")
        returned = factory.body[-1].value
        keep = len(returned.keys) - 5
        returned.keys, returned.values = returned.keys[:keep], returned.values[:keep]
        self.assertEqual(ast.dump(ast.Module(body=prefix, type_ignores=[])), ast.dump(base))

    def test_rendered_source(self):
        self.assertNotIn(b"\r", FILE.read_bytes())
        src = S["_ls_source"]()
        self.assertNotIn("@@", src)
        self.assertIn("rq_go<128, 4, 1, 0, 1, 0, 1074, 1>", src)
        self.assertIn("const int tok0 = (blockIdx.x % slabs) * NT;", src)
        self.assertEqual(src.count(", tok0, full + 8 *"), 6)
        self.assertEqual(src.count('"r"(cb + 64 * g), "r"(tok0), "r"(slab)'), 2)
        self.assertIn("r0 = tok0 + (int)((long long)ks * NT / KS)", src)
        self.assertIn("setmaxnreg.inc.sync.aligned.u32 216", src)
        self.assertIn("dim3(2 * p.NB * (p.m / NT), KS)", src)
        self.assertIn("p.ncnt < 4 * p.NB * (p.m / NT)", src)
        self.assertIn("cudaFuncAttributePreferredSharedMemoryCarveout, 100", src)
        self.assertEqual(src.count("case 0: return rq_go<"), 1)

    def test_fail_closed_source_drift(self):
        factory = S["_K"]["chenwainuo"]
        old = factory["large_source"]
        try:
            factory["large_source"] = lambda _: "drifted"
            with self.assertRaisesRegex(RuntimeError, "source drift"):
                S["_ls_source"]()
        finally:
            factory["large_source"] = old

    def test_ratio_policy(self):
        admit = S["_ls_admit"]
        self.assertTrue(admit(.98, .984))
        for r, p in ((.986, .98), (.98, .986), (math.nan, .98), (.98, math.inf), (0, .98), (-1, .98)):
            self.assertFalse(admit(r, p), (r, p))

    def test_cpu_fallback_and_no_synthetic_retention(self):
        gen = torch.Generator().manual_seed(19)
        weights = {name: torch.randn((128, 64), generator=gen).to(torch.bfloat16) for name in S["_LS_SHAPES"]}
        state = S["prepare"](weights)
        self.assertEqual(set(state), {"old_state", "raw", "large_slabs"})
        self.assertEqual(state["large_slabs"], {})
        for m, layer in itertools.product((1, 8, 16, 32, 33, 128, 130, 256, 512), weights):
            x = torch.randn((m, 64), generator=gen).to(torch.bfloat16)
            data = {"x": x, "config": dict(layer=layer, m=m, n=128, k=64)}
            self.assertTrue(torch.equal(S["kernel_fn"](data, state), x @ weights[layer].t()))

    def test_prepare_failure_falls_back_without_touching_incumbent(self):
        weights = {"o": types.SimpleNamespace(is_cuda=True)}
        for error in (S["_DTBudget"](), RuntimeError("build failure")):
            def reject(*args):
                raise error
            with patch.dict(S, {"_ls_start": lambda: {}, "_ls_tune": reject}), redirect_stdout(io.StringIO()):
                state = S["prepare"](weights)
            self.assertTrue(state["old_state"])
            self.assertIs(state["raw"], weights)
            self.assertEqual(state["large_slabs"], {})

    def run_cell(self, fail_timing=None, wrong_math=False):
        """Exercise the actual six-draw gate with simulated graph replay on CPU."""
        class Graph:
            def replay(self):
                self.run()

        class Cuda:
            current = None
            CUDAGraph = Graph

            @staticmethod
            def synchronize():
                pass

            @contextmanager
            def graph(self, g):
                self.current = g
                try:
                    yield
                finally:
                    self.current = None

        cuda = Cuda()

        class Proxy:
            def __getattr__(self, name):
                return cuda if name == "cuda" else getattr(torch, name)

        weights = {"pooler": torch.randn((256, 64)).to(torch.bfloat16)}

        def compute(data, bad):
            def value():
                out = (data["x"].float() @ weights["pooler"].float().t()).to(torch.bfloat16)
                return torch.ones_like(out) if bad else out
            out = value()
            if cuda.current is not None:
                g = cuda.current
                g.run = lambda: out.copy_(value())
            return out

        timings = []

        def abba(*args):
            timings.append(1)
            ratio = .99 if len(timings) == fail_timing else .98
            return 10., 10. * ratio, ratio, ratio

        updates = {
            "_V6_N": dict(S["_V6_N"], pooler=256),
            "_V6_K": dict(S["_V6_K"], pooler=64),
            "_dt_torch": Proxy(), "_dt_abba": abba,
            "_ls_parent_kernel_fn": lambda d, s: compute(d, False),
            "_ls_call": lambda d, s, p: compute(d, wrong_math),
        }
        with patch.dict(S, updates), redirect_stdout(io.StringIO()):
            accepted = S["_ls_cell"]("pooler", 128, {}, weights, {"cnt": torch.zeros(4)},
                                       torch.Generator().manual_seed(42), None, None, math.inf)
        return accepted, len(timings)

    def test_both_passes_all_draws_required(self):
        self.assertEqual(self.run_cell(), (True, 6))
        self.assertEqual(self.run_cell(fail_timing=5), (False, 5))

    def test_bad_math_is_never_timed(self):
        self.assertEqual(self.run_cell(wrong_math=True), (False, 0))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main(verbosity=2)
