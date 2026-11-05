"""CPU/structural tests; imports no Triton, compiles no GPU extensions.

Run using a CPU PyTorch interpreter: python kb/exp/test_decode_vms32.py
"""
import ast
import copy
import math
from pathlib import Path
import re
import statistics
import unittest

import torch

PATH = Path(__file__).with_name("decode_vms32.py")
TEXT = PATH.read_text()
TREE = ast.parse(TEXT)


def extract(names, namespace):
    nodes = [n for n in TREE.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(nodes) == len(names), (len(nodes), names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), "exec"), namespace)
    return namespace


NS = extract({"_dv_host_flags", "_dv_nvcc_flags", "_vmsgate_signature", "_vmsgate_static_check",
              "_vmsgate_repack", "_vmsgate_dequant", "_dv_error", "_dv_numeric", "_dv_admit",
              "_dv_rms", "_dv_coeff", "_dv_ref_moe", "_dv_reference", "_dv_measure", "_DVAdapter"},
             dict(torch=torch, _dv_math=math, _dv_stats=statistics, DV_PASSES=3, DV_ROUNDS=16,
                  DV_NULL=(.995, 1.005), DV_RATIO=.995, DV_ERROR_RATIO=1.10, ROUTER_MULT=.1767766952966369,
                  _dv_deadline=lambda _: None))
for node in TREE.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("VMS_SRC", "VMS_CPP", "VMSGATE_ARGS") for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), "exec"), NS)


class Structural(unittest.TestCase):
    def test_file_and_no_forbidden_runtime(self):
        self.assertLess(PATH.stat().st_size, 1_000_000)
        self.assertIsNone(re.search(r"\b(?:subprocess|Popen|multiprocessing)\b|os\s*\.\s*system\s*\(|(?:cu|cuda)Graph\w*_v[23]\b", TEXT))

    def test_all_inline_builds_explicit(self):
        found = 0
        for n in ast.walk(TREE):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("load_inline", "_tvm_load_inline"):
                found += 1
                kws = {kw.arg: kw.value for kw in n.keywords}
                for kw, wrapper in (("extra_cflags", "_dv_host_flags"), ("extra_cuda_cflags", "_dv_nvcc_flags")):
                    self.assertIn(kw, kws)
                    self.assertIsInstance(kws[kw], ast.Call)
                    self.assertEqual(kws[kw].func.id, wrapper)
        self.assertEqual(found, 13)  # twelve incumbent builds, one VMS build
        self.assertEqual(NS["_dv_host_flags"](["-O3", "-std=c++17"]), ["-O3", "-std=c++20"])
        self.assertEqual(NS["_dv_nvcc_flags"](["-O3", "-std=c++17", "-gencode=arch=compute_90,code=sm_90"]),
                         ["-O3", "-std=c++20", "-gencode=arch=compute_90a,code=sm_90a"])

    def test_cpp_python_signature(self):
        NS["_vmsgate_static_check"]()
        altered = dict(NS, VMS_CPP=NS["VMS_CPP"].replace("torch::Tensor wA,", ""))
        extract({"_vmsgate_static_check"}, altered)
        with self.assertRaises(RuntimeError):
            altered["_vmsgate_static_check"]()

    def test_adapter_local_ids_and_argument_order(self):
        class Ext:
            def vms_run(self, *args):
                self.args = args
            def __deepcopy__(self, memo):
                raise TypeError("compiled extension modules cannot be copied")
        ext = Ext()
        # Tensor stand-in supplies only the metadata touched by the adapter.
        class X:
            shape = (32, 4096)
            def data_ptr(self): return 4096
            def is_contiguous(self): return True
        x, ids, tw, out = X(), object(), object(), object()
        vs = {k: object() for k in NS["VMSGATE_ARGS"] if k not in ("x", "ids", "tw", "out", "mode", "cfg")}
        adapter = NS["_DVAdapter"](ext, vs)
        cloned = copy.deepcopy(adapter)
        self.assertIs(cloned.ext, ext)
        self.assertIsNot(cloned.vs, vs)
        adapter.okc_run(None, None, x, ids, tw, out, 0, 32, 132)
        vals = dict(zip(NS["VMSGATE_ARGS"], ext.args))
        self.assertIs(vals["ids"], ids)  # no host read/repack: unchanged local ids
        self.assertIs(vals["tw"], tw)
        self.assertEqual((vals["mode"], vals["cfg"]), (7, 0))
        with self.assertRaises(RuntimeError):
            adapter.okc_run(None, None, x, ids, tw, out, 32, 32, 132)

    def test_timing_schedule(self):
        calls = []
        def quad(base, other, *args):
            order = args[-2]
            calls.append((other, order))
            return ([100., 100.], [99., 99.] if other == "new" else [100., 100.])
        env = dict(NS, _dv_quad=quad)
        extract({"_dv_measure"}, env)
        row = env["_dv_measure"]("base", "null", "new", None, None, None, None, None, None)
        self.assertEqual(len(calls), 16 * 2 * 2)
        self.assertEqual(calls[:4], [("new", (0, 1, 1, 0)), ("new", (1, 0, 0, 1)),
                                          ("null", (0, 1, 1, 0)), ("null", (1, 0, 0, 1))])
        self.assertEqual(calls[4][0], "null")
        self.assertEqual((row["paired"], row["null"], row["old_us"], row["new_us"]), (.99, 1., 100., 99.))

    def test_static_shape_dispatch(self):
        fn = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn"][-1]
        calls = []
        ns = dict(_DV_KERNEL=lambda d, s: calls.append(s))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "dispatch", "exec"), ns)
        adapter = object()
        state = dict(dvms32_pick={(32, 1, True): adapter})
        for B, L, w in ((1, 1, 2048), (16, 1, 2048), (32, 1, 0), (4, 8, 2048), (32, 1, 2048)):
            ns["kernel_fn"](dict(config=dict(batch=B, q_len=L, tokens=B * L, window=w)), state)
        for s in calls[:-1]:
            self.assertIs(s, state)
        self.assertIs(calls[-1]["_dv_adapter"], adapter)
        self.assertNotIn("_dv_adapter", state)


class Numeric(unittest.TestCase):
    def test_admission_boundaries(self):
        valid = dict(numeric=True, null=1., null_ratio=1., paired=.994)
        rows = [dict(valid) for _ in range(3)]
        self.assertTrue(NS["_dv_admit"](rows))
        for key, value in (("null", .99499), ("null", 1.00501), ("null_ratio", .99499),
                           ("paired", 1.0), ("paired", float("nan")), ("numeric", False)):
            bad = copy.deepcopy(rows)
            bad[1][key] = value
            self.assertFalse(NS["_dv_admit"](bad), (key, value))
        self.assertFalse(NS["_dv_admit"](rows[:2]))
        self.assertFalse(NS["_dv_admit"]([dict(valid, paired=.996)] * 3))
        self.assertTrue(NS["_dv_admit"]([dict(valid, paired=v) for v in (.994, .995, .999)]))
        self.assertTrue(NS["_dv_numeric"]((.011, .02), (.01, .02), .015))
        self.assertFalse(NS["_dv_numeric"]((.01101, .02), (.01, .02), .015))
        self.assertFalse(NS["_dv_numeric"]((.01, .076), (.01, .02), .015))
        self.assertFalse(NS["_dv_numeric"]((1e-10, 0), (0, 0), .015))
        self.assertFalse(NS["_dv_numeric"]((float("nan"), 0), (.01, 0), .015))

    def test_error_floor_and_nonfinite(self):
        # Analytical known error: [3,4] -> [3.03,4.04] is exactly 1%.
        ref = torch.tensor([[3., 4.], [0., 0.]])
        out = ref * 1.01
        out[1, 0] = .01
        err, row = NS["_dv_error"](out, ref)
        self.assertAlmostEqual(err, math.sqrt(.05 ** 2 + .01 ** 2) / 5, places=6)
        self.assertAlmostEqual(row, .01 / (.1 * 2.5), places=6)
        self.assertEqual(NS["_dv_error"](out * float("nan"), ref), (float("inf"), float("inf")))

    def test_fp4_dequant_known_codes(self):
        # NVFP4 E2M1: low nibble first; scale = E4M3 block scale * FP32 global.
        packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=torch.uint8)
        scales = torch.tensor([[.5]])
        got = NS["_vmsgate_dequant"](packed, scales, torch.tensor(2.))
        expected = torch.tensor([[0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]])
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    def test_repack_is_lossless_permutation(self):
        # Each 128-row tile's eight 512-K slices concatenate blocks 0..7.
        units = torch.arange(2 * 8 * 36864, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(1, 2, 8, 36864)
        out = NS["_vmsgate_repack"](units.flatten(), 2)
        recovered = torch.empty_like(units)
        recovered[..., :32768] = out[..., :32768].reshape(1, 2, 8, 8, 4096).permute(0, 1, 3, 2, 4).reshape(1, 2, 8, 32768)
        recovered[..., 32768:] = out[..., 32768:].reshape(1, 2, 8, 8, 512).permute(0, 1, 3, 2, 4).reshape(1, 2, 8, 4096)
        self.assertTrue(torch.equal(recovered, units))

    def test_reference_moe_remote_hot_sparse(self):
        # Rank-one expert: gate=x, up=2*x, B picks inter[0]. Positive
        # outputs normalize to y/sqrt(y^2+eps), then weighted route sum.
        x = torch.tensor([[1.], [2.], [3.]])
        wa = torch.cat((torch.ones(4096, 1), 2 * torch.ones(4096, 1)))
        wb = torch.zeros(1, 4096)
        wb[0, 0] = 1.
        ids = torch.tensor([[0, -1], [0, 0], [-1, -1]], dtype=torch.int32)
        tw = torch.tensor([[.4, .6], [.3, .7], [.5, .5]])
        got = NS["_dv_ref_moe"](x, ids, tw, {"post_expert_norm": torch.zeros(1)}, {0: (wa, wb)}, None)
        y = torch.nn.functional.silu(x) * 2 * x
        expected = y / (y.square() + 1e-8).sqrt() * torch.tensor([[.4], [1.], [0.]])
        torch.testing.assert_close(got, expected)

    def test_reference_full_layer_analytic(self):
        # Manufactured solution, not an incumbent golden: zero branch weights
        # give stream2=alpha1*alpha2*stream, hidden=RMSNorm(stream2).
        torch.set_num_threads(4)
        gen = torch.Generator().manual_seed(1024)
        stream = torch.randn((32, 8192), generator=gen)
        weights = {"qkv_gate": torch.zeros(2304, 8192), "o": torch.zeros(8192, 1024),
                   "pooler": torch.zeros(4096, 8192), "unpooler": torch.zeros(8192, 4096),
                   "router": torch.zeros(256, 8192)}
        for key in ("attn_residual_gate", "ffn_residual_gate", "pre_ffn_norm", "post_ffn_norm", "next_pre_attn_norm"):
            weights[key] = torch.zeros(8192)
        weights["pre_expert_norm"] = torch.zeros(4096)
        data = dict(config=dict(rope_theta=500000., qk_scale_factor=5.4730064863875,
                                sm_scale=.125, window=2048, page_size=1),
                    attn_input=stream.to(torch.bfloat16), stream=stream,
                    positions=torch.full((32,), 3), seq_lens=torch.full((32,), 3),
                    page_table=torch.arange(128).reshape(32, 4),
                    k_cache=torch.zeros(128, 1, 2, 64, dtype=torch.float8_e4m3fn),
                    v_cache=torch.zeros(128, 1, 2, 64, dtype=torch.float8_e4m3fn),
                    k_scale=torch.tensor(1.), v_scale=torch.tensor(1.), local_start=torch.tensor([0]))
        ids = torch.arange(32, 40).repeat(32, 1)  # all remote: MoE sum is exactly zero
        ref = NS["_dv_reference"](data, weights, ids, {}, None)
        # Preserve the two FP32 products before BF16 rounding: mathematically
        # .75*stream, but reassociation flips exact BF16 halfway values.
        alpha = torch.tensor(.75).sqrt()  # alpha1=alpha2=sqrt(.75), beta=.5
        expected_stream = alpha * (alpha * stream)
        expected_hidden = (expected_stream * torch.rsqrt(expected_stream.square().mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)
        torch.testing.assert_close(ref["hidden"], expected_hidden, rtol=0, atol=0)
        torch.testing.assert_close(ref["update"], torch.zeros_like(stream), rtol=0, atol=5e-7)
        # Nonzero attention branch: uniform heads and constant O projection
        # yield a normalized all-ones branch, so update=alpha2*beta1.
        data["attn_input"].fill_(1.)
        data["v_cache"].fill_(2.)
        for row in range(128):
            weights["qkv_gate"][1152 + row, row] = 1.
        for row in range(8192):
            weights["o"][row, row % 1024] = 1.
        ref = NS["_dv_reference"](data, weights, ids, {}, None)
        expected_update = torch.full_like(stream, math.sqrt(.75) * .5)
        torch.testing.assert_close(ref["update"], expected_update, rtol=2e-6, atol=5e-7)
        torch.testing.assert_close(ref["v"].float(), torch.ones(32, 2, 64), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
