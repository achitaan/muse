"""No Torch/CUDA required: layout, tensor-instruction semantics and gate tests.

This does not establish GPU rounding, code generation, barrier liveness or speed.
"""
from __future__ import annotations

import ast
import collections
import importlib.util
import math
from pathlib import Path
import random
import shutil
import statistics
import struct
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/moe_verify_sync"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("build_verify", ROOT / "scripts/build_moe_verify_sync.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
SOURCE = builder.build()
TREE = ast.parse(SOURCE)
CUDA = next(ast.literal_eval(n.value) for n in TREE.body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))


def functions(*names):
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<selected functions>", "exec"), ns)
    return ns


def bf16(bits):
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def dq4(word):
    def rot(q, sh):
        return ((q << sh) | (q >> (32 - sh))) & 0xffffffff
    mask = 0x81c081c0
    return (word & mask, rot(word, 10) & mask, rot(word, 13) & mask,
            ((word & 0x001b001b) * 0x1084) & mask)


def xoffset(token, k, ntt=1):
    return (((k // 256 * ntt + token // 8) * 32 + k % 256 // 8) * 64
            + token % 8 * 8 + k % 8)


def emulate_stage(seed, s0=0, s1=16):
    """Pack exactly one 36-KiB WGS stage, then read the new core's operands.

    Compare the per-lane m16n8 fragments with dense 64x8 multiplication.
    Values are dyadic rationals, keeping this a strict equality test.
    """
    rng = random.Random(seed)
    ns = functions("_stored_pos", "_scatter_code_bits")
    ns["_POS"] = ns["_stored_pos"]()
    pack_code = ns["_scatter_code_bits"]
    codes = [[rng.randrange(16) for _ in range(256)] for _ in range(64)]
    scales = [[rng.choice((0x28, 0x30, 0x38, 0x40, 0x48)) for _ in range(16)] for _ in range(64)]
    x = [[rng.randrange(-8, 9) / 8 for _ in range(256)] for _ in range(8)]
    # Use bf16 bits, like the actual TMA x slice; invalid padding is harmless
    # because columns are independent and epilogues mask token >= count.
    xbuf = bytearray(4096)
    for token in range(8):
        for k in range(256):
            bits = struct.unpack("<I", struct.pack("<f", x[token][k]))[0] >> 16
            struct.pack_into("<H", xbuf, xoffset(token, k) * 2, bits)
    piece = bytearray(9216)
    for lg in range(4):
        for wp in range(4):
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for js in range(4):
                    word = 0
                    for reg, (h, q) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
                        for up in range(2):
                            row = wp * 16 + h * 8 + g
                            k = (lg * 4 + js) * 16 + 8 * q + 2 * t + up
                            word = pack_code(word, codes[row][k], reg, up)
                    addr = (((lg * 4 + wp) * 32 + lane) * 4 + js) * 4
                    struct.pack_into("<I", piece, addr, word)
            for g in range(8):
                for js in range(4):
                    for h in range(2):
                        addr = 8192 + (((lg * 4 + wp) * 8 + g) * 4 + js) * 2 + h
                        piece[addr] = scales[wp * 16 + h * 8 + g][lg * 4 + js]
    a = [[0.] * 256 for _ in range(64)]
    b = [[0.] * 8 for _ in range(256)]
    for wp in range(4):
        for lane in range(32):
            g, t = lane >> 2, lane & 3
            for step in range(s0, s1):
                lg, js = divmod(step, 4)
                addr = (((lg * 4 + wp) * 32 + lane) * 4 + js) * 4
                word, = struct.unpack_from("<I", piece, addr)
                sc = 8192 + (((lg * 4 + wp) * 8 + g) * 4 + js) * 2
                sh = (piece[sc] * 16 + 0x5c00, piece[sc + 1] * 16 + 0x5c00)
                for reg, raw in enumerate(dq4(word)):
                    h, high_k = reg % 2, reg // 2
                    for up in range(2):
                        # Undo the common 2^-62 operand scale: it is the same
                        # dequant path and expert global scale in both cores.
                        a[wp * 16 + h * 8 + g][step * 16 + high_k * 8 + 2 * t + up] = (
                            bf16((raw >> (16 * up)) & 0xffff) * bf16(sh[h]) * 2**62)
                bp = step * 256 + g * 16 + t * 4
                for high_k in range(2):
                    for up in range(2):
                        raw, = struct.unpack_from("<H", xbuf, bp + 128 * high_k + 2 * up)
                        k = step * 16 + 8 * high_k + 2 * t + up
                        assert b[k][g] in (0., bf16(raw))
                        b[k][g] = bf16(raw)
    vals = (0., .5, 1., 1.5, 2., 3., 4., 6.)
    expected = [[0.] * 8 for _ in range(64)]
    actual = [[0.] * 8 for _ in range(64)]
    for row in range(64):
        for k in range(s0 * 16, s1 * 16):
            c = codes[row][k]
            sc = scales[row][k // 16]
            scale = (1 + (sc & 7) / 8) * 2**((sc >> 3) - 7)
            value = vals[c & 7] * (-1 if c & 8 else 1) * scale
            assert a[row][k] == value, (row, k, a[row][k], value)
            for token in range(8):
                expected[row][token] += value * x[token][k]
                actual[row][token] += a[row][k] * b[k][token]
    assert actual == expected
    # Each output register is (row g/g+8, token 2t/2t+1), unchanged from WGMMA.
    coverage = set()
    for wp in range(4):
        for lane in range(32):
            for reg in range(4):
                pair = (wp * 16 + (lane >> 2) + 8 * (reg // 2),
                        2 * (lane & 3) + reg % 2)
                assert pair not in coverage
                coverage.add(pair)
    assert len(coverage) == 64 * 8


class VerifyTests(unittest.TestCase):
    def test_generated_file_and_scope(self):
        self.assertEqual(SOURCE, (ROOT / "candidates/moe_verify_sync.py").read_text())
        self.assertIn("if M in (128, 256):", SOURCE)
        self.assertEqual(CUDA.count("case 1: wgs_seg_sync8"), 1)
        self.assertNotIn("wgs_seg_sync8<PB,", CUDA)
        for n in range(2, 8):
            self.assertIn(f"case {n}: wgs_seg<PB, {n}>", CUDA)
        self.assertIn("if (nj == 0)", CUDA)
        self.assertIn("for (int st = s0; st < s1; ++st)", CUDA)
        self.assertEqual(CUDA.count("__device__ __forceinline__ void wgs_seg_sync8"), 1)

    def test_incumbent_arithmetic_unchanged(self):
        bt = ast.parse(builder.BASE.read_text())
        base_cuda = next(ast.literal_eval(n.value) for n in bt.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "CUDA_SRC" for t in n.targets))
        def body(text, start, end):
            return text[text.index(start):text.index(end, text.index(start))]
        for start, end in (("__device__ __forceinline__ void dq4", "// ---------------------------------------------------------------- routing"),
                           ("__device__ __forceinline__ void wgs_seg(", "// Verify experiment:"),
                           ("__device__ __forceinline__ void wgs_epi_tile", "// one segment")):
            if end == "// Verify experiment:":
                self.assertEqual(body(base_cuda, start, "template <bool PB>\n__device__ __forceinline__ void wgs_phase"),
                                 body(CUDA, start, end))
            else:
                self.assertEqual(body(base_cuda, start, end), body(CUDA, start, end))

    def test_tensor_fragments(self):
        for seed in range(16):
            emulate_stage(seed)
        for s0, s1 in ((0, 1), (1, 16), (3, 8), (8, 16), (15, 16)):
            emulate_stage(100 + s0, s0, s1)

    def test_all_stage_and_slot_addresses(self):
        for ntt in range(1, 9):
            offsets = {xoffset(t, k, ntt) for t in range(8 * ntt) for k in range(4096)}
            self.assertEqual(len(offsets), 8 * ntt * 4096)
            self.assertEqual(max(offsets), len(offsets) - 1)
        for step in range(16):
            for high in range(2):
                words = [(step * 256 + (lane >> 2) * 16 + (lane & 3) * 4 + high * 128) // 4
                         for lane in range(32)]
                self.assertEqual(len({x % 32 for x in words}), 32)

    def test_ring_epoch_model(self):
        # Unordered warps may progress independently, but producer reuse needs
        # all 16 arrivals. Vary stage ranges and mix sync8/old-wide segments.
        rng = random.Random(8441)
        for trial in range(100):
            stage_count = rng.randrange(1, 300)
            produced, consumed = 0, [0] * 16
            weights, xs = [None] * 3, [None] * 3
            xseen, wseen = [set() for _ in range(3)], [set() for _ in range(3)]
            steps = 0
            while min(consumed) < stage_count:
                choices = [w for w in range(16) if consumed[w] < produced]
                ws = produced % 3
                can_produce = (produced < stage_count and
                               (weights[ws] is None or len(wseen[ws]) == 16) and
                               (xs[ws] is None or len(xseen[ws]) == 16))
                if can_produce:
                    choices.append(16)
                self.assertTrue(choices)
                who = rng.choice(choices)
                if who == 16:
                    weights[ws] = xs[ws] = produced
                    wseen[ws].clear(); xseen[ws].clear()
                    produced += 1
                else:
                    seq = consumed[who]
                    sl = seq % 3
                    self.assertEqual(weights[sl], seq)
                    self.assertEqual(xs[sl], seq)
                    wseen[sl].add(who); xseen[sl].add(who)
                    consumed[who] += 1
                steps += 1
                self.assertLessEqual(steps, stage_count * 17)
            self.assertEqual(consumed, [stage_count] * 16)

    def test_gate_rejections(self):
        admit = functions("_verify_admit")["_verify_admit"]
        rows = [[(i, .98, .98, .98, .98) for i in range(8)] for _ in range(2)]
        self.assertTrue(admit(rows))
        self.assertFalse(admit(rows[:1]))
        for value in (1.006, float("nan"), float("inf"), 0., -1.):
            bad = [list(r) for r in rows]
            bad[1][7] = (7, value, .98, .98, .98)
            self.assertFalse(admit(bad))
        for j in range(1, 5):
            bad = [[tuple(v if k != j else .99 for k, v in enumerate(r)) for r in rr] for rr in rows]
            self.assertFalse(admit(bad))
        self.assertFalse(admit([rows[0][:-1], rows[1]]))

    def test_prepare_only_timing(self):
        kernel = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn"][-1]
        calls = [ast.unparse(n.func) for n in ast.walk(kernel) if isinstance(n, ast.Call)]
        self.assertFalse(any("synchronize" in x or "item" in x or "_verify_" in x or "cpu" in x for x in calls))
        gate = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "_verify_time_pair")
        calls = [ast.unparse(n.func) for n in ast.walk(gate) if isinstance(n, ast.Call)]
        self.assertIn("flush.amax", calls)
        self.assertNotIn("flush.zero_", calls)

    def test_timed_dispatch_mock(self):
        import types
        calls = []
        class Tensor:
            def __init__(self, m):
                self.shape, self.device = (m, 4096), "fake-device"
            def contiguous(self):
                return self
            def to(self, dtype):
                return self
            def data_ptr(self):
                return 16
            def __getitem__(self, sl):
                return Tensor(sl.stop - sl.start)
        class Ext:
            def __init__(self, name):
                self.name = name
            def okc_run(self, *args):
                calls.append((self.name, args[2].shape[0]))
        torch = types.SimpleNamespace(int32="int32", float32="float32", bfloat16="bf16",
                                     empty=lambda shape, **kwargs: Tensor(shape[0]))
        ns = functions("kernel_fn")
        ns.update(torch=torch, HIDDEN=4096, CHUNK_TOKENS=384, _MOE_SEL={},
                  _moe_default=lambda m, sfx: "incumbent", _ext=lambda key: Ext(key))
        candidate = Ext("candidate")
        state = {"w": [], "scratch": [], "lo": 0, "cnt": 32, "grid": 132,
                 "verify_sync": {128: candidate, 256: candidate}}
        for m in (1, 8, 16, 17, 32, 128, 256, 384, 512):
            data = {"hidden_states": Tensor(m), "topk_ids": Tensor(m), "topk_weights": Tensor(m),
                    "config": {"local_start": 0, "local_count": 32}}
            calls.clear()
            first = ns["kernel_fn"](data, state)
            second = ns["kernel_fn"](data, state)
            self.assertIsNot(first, second)
            expected = "candidate" if m in (128, 256) else "incumbent"
            self.assertTrue(all(name == expected for name, _ in calls))
            self.assertEqual(sum(n for _, n in calls), m * 2)
        state["verify_sync"].clear()
        ns["_MOE_SEL"][128] = "incumbent-selected-fallback"
        calls.clear()
        ns["kernel_fn"]({"hidden_states": Tensor(128), "topk_ids": Tensor(128), "topk_weights": Tensor(128),
                          "config": {"local_start": 0, "local_count": 32}}, state)
        self.assertEqual(calls, [("incumbent-selected-fallback", 128)])

    def test_timing_protocol_mock(self):
        ns = functions("_verify_time_pair")
        trace = []
        class Graph:
            def __init__(self, arm):
                self.arm = arm
            def replay(self):
                trace.append(("replay", self.arm))
        class Flush:
            def amax(self):
                trace.append(("flush",))
        class Event:
            def record(self):
                pass
            def synchronize(self):
                pass
            def elapsed_time(self, other):
                return (100., 98.)[trace[-1][1]]
        for reverse, order in ((False, (0, 1, 1, 0)), (True, (1, 0, 0, 1))):
            trace.clear()
            ratio, paired = ns["_verify_time_pair"]((Graph(0), Graph(1)), Flush(), (Event(), Event()), reverse)
            self.assertAlmostEqual(ratio, .98)
            self.assertAlmostEqual(paired, .98)
            self.assertEqual([row[1] for row in trace if row[0] == "replay"], list(order) * 12)
            self.assertEqual(len(trace), 96)
            self.assertTrue(all(trace[i][0] == "flush" and trace[i+1][0] == "replay" for i in range(0, 96, 2)))

    def test_fallback_mock(self):
        # Execute the final prepare wrapper with a fake failed selector. An
        # exception after partial selection may not leak an unvalidated arm.
        import contextlib
        import io
        final = [n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == "prepare"][-1]
        state = {"w": ["weights"], "scratch": "production scratch"}
        def fail(s):
            s["verify_sync"][128] = "unvalidated"
            raise RuntimeError("injected compile/validation failure")
        class Cuda:
            @staticmethod
            def synchronize():
                pass
        class Torch:
            cuda = Cuda
        ns = {"_verify_original_prepare": lambda w: state, "_verify_select": fail, "torch": Torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[final], type_ignores=[])), "<prepare wrapper>", "exec"), ns)
        with contextlib.redirect_stdout(io.StringIO()):
            result = ns["prepare"]({"test": "weights"})
        self.assertIs(result, state)
        self.assertEqual(result["verify_sync"], {})
        self.assertEqual(result["scratch"], "production scratch")

    def test_cpp_template_syntax(self):
        compiler = shutil.which("clang++")
        if not compiler:
            self.skipTest("no host C++ compiler; NVRTC unavailable too")
        # Host syntax only: mocks do not pretend to validate PTX or CUDA codegen.
        prefix = r'''
#define __device__
#define __forceinline__ inline
using uint32_t = unsigned int;
using size_t = unsigned long;
struct uint4 { uint32_t x,y,z,w; };
struct uint2 { uint32_t x,y; };
struct Dim { int x; } threadIdx;
#define WARP_IDX(tid) ((tid) >> 5)
constexpr int WGS_PIECE=9216, WGS_NS=3, WGS_NX=3, UNIT_BYTES=36864;
constexpr int WGS_XS=32768, WGS_FIXW=4096, BW_F=0, BW_E=3, BX_F=6, BX_E=9;
constexpr int OKC_WGSDFIN=2;
struct Params { uint32_t zero; };
struct Smem { unsigned long long bars[24]; };
struct WgsPend { int kind,u,slot,nst; };
void wgs_wait(unsigned long long*, uint32_t, int);
void mbar_arrive1(unsigned long long*);
void __syncwarp();
void dq4(uint32_t,uint32_t&,uint32_t&,uint32_t&,uint32_t&);
uint32_t __byte_perm(uint32_t,uint32_t,uint32_t);
uint32_t bmul2l(uint32_t,uint32_t), bmul2h(uint32_t,uint32_t);
void mma_bf16(float*,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t);
template <bool B> void wgs_flush(const Params&,Smem&,WgsPend&);
template <bool B> void wgs_epi_tile(const Params&,const Smem&,int,int,int,const float*);
float* wgs_fix(const Params&,int);
'''
        postfix = r'''
template void wgs_seg_sync8<false>(const Params&, Smem&, unsigned char*,int,int,int,int,int&,WgsPend&);
template void wgs_seg_sync8<true>(const Params&, Smem&, unsigned char*,int,int,int,int,int&,WgsPend&);
'''
        stub = OUT / "sync8_syntax.cpp"
        stub.write_text(prefix + (ROOT / "candidates/moe_verify_sync_core.cuh").read_text() + postfix)
        result = subprocess.run([compiler, "--target=x86_64-unknown-linux-gnu", "-x", "c++", "-std=c++17", "-fsyntax-only", str(stub)],
                                capture_output=True, text=True)
        (OUT / "syntax.log").write_text(result.stdout + result.stderr)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
