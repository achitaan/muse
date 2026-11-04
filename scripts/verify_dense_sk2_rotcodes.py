#!/usr/bin/env python3
"""Offline checks for the actual candidate; no CUDA, OKC or harness imports.

Expected payloads come from the BF16 bit format and the public x @ W.T
equation. The numerical model is NOT an emulator of Hopper's MMA rounding.
"""
import ast
import json
from pathlib import Path
import re
import subprocess
import types

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidates/dense_sk2_rotcodes.py"
BASE = ROOT / "board_best/dense_kbv2_scored_1.5886.py"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
report = {}
source = CANDIDATE.read_text()
tree = ast.parse(source)
base_tree = ast.parse(BASE.read_text())
factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_make_vjeux")


def extract(nodes, names, env):
    selected = [n for n in nodes if (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)
                or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(CANDIDATE), "exec"), env)


env = {"torch": torch, "_dt_torch": torch}
extract(factory.body, {
    "CUDA_SRC", "MBAR_HELPERS", "MAP_SRC", "SK2_SRC", "CPP_SRC", "SK2_CONFIGS", "SK2_PICK", "SK2_K",
    "SK2_DEFAULT_RING_CAP_KB", "SK2_RING_SMEM_BUDGET_BYTES", "SK2_MIN_STAGES", "SK2_MAX_STAGES",
    "EXPONENT_WINDOW", "EXPONENT_VALUES", "MAX_ESCAPE_DENOMINATOR", "BASE4_REPLICATE",
    "_sk2_geom", "_sk2_name", "_sk2_tokr", "_sk2_src", "sk2_tileunits", "_exponent_window", "_escapes",
    "sk2_compress", "sk2_stream", "sk2_table", "sk2_balance_perm", "sk2_table_add_perm", "UNPOOLER_SLICE_ROWS",
}, env)
# The factory constants are read in source order; UNPOOLER_SLICE_ROWS precedes SK2_CONFIGS.
env["_K"] = {"vjeux": {
    "sk2rot_sources": tuple(env[k] for k in ("CUDA_SRC", "MBAR_HELPERS", "MAP_SRC", "SK2_SRC")),
    "sk2rot_configs": env["SK2_CONFIGS"], "sk2rot_name": env["_sk2_name"], "sk2rot_geom": env["_sk2_geom"],
}}
extract(tree.body, {"_SR_CELLS", "_SR_DRAWS", "_SR_GAIN", "_SR_CPP", "_sr_source", "_sr_rotate", "_sr_call",
                     "_sr_valid", "_sr_accept"}, env)


def static_checks():
    # All production AST is unchanged except the factory's helper-export return.
    for old, new in zip(base_tree.body, tree.body):
        if isinstance(old, ast.FunctionDef) and old.name == "_make_vjeux":
            assert ast.dump(ast.Module(body=old.body[:-1], type_ignores=[])) == ast.dump(ast.Module(body=new.body[:-1], type_ignores=[]))
        else:
            assert ast.dump(old) == ast.dump(new)
    cuda = env["_sr_source"]()
    assert "@@" not in cuda
    assert len(re.findall(r"if \(cfg == \d+ && tokr == (?:16|32)\)", cuda)) == 8
    # Everything in the SK2 body, including all barriers and math, is literally unchanged.
    expected = env["SK2_SRC"].replace("@@SK2_DISPATCH@@", "DISPATCH")
    actual = cuda[cuda.index("\n#define SK2_WAIT"):]
    actual = re.sub(r"  if \(cfg == .*?\n  TORCH_CHECK", "  DISPATCH\n  TORCH_CHECK", actual, flags=re.S)
    # First dispatch is not indented in the replacement, but the placeholder was.
    actual = actual.replace("rot_decode8(", "decode8(").replace("rot_sk2_kernel", "sk2_kernel")
    actual = actual.replace("launch_rot_sk2", "launch_sk2").replace("void rot_sk2(", "void sk2(")
    assert expected.lstrip("\n") == actual.lstrip("\n")
    # Assembly matches the parent's own exact specialization string for these cells.
    parent_lines = [line.strip() for line in env["_sk2_src"](True).splitlines() if line.strip().startswith("if (cfg ==")]
    new_lines = [line.strip().replace("launch_rot_sk2", "launch_sk2") for line in cuda.splitlines() if line.strip().startswith("if (cfg ==")]
    assert all(line in parent_lines for line in new_lines)
    assert set(m for _, m in env["_SR_CELLS"]) == {16, 32}
    forbidden = {"subprocess", "requests", "urllib", "inspect", "importlib.machinery"}
    tail = tree.body[len(base_tree.body):]
    for n in ast.walk(ast.Module(body=tail, type_ignores=[])):
        if isinstance(n, ast.Import):
            assert not any(a.name in forbidden for a in n.names)
    (RESULTS / "dense-sk2-rotcodes-generated.cu").write_text(cuda)
    report["static"] = {"incumbent_ast_preserved": True, "kernel_body_preserved": True, "native_specializations": len(new_lines)}
    return cuda


def rotate_np(c):
    c = np.asarray(c, dtype=np.uint32)
    signs = c & np.uint32(0x80808080)
    return ((c & np.uint32(0x77777777)) | ((c & np.uint32(0x08080808)) << np.uint32(4))
            | (signs << np.uint32(4)) | (signs >> np.uint32(28)))


def decode_np(low, c, base, rotated):
    c = np.asarray(c, dtype=np.uint32)
    if rotated:
        other = (c >> np.uint32(4)) | (c << np.uint32(28))
        ha = (c & np.uint32(0x87878787)) + np.uint32(base * 0x01010101)
        hb = (other & np.uint32(0x87878787)) + np.uint32(base * 0x01010101)
    else:
        ha = ((c & np.uint32(0x07070707)) | ((c << np.uint32(4)) & np.uint32(0x80808080))) + np.uint32(base * 0x01010101)
        hb = (((c >> np.uint32(4)) & np.uint32(0x07070707)) | (c & np.uint32(0x80808080))) + np.uint32(base * 0x01010101)
    high = np.stack([((h >> np.uint32(i * 8)) & np.uint32(255)) for h in (ha, hb) for i in range(4)], axis=-1)
    return (high.astype(np.uint16) << np.uint16(8)) | np.asarray(low, dtype=np.uint16)


def codec_checks():
    rng = np.random.default_rng(0x5C21632)
    # Every BF16 payload under every legal window: independent expected clamped high bytes.
    payload = np.arange(65536, dtype=np.uint16)
    exhaustive = 0
    for base in range(121):
        values = np.stack([np.roll(payload, i * 7937) for i in range(8)], axis=-1)
        hi = values.astype(np.uint32) >> np.uint32(8)
        offs = np.clip((hi & np.uint32(127)).astype(np.int32) - base, 0, 7).astype(np.uint32)
        codes = (hi >> np.uint32(7)) * np.uint32(8) | offs
        c = sum((codes[:, i] << np.uint32(i * 8)) | (codes[:, i + 4] << np.uint32(i * 8 + 4)) for i in range(4))
        low = values & np.uint16(255)
        expected = (((hi & np.uint32(128)) | (offs + base)) << np.uint32(8)) | low
        # Expected bits are from the BF16 sign/exponent/mantissa definition, not the old decoder.
        assert np.array_equal(decode_np(low, c, base, False), expected)
        assert np.array_equal(decode_np(low, rotate_np(c), base, True), expected)
        exhaustive += values.size
    words = np.concatenate([np.array([0, 0xFFFFFFFF], dtype=np.uint32), np.uint32(1) << np.arange(32, dtype=np.uint32),
                            rng.integers(0, 2**32, 1_000_000, dtype=np.uint32)])
    rot = rotate_np(words)
    s = rot & np.uint32(0x08080808)
    inv = (rot & np.uint32(0x77777777)) | ((rot & np.uint32(0x80808080)) >> np.uint32(4)) | (s >> np.uint32(4)) | (s << np.uint32(28))
    assert np.array_equal(inv, words)
    # The actual Torch transformer must agree with the independent NumPy bit permutation.
    codes = np.resize(words, (1001, 64)).astype(np.uint32).view(np.uint8)
    low = rng.integers(0, 256, (1001, 512), dtype=np.uint8)
    stream = torch.from_numpy(np.concatenate([low, codes], axis=-1)).reshape(7, 143, 768)
    new = env["_sr_rotate"](stream, 61 * 0x01010101)
    assert torch.equal(stream[..., :512], new[..., :512])
    assert np.array_equal(new.numpy().reshape(-1, 768)[:, 512:].copy().view(np.uint32), rotate_np(codes.copy().view(np.uint32)))
    assert new.data_ptr() != stream.data_ptr()
    rejected = 0
    for st, b in [(stream, 121 * 0x01010101), (stream, 0x101), (stream.to(torch.int32), 0),
                  (stream[..., :767], 0), (stream.transpose(0, 1), 0)]:
        try:
            env["_sr_rotate"](st, b)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("malformed stream admitted")
    report["codec"] = {"exhaustive_payload_position_window_cases": exhaustive, "random_and_basis_word_cases": words.size,
                       "actual_torch_word_cases": codes.size // 4, "malformed_packs_rejected": rejected}


def cpu_cpp(cuda):
    # Compile the actual decoder expressions, replacing only CUDA intrinsics with CPU semantics.
    new = cuda[cuda.index("__device__ __forceinline__ void rot_decode8("):]
    new = new[:new.index("\n}") + 2].replace("__device__ __forceinline__", "inline")
    old = env["CUDA_SRC"][env["CUDA_SRC"].index("__device__ __forceinline__ void decode8("):]
    old = old[:old.index("\n}") + 2].replace("__device__ __forceinline__", "inline")
    cpp = r"""
#include <cstdint>
#include <cstdio>
#include <cassert>
uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t d=0;
  for (int i=0;i<4;i++) { uint32_t idx=(sel>>(4*i))&7, v=(idx<4?a:b)>>(8*(idx%4)); d|=(v&255)<<(8*i); }
  return d;
}
uint32_t lop3_sel(uint32_t a,uint32_t b,uint32_t c) { return (a&~c)|(b&c); }
uint32_t __funnelshift_r(uint32_t a,uint32_t b,int s) { return (a>>s)|(b<<(32-s)); }
""" + old + "\n" + new + r"""
int main() {
  uint64_t count=0;
  for(uint32_t base=0;base<=120;base++) for(uint32_t p=0;p<65536;p++) {
    uint32_t low[2]={}, c=0, expected[4]={};
    for(int i=0;i<8;i++) {
      uint32_t v=(p+uint32_t(i)*7937u)&65535u, hi=v>>8;
      uint32_t o=(hi&127)<base?0:((hi&127)>base+7?7:(hi&127)-base);
      uint32_t code=(hi>>7)*8+o, dec=(((hi&128)|(base+o))<<8)|(v&255);
      low[i/4]|=(v&255)<<(8*(i%4)); c|=code<<(8*(i%4)+4*(i/4)); expected[i/2]|=dec<<(16*(i%2));
    }
    uint32_t sign=c&0x80808080u;
    uint32_t rot=(c&0x77777777u)|((c&0x08080808u)<<4)|(sign<<4)|(sign>>28), a[4],b[4];
    decode8(low[0],low[1],c,base*0x01010101u,a);
    rot_decode8(low[0],low[1],rot,base*0x01010101u,b);
    for(int i=0;i<4;i++) assert(a[i]==expected[i] && b[i]==expected[i]);
    count++;
  }
  std::printf("%llu word/window cases\n",(unsigned long long)count);
}
"""
    path = RESULTS / "dense-sk2-rotcodes-decoder.cpp"
    binary = RESULTS / ".dense-sk2-rotcodes-decoder"
    path.write_text(cpp)
    subprocess.run(["clang++", "-O2", "-std=c++20", str(path), "-o", str(binary)], check=True)
    result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
    binary.unlink()  # generated executable only; keep the readable source as evidence
    report["cpu_cpp"] = result.stdout.strip()


def unpack_stream(stream, cfg, base):
    name, layer, _, R, NT, NWR, WSK, UPW, _, _ = cfg
    a = stream.numpy()
    nst, ncta = a.shape[:2]
    n, k = ncta * R, nst * WSK * UPW * 64
    decoded = np.zeros((n, k), dtype=np.uint16)
    visits = np.zeros((n, k), dtype=np.uint8)
    for st in range(nst):
        for cta in range(ncta):
            for w in range(NWR * WSK):
                for uu in range(UPW):
                    unit = st * WSK * UPW + (w % WSK) * UPW + uu
                    for nt in range(NT):
                        raw = a[st, cta, w, uu, nt]
                        low = raw[:512].reshape(32, 16)
                        words = raw[512:].copy().view(np.uint32).reshape(32, 2)
                        vals = np.concatenate([decode_np(low[:, :8], words[:, 0], base, True),
                                               decode_np(low[:, 8:], words[:, 1], base, True)], axis=1)
                        # Independent documented B-fragment map; not sk2_tileunits' reshape/permute.
                        for lane in range(32):
                            row = cta * R + ((w // WSK) * NT + nt) * 8 + (lane >> 2)
                            for v in range(16):
                                col = unit * 64 + 16 * (v >> 2) + 8 * ((v >> 1) & 1) + 2 * (lane & 3) + (v & 1)
                                decoded[row, col] = vals[lane, v]
                                visits[row, col] += 1
    assert np.all(visits == 1)
    return decoded


def numeric_model(decoded_bits, tab, cfg, m, nstage, x):
    # Model SK2's real K partition and correction ordering with NumPy FP32 dot16s.
    # NumPy's inner-dot16 rounding is not guaranteed to match mma.sync on hardware.
    _, _, _, R, NT, NWR, WSK, UPW, NACC, _ = cfg
    if m == 32 and nstage >= 4 and NT == 4:
        NACC = 1
    decoded = (decoded_bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    n, k = decoded.shape
    nst = k // (WSK * UPW * 64)
    nesc = min(4, nstage)
    acc = np.zeros((WSK, NACC, m, n), np.float32)
    for st in range(nst):
        for wk in range(WSK):
            for uu in range(UPW):
                unit = (st * WSK + wk) * UPW + uu
                for j in range(4):
                    at = unit * 64 + j * 16
                    h = (uu * 4 + j) % NACC
                    acc[wk, h] += x[:, at:at + 16] @ decoded[:, at:at + 16].T
    corr = np.zeros((nesc, m, n), np.float32)
    for cta, table in enumerate(tab.numpy()):
        for ek, eb in table[1:int(table[0, 0]) + 1]:
            row, col = (int(ek) >> 16) + cta * R, int(ek) & 65535
            actual = np.array([np.uint32(int(eb) << 16)], np.uint32).view(np.float32)[0]
            ew = (col // (WSK * UPW * 64)) % nesc
            corr[ew, :, row] += (actual - decoded[row, col]) * x[:, col]
    out = np.zeros((m, n), np.float32)
    for ew in range(nesc):
        out += corr[ew]
    for wk in range(WSK):
        partial = acc[wk, 0].copy()
        for h in range(1, NACC):
            partial += acc[wk, h]
        out += partial
    return torch.from_numpy(out).to(torch.bfloat16)


def packed_and_numerical():
    rng = np.random.default_rng(0x1632)
    worst, counts, escapes, numbers = 0.0, 0, 0, []
    configs = env["SK2_CONFIGS"]
    for layer, m in env["_SR_CELLS"]:
        cfg = next(c for c in configs if c[0] == env["_sk2_name"](layer, m))
        _, _, _, R, NT, NWR, WSK, UPW, _, _ = cfg
        n, k = R * 2, env["SK2_K"][layer]
        hi = rng.integers(57, 65, (n, k), dtype=np.uint16) | (rng.integers(0, 2, (n, k), dtype=np.uint16) << 7)
        low = rng.integers(0, 256, (n, k), dtype=np.uint16)
        bits = (hi << 8) | low
        # Sparse true escapes on both sides of the chosen window, plus zero.
        bits.reshape(-1)[::2003] = np.uint16(0x3881)
        bits.reshape(-1)[1000::2003] = np.uint16(0xC181)
        bits[0, 0] = 0
        w = torch.from_numpy(bits.copy().view(np.int16)).view(torch.bfloat16)
        original = w.clone()
        perm = env["sk2_balance_perm"](w, n, R) if layer == "qkv_gate" else None
        if perm is not None:
            w = w[perm].contiguous()
        comp = env["sk2_compress"](w)
        assert comp is not None
        stream = env["sk2_stream"](comp["packed"], n, R, NT, NWR, WSK, UPW)
        tab, pmax = env["sk2_table"](comp, n, R)
        new = env["_sr_rotate"](stream, comp["base4"])
        decoded = unpack_stream(new, cfg, comp["base4"] & 255)
        corrected = decoded.copy()
        for cta, table in enumerate(tab.numpy()):
            for ek, eb in table[1:int(table[0, 0]) + 1]:
                row, col = (int(ek) >> 16) + cta * R, int(ek) & 65535
                corrected[row, col] = int(eb)
                escapes += 1
        # Expected payloads are ORIGINAL BF16 data, independently indexed above.
        assert np.array_equal(corrected, w.view(torch.int16).numpy().view(np.uint16))
        if perm is not None:
            perm_tab, pm2 = env["sk2_table_add_perm"](tab, pmax, perm, R)
            output_map = perm_tab[:, pmax:pmax + R // 2].reshape(-1).long()
            assert torch.equal(output_map, perm)
            assert (perm_tab[:, 0, 1] == pmax).all()
            assert pm2 % 2 == 0
        stages = env["_sk2_geom"](cfg, m)[1]
        normal = torch.from_numpy(rng.standard_normal((m, k)).astype(np.float32)).to(torch.bfloat16)
        outlier = normal.clone()
        outlier[:, ::97] *= 32
        probes = torch.zeros_like(normal)
        for row in range(m):
            probes[row, (row * 173 + 11) % k] = 8 if row % 2 else -32
        for x in (normal, outlier, probes, torch.zeros_like(normal)):
            result = numeric_model(decoded, tab, cfg, m, stages, x.float().numpy())
            if perm is not None:
                remapped = torch.empty_like(result)
                remapped[:, perm] = result
                result = remapped
            # Source of expected numerical values: public equation, FP64 from actual BF16 inputs/weights.
            ref = x.double() @ original.double().T
            err = float((result.double() - ref).norm() / ref.norm().clamp_min(1e-30))
            per = (result.double() - ref).norm(dim=1)
            rn = ref.norm(dim=1)
            rowerr = float((per / torch.maximum(rn, rn.mean() * 0.1).clamp_min(1e-30)).max())
            assert err < 0.008 and rowerr < 0.04, (layer, m, err, rowerr)
            if x is probes:
                assert torch.equal(result, ref.to(torch.bfloat16))
            worst = max(worst, err)
            counts += 1
        numbers.append({"layer": layer, "m": m, "modeled_n": n, "native_k": k, "nstage": stages})
    report["numerical"] = {"models": counts, "escapes": escapes, "worst_rel_l2": worst, "geometries": numbers,
                           "caveat": "FP32 NumPy dot16 model; not actual GPU MMA rounding, barriers or performance"}


def wrapper_and_gate():
    # Execute the actual final wrappers with a mock native call: fresh output, correct arguments, no parent for adopted cells.
    final_kernel = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "kernel_fn"][-1]
    e = env.copy()
    extract(tree.body, {"_V6_K"}, e)
    calls, fallback = [], []
    e["_ctx_bind"] = lambda x: calls.append("bind")
    e["_sr_parent_kernel"] = lambda d, s: fallback.append(d["config"].copy()) or "parent"
    exec(compile(ast.Module(body=[final_kernel], type_ignores=[]), str(CANDIDATE), "exec"), e)
    class Mod:
        def rot_sk2(self, *args):
            calls.append(args)
            args[2].zero_()
    mod = Mod()
    for layer, m in env["_SR_CELLS"]:
        k, n = e["_V6_K"][layer], {"o": 8192, "unpooler": 8192, "pooler": 4096, "qkv_gate": 2304}[layer]
        plan = (17, torch.zeros(1), torch.zeros(1), 2, 61 * 0x01010101, n)
        state = {"sk2rot": {(layer, m): plan}, "sk2rot_mod": mod, "vjeux": {"dummy": torch.zeros(1)}}
        data = {"x": torch.zeros((m, k), dtype=torch.bfloat16), "config": dict(layer=layer, m=m, n=n, k=k)}
        a, b = e["kernel_fn"](data, state), e["kernel_fn"](data, state)
        assert a.shape == (m, n) and a.dtype == torch.bfloat16 and a.data_ptr() != b.data_ptr()
        assert calls[-1][9:12] == (17, m, 0)
        for changed in ({"m": 8}, {"n": n - 8}, {"k": k - 64}):
            cfg = dict(data["config"], **changed)
            assert e["kernel_fn"](dict(data, config=cfg), state) == "parent"
        assert e["kernel_fn"](dict(data, x=data["x"].float()), state) == "parent"
        assert e["kernel_fn"](dict(data, x=data["x"].T.contiguous().T), state) == "parent"
        assert e["kernel_fn"](data, {"sk2rot": {}}) == "parent"
    # Admission boundary: no partial, NaN, zero, marginal or one-pass wins.
    accept = env["_sr_accept"]
    assert accept([(0.97, 0.97)] * 3)
    assert accept([(0.975, 0.975)] * 3)
    assert not accept([(0.97, 0.97)] * 2)
    for bad in [(0.976, 0.96), (0.96, 0.976), (float("nan"), 0.96), (0.0, 0.0), (-1.0, -1.0)]:
        assert not accept([(0.97, 0.97), bad, (0.97, 0.97)])
    report["wrappers"] = {"fresh_output_pairs": len(env["_SR_CELLS"]), "fallback_cases": len(fallback), "gate_cases": 8}


def control_flow_checks():
    """Exercise the actual three-draw gate and prepare fallback, with CPU graph mocks."""
    print("CPU GRAPH MOCK CONTROL TESTS: SR timings below are synthetic, NOT H200 measurements")
    e = env.copy()
    fake_cuda = types.SimpleNamespace(active=None, synchronize=lambda: None, CUDAGraph=None,
                                      get_device_capability=lambda dev: (9, 0))

    class Graph:
        def __init__(self):
            self.actions = []

        def replay(self):
            for action in self.actions:
                action()

    class Capture:
        def __init__(self, graph):
            self.graph = graph

        def __enter__(self):
            fake_cuda.active = self.graph

        def __exit__(self, *args):
            fake_cuda.active = None

    fake_cuda.CUDAGraph, fake_cuda.graph = Graph, Capture

    class TorchProxy:
        cuda = fake_cuda

        def __getattr__(self, name):
            return getattr(torch, name)

        def ones(self, shape, **kwargs):
            # A timing-control mock does not need the real 512 MiB flush allocation.
            return torch.ones(16, **kwargs)

    e["_dt_torch"] = TorchProxy()
    extract(tree.body, {"_DTBudget", "_dt_check", "_dt_errors", "_dt_close", "_dt_inputs", "_sr_valid", "_sr_call", "_sr_cell"}, e)
    e["_dt_time"] = types.SimpleNamespace(monotonic=lambda: 0.0)
    e["_V6_K"], e["_v6_n"] = {"o": 16}, lambda layer, m: 8
    weight = torch.arange(128, dtype=torch.float32).reshape(8, 16).remainder(17).sub(8).to(torch.bfloat16)
    mode = {"wrong": False, "twin_wrong": False, "post_timing": False, "ratio": (0.97, 0.97)}
    traces = {"timings": 0, "captures": 0}

    def enqueue(x, out, candidate=False, twin=False):
        def action():
            expected = (x.double() @ weight.double().T).to(torch.bfloat16)
            if mode["wrong"] and candidate or mode["twin_wrong"] and twin:
                expected = expected + 1
            out.copy_(expected)
        action()
        if fake_cuda.active is not None:
            fake_cuda.active.actions.append(action)
            traces["captures"] += 1

    class Native:
        def sk2(self, *args):
            enqueue(args[1], args[2], twin=True)

        def rot_sk2(self, *args):
            enqueue(args[1], args[2], candidate=True)

    native = Native()
    plan = (1, torch.zeros(1), torch.zeros(1), 2, 0, 8)
    e["_K"] = {"vjeux": {"warm": lambda: native, "sk2rot_name": lambda layer, m: "native"}}
    state = {"vjeux": {"dummy": torch.zeros(1), "sk2": {"native": plan}}}

    def parent(data, state):
        out = torch.empty((data["x"].shape[0], 8), dtype=torch.bfloat16)
        enqueue(data["x"], out)
        return out

    def abba(*args):
        traces["timings"] += 1
        if mode["post_timing"]:
            mode["wrong"] = True
        return (10.0, 9.7, *mode["ratio"])

    e["_sr_parent_kernel"], e["_dt_abba"] = parent, abba
    run = lambda: e["_sr_cell"]("o", 16, plan, native, state, {"o": weight}, None, None,
                                    torch.Generator().manual_seed(1234), 100.0)
    assert run()
    assert traces == {"timings": 3, "captures": 9}
    traces.update(timings=0, captures=0)
    mode["ratio"] = (0.976, 0.96)
    assert not run() and traces["timings"] == 1
    mode.update(ratio=(0.97, 0.97), twin_wrong=True)
    traces.update(timings=0, captures=0)
    assert not run() and traces["timings"] == 0  # numerical tolerance cannot replace exact twin equality
    mode.update(twin_wrong=False, post_timing=True)
    traces.update(timings=0, captures=0)
    assert not run() and traces["timings"] == 1  # fresh-input postcheck is not dead code
    mode.update(wrong=False, post_timing=False)
    # Full prepare: optional build errors, local cell errors, and deadline all preserve the incumbent.
    final_prepare = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "prepare"][-1]
    exec(compile(ast.Module(body=[final_prepare], type_ignores=[]), str(CANDIDATE), "exec"), e)
    e["_SR_BUDGET_S"] = 285.0
    e["_ctx_bind"] = lambda x: None
    e["_sr_rotate"] = lambda stream, base: stream.clone()
    e["_sr_parent_prepare"] = lambda weights: {"incumbent": "preserved", "vjeux": {"sk2": {"native": plan}}}
    e["_SR_CELLS"] = (("o", 32), ("o", 16))
    e["_sr_start"] = lambda: {"thread": types.SimpleNamespace(join=lambda **kw: None), "error": "expected build failure"}
    weights = {"o": types.SimpleNamespace(device=torch.device("cpu"), is_cuda=True)}
    result = e["prepare"](weights)
    assert result["sk2rot"] == {} and result["incumbent"] == "preserved"
    e["_sr_start"] = lambda: {"thread": types.SimpleNamespace(join=lambda **kw: None), "mod": native}
    e["_sr_cell"] = lambda layer, m, *args: m == 32
    result = e["prepare"](weights)
    assert set(result["sk2rot"]) == {("o", 32)} and result["sk2rot_mod"] is native
    assert not any(key in result for key in ("flush", "sink", "graphs", "xs", "refs", "packs"))

    def budget_after_first(layer, m, *args):
        if m == 16:
            raise e["_DTBudget"]()
        return True

    e["_sr_cell"] = budget_after_first
    result = e["prepare"](weights)
    assert set(result["sk2rot"]) == {("o", 32)} and result["incumbent"] == "preserved"
    report["control_flow"] = {"actual_three_draw_gate": "passed", "pre_timing_twin_failure": "rejected",
                              "post_timing_corruption": "rejected", "marginal_win": "rejected",
                              "build_failure_fallback": "passed", "partial_adoption_and_deadline": "passed"}


def bandwidth():
    times = {("qkv_gate", 16): 9.70, ("o", 16): 5.22, ("pooler", 16): 13.98, ("unpooler", 16): 13.79,
             ("qkv_gate", 32): 9.80, ("o", 32): 5.76, ("pooler", 32): 14.97, ("unpooler", 32): 14.65}
    shapes = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}
    rows = []
    for (layer, m), us in times.items():
        n, k = shapes[layer]
        b = n * k * 3 // 2
        rows.append(dict(layer=layer, m=m, encoded_weight_bytes=b, time_us=us,
                         weight_only_TBps=b / us / 1e6, fraction_of_4p8_TBps=b / us / 4.8e6,
                         weight_stream_floor_us=b / 4.8e6))
    report["bandwidth"] = rows


if __name__ == "__main__":
    torch.set_num_threads(2)
    cuda = static_checks()
    codec_checks()
    cpu_cpp(cuda)
    packed_and_numerical()
    wrapper_and_gate()
    control_flow_checks()
    bandwidth()
    report["unverified"] = ["CUDA host/device compilation", "NVRTC/ptxas", "GPU registers/spills/SASS", "barrier execution",
                            "actual H200 numerics/graphs", "GPU prepare time", "H200 timing"]
    path = RESULTS / "dense-sk2-rotcodes-offline.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
