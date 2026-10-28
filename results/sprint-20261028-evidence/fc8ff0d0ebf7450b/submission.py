# okc: problem=muse_bf16_proj
# Derived from NathenFernandes/muse-bf16-proj commit 7385e5d8f70980cfd5322ea254edc0b2b8ca90f7.
# Upstream v18 with its additive lossless decoder extended to four more kernel families; MIT attribution retained.
# MIT License
# 
# Copyright (c) 2026 Nathen Fernandes
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# Large CUDA/C++ source strings are stored zlib-compressed to fit the 1 MB submission limit;
# _zsrc() restores each one byte-for-byte at import. Nothing else changes.
import base64 as _zsrc_b64, zlib as _zsrc_zlib
def _zsrc(payload: str) -> str:
    return _zsrc_zlib.decompress(_zsrc_b64.b64decode(payload)).decode()
"""Dense projections: small_m (12-bit lossless or BF16 register-direct GEMV) for the shapes it claims, then the
register-direct swap-AB GEMV for small m, 12-bit lossless pooler weights at m = 1, and a TMA + wgmma router for
64 < m <= 256. Shapes no kernel claims fall back to SGLang's own dispatch."""

import hashlib
import zlib
from typing import TypedDict

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config




# ---- small_m/small_m_kernels.py ----


SM_CUDA_TEMPLATE = _zsrc('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')

SM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

SM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
SM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
SM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
SM_ESC_CAP = 32


def sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)


def sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7))
        cases.append(f"    case {i}: return sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return SM_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def sm_build(instances, name="small_m_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, SM_CPP + extra_cpp, cuda_sources=sm_cuda_source(instances) + extra_cuda,
                       functions=["sm_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def sm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def _to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def _to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = _to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = _to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def sm_c12_unpack(data, base, esc):
    """Inverse of sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(SM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def sm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def sm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [sm_units(hi), sm_units(lo)]
    else:
        mats = [sm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = sm_c12_bytes(v)
        if not torch.equal(sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(sm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = sm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > SM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = _to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.sm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


# ---- deliverable API ----
# Exact token counts take precedence over token-tile entries. Geometry and reduction mapping are H100-validated;
# the compressed m=16/32 configurations have not yet been measured on the H200 judge.
SM_TABLE_M = {
    ("router", 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8)),
}
SM_TABLE_MT = {
    ("pooler", 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)),
    ("qkv_gate", 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)),
    ("o", 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("o", 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("unpooler", 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)),
    ("router", 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)),
}
SM_MAX_M = 32

SM_TABLE_M.update({
    ("router", 8): SM_TABLE_MT[("router", 1)],
    ("qkv_gate", 8): SM_TABLE_MT[("qkv_gate", 1)],
    ("pooler", 8): SM_TABLE_MT[("pooler", 1)],
    ("o", 8): SM_TABLE_MT[("o", 1)],
    ("unpooler", 1): SM_TABLE_MT[("unpooler", 1)],
    ("unpooler", 8): SM_TABLE_MT[("unpooler", 1)],
})
SM_TABLE_MT[('pooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT[('pooler', 4)] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
SM_TABLE_MT[('qkv_gate', 2)] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
SM_TABLE_MT[('unpooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))


def sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


def sm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = sm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = sm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = sm_verify_onehot(ext, inst_id[inst], sp, sm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


def prepare_small_m(weights, table_m=None, table_mt=None, name="small_m_gemv"):
    table_m = dict(SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if sm_bf16_twin(inst):
            instances.append(sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = sm_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and sm_bf16_twin(inst):
            plan = sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, sm_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    return st


def run_small_m(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= SM_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", sm_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != SM_LAYER_K[layer] or not x.is_cuda:
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    if sm_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out


COMPRESSED_PAIRS = {("pooler", 1)}


# ---- m32/m32_kernels.py ----


M32_CUDA_TEMPLATE = _zsrc('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')

M32_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int m32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

M32_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
M32_NL = {0: 4, 1: 3, 2: 8, 3: 6}
M32_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
M32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
M32_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
M32_ESC_CAP = 32


def m32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)


def m32_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,)))
        cases.append(f"    case {i}: return m32_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return M32_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def m32_build(instances, name="m32_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, M32_CPP + extra_cpp, cuda_sources=m32_cuda_source(instances) + extra_cuda,
                       functions=["m32_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def m32_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def m32_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def m32_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def m32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = m32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def m32_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(M32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = m32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def m32_c12_unpack(data, base, esc):
    """Inverse of m32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(M32_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def m32_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def m32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [m32_units(hi), m32_units(lo)]
    else:
        mats = [m32_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([m32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = m32_c12_bytes(v)
        if not torch.equal(m32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(m32_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def m32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def m32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = m32_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, M32_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > M32_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = m32_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def m32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


# H100-selected candidates; H200 confirmation is recorded separately.
M32_TABLE_M = {}
M32_MAX_M = 32
M32_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def m32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


def m32_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def m32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def m32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if m32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def m32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if m32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def m32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def m32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = m32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = m32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = m32_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = m32_verify_onehot(ext, inst_id[inst], sp, m32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            m32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


def prepare_m32(weights, table_m=None, table_mt=None, name="m32_gemv"):
    table_m = dict(M32_TABLE_M if table_m is None else table_m)
    table_mt = dict(M32_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = m32_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, M32_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if m32_bf16_twin(inst):
            instances.append(m32_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = m32_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and m32_bf16_twin(inst):
            plan = m32_plan(ext, inst_id, packs, streams, weights[layer], layer, key, m32_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    st["outputs"] = {layer: torch.empty((M32_MAX_M, w.shape[0]), dtype=w.dtype, device=w.device)
                     for layer, w in weights.items() if any(key[0] == layer for key in st["plans"])}
    return st


def run_m32(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= M32_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", m32_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != M32_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=st["outputs"][layer].dtype, device=x.device)
    if m32_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out


# ---- wide2/wide2_kernels.py ----



def mma(n):
    regs=','.join(f'%{i}' for i in range(n//2))
    outs=','.join(f'"+f"(a[{i}])' for i in range(n//2))
    return f'''__device__ void mma{n}(float* a,uint64_t x,uint64_t y) {{
      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(y));
    }}'''


CUDA=_zsrc("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")
CPP=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int launch_ss(const void*,const void*,void*,int,int,int,int,int,int,int,int,cudaStream_t);
int launch(torch::Tensor x,torch::Tensor w,torch::Tensor y,int bn,int split,int stages,int hint,int plain) {
  return launch_ss(x.data_ptr(),w.data_ptr(),y.data_ptr(),x.size(0),y.size(1),x.size(1),bn,split,stages,hint,plain,at::cuda::getCurrentCUDAStream());
}
'''


def wide2_build():
    source=CUDA.replace('MMA_FUNCTIONS','\n'.join(mma(n) for n in [64,128,256]))
    return load_inline('wide2_final_v1',CPP,cuda_sources=source,functions=['launch'],
        extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'])


CANDIDATE_CONFIGS = {
    (128, "qkv_gate"): (128, 3, 8, 1),
    (128, "o"): (128, 1, 8, 1),
    (128, "pooler"): (128, 2, 8, 1),
    (128, "unpooler"): (128, 1, 9, 1),
    (256, "qkv_gate"): (256, 3, 5, 1),
    (256, "o"): (256, 1, 5, 1),
    (256, "pooler"): (256, 2, 5, 1),
    (256, "unpooler"): (256, 1, 5, 1),
}


def prepare_wide2(weights) -> dict:
    device = weights["qkv_gate"].device
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0):
        return {"configs": {}}
    h200_keys = [(128, "qkv_gate"), (128, "pooler"), (256, "pooler")]
    keys = h200_keys
    configs = {key: CANDIDATE_CONFIGS[key] for key in keys}
    return {
        "ext": wide2_build(),
        "weights": {name: weight.contiguous() for name, weight in weights.items() if name != "router"},
        "configs": configs,
    }


def run_wide2(x, layer, n, st):
    m = x.shape[0]
    cfg = st["configs"].get((m, layer))
    if cfg is None or m > 256 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    weight = st["weights"].get(layer)
    if weight is None or weight.shape != (n, x.shape[1]) or weight.dtype != torch.bfloat16:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    rc = st["ext"].launch(x, weight, out, *cfg, 0)
    if rc:
        raise RuntimeError(f"wide2 launch failed: {rc}")
    return out


# ---- router/router_kernels.py ----



def _mma(n):
    regs = ', '.join(f'%{i}' for i in range(n // 2))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void rr_mma_{n}(float* d, uint64_t a, uint64_t b) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
}}'''


ROUTER_CUDA = _zsrc("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")

ROUTER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int rr_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int rr_scalar(int64_t,int64_t,int64_t,int,int64_t,int);
int rr_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);
'''


def build_router(configs):
    src = ROUTER_CUDA.replace('@@MMA@@', '\n'.join(_mma(n) for n in sorted({c[0] for c in configs})))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rr_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
    cases = '\n'.join(f'case {i}: return rr_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
    src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', ROUTER_SCALAR + ROUTER_SMALL)
    name = 'router_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, ROUTER_CPP, cuda_sources=src, functions=['rr_launch','rr_scalar','rr_small'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)


def pack_router(w, nr):
    hi = w.to(torch.bfloat16)
    lo = (w-hi.float()).to(torch.bfloat16)
    return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)


def prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext':build_router(configs), 'configs':configs,'weight':w,
            'packed':{nr:pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}


def run_candidate(x, st, index):
    out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rr_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rr_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out


# Configuration: WGMMA N, K splits, stages, swap operands, reduction, pipeline flags.
# Reduction 0 pulls, 1 pushes with a final barrier, 2 relies on destination byte completion.
# Pipeline bits: 1 keeps one MMA group in flight, 2 adds a producer, 4 initializes reduction first.
ROUTER_CONFIGS = [(32,16,4,1,2,1), (32,8,4,0,2,1), (64,4,8,1,2,3)]


def prepare_router(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {'enabled':False}
    st = prepare_candidates(weights, ROUTER_CONFIGS)
    st['small'] = {(16,8):pack_small_router(w,16,8)}
    st['enabled'] = True
    for m in (1,8,16,32,128,256):
        x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
        run_router(x,'router',256,st)
    torch.cuda.synchronize(w.device)
    return st


def run_router(x, layer, n, st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return run_candidate(x,st,-1)
    if m <= 16:
        return run_small_candidate(x,st)
    return run_candidate(x,st,0 if m <= 32 else 1 if m <= 128 else 2)


ROUTER_SCALAR = _zsrc("eNqdVm1v4zYM/p5focOAwo59PdtNi2tt+bABu2Ho2g9LvgWGodhyLzi/ZJIcOy3630fKL0matrtbPiQSRfIhH0pkJooXm5wpHqxLRe5sgj/3i+53vggncfyQVyuWxzHZVuuUCBHLhOVMxN+5KHluJFUpFcnyiqkpiWPBpRLrRIF+Y5PuMI7LbbzSKu7VC6XWnpD/+Lzme9dFWJjkCew1DG83Qgtv6Wf32vNBjju1Tqn6JjhL/0zb89YmomroKq+S7/0+ZyWnoHV24dqkYWKDmzC8RAcamrAkWd5Fy1lEn55H6Yw0zXK+iFBQA5BH2hb3oIkiVquK5CldnkUGhiHzStm4WHchEzRHYUQFBzkXG8FVnDCpggNKZ9PQaCwIeXprLtfaMyG/bAR7KBipS1HluRZlldAwijq+Cu58ZVkDDtCQGSoAqiBAjbh8G1RnApitpQ4Rn+Fbp34CPQBLAJbBfOFLRM5TQ9pAoyWn9wvz1JC4B6YrJnUBfFwEt59memHR+QKN+yzeyfkYeki6L9KWIs99GkOlWF+kNxy/QSdD3pBDJNDfc6ujNaQ1X5gQMCbQE3B8YP484lC4p/GNHD0ljzw6dHpSyWMdKOYZRn7ejgGcunF/3M3uwI0+9Ejm0Hiv6XVS49Ex7cx99cQ98IGPC66kE9GsYJmxhSeZOfA1yk913VF3h7o7e5Sf6nqj7iNEs/frvaJ7Meo2qDv6vdjrPu+fwwRplN+Y4Ck0pK5TyLqQ2CruF58uvOjdB/NOoTtfW2qMHFj7DC1jzMB6Gd87jySl7pWf+tDYqGuSrUUx9iyP06opY7krE8Nps/5jb+209wiB6f5IHbNLDuCwR0Z06+9JAPOuxUpD24GVbqRo9fRTLevljR+owCACzeqXIQwURTfOefZD7+p/5n/MwG6ppt7lFfbjnoDhLjxPJogyTkfEvJrFCobgsGrHVVV3o6AYJTDXONNb8k/Nxa4jQM8QHLP05dQNXPvS9Wwv7NnWRpR6A3FJnbKvdZn8qmBgrmrFJcnYWNHh9A+u9grGWcZs9G5+oKgwr5OES2kSaAi1KMlH1+kc9PuMnZd18Td/kFPXcRwLMzalSm9uinUZyPUjj1VogBaMWpbPYf/bDmDs6+trc7g6EEvxAesxYOihi0EEQQBE6yQdWwc81xTFiIGLMAyNo/8eZmMbr/3dMFvb6DWQ90KD93joF0j4C9rd70LATTHpYe5fnJuPn32o7b8VE84w")

ROUTER_SMALL = _zsrc('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')


def pack_small_router(w, s=16, wk=8):
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    def units(v):
        v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
        return v.permute(0,1,3,2,4)
    data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
    ku=128//s//wk
    return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()


def run_small_candidate(x,st,s=16,wk=8):
    out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    rc=st['ext'].rr_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'router small launch failed: {rc}')
    return out



V9_SELECTED_ROUTES = []


# ---- qkv/qkv_kernels.py ----



def qkv_mma(n):
    regs = ",".join(f"%{i}" for i in range(n // 2))
    outs = ",".join(f'"+f"(a[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{
      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));
    }}'''


QKV_CUDA = _zsrc("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")


def build_qkv(configs, profile=False):
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
    source = QKV_CUDA.replace("@@MMA@@", "\n".join(qkv_mma(n) for n in ns))
    source = source.replace("@@CALL@@", " ".join(f"if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);" for n in ns))
    source = source.replace("@@SWAP@@", " ".join(f"if constexpr(TM=={n})q_mma_{n}(a,dw,dx);" for n in ns))
    source = source.replace("@@CASES@@", "\n".join(
        f'case {i}: return q_launch_t<{",".join(map(str,c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);'
        for i,c in enumerate(configs)))
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint q_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
    return load_inline("qkv_"+hashlib.sha256(source.encode()).hexdigest()[:16],cpp,cuda_sources=source,
        functions=["q_launch"],no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=["-O3","-std=c++20","-gencode=arch=compute_90a,code=sm_90a","--ptxas-options=-v"])


def qkv_pack(w, groups, kt=1):
    rows = (2304 + groups - 1) // groups
    if kt==2:
        return w.view(groups, rows, 128, 64).permute(0,2,1,3).contiguous().reshape(-1,64)
    if 2304 % groups == 0:
        return w.contiguous()
    out = torch.zeros((groups, rows, 8192), dtype=torch.bfloat16, device=w.device)
    for g in range(groups):
        lo, hi = g * 2304 // groups, (g + 1) * 2304 // groups
        out[g, :hi-lo].copy_(w[lo:hi])
    return out.reshape(-1,8192)


def prepare_qkv_candidates(weights, configs, profile=False):
    w = weights["qkv_gate"]
    return {"ext": build_qkv(configs, profile), "configs": list(configs),
        "packed": {(g,kt):qkv_pack(w,g,kt) for g,kt in {(c[1],2 if c[5]&32 else 1) for c in configs}}, "device":w.device}


def run_qkv_candidate(x, st, index, stats=None, query=0):
    out = torch.empty((x.shape[0],2304), dtype=torch.bfloat16, device=x.device)
    cfg = st["configs"][index]
    rc = st["ext"].q_launch(index, st["packed"][cfg[1],2 if cfg[5]&32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(),
        x.shape[0],torch.cuda.current_stream().cuda_stream,query,stats.data_ptr() if stats is not None else 0)
    if query:
        return rc
    if rc:
        raise RuntimeError(f"qkv launch {cfg} failed: {rc}")
    return out


# ---- qkv_sm/qkv_sm_kernels.py ----


QKV_SM_CUDA_TEMPLATE = _zsrc('eNrNPf1X27iyv/evULuHXhvcNE5SSgmlS+nH3UOh+whs9x0em+M4Dklx7GA7H9Dlf38zI8mWbTkJ7e69yzlJZHk0Go1GM6PRB49+GgWuP+17bM+d9p1uNA2S0dirDfcfFd70BvZ2PjtO+qMgyec5/lUYjZLhWM1MbideN4mcURLn0IbhxIucZDTzuldROJ3EiOvRT31vMAo89j9Hv3U7x+9Yt9v3ZiPX63YhOQgj1xsFPgB0u48eSRigo8VurmfdeNz1+3PDDYM4odxNNjHZt0eMg8zakHLiMZuFPlTse8YTv1+78sOe49fcq9qsVYubDfZto25t2NZGw9po3lvsYqN12X4CJXN/u+zJ6+iJMastTEsmb7PkXZacmwjsPzEmpokERF4yjQIk5j5rwjTfhoVowywc9dM2TFc3InCxEdOqRvwdVDcb3SQj3BOEyzc54gl0Ff1IPNAOFNsqxd9DziQaJ0aa6VgZQE9JxwUC+4JA4wmWr/UEPcBMBtxkG82Mqj5RBQkHGQe/PfEb56js56jEPpUUjseOMfBDJ2HGU9e8aF1arMi/HNl19cHmhBOpgKgW3wL7HH90FXj92tjeDnauYdBG4bzmhn5t0GzUaBTT16AsIt82WtbGC2tj29p4SY871sYrSihg6UCAdm8NnhjuRf0S28zTtpJuKOnmpZkVQ3aJUpS0s2QjSzZlsleXCRuZCox8/pztsLk3uhomMQsHLAR1AS1lceJNdpkfzlnvNvFiGPPsU50ZM8efwlP9WdNkTtBnn+w0r/XsJSBvPeuNEuB7n5c5rGm7qu+5mSx9Ujvik608HCrpMxXqTIGCXuV9l+9s5o7Za3bInrL64qX4m7bLYEM/BDhVyLEmrMAdmzr44agSnu3vM3ubSmGvFOCwnVCbBQS9sFv1qcn+ZIZxyPb2mN0widCder2On6nAYS/B8bK53VBx7OhRNEoogFRohI6Mlh5FcwmKjIrDUlkQr5zFCUJpbnLCEHgzLzIy7VxQaGDtJu0nZptxaT1BaJY4115gsd40YZNk4cTMdYIghIconHlslOwyL+jHLBl6rOfEI5f1/NC9ZnEIIu3Ai1EcTz02H3oRfIOVTbxAL6lh1Pei7gQoTwz4sLFVtiMkF0lZFY/DWaqBNzZcpCBTdwnXaqMBMD8B1qXjwRyb7PVrZO2A/wF7c5yamEsVdTz2xl2otmT1pAbNaup23VnidK+8wItGbjcJu/HQibw+VsGWVBE5wbVRaH+0uv1TUCpR100cLJ9xIsrp92hp48bOxNFbIcRZYXxUkqB8jTdyd1dQlNFIVmmJRaI61jRGPSfqjoKRYjMhR6HXDcE/TI2OSiLARSMgC0tntCZOrbfdAktev0RKkcpd0uROJKgTGIFkkIAwun1iLqfOW0w8t5K+ZLGUOPyZeTWOo5ssSoR2rWW0IvZ1CB14ASgP4mNZM9DLmiSJoGqR53tO7NVE31LVakUg1rp6XL/LW6SpRjZZiotoOtTkLMA5EE7CA2qaO9rmFOtBsGXYtejjpOug79IylJFhMfKLwN9VBwz2hqaLY6iUvJ/iMJGMhpxwPPG9BGYii91d8g/QU0Y3iHf5t8y/g0+LXOYXOr9/l48qFIpB6kkPUkd6kPrRA+5Gi2FIlFeJTy8M/ZycJ9FtTsgLWiK8LquJb/8XMOjgK1abAAvYpA3PqegDPurC2sQBy3Fbc9yb6ShKRa40Eibc/wZrCWhKPHjCYs+fpDoIGAffWOF9qofCa6mIyg1PVVF4zR6D1Shpz+1Wpj2vcEKqKG56V2W4gHKuuPl0AotGQjf6qfkSlSfSl3wLOqgPjWcO60Uh2GgG5juIHTcZhQHXeSwehvOYTSdQKYBx/4ANnKmP00zgoNNHRxSt13Q8Cq7YNIBJ7CgeQkf0HeCpVu6xQ7rAHk1Po4ktygO+NAX1eSeP86SeeTyCaQg1HwJ7mPFYi406VqlLMvsZYttnjTr/m/q+Cb4QejVGtd5z+v0Wn8y0nqKZ4/TJjB4pj9qCbb1mvdqiDelbnr7F9B1P32F6ztPznMIQ2kCqjOHA7y7CqGXkawFlgR5PyLkoenoMjleXQxjdtGgXFYZRXwykywJeUm1hQVmLrYC6JajiqOB/K4rerVXBHKGqOR0nYeRBU4JZt0ftsrc3ccjymeQMOb05AWno8lY3JFA3CmAiXaXlOVYC1WKbUX8kHuhRJ/HYHjL6w/HZPsy7oqmbiIjNhzEMrAyo/Havvg8Y4wTGrctlBAxyRN128gmqaVns5AB+bahuBSZ7Oabm+pgayzHtCEyN1ZiayzFt5zBpuFk19wzT0ApFgYync/MiqxVL7u6efLpUjKXx9EADcsA2Yb53uWSWqpPr9P25Cnxup9oqa6oBFdFkgP35JxPpBsL9NImcq7EDujEKfZ9qGYQAjy33GdoB+Nlj5Ua12daWz2vCv4OLFrTBZ1uMpqvZ1GB+4V+CZimB2Rqw2zJYQwN2VwZrasDmHOwevu+Z58feiuYOeXOHuuYeYHOHWXPVYOYUys2RBqBl2M4BZE4w2oEhewNdBYYPOzexZY6NObYsqAY2YOqPyo9+b/G3Qc8JFbfYAbJxG2tFtptVGO4EhrnAcFuBoaXHYAsa7IyGuwoMO5UY7gQGScO8AoPdMJVeE77A+9h1Jh5MxMFE7oKXc0ORjh32J/OdwKOHJjyAmzVCv4liGPA8dPwBpl9C+iVFk37/fIoeAU7mh6OrIUWk/hUzGCFhANgZwMS1wvA/OdWP/omTuEMjP65PTtUh7GmjSegMgqh6GOKhSEvT1kWTFhGHoqjTB5i5Q0OMHeCQwYu+xKI25G/yQFFJsFOxjrhYA1ugLSjGUeZhEDHqwAFP5+AiumR/vAYKpC920P/quMgfZHcMLQDHypF5wgWI/TCJLTYfJUPmOe6Qdw04XuR5JUMwWggC3I4n/ih5dsTA6U1Gjp9juOs7cczej0eJnungDk9dL+debLI4crmDcfIL//1V/HTOTnkqGfV5Ikgsws48+OLdk/Jp1F8ApwC0TUng1i88BX6PnGCzdP4DoKoHU68NwN9Wv4Qcp+gnvBsmgPhX7IWJmXPRwEeCdlwY8egOZkPmBPoV6YcOBxIuBTKk2oBn8EYoRw4Q0OfBDku4mwyeJXTE6RnIxbMonPMMNoFxMXci8CA6ZyAkIOgYVR34GK+llxK0xt79cvCRGeB0Ys+FgX9r7jKbBSE7Pj6wsLoGPix4AMxiLXyijkHX3IKhxyNjWNLCkQ29m9yyay8KPN9iMD0xSEpA+KHXgYhwmkymIB7o6QA+mC6kAA5gP3qLdca4VgW0PsO4H4WFRSQOBAvHsxdAF9uNHVG2ZSJVOcy1sl3nQiF/T8VvRz6/f8cTyBAhYWfHB7/z5OGx+D0Rv0fi98upTMico3OR6PDf30XRs4/7j7pdPiuSocxu13emgTvs9nACFHe7BtWZCsuVN54Zj4omqAsjAzg0chPAM59IN7/gjOagFjL+mMtFjlmPygasAObFYsiB9EZXgjdZ8ihLfjlV0kr+0TlPL3d4lns4KQcYReiPwZjC1y4nqp0DOEGAEwQ4AQBB6hFmHmHmEWQelUt9OUUI+H5D37tpa75Q0S9U9gsWlk07Oiec54T0HLGe59Dm/c6yVyUc0bL/YYGYgteGQ30TBCePkCIVh3YD3gvfzlb8vKYG+N+/fPosoPfR7y1DnHyG4Q4gpA2eYjjh5PPv2XMDn2GEZDkwRTj9nD2Cd/7++Nez/1VQbGuqOf3c7ZwBogys2UBE3cPP75TMbZyAfC5AwnDXYPz1vPNvtJ1EHLQNOSHSTSWNSxRP2WPeCEwB8fhDRCPaaYz67/M0QbbjcnngjMGiJRiQCYP+CLWd4+8hJ8W0zMoPOOg3LIa4EMsmDi5AFXnQ/140gXlw13XiZI9e7hs49Eirewt4HdBsFOM/tOJB0TtQBmDrQXvHFMpjLrxnGKW/uMRy2gKgRNNgRIxxjbyEL/iiHLDr9+OUWzzZzJIvTJBnlL9N0O6bzDhiz5lx2EEh74CMX0Wj/rvRuHZrkg8IEA3IrXOachooXugYkAfaN7BJZp7OuaSTE9QSI3KTht8mWrtNRrOoRr21IyvnujGe6+qkd7wqIJqzQa6jKNMmkgZp+7MoTzqfyClidB5QVoboIf3SX6DfPJlzpwJ9thcWmWDIgNznMIrBAb7iTxv0RD4TB0e/sF1AHyQ5dSQbf/YRvprIcrKI2BOLYtGTL1CUyqBVmqNPSKRsMDRX82v5/Byei2U7WGu5s4tgH39XdCzIB9VGHYNCmxUs0fYRykHhTagIOvFL8fUVctCgpiFPbwEIoLdYmrMwqRxkYROKxSOaemXlF8gD9KvmkUkEFgu49VJ9wGaAv0b4o/Mi/Nlv6AuOAgPnojBILGagSIGNwkVcs62xphhSzOKC6TrbUxqgokRuPYp8R+m1K+KJui6blArI3NqanJCRt0/vMQgAio5kFdJZ8RKV6KHHFaq0hSohmzcYRoc9wzZjT2Bva199OU3pYeX1LVo2sisgxBoTwXC6hHBJ4lqpimrhfPo3mhk1dqZlfOpSkHx5n5+3Kqs4cjqq4f4KlVmKFbBr5CYqz46sN50ezOTcYwYThEzVQonnbAdzc7OQgtrBOCAOOQ5tgvtHORtpTrtQint+MOEAuAVOKsW0A8UYzNhYdtiROs5uOT1bgB2Iy1ByZP0YyYgRW4LdQpA7Zhlcx0M94zQiGTuacQNVKy0srkG4E7EM5V6JFRWxr0msMfElFXBOcGWM7xOidSQfEpFbWiUR8sINwGbJrkyFYUGaMlJx7xgiK4rbal5UtSWEiXXCt+iVlvQEcqKwS6FsbpJiQwmuKBpmfNtFs1NffBB/U52uQf+S+0mmLKFsMIMZgSJHoLPJJG2RSYNaSzOW+Q2anEmxjMEt+ScqboryqgrlwCzGqUgfaZDFP5ZKq8UKfsiiXhB8UPqbOPOEHDLCYH5R/rFLwD9+BboFvQqKtHDWYLh3ftE5u7w4hc/Jp8tMa7fYYkZvjuHTEC+cacK3jEDFF08vacxjQITPhuKKUKyiIL7y+MFXUA9gs9jW1ldTX0IfwE0DtmgIoFWPUWOCoCVXwgrML5Ccy4uvlxf+pdq3cwP6KuMUKl7Jfcj9yhkOHWWqXC8Kz2OcPGRethD1eibq21nyVZa062YlZ5R2BrydAalObGegUyJFGUBHDAQhr/0MYAiaccgOTFLCUiTAmjfMgm6MSWpU3QNdz/kYXOY2bqEGWEzMKlC7BIr1ZtJH7t2OqWqgnCq5by8NbcecQTEwqEMMAoFDYTRii8UCqWaf0thi6azkIYq7QmnR0iqpLCCmQmlV6Kss9sYc1704OeDjDkdY67K9blhfhvH/hqGmEcGqMkopl5dyoVQLC4G5weYNaRiCWLgoFvXaoOASDkK/n7mEaXawSLRqHCryJlUupKrWOQKxDFpeAkWtb9FCQ6FpzM6zfMSbNYJmgb8Mv+C8dM5W6zidmOocnhgrGOH4y7kVRofHO7A9AAJD5+jcZD0Qpesq9wNb/W2NjnqASKzSwIWVPOrLP9QYvAGqWOhhmK38wZTH2/zjXf5xnlMvyJAYmQRc2SNOpCOe5+agV45+xetRVWs2paHBuHNZIKGkMV5mGmNHUfT20n54gK7/e/X9qKTw805qi93VX+dVvsXu7HzW1lLFzv963IC8vqvjtowetxH4dCueGvR0J56a9DTXYmnBOzvF8oKeJJZtepJYXtJTAct97mktOckXSkXkgPtKYrG7/X3DrtKBya3KU9xUDiQLasZvub8ag8nntrkOAaulLS/j4OasEGQFuceRe4B8B3F7Ju8Gj7T+ojnYqeNnWugPPgjz9T7QodLKGg6PGXKMnhSxK7zIJLDwIhPGwosquVTB7EtFRAsvMmktvMgEt/BiiQzf/706UmNf11VtOmn/VqlqaEhNcTU8jd62KLBzRfv3MD6TugLvp/mQDOmxU1R6X0uahzaSoyeA4asd2kKOtZRJYawfajI5Bh7KfE0NMzN8TbHibZq5dfQ9oRT2DT5SAbxEF/7BGJno8rnj4uHiKijUN6tcGG+C0/qSo5QfZWKPno4VRfD7dYWtZA1bGnVipCptXUXyQO8kV/aGl70h9xijZ1j+xtQwOZ5fGAZFiVPZkVF3nDjfmOqcHwakdiseLZqjoaSevrjZbAn1jMktW31oqA/NS3OZWdJOGwo8ffzf5ql2sCjbxzAG+DAm68ZCytfXM9SoCm8h47at8hcy7toqjyFjrhkL3y3e23pnr76uRlQYKeVyvd570HRsjZ4vs1k5K4hTtrqYsnGRxZ7CHrOEpdy8ucyS0BN5X0ln0JU59fcxam1BD8TOlAC1p0GeMPi7z3EzJLBtopNaJfQc4oqD9KG5+xxMSJcrsbSXGMkz5Mo02CW0RtLJbpHbrVH3ypJMiPMoY35NCzDkh5syXneTVpZiJCNDPvtOFdbs5E+/r1uK4XHs8FqudnbSYHboa0hVz4L2bKvXsHrNMlS+hwU7dLJVPIg7dpJotCgcJd0JdmqLhgxp94AH/EwoHZJoKKd0+VnN1/ywpjyIBM02NQ1BSfwhglplgiqOO+fIkqlGmmoWSf2B4atVvPoBjLLbKI9imDH0bK1LUupUXPRbhpz2vpYqaMCnaa7SvUt18ZI9rn9RNGOduXc2FBaFaIDWSOyjRgCvFZJ7woF9LCZRq7TOA6bvVWqgFCJYNWiXL/lpxIOiDrG61KDTKnrVlluK4LtnyqNVE9fgNSyJV7A0VoHBiR9rs3Zs3eH6SnlxTImTbO4Xw+FpWXutsrntxppRUEFUIRzPa3tQ4L1qZDKU+Cxis1DiNQslWrOoitVQ+SxWs1AiNQslTrPQR2mQnvXsvdZVWC4BtBdspUexMuqumw/I8Cc5odlWdPi9uWR/6LL52YGqVw14pT8DpLrHpWJNrSudxmaXaqo31MGc4l3h5dGTCVSuW5S3Si1OOQ+T8FU+KbXW+u7mWD/cmjUsKO6nK9lPmP+AGi82h9y+f2Kb7v/K+EDl1LawAqjbocKXN/QLH7TDUh37OE3m+9MK24IUo47bL3BUgLciUiDj9pTvOBGPDXjsZ4/NXHSl4H5f547tjNmfDE9DrPZgFCW2rYtYFA/r92t+mB3Pxw84XRsv6Diu8tIWZ/e1LxviKLJ8+aRazzzJFWzK88tUkBzgLXkZwJa8nwYTrkykFwZci5PKN3kxEwuADrC3Bx8XPv0qwdD0PO2jLXa94m2QAJQDn/wsIu5i3aJtdblZnvC3ul3sza4T85MRBlH6lKLZ8hKKbBdMUZMhGbxpGP9+5TVfvnzVe5VdW5ESYDE7PWGRtTPbn1nCzZeOo1WLxv+UReBIXf+lvdpvmKrR8cUWK2lGzN4tAuo3ZKQmfSXvI0RWFwsEtlyLThWO6pefvpW7TMW+z3YJJArn6IbltmOevlV3KfId2eBhyp19tF0Rd9zjbHmd7SAPmt58b8hoWffpg/bZzEU3a0GLogxBGFYuRTHalbjwxA7gwpEowoTF4Yg+MqJ11a2o+bio3FT/Rp5LxqGLQ9jIZNAsyATIGLZjj43LPp1GVeSnJSeoPMK5pch4ZWRXHdBV6ozv4Fd5/vw5+zD1fX4kbVec6zLQEuSYnkZRa+wwHE8cl2/IimkSb5vFgme/YURI9tZVLa2OdnbhyazuAGvNtnfRQS15Xky3IYk2CKDYYU24LAJM8WgbJWbwDeMBbV6HzA2+t5ioEVl48OusbFzFEfEZX2D7Rufn6Xg8HXSf1eb37e/TSnmxLgp1kBPkYKkYF4WYtMIWE6wWYhysIcQU2hey+ADZA+YUxC4Tuvu2pmtdISEP7N2rtHNfFvsWHIIk69uz34pde/bbP6ZnsUeTH+jKq/9iFxZ8H2lRso3sdr718lAp7jHXnRsR7wsnR/B4QHaEpMyak19yhpHv72422v/ZWF1BWWMTLwxD8re06IRUbrGCqVqxwKeeiZUKHjdZy7StpBtKurC8t2Q5T9+hRRHOnxfGllp0PBhPv+AvnQjG08Cp4tYtyeRxStFw4ymeNyQJycTz5Eu5d7+LoioFQ/Ve4Glgce1HaSAWr25Nb1wSmy5dtE46sGQ4irsCtrh46voU5jeqhz15NLjMD6Dk1xWOnhQL4GljKAFjYouJoyHKiYgitGeLUC8yyYDKtvjSDmAp1FA4yR1dcyhSRumhbs/WHOou9jGtxG5Cc8bORB6qozZR12/xrqmX/Knc7NTms9OOCLEVznqvQn5TDiukkqqeAK+KMWh77X7ZRj5VWaGFUI7VlBUXHYWVmeAZFVba8kdTpFNrkTlMvdO2RutGnjv7Ua0L7t/Z0GM+qA4v+lfMwnmQ3jWA2zfEqXPOeXE2vE2n1aHecZh47PDsIGaTaTzk1xmk95it0kKF81FajZTe2lY2ginHcVcbiA6eYVqx/SfKHSSjOw3phFR91XrZX7LNYZ3loII4BJrZDy51JBWBwTxTggquFKvhCkCRYMOYRw+wbbg8XSzB/f90unbV1oezhSDz01AkzpmJ4AfwcscJcazRkG8vaVLFnRM/YF/zPE7PBQJFnPCZHpiURul+QFX6dEezUBwtvAQsyk43rlpKuV+i21KS+Yae/CRxje1AxYveylRpR3fLLNhw7F203a5FJ1e5IXdX+hZ5Nj4Il5iFrO+qpF5sTg47XPDW1UD/FUVSoQ60uoNTqpVY7uNKD/c6HXH64a3Fno74qi1tf6Hf+6BtbcstygMsxWo31bU0IrnGTHilShQasVINrqme/nrFdK8x0H+v0qlWCOvrgsq1IXFXkLgCj3xUhSHhJAEHm3edAoIG55ktL5fGg8npPAtf971ZdqgI/x/FRy95R5dwGk/hnZm94Lnw+iBJolFvmgAEtE68wsxjZ9Ehh+yYzoP96kVvcSbxGQmzmMR2X/wfA+Dr3Uy96BZm9PzaHHTlbAETw6hcMLwxdOYxMbuJmSFSooDJQAnRrIXfkdQ5boMfJxFE3lW8adfr9S2AcHx+hf0/8yKhcteKi4TounG6/wMvBpLpRZak9RT5kN3sIwgV1/XI23zkTT7yFh+rSiUSAzK8Sd3OHqbqQ5zAgBG1UWeKu4Qx9oaXRikXqHrj2R5xGz/IZeQwcZdzVnAVOYrcRE4SF4mDxD3kHHINOCblmguQvA9RiuwH4F0qrTEbOOnZOUO+VuU5Np4OHIvoNVFFIExn6rpeHEtlAUNJjBXxPHBqwXR8CiKGgTIQMtSLOI+mO2ZgyrvHleS+AZAkfx14fosiaLFXr16lQ4IM4+ExLpOMsfLDY5PWTw5PMOuEsk5E1hFmHVHWkamQ1kjvkMerRXDB5ZSAvpyKgl+OxDIM5R5JdOeE75wjPBe5HczsUF6nVEnVpSR0uwlFUcSFJpZ+PQmJhLw9fscRECRSUH6ficuy0hplgRO2AdiyF1lzhRia3O3iu7Bo/xVQf5QVaKYFOmf4qinavUEHG6F6SO9RWpZoyRJjSeEYCBT3N9GW1xT0RcYXusKIAjXojD0W+hXRPIZcyVY7K/tKvuevjQ7URhPRDqP/IVGALNxbIkVNXF5CtGUXmDxn/Dy3PHQ/Zio9hrwKaic9U6Q4kWAqCB54uk+bfnWBMsmLF3Z6YxBLd0MbWYSWMKXTKLrrBQVTh0ePozA1z+Hj187o7xORiHN3iihV5AnD/Xw4NDaL0RFRSA7WFTvqAOlWhlWRmMK+wU2WinG6WyRfVgwuSYv23lm8MamdN/xgN7v92yBn/mWe4gQs0ZOrdCSHyurReSPPuDYcyP+zkGq/fBUd1a1A5FaZMHAv3t0Gznjkql4GIrQkEaai4H3zQdhPwuDXMEqcnu8dcucCMR/4fjj3+vJKm/uiwolRHcjK00HK70iDGj6R9T4Mg8HoCsaejzG5b/f5tykFzElgioEvfbcmLloC+P5o3DTwMqbTt+RGckoARF4UJWHQq7SV933BLOiOt+LirXgs3nF7jWJPRo6esDmUyJSRFBKgrH5Zo4uxNIQLfgEpXhCPQuFR8zIz/P9f6fvagoKq1e/Bhssosvb9XVYeWuEABdgsJ0mzwBgfiFxb7bHUQbBV77fg+76PojDq8jOdoInfUOZn151OnMC9Bfk7ID9UtDcm/5eL01PfNaucqN0KNOQcx+AlH0/9ZDSJQnQ0wkjBij0aj82cc/7JiROiU05OhNDRQT3FYwHqZ1D1MztlQ9ZzR3SN5/uFAWTLuir26oqtunM8Al0BUrz5bVF1gXupPF06RJfmVWNPL6/cx+tpcDO4hf4s+BroFKLPUem/KmclwXOlDUzyFCFuGVQut5K365kIh+u6jTzAtFR8ZY0ZzmmKU/0nCeXuLHcgHrF5tkOTpNKUgM/agUOZE/6fmBr86Iwgno8Sd4hGIaaVm59/PjzovO/8/DPV0/foPz/sZu7OK3lF7v8DJNjW8g==')

QKV_SM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int qkv_sm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

QKV_SM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")
QKV_SM_NL = {0: 4, 1: 3, 2: 8, 3: 6}
QKV_SM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}
QKV_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
QKV_SM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}
QKV_SM_ESC_CAP = 32


def qkv_sm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)


def qkv_sm_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst if len(inst) == 16 else inst + (0, 1) if len(inst) == 14 else inst + (0,) * 8 + (1,)))
        cases.append(f"    case {i}: return qkv_sm_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return QKV_SM_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))


def qkv_sm_build(instances, name="qkv_sm_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, QKV_SM_CPP + extra_cpp, cuda_sources=qkv_sm_cuda_source(instances) + extra_cuda,
                       functions=["qkv_sm_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_90a,code=sm_90a"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def qkv_sm_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)


def qkv_sm_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)


def qkv_sm_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


def qkv_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = qkv_sm_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)


def qkv_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = qkv_sm_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)


def qkv_sm_c12_unpack(data, base, esc):
    """Inverse of qkv_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(QKV_SM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val


def qkv_sm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v


def qkv_sm_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [qkv_sm_units(hi), qkv_sm_units(lo)]
    else:
        mats = [qkv_sm_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([qkv_sm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = qkv_sm_c12_bytes(v)
        if not torch.equal(qkv_sm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(qkv_sm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}


def qkv_sm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk


def qkv_sm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = qkv_sm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, QKV_SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > QKV_SM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = qkv_sm_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}


def qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.qkv_sm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)


# H100-selected candidates; H200 confirmation is recorded separately.
QKV_SM_TABLE_M = {}
QKV_SM_MAX_M = 32
QKV_SM_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def qkv_sm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0


def qkv_sm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full


def qkv_sm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None


def qkv_sm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if qkv_sm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))


def qkv_sm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if qkv_sm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True


def qkv_sm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows


def qkv_sm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = qkv_sm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = qkv_sm_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = qkv_sm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = qkv_sm_verify_onehot(ext, inst_id[inst], sp, qkv_sm_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            qkv_sm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


def prepare_qkv_sm(weights, table_m=None, table_mt=None, name="qkv_sm_gemv"):
    table_m = dict(QKV_SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(QKV_SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, "m", m), v) for (layer, m), v in table_m.items()] + \
              [((layer, "mt", mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == "m" else 0
            rows = qkv_sm_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, QKV_SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if qkv_sm_bf16_twin(inst):
            instances.append(qkv_sm_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {"plans": {}}
    if not instances:
        return st
    ext = qkv_sm_build(instances, name)
    st["ext"] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = {}, {}
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and qkv_sm_bf16_twin(inst):
            plan = qkv_sm_plan(ext, inst_id, packs, streams, weights[layer], layer, key, qkv_sm_bf16_twin(inst), cfg)
        if plan is not None:
            st["plans"][key] = plan
    del packs
    return st


def run_qkv_sm(x, layer, n, st):
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if not 9 <= m <= QKV_SM_MAX_M:
        return None
    sp = st["plans"].get((layer, "m", m)) or st["plans"].get((layer, "mt", qkv_sm_mt(m)))
    if sp is None or n != sp["rows"] or x.dtype != torch.bfloat16 or x.shape[1] != QKV_SM_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if qkv_sm_launch_raw(st["ext"], sp["inst_id"], sp, x, out, n, sp["cfg"]) != 0:
        return None
    return out



# H100 finalists. H200 routing stays disabled until the dated judge gate is used.
QKV_LARGE_CONFIGS = [(64,36,128,3,8,27,2), (64,36,128,3,8,25,2), (64,36,128,3,4,59,2)]
QKV_SMALL_INST = (1,2,1,1,4,0,256,0,2304,8192,1,8,8,2,7,1)
QKV_SMALL_GEOMETRY = (1,8,8,2)


def prepare_qkv(weights) -> dict:
    w = weights.get("qkv_gate")
    if w is None or w.shape != (2304,8192) or w.dtype != torch.bfloat16 or not w.is_cuda:
        return {"is_enabled":False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {"is_enabled":False}
    small = prepare_qkv_sm({"qkv_gate":w}, table_m={},
        table_mt={("qkv_gate",2):(QKV_SMALL_INST,QKV_SMALL_GEOMETRY)}, name="qkv_small_final")
    large = prepare_qkv_candidates(weights,QKV_LARGE_CONFIGS)
    warm = torch.zeros((128,8192),dtype=torch.bfloat16,device=w.device)
    for index in range(len(QKV_LARGE_CONFIGS)):
        run_qkv_candidate(warm,large,index)
    torch.cuda.synchronize(w.device)
    return {"is_enabled":"H100" in torch.cuda.get_device_name(w.device), "device":w.device,
        "small":small, "large":large, "large_index":0}


def run_qkv(x, layer, n, st) -> torch.Tensor | None:
    if not st.get("is_enabled") or layer != "qkv_gate" or n != 2304:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda:
        return None
    if x.device != st["device"] or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if 9 <= m <= 16:
        return run_qkv_sm(x,layer,n,st["small"])
    if m == 128:
        return run_qkv_candidate(x,st["large"],st["large_index"])
    return None

# ---- router2/router2_kernels.py ----



def rt2_mma(n):
    regs = ', '.join(f'%{i}' for i in range(n // 2))
    outs = ', '.join(f'"+f"(d[{i}])' for i in range(n // 2))
    return f'''__device__ __forceinline__ void rt2_mma_{n}(float* d, uint64_t a, uint64_t b) {{
    asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(a), "l"(b));
}}'''


rt2_ROUTER_CUDA = _zsrc("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")

rt2_ROUTER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int rt2_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
int rt2_scalar(int64_t,int64_t,int64_t,int,int64_t,int);
int rt2_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);
'''


def rt2_build_router(configs):
    src = rt2_ROUTER_CUDA.replace('@@MMA@@', '\n'.join(rt2_mma(n) for n in sorted({c[0] for c in configs})))
    for tag, a, b in [('H', 'dh', 'dx'), ('L', 'dl', 'dx'), ('NH', 'dx', 'dh'), ('NL', 'dx', 'dl')]:
        src = src.replace('@@CALL_' + tag + '@@', ' '.join(f'if constexpr(BN=={n}) rt2_mma_{n}(acc,{a},{b});' for n in sorted({c[0] for c in configs})))
    cases = '\n'.join(f'case {i}: return rt2_launch_t<{",".join(map(str,c))}>(w,x,out,m,stream,query);' for i,c in enumerate(configs))
    src = src.replace('@@CASES@@',cases).replace('@@SCALAR@@', rt2_ROUTER_SCALAR + rt2_ROUTER_SMALL)
    name = 'router2_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, rt2_ROUTER_CPP, cuda_sources=src, functions=['rt2_launch','rt2_scalar','rt2_small'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True, with_pytorch_error_handling=False,verbose=False)


def rt2_pack_router(w, nr):
    hi = w.to(torch.bfloat16)
    lo = (w-hi.float()).to(torch.bfloat16)
    return torch.stack([p.view(256//nr,nr,128,64).permute(0,2,1,3) for p in (hi,lo)],dim=2).contiguous().view(-1,64)


def rt2_prepare_candidates(weights, configs):
    w = weights['router'].contiguous()
    return {'ext':rt2_build_router(configs), 'configs':configs,'weight':w,
            'packed':{nr:rt2_pack_router(w,nr) for nr in {64 if c[3] else c[0] for c in configs}}}


def rt2_run_candidate(x, st, index):
    out = torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if index == -1:
        rc = st['ext'].rt2_scalar(st['weight'].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    else:
        cfg = st['configs'][index]
        w = st['packed'][64 if cfg[3] else cfg[0]]
        rc = st['ext'].rt2_launch(index,w.data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],stream,0)
    if rc:
        raise RuntimeError(f'router launch failed: config={index}, rc={rc}')
    return out


# Configuration: WGMMA N, K splits, stages, swap, reduction, pipeline, W/x L2 promotion.
# Reduction 0 pulls, 1 pushes with a final barrier, 2 relies on destination byte completion.
# Pipeline bits: 1 keeps one MMA group in flight, 2 adds a producer, 4 initializes reduction first.
rt2_ROUTER_CONFIGS = [(32,16,4,1,2,1,256,256), (32,8,4,0,2,1,256,256),
                      (64,8,6,1,2,1,64,64), (64,4,8,1,2,3,256,256)]


def prepare_router2(weights) -> dict:
    w = weights.get('router')
    if w is None or w.shape != (256,8192) or w.dtype != torch.float32 or not w.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(w.device) != (9,0):
        return {'enabled':False}
    st = rt2_prepare_candidates(weights, rt2_ROUTER_CONFIGS)
    st['small'] = {(16,8):rt2_pack_small_router(w,16,8)}
    st['enabled'] = True
    for m in (1,8,16,32,64,128,256):
        x = torch.zeros((m,8192),dtype=torch.bfloat16,device=w.device)
        run_router2(x,'router',256,st)
    torch.cuda.synchronize(w.device)
    return st


def run_router2(x, layer, n, st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256:
        return None
    if x.ndim != 2 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    m = x.shape[0]
    if not 1 <= m <= 256:
        return None
    if m == 1:
        return rt2_run_candidate(x,st,-1)
    if m <= 16:
        return rt2_run_small_candidate(x,st)
    return rt2_run_candidate(x,st,0 if m <= 32 else 1 if m <= 64 else 2 if m <= 128 else 3)


rt2_ROUTER_SCALAR = _zsrc("eNqdVm1v4zYM/p5focOAwo59PdtNi2tt5bABu2Ho2g9LvgWGodhyLzi/ZJIcOy3630fKb0nTdnfLh0SiSD7kQ4nMRPF8mzHFg02hyJ1N8Od+2f4ulvNJFD1k5ZplUUR25SYhQnmRjFnGRPSdi4JnRlwWUpE0K5makigSXCqxiRUY1DZpD6Oo2EVrreJevVBq7An5j89rvvdtiLlJnsBew/BmK7Twln52rz0f5LhTm4Sqb4Kz5M+kOW9sIsqarrMy/t7tM1ZwClpnF65Naia2uJnPL9GBhiYsjld34WoW0qfnQTojdb1aLEMUVADkkabBPWiiiFWqJFlCV2ehgWHIrFQ2LjZtyATNURhSwUHOxVZwFcVMquCA0tl0btQWhDy9NVcb7ZmQX7aCPeSMVIUos0yL0lJoGEUdXwV3vrKsHgdoSA0VAFUQoEZcvQ2qMwHMxlKHiM/wrVM/ge6BJQDLYLH0JSJniSFtoNGS0/uleWpI3APTNZO6AD4ugttPM72w6GKJxl0W7+R8DN0n3RVpR5HnLo2+Uqwr0huO36CTIW/IIRLoj9zqaA1pLZYmBIwJdAQcH5g/j9gX7ml4I0dPySOPDp2eVPJYB4p5hpGfN0MAp27cH3ezP3CjDz2SOjQaNb1Wajw6pp26r564Bz7wccGVdEKa5iw1dvAkUwe+Bvmprjvo7lF3bw/yU11v0H2EaEa/3iu6F4NujbqD34tR93l8DhOkUX5jgifQkNpOIatcYqu4X3668MJ3H8w7hW597agxcGCNGVrGkIH1Mr53HklC3Ss/8aGxUdckO4ti7GkWJWVdRHJfxIbTpN3H3tlJ5xEC0/2ROmabHMBhjwzpzh9JAPO2xUpD24GVbqRo9fRTLevlje+pwCACzeqXPgwUhTfOefpD7+p/5n/MwH6lpt7lFfbjjoD+LjxPJogyjkcEvZpFCqZgv2qGVVm1syAfJDDYONNb8k/Fxb5lQA8RnLP0ZO4Grn3perY37/jWVpR6PXVxlbCvVRH/qmBkrivFJUnZUNP+9A+uRgXjLGU2ejc/UFRYVHHMpTQJtIRKFOSj67QOun3Kzosq/5s/yKnrOI6FKZtSJTc3+aYI5OaRR2pugBYMW5YtYP/bHmDs6+trs788EEv+ASvSY+ixi0EEQQBU6yQdWwe80BxFiIGL+XxuHP37MGvbeO0Ph9nYRqeBxOcavMNDv0DCX9DwfhcC7opJD3P/4tx8/OxDdf8FLxbOzA==")

rt2_ROUTER_SMALL = _zsrc('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')


def rt2_pack_small_router(w, s=16, wk=8):
    hi=w.to(torch.bfloat16);lo=(w-hi.float()).to(torch.bfloat16)
    def units(v):
        v=v.view(16,2,8,128,4,4,2,2).permute(0,3,2,4,5,6,1,7).reshape(16,128,32,4,8)
        return v.permute(0,1,3,2,4)
    data=torch.cat([units(hi),units(lo)],dim=2).contiguous()
    ku=128//s//wk
    return data.view(16,s,wk,ku,8,32,8).permute(3,1,0,2,4,5,6).contiguous()


def rt2_run_small_candidate(x,st,s=16,wk=8):
    out=torch.empty((x.shape[0],256),dtype=torch.float32,device=x.device)
    rc=st['ext'].rt2_small(st['small'][s,wk].data_ptr(),x.data_ptr(),out.data_ptr(),x.shape[0],s,wk,torch.cuda.current_stream().cuda_stream,0)
    if rc:raise RuntimeError(f'router small launch failed: {rc}')
    return out

# ---- olayer/olayer_kernels.py ----




OLAYER_CUDA = _zsrc("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")

OLAYER_CPP = '''#include <pybind11/pybind11.h>
#include <stdint.h>
int ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);
'''


def olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join(f'%{i}' for i in range(n // 2))
        outs = ','.join(f'"+f"(a[{i}])' for i in range(n // 2))
        functions.append(f'''OD void ol_mma{n}(float* a,uint64_t w,uint64_t x) {{
          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n//2}, %{n//2+1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));
        }}''')
    cases = '\n'.join(f'case {i}: return ol_launch_t<{",".join(map(str,tuple(c)+(1,) if len(c)==6 else c))}>(x,w,out,m,stream,query);' for i,c in enumerate(configs))
    calls = '\n'.join(f'if constexpr(BN=={n}) ol_mma{n}(a,aw,ax);' for n in ns)
    return OLAYER_CUDA.replace('@@MMA@@','\n'.join(functions)).replace('@@CASES@@',cases).replace('@@CALL@@',calls)


def build_olayer(configs):
    for cfg in configs:
        bn,bk,st,pipe,pack,hint,*groups = cfg
        tw = groups[0] if groups else 1
        if (bn<16 or bn>256 or bn%8 or bk not in (64,128,256) or st<1
                or st*((64+bn*tw)*bk*2+16)>227000 or pipe not in (0,1,2,3)
                or (pipe&1 and st<2) or pack not in (0,1,2) or hint not in (0,1)
                or tw not in (1,2) or bn*tw>256 or (tw>1 and not pipe&2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = olayer_source(configs)
    name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name,OLAYER_CPP,cuda_sources=source,functions=['ol_launch'],
                       extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a','--ptxas-options=-v'],
                       no_implicit_headers=True,with_pytorch_error_handling=False)


def pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64,device=weight.device)[None,:] ^ ((torch.arange(64,device=weight.device)[:,None]&7)*8)
    tiled = weight.reshape(128,64,16,64).permute(0,2,1,3)
    return tiled.gather(-1,columns.expand(128,16,64,64)).contiguous()


# Token tile, K tile, stages, pipeline flags, packed weight, L2 hints.
OLAYER_PRIMARY_CONFIGS = [(16,256,3,3,1,1),(32,128,8,1,1,1),(64,128,4,3,1,1),
                         (128,128,4,3,1,1),(128,64,5,3,1,1,2)]
OLAYER_NEARBY_CONFIGS = {
    bn: (bn,128,4,3,1,1) if bn<=128 else (bn,64,min(8,227000//((64+bn)*128+16)),3,1,1)
    for bn in range(72,249,8)
}
OLAYER_CONFIGS = list(dict.fromkeys(OLAYER_PRIMARY_CONFIGS+list(OLAYER_NEARBY_CONFIGS.values())))


def prepare_olayer_candidates(weights,configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled':True,'ext':build_olayer(configs),'configs':configs,'weight':weight,
            'packed':pack_olayer(weight) if any(c[4]==1 for c in configs) else None,
            'tiled32':weight.reshape(256,32,16,64).permute(0,2,1,3).contiguous() if any(c[4]==2 for c in configs) else None}


def run_olayer_candidate(x,st,index) -> torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack==2 else st['packed'] if pack==1 else st['weight']
    out = torch.empty((x.shape[0],8192),device=x.device,dtype=torch.bfloat16)
    rc = st['ext'].ol_launch(index,x.data_ptr(),weight.data_ptr(),out.data_ptr(),x.shape[0],torch.cuda.current_stream().cuda_stream,0)
    if rc:
        raise RuntimeError(f'olayer launch failed: {rc}, config={st["configs"][index]}')
    return out


def prepare_olayer(weights) -> dict:
    weight = weights.get('o')
    if weight is None or weight.shape != (8192,1024) or weight.dtype != torch.bfloat16 or not weight.is_cuda:
        return {'enabled':False}
    if torch.cuda.get_device_capability(weight.device) != (9,0):
        return {'enabled':False}
    st = prepare_olayer_candidates(weights,OLAYER_CONFIGS)
    st['token_configs'] = {bn:OLAYER_CONFIGS.index(cfg) for bn,cfg in OLAYER_NEARBY_CONFIGS.items()}
    for index,cfg in enumerate(OLAYER_CONFIGS):
        m = cfg[0]*(cfg[6] if len(cfg)>6 else 1)
        run_olayer_candidate(torch.zeros((m,1024),dtype=torch.bfloat16,device=weight.device),st,index)
    torch.cuda.synchronize(weight.device)
    return st


def run_olayer(x,layer,n,st) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'o' or n != 8192:
        return None
    if x.ndim != 2 or x.shape[1] != 1024 or x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous():
        return None
    if x.device != st['weight'].device or x.data_ptr()%16 or not 1 <= x.shape[0] <= 256:
        return None
    m = x.shape[0]
    if 65<=m<=224:
        index = st['token_configs'][((m+7)//8)*8]
        return run_olayer_candidate(x,st,index)
    return None

# ---- cpstage/cpstage_kernels.py ----


CPS32_CUDA_TEMPLATE = _zsrc('eNrNPf1X27iyv/evUHsPvTZx0zhJKSXALmW7e/dQaA+h232Hx+Y4jkNSHDvYzgd0+d/fzEiyZVtOQrv77nJOiCxLo9FoZjQz+siTf40D158NPLbvzgZOL5oFyXji1UeHTwpv+kN7J58dJ4NxkOTzHP86jMbJaKJmJndTr5dEzjiJc2DDcOpFTjKee73rKJxNY4T15F8DbzgOPHbaav7Eer2BNx+7Xq8HyWEYud448OFtr/fkCRUADNrMncatZs8fLAw3DOKEMrfZ1GRfnzBeYt6BlBNP2Dz0oUXfM575g/q1H/Ydv+5e1+ftOoBgX7ca1pZtbTWtrdaDxS632ledZ1Az97fHnh1Ez4x5fWlaMnmXJe+z5MLEwv4zY2qaiEDkJbMoQGQeBPqzHP5Lgf88HA9S/GfrOxC42IFZVQf+cowB2SRF2hNIyxc5xKnkOtwRccAbsLVVbB+LyjSaJEaa51jZ+76SjgvIDQRyxjOsX+8LXICIDKjItloZRgPCCBIOEgy+++I7zmE4yDDEgRTYTSaOMfRDJ2HGc9e8bF9ZrEi3HMoN9cHmSBOaAKge3wHZHH98HXiD+sTeCXZvQDqjcFF3Q78+bDXrJK70b1hmi69bbWvrlbW1Y229psdda+sNJZRiKeNDn2vDZ4Z72bjC/vK0raSbSrp1ZWbVkFSiFiXtLNnMki2Z7DdkwkaCAhFfvmS7bOGNr0dJzMIhC0EvQE9ZnHjTPeaHC9a/S7wYZJy9bzBj7vgzeGq8aJnMCQbsvZ3mtV+8BuDtF/0xsEs44HWO66VhGnhuxkPv1UF4bysPx0r6Qi11oZSCEeXjlh9o5k7YATtmz1lj+Vr8zTrlYiM/hHIKb2NDCN+dmLrio3FVcXZ4yOwdqoTjkS+GnYSmLMDmld1uzEz2JzOMY7a/z+ymSVjuNhoN/MwECLsaxOvWTlMFsauH0CxCADyhAzok2noIrWoIGQ7HparAVLk5JQjlhKJyQeDNvcjI1HBBe8FcNu08MzuMs+gZlmaJc+MFFuvPEjZNlk7MXCcIQniIwrnHxske84JBzJKRx/pOPHZZ3w/dGxaHwMcOvBjH8cxji5EXwX+YQxMvKLNnGA28qDcFpBMDPmxilScL4oakrHMn4TxVtVtbLrae6baEq7DxEMieANVSITAnJjs4QKoO+R9QViXS1KxUyPHEm/SgxdKsJjVl1kiv584Tp3ftBV40dntJ2ItHTuQNEDyrAB85wY1R6Ha0vtszUB5Rz00crJ8RIMrp8KiyUxNn6uhnGYRXMbmo6ED9Ou/c3p7AJsOPZp0VMw61scFk03ei3jgYK/Mh5Ci4uiEYeumkoqIH5aIxoIS1MzwTp97facMM3bhCLBHDPdLUTiQwExABXRj1MLp7ZlZj5i2nnluJW7JciRh+zb06h9FLliUke9YqPBH6OiSHXgDqgehXFn56WZfoUKl65PmeE3t1MZ7UrNoIsHCxDdfv8Z5ompBdlewhugytOEuY8MXEv2ErC0fbjWIbWGwV5BLoOOk5aIe0DUUCLEY2DtirqmAg9TVDGkODZMkUxUESF3LCydT3EnAflnt7NNejpYsmDR/ir5mdBp82mbyvdDb7HpceZIJhagkPU0N4mNrBQ24GC3EjzLXs0g9DX+XpJLrLMXRBE4Q3ZVXw9X8DBoN6zepT6D6bduA5ZXOAR0NXnzowGdzVHfd2No5SFitx/ZTbzzD/AZhS/5+x2POnqZ4BosF/bPAh1TXhjVQ25U6n6ia8YU9hMshrx512qh2v0XtUlDK9qpqLAGuulLkrgFUjofv8dEYSDSfSHnwLemYAHWcO60chTLkMZuMgdtxkHAZcr7F4FC5iNptCo1CMz/Zs6Mx8dA2Bes4AjUmclWaTcXDNZgF4nON4BIMwcICeJV7HgegBWTQjjDNmgQ3wnSkQzxtqnByN1HIR5MJCixEQhhlPdbBoNLOGJJFfIKhD1mzwv5nvm2DRoHFi6PWaMxi0uRPSfo5TF0dMZvRJSdSXrHbA+vVlB9J3PH2H6Xuevsf0gqcXmWIQYi9Uw2jo95Zh1DbyLYBSQKMl5JQTIzsBu6nHSxi9tGoPFYPRWA6l1QGGTn1pQV2LrSl1R6WKEsD/1lS936iBBZbSUzhOwsiDbgTzXp/6ZO9so2hyz2+OFN6ewvD3eI+bslAvCsDh1WlwDpGKaSHNaQwSD/Skk3hsHwn88+nFIfhI0cxNMIzy8wSkJytReLXfOARYcQKS6XKOgKk1ooE6ew8NtC12dgTfNjS0Coy9GkxrQzDN1WB2BZjmGjCt1WB2cmA05NM5hmEa66CQjPF8YV6K5rDK3t7Z+ytl2jOeHxXfH7FtcMSuVjiPOrZN339SC3+yUwWUddCAhshcZ3/+yUS6ieX+NY2c64kDqi4KfZ9aGYZQHvvrM1Tp8LXPCt3psFrN583g39FlGzrgsxojJzIz3xeX/hVojFIxW1PsrlysqSl2Xy7W0hRb8GIP8P+BeX7srenriPd1VOrrEfZ1lPVVjSTOoNICEQBERp1cgcxqRa0+Yj/AIMEMhsOa2DLHxhw7rZgFGsAZR6VG33f43aTnhGpb7AhJuIONIsnNCgD3AsBCALirANDWArAFBnaGwX0FgN0qAPcCgMRgUQHAbprKcIkJ/V3sOlMPnGOY7vbATLmlyMMuuO++E3j00IIHsJPGaPhQUAGeR44/xPRrSL+msM7vH85xWkcHezS+HlFo6N8xA7kIA4DOoExcL4j62XlZ0qdO4o6MvBifnatC62nDOmjJAX96GG2hsEfL1oV1lhEvReGfn8Gbhk4Yu0Adg1d9jVVtyN/mMZsSN6e8HHFeBpJAP5B9o8xSIGRUaQFz5egyumJ/HAAG0pg6GnxxXKQNkjqGHoBl5Mg8MaXHfpjEFluMkxHzHHfEhwUsJzKdkhHMRlgEKB1P/XHy4oSBxZqMHT9HbNd34pi9m4yTMsHBjp25Xs5W2GZx5HJr4exX/v1RfHUvznkqGQ94Ikgsgsw8+MeHJqXReLAEKkHRDiWBUr/yFBgw0gNmqdMCRVVzpFEfgqGs/hP8m4Kf8iGYAuCPOAJTUzW1wN6Bblwa8fgePBhzCkOK6MNYAwZXAhYibcAzWBaUI+UClHewyxJu4oJtCGNwfgEs8SIKFzyDTUEcFk4EVkH3AvgD+BujmkMf46X0Uhats59+PfqFGWA44qCFgX9n7jGbBSE7PT2ysLkmPix5LMpibXyicUGz2gKJ40EqrGmhQMPAJnfsxosCz7cYuBUGMQjwPQw4IBHOkukMOAOtF4AHpn5awAHoJ2+xzRgXhQDXFxiBo7CsCIoBT6EYewGMsN3cFXXbJmKVg1wvT92cJ+T3ufjuyud3P/EEEkQw2MXp0e88eXwqvs/E94n4/nwuEzLn5JNIdPn376LqhYB6/PHwSa/HPRsZXOz1fGcWuKNeH52YuNczqG3JM9feZG48KU4+PZAPINTYTQDMYiqt9oKNmSu1lFHBXC4SznpSnroKxbxYCN6k50TXgkRZ8iRLfj5X0kr+ySeeXm3krLZqUgowCpafwjQK//Y4Up1cgTMscIYFzqCAQPUEM08w8wQyT8q1Pp9jCfj/A/3fS3vzmap+prqfsbLs2skngvmJgH5CqJ9yYPMWZsGSEvZmweawgE8vMPsCVMPFL3lQFGE4tpvwXlhytmLVtTSF//Pr+w+i9CHatuUSZx9A3qEIqYPnGAc4+/B79tzEZxCRLAes//MP2SOY3+9OP178jwJiR9PM+Yde9wIAZcVaTQTUO/7wk5K5g77Fh0JJkHcNxI+fuv/BeZOQg74hJUS6paRxseA5e8o7gSlAHr8IaQQ7i1EBfpglSHZcmA6cCcxmCUZSwmAwRnXn+PtISeFrWXlRg3HDaggLoWyjWAGoyIOR96IpOLU914mTfXp5aKDQkVr3lvA6INcSAze0+EDhNtACMM+D+o4p9sZceM8wdH55hfW0FUCLpuGEGAMTed5e8lUxINfvpym1eLKVJV+ZwMnIf9ug3reZccJeMuO4i+zdBe6+jsaDn8aT+p1Jph+UaEJug+OU0z3xUkeAfKFDA7tk5vGMnIXE9PgjNAByuI3yt40z3TbK0TZ7ZTfzldIqae2a7GdbCPM2Sa6Egk5Xs9HeRez5Igfp1Xihw5recWQBLiekXBlR3CziJ2k9ZJGe1AXJKXE0P5DbRmhf/TpYosE9XXCzBC2+VxbN4pABuS9pBkmu+dMWPZHFxYujVdkpgA+SnCqTvb/4Bf61cNBoUsWxXBarnn2GqoLsgAValITKFsMZb3Ejn1/Cc7FuF1sts0ux2C+/K/oZOIxao5FBts8qlnD7BepB5W1oCEbxc/H1NVLQoK4hTe+gEJSusTRnaVI9yMIuFKtH5LBl9Zec/6BwZBKCxQpuo9QekBnK32D5k0/F8he/oTU5Dgx0X0HMLGYgS8H8hquxZkczE2NUMY0Npqtnz0nCRYXcahMZn9LkV7gTlWXmyYqS6qKZ9OPIU6DXGDIARUmcCumsdglHtO7jClXcRpWS+RyG0WUvsMc4DjjW2lefz1N8WGn9ipaGbH0BsYxERThWgrEkau1UwbXRA/+NfKrm7qwETl3wke8ecr6uslwjXVgN4deo21Jsgd0gJVHxdmWzqWsxl37LHGMVqZqGGi/ZLubmPJiCwsHAIAobL22C0Ug5W2lOp1CL24vgrUC5JWpU4bMgA8MUOJGDdaJK2B3HpwbQAbkMJAc2iBGNGKElOChUctcsF9fRUE84DTvGTllioGWlg8W1B3cqlp3ca7GKIvYiiTUlvowCdg2ugvH9PbRu5EMicksrI4JbuObfLk0oMzGjIE4pprjVC2EVeW09Jaq6EoI7nvBtdKXVOwGcEOxRRJtPRbGhhGMU1TK56+F001j+LP5mOiWDlim3sExZI9sSBk6EwkSgqmkmqtFMBo2WnJzFLc4002IdI7MDWk1T1Fc1Jy/MYvReBoiCrP5LqbZarWDALBsFrgddv40+K+TQ3AuzLjI/jggY1m9AraA1QeEZThnqBBgkEVqXjzErgDlnSYiR6muvJ3c6HbDL51ekBDCywp2qeJyJQWEgPrLDvMZGtR6PQW2cfKqI/RZ0zZeDRod92YdJj9VqX8zqGmrUGOv4+6UgcYWcAv8fFMRU0gtDXwb2dPvscw3nanP7/ALQ2D57D4C3W82aYBtdA8KYi9yDxW1Njl883hYsUTO+pGCKAL5XLaBIr9MLqnSrqe+QYw7kobMyzn2Lo3O7f0xRqVszz2DGrXUrwNFKxuKye3F1eQ6fs/dXmZ3RZss5vTmFT/NKYVcMBum5tJLfMl7jIbMvwJ7r+E23QpHyW0kIuN2CvA9i+hRnf1CcybXIX1willeXX64u/StFVy0M0D2Z5KMNIbUJ5H7hCgQUj6lqkaIIPkUvOnM3heZuZJp7J0u+yZJ2w1whn2nvA977gOwA7H2gmxGLOg39CVBs+ancAHqgNQrZgUkWhVRxYJQ2zcJEH5MWVJkZGIKTMbhSNxLifLacmlUl7WJJ4bIJXUo+yq6pzqaPYfWYkycG8nSJPMCEyKBGbDGpZMu75yYWS33zx5ggFYJLOwRIbAGXCsGtmHuzCDRzXPfy7IiLIgpd+6qz6WqWXMD6G6RPw38bzCkur+VCrTZWAg2J3RuRCAJTuMgUjfqw4NcMQ3+Q+TVpdrBMtCYJNORNq/wg1UThAMTKfnlVHy0Yi+IDha4xO0/yMe8Wn13xG8zw7sV6tafjUp3pHmMDYxS+nIFsdHnUD/sDRUBycG7vAyvdVBnS2OuNJv/NWWKdUi5MsDSWf6irUAaoYaGDweP+gymPd/nH+/zjIqdchIFTw3UPbuWkAs9zc6XXCb9iv6tqNfPKSRZ3rwoYlBTG60xh7CpK3l45DI/Q83+vrh+XlH3eimuz+8ZBTt1b7N7O5dRWKnX+1+dTx8F9A7cY9fn0gE934qlJT/fiqUVPCy2UNryzUyiv6ElC2aEnCeU1PS2qzLJNmSRfZ50VvtlssdUQBmXwDGC8sAt25CZ88wjr/RH2e9nTBhtbb8HXDDTgx1vHH83HmfDc/nzO5geZqJfLlI4OCfN83qbjOuvPPGVHnrITT7kDT9E6Ez4/8g86LXHEDWixxafzbYq30nxVNyHR8pHUpBY0jP/laQ9cTftkm521jJr3CGj0rHhcgzHcBPf1uirfKhjIa9SgAtzjwD0AvouwPZOLsUcmw7I13G3gZ1aQZ67C8+0+0hTX6ipUrnOkNj0paqvwItNghReZMiu8qNJrajH7SlFxhReZtiu8yBRf4cUKHfjwt06wGtts03lRJydrog0z3Ed0kK0LYWj7mrYuY4Q6NSPfzfJBaZoEz3HG/FJSAHQ0Bq1IDN/v0qEYbEWnOAehJpND4Es5B9QxM4PXEvuFTFPdhLQvtMmhwWUcSmtUKEMJmeryuc3r4e4U0JM/rLN+vSlqwJKNnZcxsU9ZR4lVKnMlq5UsqbZGmRipLtxUjTzSsM3HT3jdW/KscAGBx1E0RI4Xl4ZBi2Qp68hVRwwg3ppq7BPEUbsxmXYd4VxII315u90Wih2TNVt9aKoPrStzlVGj9TgLNH3636apVlaUfba4EPI4IutkIaXrwRz1qUJbyLjrqPSFjPuOSmPIWGhk4ZvZe0fvKDQ2VYgKISVfbjZ6j/LkNxh5zbilh53R2W8IZ59zLA4UDpglpsnt26ssCQORt7R1s7kSjfk2Om3M54HY2Reg8jTIiQLT/CVuGAeqTavNZVp+C3HBVbpf3PMKpqTKlSWF17igYcidPTAr4Vwk/bM2eWwaba8sSIfogRuLG1p+JhfOlMsWt2ljKUSaYsjd2+2sM/WdwUCzEM2X8sIbuVmkm67nhb4GU/Use9+2+k2r39I4ALkBFtTQcVbRC5g4STReFo7C7wa79WVTOgh9IAH3DsgvaCp+AT9rfsAPm8tDltBrU9MRZMTvQqhdRqjCXcmhJVPNNNUqovodwqtXuzrxRc5tlmUYHI2+rbVHSmOKGx5WwKaTAiX4Tfi0zHV6d6UeXnEi4C8Kgm0Ss8kEYVmIImkniENUB2CwQnJf2K5Phfu0TuU8IuxTpQNKoaV1Irt6z4OGOyhaFavLrTqdotdrueVYvvWwLKuaeBhvYUWci6UxLoxqfV+ftZJ1j2vM5e0BSnxt+7C4hpLWtTeqmzuioZGCCqTySzi8sccs1lTJJUN+zyJ9SyXOt1SifMuqGB/Vz2J8SyXCt1Tie0t9dA/x2Wyq11oJq8efttGuNSbWLtXoPAEZMyfzMzu7A9+3V+wPXTY/Z1X1qgmv9GchVcO4VK2lNaLTgP5KPfUDDTDHeE8YePRkApabVuW9UqtTzuP4e401Sp21vrk31nd3ZoPZE3ciF+dO8HtAhRd7Q/beP7FLD39lXKDSpS2sGeu258nQ5zev4eoW2PRLb7TTvbgnh+/yPSguE2RmN+gIFDEwfEQKBMae8d174rEJj4PssZUL0hTM+JvceckJ+5PhibT1xpCiEXd0gY/ijSaDuh9md5jgB+y3rVd0n4Hy0hYXnGhfNsU9DvLls2ql9SxXsSUvf6CKZEnX5I0pNXlJFyZcmUhvVbkR1zzc5rlWLEE7QN4+fFz4DKr4TDPydJ6hOPSK4UIMUIqe8nPeeJigRnuTc86isNx6PRzMnhPzA2oGIfqcIuLybp5sl1BRKyIWvGcYQ3/jtV6/ftN/k97mk7ZvMTs955b1MtvjXgLNty5E6zYt/FM2IUTq/gM6MfMDU2cHfFFjJTWL2XvFgvrdQKl1sJb0EQJriDUGW+6FSLWXauCfv5U79cXe+U6pSBQu0J7LbWk/f6vu9ebnYsBUlTukadM3nnhCp3uTvUiP8pO+Ne60avi+PlntAuncH5yeFAEEqXIpFlK5o5AOTgIsFEQRayxKIxrbCNZVt/Png6vyaNMP8roHlFyUYCPjQbPAE8Bj2I99Nimbh2VFkXdvzlB1hAtLYfHK6LAqz1W6jB+jUkn+8iX7eeb7/Ezwnjhca+A0kKN5Gomts+NwMnVcvkUwpliAbRYrXvyGcSU5WNf1bJMk7jXE87G9IbaabTik47Ly0K5uMxwtcCPXYUu4sgJE8Wg/OmbwMzcBbdSFzC1+PIOwEVl4/PaiPLOKyzfmfInuK91IQheO0NUh8/riofNtSinP1UWeDnJ8HKzk4iIPk1KoMUFqwcXBBjxMywOCFTdnPaBNgeuU3a8dzci6gkEeObjX6di+Lg4tGANJNrQXvxVH9uK3f8zA4oAm3zGS1/+9ESyYPXI6yc4C2fnOy3P9eFBHt0NevC/skcfzVepm+SJlzn7NzYr8mAzu9P5/jfgVNDV28ZJ2xBB9S8tWiGWNFeapNUuE6rUEUr3jrnSZtpV0U0kXFghXLAjqB7TEweqNDdhRiy5owNOD+E13MuB9DKnW1i3q5EFKznDjGZ71JgbJuPPsc3lwvwWhKu1CzV7ifQziMqWSFBZvqU7vqRO+ooszk65YMhrHPVG2uPjq+rRQYFTLPBkzuEsAipJJVzi7V6yA9z1ADZCIGhNn65RjZcXSni3CxUgkAxqr8bUhgFJooXCVRnTDS5EmSm/V8GzNrRrFIaaV3G3ozsSZylPN1Cca+RofmkbJlMq5pTZ3S7siUJe/bGMd7NtydCLlU/UKjso9grpBe1i1h1TVVDg7KAcTy1qLLiGQmWAUFVbq8sf7pDlr0VSY2qUdjcqNPHf+vSoXLL+Lkcd80Bte9O+YhYsgvecFd3+Iaz845cXlHB26LgTanYSJx44vjmI2ncUjfpVMevnjOhVUOF+qU0fpLZfl+S8lOG6mA8bBQ6Br9g5F6ilcuu6VTpg21i23/SV7JDZZTyowQ6DxenCtJKkILuZpElQQpdgMl36Ffw1jET1iWsPF7WINbvinbtp1Rx8RF2zMz5MSM2fTAz++nDuKjZJGAt9Z0aWKG3++Y2rN0zg9VQ0YccTn+sKkMoo3qiq8pznairxo4WWKUXYufN1SzMMKtZbiy7cC5V3DDTYSFe7ILCOlleu2mZ+7cWBxznYtOvDPJ3B3rUmRo+CjQAnPY3MDJTVdcxzY5Sy3qer5r6iQCkWg1RocUy2vcsNWmrU3qazpBVsLPZX1qp1wf6Gx+6jdcCtnks1niLWmqWtp+HED13etJhSKsFL7baiV/mJ99KCZk/9WXVOpCDbXAZVLSuJyNnGtKHn8GTXCaQLmNB80pQTOMC9sebc+3uWQ+lT4euDNs9Nr+EM7v3jJT3RhsfEc3pnZC54Lr4+SJBr3ZwmUgM6JV5h56iy7ZH+d0qLVRy96i37DB0TMYhLaQ/G3VMC0u5150R147/x6MrTcbFEmBmFcMrxdeQ5swX2ZmBkiJSqYDHQP+Sj8TrruaQfMNgkg8q7jbbvRaNSghOPzn+z4Z1/cVhphcW8b/egC3bqEF7HJ9DJL0vqJfMhuUhP4iuvR5O1p8uY0eWuaVaUQiQ4Z3KRhZw8z9SFOQGhEazSm4tp1jLbhXX3ZxdPeZL5PNMcP0hrpTDTm9BW0RboiTZGeREuiI9EQ6Ye0Q7oBzfKXNE1Df+zeQXOcs14yGH8swR+3DuQzygTPkxfVSnb/GQiecnrMhk4nuwVbvFZlITaeDx2LOmmibsEy3ZnrenEstQyIoZAz8Tx06sFscg7sifE0YFDUpuhx03Vg4Bzvc9V6aEBJ4t0uPL9F9rXYmzdvUnGiufT4FNdSJtj48alJiyzHZ5h1RllnIusEs04o68RUUGumP7+Bdzjhqsw5Ffp8Lip+PhFrNZR7IsF9InifOMBPIreLmV3K65Yaqbr9ia6RonCLuDnK0i86IZKQt8+vowOERArqHzJxoWHaoqxwxrYAWvYi667gXbMr7/6CztBuL8D+JKvQSit0L/BVS/R7i07fQvOQ3qe0rNGWNSYSwwkgKK7ao921adFXGV34vf6+8HWeCt2MYJ5CriSrndV9I9/z10YXWiOftcvoV3cKJQsXRElWE7dEEW7ZTVEvGb9xQF5zMmEqPoa8tW83Pbuk2J14xAvLA00PaX+xLqImafHKTi93Y+nGayOL5BKk1OeiS7WQMXVw9DAKXnwOHr/fS395kwScu8BJaSKPGO4eRNHYLgZSRCUprGv27wHQWgZV4ZjCLsVtlrJxuj8lX7dwo135NnC8mI5GOFeNbsNTxil/l2RGZ2Wegmm6N7gLctaGzFNsjhWqdZ1a5aWydjS2zwuuP4fyF25SfZlvoasaMQjbKuMFxsxPd4EzGbuqTYMALYmDqUwJvvko6Gdh8DGMEqfve8fclEHIR74fLryBvHDsoaiiYlQgsvFUrPkFmNDCezISjsNgOL4GafUx4Pf1If82xYA5ibhTxXfr4g48KD8YT1oG3pN3/pZsVo4JFJF3+MkyaMLayvuBIBYMx1txQWI8Ee+4WYCCQtMiPWF3KJGpL8kjgFnjqk53FmoQF/QCVLwgHofCeOd15vhriun7+pICttXv0UDornh/n9WHXjiAAXbLSdIq9lo0u2DODmb+OLj+SCbJx8gbehFe96YCUdqtrkDaGuVN7L6FNisaOTS4/WOmuIOdcSTQb6qsldo+LVyKIH8Bj2Uj7HdRFEZoRR7g0wfXnU2dwL0DuTgia5xs+xiM/NOZn4ynUYi2ThihU0BMDwwST8yOZFPv4ECxiX6Y772w8acYini0U77mltCkZKXZqvtScF5SnHFMfFDgetQF2WJyYLiEPvdds8r83WOPpoCAikIST8ycd/UeBo/wlK6lJBB2TiERYD+HppFMYsAyLjuhe6/fLQ1AW7ZVsT1b7M5e4HUJFUWKN6Uuq369pFSfLtqjS2aroafXPB/irWy4/99CTwQMPrTn0fCr9DyUk7Hgc9BOM3loFHeKKpc5yttoTSyHa/DNfIFZqfraFjOYsxSm+otA5eEsDyAeqXqxS15u0Zfj8RYgUOY9/X/4dN/rysWLceKOUDhjWmf78cfjo+677o8/UjsDj37laC8zOd/IK+X/D/aQcM0=')

CPS32_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int cps32_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

CPS32_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")

CPS32_NL = {0: 4, 1: 3, 2: 8, 3: 6}

CPS32_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}

CPS32_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

CPS32_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}

CPS32_ESC_CAP = 32

def cps32_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def cps32_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def cps32_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cps32_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

def cps32_bf16_bytes(v):
    r, c = v.shape[:2]
    words = cps32_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def cps32_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = cps32_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def cps32_c12_unpack(data, base, esc):
    """Inverse of cps32_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPS32_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def cps32_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def cps32_pack_format(w, fmt, layout=0):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [cps32_units(hi), cps32_units(lo)]
    else:
        mats = [cps32_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([cps32_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = cps32_c12_bytes(v)
        if not torch.equal(cps32_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(cps32_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def cps32_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def cps32_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = cps32_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, CPS32_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > CPS32_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = cps32_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cps32_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

CPS32_TABLE_M = {}

CPS32_MAX_M = 32

CPS32_TABLE_MT = {
    ('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)),
    ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)),
    ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)),
    ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)),
    ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1)),
}

def cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cps32_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def cps32_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cps32_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if cps32_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cps32_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if cps32_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def cps32_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def cps32_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * key[2]
    rows = cps32_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    layout = 9 if len(inst) == 16 and inst[14] >= 9 else 5 if len(inst) == 16 and inst[14] == 5 else 0
    pkey = (layer, rows, fmt, layout)
    if pkey not in packs:
        packs[pkey] = cps32_pack_format(w[:rows].contiguous(), fmt, layout)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = cps32_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        total_mt = inst[1] * (inst[15] if len(inst) == 16 else 1)
        lo = 1 if total_mt == 1 else 4 * total_mt + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * total_mt) // 2, 8 * total_mt})
        streams[vkey] = cps32_verify_onehot(ext, inst_id[inst], sp, cps32_exact(w_rows, fmt), cfg, fixed_m or 8 * total_mt) and \
            cps32_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)



CPSSM_CUDA_TEMPLATE = _zsrc('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')

CPSSM_CPP = r"""
#include <pybind11/pybind11.h>
#include <stdint.h>
int cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

CPSSM_FIELDS = ("fmt", "mt", "rt", "st", "red", "diag", "ntmax")

CPSSM_NL = {0: 4, 1: 3, 2: 8, 3: 6}

CPSSM_FMT_NAME = {0: "bf", 1: "c12", 2: "hl", 3: "hlc"}

CPSSM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

CPSSM_LAYER_K = {"qkv_gate": 8192, "o": 1024, "pooler": 8192, "unpooler": 4096, "router": 8192}

CPSSM_ESC_CAP = 32

def cpssm_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 0xFF
    sign = v >> 15
    eh = (v >> 8) & 0x7F
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return data, base, (er, ec, el, ev, xor7)

def cpssm_c12_unpack(data, base, esc):
    """Inverse of cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 0xFFFFFFFF
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(CPSSM_SIGN_POS, device=dev).long()
    d = ((code[..., None] >> nib) & 7).reshape(r, c, 32, 32)
    sg = ((code[..., None] >> pos) & 1).reshape(r, c, 32, 32)
    val = ((sg << 15) | ((base + d) << 8)).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def cpssm_table(base):
    t0 = base | (base + 1) << 8 | (base + 2) << 16 | (base + 3) << 24
    t1 = (base + 4) | (base + 5) << 8 | (base + 6) << 16 | (base + 7) << 24
    v = t0 | t1 << 32
    return v - 2**64 if v >= 2**63 else v

def cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [cpssm_units(hi), cpssm_units(lo)]
    else:
        mats = [cpssm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = cpssm_c12_bytes(v)
        if not torch.equal(cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(cpssm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > CPSSM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = cpstage___to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.cpssm_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

CPSSM_TABLE_M = {
    ("router", 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8)),
}

CPSSM_TABLE_MT = {
    ("pooler", 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)),
    ("qkv_gate", 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)),
    ("o", 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("o", 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)),
    ("unpooler", 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)),
    ("router", 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8)),
}

CPSSM_MAX_M = 32

CPSSM_TABLE_M.update({
    ("router", 8): CPSSM_TABLE_MT[("router", 1)],
    ("qkv_gate", 8): CPSSM_TABLE_MT[("qkv_gate", 1)],
    ("pooler", 8): CPSSM_TABLE_MT[("pooler", 1)],
    ("o", 8): CPSSM_TABLE_MT[("o", 1)],
    ("unpooler", 1): CPSSM_TABLE_MT[("unpooler", 1)],
    ("unpooler", 8): CPSSM_TABLE_MT[("unpooler", 1)],
})

CPSSM_TABLE_MT[('pooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))

CPSSM_TABLE_MT[('pooler', 4)] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))

CPSSM_TABLE_MT[('qkv_gate', 2)] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))

CPSSM_TABLE_MT[('unpooler', 2)] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def cpssm_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def cpssm_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = cpssm_verify_onehot(ext, inst_id[inst], sp, cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def cpstage___to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def cpstage___to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)


CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')


def cps_build(family, instances, depth):
    is32 = family == 'm32'
    prefix = 'cps32_' if is32 else 'cpssm_'
    template = CPS32_CUDA_TEMPLATE if is32 else CPSSM_CUDA_TEMPLATE
    cpp = CPS32_CPP if is32 else CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == (16 if is32 else 14) else inst + (0, 1) if is32 and len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name, cpp, cuda_sources=source, functions=[prefix+'launch'],
                       extra_cuda_cflags=['-O3','-lineinfo','-Xptxas=-v','-gencode=arch=compute_90a,code=sm_90a'],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)


def prepare_cps(weights, depth=None, layers=CPS_LAYERS, buckets=(2,4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple(layer for layer in layers if layer in ('pooler','unpooler'))
        for bucket,selected_depth in ((2,2),(4,1)):
            if bucket in buckets and selected_layers:
                part = prepare_cps(weights,selected_depth,selected_layers,(bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0,1,2,3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm','m32'):
        table = CPSSM_TABLE_MT if family == 'sm' else CPS32_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and layer in CPS_LAYERS and bucket == 1 and inst[4] in (2,3)
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and (layer,bucket) in CPS32_TABLE_MT:
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0,rows,CPSSM_LAYER_K[layer],*cfg)
            entries.append(((layer,'mt',bucket),(inst,cfg)))
        if include_policy and family == 'sm':
            for (layer,m),(inst,cfg) in CPSSM_TABLE_M.items():
                if layer not in CPS_LAYERS or layer not in weights or inst[4] not in (2,3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m,rows,CPSSM_LAYER_K[layer],*cfg)
                entries.append(((layer,'m',m),(inst,cfg)))
        if not entries:
            continue
        twin = cpssm_bf16_twin if family == 'sm' else cps32_bf16_twin
        instances = list(dict.fromkeys(inst for _,(inst,_) in entries for inst in (inst,twin(inst)) if inst is not None))
        ext = cps_build(family,instances,depth)
        ids = {inst:i for i,inst in enumerate(instances)}
        plan_fn = cpssm_plan if family == 'sm' else cps32_plan
        packs, streams = {}, {}
        for key,(inst,cfg) in entries:
            sp = plan_fn(ext,ids,packs,streams,weights[key[0]],key[0],key,inst,cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext,ids,packs,streams,weights[key[0]],key[0],key,chosen,cfg)
            if sp is not None:
                st['plans'][key] = dict(sp,ext=ext,family=family,inst=chosen,depth=depth)
    return st


def cps_launch(sp, x, out, n, policy=0, query=0):
    fn = cps32_launch_raw if sp['family'] == 'm32' else cpssm_launch_raw
    return fn(sp['ext'],sp['inst_id'],sp,x,out,n,sp['cfg'],query+policy*100)


def run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer,'m',m)) or st['plans'].get((layer,'mt',cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or x.shape[1] != CPS32_LAYER_K[layer] or not x.is_cuda:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = torch.empty((m,n),dtype=torch.float32 if layer == 'router' else torch.bfloat16,device=x.device)
    rc = cps_launch(sp,x,out,n,st.get('policy',0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out


def cps_resources(st, layer, m, policy=0):
    sp=st['plans'].get((layer,'m',m)) or st['plans'][layer,'mt',cps32_mt(m)]
    x=torch.empty((m,CPS32_LAYER_K[layer]),dtype=torch.bfloat16,device=sp['wp'].device)
    out=torch.empty((m,sp['rows']),dtype=torch.float32 if layer=='router' else torch.bfloat16,device=x.device)
    attrs=cps_launch(sp,x,out,sp['rows'],policy,2)
    return {'registers':attrs//1000,'local_bytes':attrs%1000,
            'active_clusters_or_blocks':cps_launch(sp,x,out,sp['rows'],policy,1),
            'blocks_per_sm':cps_launch(sp,x,out,sp['rows'],policy,3),
            'dynamic_smem':cps_launch(sp,x,out,sp['rows'],policy,4),
            'fmt':sp['fmt'],'inst':sp['inst'],'cfg':sp['cfg']}



"""BF16 cuBLASLt projections with offline descriptors and a cold-L2 tuning probe."""
import statistics
import time
import torch
from torch.utils.cpp_extension import load_inline
tt_lt__LT_CPP = _zsrc("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")
tt_lt__LT_EXT = None
tt_lt__LT_WORKSPACE_BYTES = 128 << 20
tt_lt__LT_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
tt_lt__LT_H200_FIXED = {(256, 'o'): (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (128, 'unpooler'): (2, 0, [66, 18, 1, 0, 0, 3, 35, 0, 5]), (256, 'unpooler'): (3, 0, [66, 20, 1, 0, 0, 4, 35, 0, 2]), (128, 'qkv_gate'): (0, 0, [66, 92, 1, 0, 0, 0, 35, 0, 5]), (256, 'qkv_gate'): (0, 0, [66, 124, 1, 0, 0, 0, 35, 0, 2]), (128, 'pooler'): (2, 0, [66, 15, 1, 0, 0, 4, 35, 0, 2]), (256, 'pooler'): (3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2])}
tt_lt__LT_H200_VERSION = 130101
tt_lt__LT_SEARCH_SEEDS = {(128, 'qkv_gate'): [(2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 4])], (128, 'o'): [(2, 0, [66, 18, 1, 0, 0, 4, 35, 0, 2]), (2, 0, [66, 17, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 2, 35, 0, 5]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (128, 'pooler'): [(2, 0, [66, 15, 1, 0, 0, 1, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 15, 1, 0, 0, 2, 35, 0, 5])], (128, 'unpooler'): [(2, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (2, 0, [66, 18, 1, 0, 0, 6, 35, 0, 5]), (2, 0, [66, 28, 1, 0, 0, 2, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 5, 35, 0, 2])], (256, 'qkv_gate'): [(0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 9]), (1, 0, [66, 134, 1, 0, 0, 2, 35, 0, 9]), (0, 0, [66, 134, 1, 0, 0, 0, 35, 0, 2])], (256, 'o'): [(1, 0, [66, 20, 1, 0, 0, 7, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 4, 35, 0, 5]), (3, 0, [66, 20, 1, 0, 0, 5, 35, 0, 3]), (2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5])], (256, 'pooler'): [(3, 0, [66, 18, 1, 0, 0, 0, 35, 0, 2]), (3, 0, [66, 18, 1, 0, 0, 4, 35, 0, 3]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 8]), (0, 0, [66, 183, 1, 0, 0, 0, 35, 0, 9])], (256, 'unpooler'): [(2, 0, [66, 20, 1, 0, 0, 0, 35, 0, 5]), (2, 0, [66, 20, 1, 0, 0, 5, 35, 0, 2]), (3, 0, [66, 20, 1, 0, 0, 0, 35, 0, 3])]}

def tt_lt__lt_build():
    global tt_lt__LT_EXT
    if tt_lt__LT_EXT is None:
        tt_lt__LT_EXT = load_inline(name='muse_cublaslt_v1', cpp_sources=tt_lt__LT_CPP, with_cuda=True, extra_cflags=['-O3', '-std=c++20'], extra_ldflags=['-l:libcublasLt.so.13'], verbose=False)
    return tt_lt__LT_EXT

def tt_lt__lt_state(weights: dict) -> dict:
    ext = tt_lt__lt_build()
    device = weights['qkv_gate'].device
    workspace = torch.empty(tt_lt__LT_WORKSPACE_BYTES, device=device, dtype=torch.uint8)
    return {'ext': ext, 'version': ext.version(), 'context': ext.Context(workspace), 'configs': {}, 'weights': {name: weights[name].contiguous() for name in tt_lt__LT_LAYERS}, 'packed': {}, 'workspace': workspace}

def tt_lt__lt_plan(st: dict, layer: str, m: int, layout: int, epilogue: int):
    w = st['weights'][layer]
    n, k = w.shape
    if layout >= 2:
        if layer not in st['packed']:
            st['packed'][layer] = w.t().contiguous()
        w = st['packed'][layer]
    return st['ext'].Plan(st['context'], w, m, n, k, layout, epilogue)

def tt_lt_prepare_cublaslt(weights: dict) -> dict:
    device = weights['qkv_gate'].device
    props = torch.cuda.get_device_properties(device)
    if (props.major, props.minor) != (9, 0) or not tt_lt__LT_H200_FIXED:
        return {'configs': {}}
    st = tt_lt__lt_state(weights)
    if tt_lt__LT_H200_VERSION is not None and st['version'] != tt_lt__LT_H200_VERSION:
        return {'configs': {}}
    for key, descriptor in tt_lt__LT_H200_FIXED.items():
        m, layer = key
        layout, epilogue, config = descriptor
        plan = tt_lt__lt_plan(st, layer, m, layout, epilogue)
        index = plan.restore(list(config))
        if index < 0 or plan.metadata()[index][:9] != list(config):
            continue
        x = torch.zeros((m, st['weights'][layer].shape[1]), device=device, dtype=torch.bfloat16)
        out = torch.empty((m, st['weights'][layer].shape[0]), device=device, dtype=torch.bfloat16)
        if plan.run(x, out, index) == 0:
            st['configs'][key] = (plan, index)
    torch.cuda.synchronize(device)
    return st

def tt_lt_run_cublaslt(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    entry = st['configs'].get((x.shape[0], layer))
    if entry is None or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    w = st['weights'][layer]
    if x.device != w.device or w.dtype != torch.bfloat16 or tuple(w.shape) != (n, x.shape[1]) or x.data_ptr() % 256:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = entry[0].run(x, out, entry[1])
    if rc:
        raise RuntimeError(f'cuBLASLt launch failed: status {rc}')
    return out

"""Experimental lossless BF16 register-A WGMMA projections, one launch per call."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def tt_mw_mma(n):
    nr = n // 2
    regs = ','.join((f'%{i}' for i in range(nr)))
    outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
tt_mw_CUDA_PREFIX = '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t l0,uint32_t l1,uint32_t c,uint32_t t0,uint32_t t1,uint32_t* a) {\n  uint32_t cm=c&0x77777777u,lo=prmt(t0,t1,cm),hi=prmt(t0,t1,cm>>16);\n  a[0]=prmt(l0,lo,0x5140u)|((c<<12)&0x80008000u);\n  a[1]=prmt(l0,lo,0x7362u)|((c<<8)&0x80008000u);\n  a[2]=prmt(l1,hi,0x5140u)|((c<<4)&0x80008000u);\n  a[3]=prmt(l1,hi,0x7362u)|(c&0x80008000u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
tt_mw_CUDA_CORE = _zsrc("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")
tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def tt_mw_cuda_source(instances):
    cases = []
    attrs = []
    for i, inst in enumerate(instances):
        bn, r, wk, pf, q, c, cap = inst
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    return tt_mw_CUDA_PREFIX + '\n'.join((tt_mw_mma(n) for n in (16, 32, 64))) + tt_mw_CUDA_CORE + tt_mw_CUDA_MAP + tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def tt_mw_build(instances):
    src = tt_mw_cuda_source(instances)
    name = 'm32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])
tt_mw_SM_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

def tt_mw__to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_mw__to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_mw_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_mw_SM_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_mw_sm_c12_unpack(data, base, esc):
    """Inverse of sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor(tt_mw_SM_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_mw_sm_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = tt_mw_sm_c12_bytes(v)
        if not torch.equal(tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = tt_mw__to_i32(entry)
    else:
        data = tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def tt_mw_launch_raw(ext, idx, pk, x, out):
    return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def tt_mw_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))
tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def tt_mw_prepare_m32w(weights, configs=None) -> dict:
    configs = tt_mw_CONFIGS if configs is None else configs
    device = next(iter(weights.values())).device
    state = {'plans': {}, 'device': device}
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
        return state
    instances = []
    for inst, split in configs.values():
        for c in (inst[5], False):
            twin = inst[:5] + (c,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    ext = tt_mw_build(instances)
    state['ext'] = ext
    packs = {}
    for (m, layer), (inst, split) in configs.items():
        if layer not in weights:
            continue
        w = weights[layer]
        for compressed in (inst[5], False):
            actual = inst[:5] + (compressed,) + inst[6:]
            idx = instances.index(actual)
            if ext.info(idx) % 10000:
                continue
            key = (layer, inst[1], inst[2], split, compressed)
            if key not in packs:
                packs[key] = tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
            pk = packs[key]
            if pk is None or not tt_mw_verify_onehot(ext, idx, pk, w, m):
                continue
            if m == 32 and (not tt_mw_verify_onehot(ext, idx, pk, w, 33)):
                continue
            state['plans'][m, layer] = (idx, pk)
            break
    return state

def tt_mw_run_m32w(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
        return None
    m = x.shape[0]
    if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    key = (16 if m <= 16 else 32, layer)
    plan = st['plans'].get(key)
    if plan is None:
        return None
    idx, pk = plan
    if n != pk['n'] or x.shape[1] != pk['k']:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    rc = tt_mw_launch_raw(st['ext'], idx, pk, x, out)
    if rc:
        raise RuntimeError(f'm32w launch failed: {rc}')
    return out

V9_SELECTED_ROUTES = [{'m': 32, 'layer': 'o', 'name': 'ol_1', 'module': 'ol_v8', 'config': [32, 128, 8, 1, 1, 1]}, {'m': 32, 'layer': 'pooler', 'name': 'cps', 'module': 'cps_v8'}, {'m': 32, 'layer': 'unpooler', 'name': 'mw', 'module': 'mw'}, {'m': 128, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 128, 'layer': 'unpooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'pooler', 'name': 'lt', 'module': 'lt'}, {'m': 256, 'layer': 'qkv_gate', 'name': 'lt', 'module': 'lt'}]


def prepare_v9(weights: Weights) -> dict:
    state = {name: w.contiguous() for name, w in weights.items()}
    hi = weights['router'].to(torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = (weights['router']-hi.float()).to(torch.bfloat16).t().contiguous()
    state['router_weight'] = weights['router']
    state['sm'] = prepare_small_m(weights)
    state['m32'] = prepare_m32(weights)
    state['wide2'] = prepare_wide2(weights)
    state['router'] = prepare_router(weights)
    state['qkv'] = prepare_qkv(weights)
    if state['qkv'].get('large') is not None:
        state['qkv']['is_enabled'] = True
    state['router2'] = prepare_router2(weights)
    state['olayer'] = prepare_olayer(weights)
    state['cps'] = prepare_cps(weights)
    state['new_lt'] = tt_lt_prepare_cublaslt(weights)
    state['new_mw'] = tt_mw_prepare_m32w(weights)
    for row in V9_SELECTED_ROUTES:
        m, layer = row['m'], row['layer']
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        x = torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device)
        candidate_v9({'x': x, 'config': {'m': m, 'n': n, 'k': weight.shape[1], 'layer': layer}}, state)
    torch.cuda.synchronize(weights['qkv_gate'].device)
    return state





def candidate_sglang(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor:
    if layer != 'router':
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= 8:
        return F.linear(x.float(), state['router_weight'])
    output = torch.mm(x, state['router_hi_t'], out_dtype=torch.float32)
    output += torch.mm(x, state['router_lo_t'], out_dtype=torch.float32)
    return output


def candidate_v8(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    output = None
    if layer == 'router':
        if 2 <= m <= 8:
            output = run_small_m(x, layer, n, state['sm'])
        elif 65 <= m <= 192:
            output = run_router2(x, layer, n, state['router2'])
        if output is None:
            output = run_router(x, layer, n, state['router'])
    elif layer == 'qkv_gate' and m == 128:
        output = run_qkv(x, layer, n, state['qkv'])
    elif layer == 'o' and 65 <= m <= 224:
        output = run_olayer(x, layer, n, state['olayer'])
    elif layer in ('pooler', 'unpooler') and 9 <= m <= 16:
        output = run_cps(x, layer, n, state['cps'])
    if output is None and 9 <= m <= 32:
        output = run_m32(x, layer, n, state['m32'])
    if output is None and m <= 32:
        output = run_small_m(x, layer, n, state['sm'])
    if output is None and 64 < m <= 256:
        output = run_wide2(x, layer, n, state['wide2'])
    return candidate_sglang(x, layer, n, state) if output is None else output





def candidate_base(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    if layer == 'router' and x.shape[0] <= 16:
        output = run_router(x, layer, n, state['router'])
        return candidate_sglang(x, layer, n, state) if output is None else output
    return candidate_v8(data, state)



def candidate_v9(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    if layer == 'o' and m == 32 and n == 8192 and x.shape[1] == 1024 and x.dtype == torch.bfloat16 and x.is_contiguous() and x.data_ptr() % 16 == 0:
        output = run_olayer_candidate(x, state['olayer'], OLAYER_CONFIGS.index((32, 128, 8, 1, 1, 1)))
        if output is not None:
            return output
    if layer == 'pooler' and m == 32:
        output = run_cps(x, layer, n, state['cps'])
        if output is not None:
            return output
    if layer == 'unpooler' and m == 32:
        output = tt_mw_run_m32w(x, layer, n, state['new_mw'])
        if output is not None:
            return output
    if layer == 'pooler' and m == 128:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'unpooler' and m == 128:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'pooler' and m == 256:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    if layer == 'qkv_gate' and m == 256:
        output = tt_lt_run_cublaslt(x, layer, n, state['new_lt'])
        if output is not None:
            return output
    return candidate_base(data, state)



"""H200 small-m table refinements. Unclaimed shapes return None."""
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_sm2_SMALLM2_CUDA_TEMPLATE = _zsrc('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')
tt_v10_sm2_SMALLM2_CPP = '\nint smallm2_router_launch(long,long,long,int,long,int);\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smallm2_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v10_sm2_SMALLM2_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
tt_v10_sm2_SMALLM2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v10_sm2_SMALLM2_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
tt_v10_sm2_SMALLM2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_sm2_SMALLM2_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v10_sm2_SMALLM2_ESC_CAP = 32

def tt_v10_sm2_smallm2_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def tt_v10_sm2_smallm2_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in inst + (0,) * (15 - len(inst))))
        cases.append(f'    case {i}: return smallm2_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return tt_v10_sm2_SMALLM2_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v10_sm2_smallm2_build(instances, name='smallm2_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, tt_v10_sm2_SMALLM2_CPP + extra_cpp, cuda_sources=tt_v10_sm2_smallm2_cuda_source(instances) + extra_cuda, functions=['smallm2_launch', 'smallm2_router_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v10_sm2_smallm2_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_sm2_smallm2_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_sm2_smallm2_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_sm2_smallm2_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v10_sm2_smallm2_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_sm2_smallm2_c12_bytes(v):
    return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def tt_v10_sm2_smallm2_c12_unpack(data,base,esc):
    return b2q_tt_v10_m32x_m32x_c12_unpack(data,base,esc)

def tt_v10_sm2_smallm2_table(base):
    return b2q_tt_v10_m32x_m32x_table(base)

def tt_v10_sm2_smallm2_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v10_sm2_smallm2_units(hi), tt_v10_sm2_smallm2_units(lo)]
    else:
        mats = [tt_v10_sm2_smallm2_units(w)]
    if fmt in (0, 2):
        data = torch.cat([tt_v10_sm2_smallm2_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v10_sm2_smallm2_c12_bytes(v)
        if not torch.equal(tt_v10_sm2_smallm2_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(tt_v10_sm2_smallm2_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_sm2_smallm2_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v10_sm2_smallm2_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v10_sm2_SMALLM2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v10_sm2_SMALLM2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v10_sm2_smallm2_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp['fmt'] == 4:
        return ext.smallm2_router_launch(sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smallm2_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_sm2_smallm2_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v10_sm2_smallm2_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v10_sm2_smallm2_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v10_sm2_smallm2_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_sm2_smallm2_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v10_sm2_smallm2_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v10_sm2_smallm2_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def tt_v10_sm2_smallm2_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = tt_v10_sm2_smallm2_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = tt_v10_sm2_smallm2_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = tt_v10_sm2_smallm2_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = tt_v10_sm2_smallm2_verify_onehot(ext, inst_id[inst], sp, tt_v10_sm2_smallm2_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v10_sm2_smallm2_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)
tt_v10_sm2_SMALLM2_ROUTES = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1)), ('pooler', 8): ((1, 1, 1, 1, 0, 0, 1024, 0, 4096, 8192, 2, 16, 8, 1, 6), (2, 16, 8, 1))}

def tt_v10_sm2_prepare_smallm2(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in tt_v10_sm2_SMALLM2_ROUTES.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
    ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_final_v3')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

def tt_v10_sm2_run_smallm2(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != tt_v10_sm2_SMALLM2_LAYER_K[layer]:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = tt_v10_sm2_smallm2_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None
tt_v10_sm2_SMALLM2_ROUTES = {key: value for key, value in tt_v10_sm2_SMALLM2_ROUTES.items() if key == ('pooler', 8)}

"""Hopper BF16 projections tuned with cold-L2 H200 graph replays."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_tune_CUDA = _zsrc("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")
tt_v10_tune_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,int,int,int,uint64_t,int);\n'

def tt_v10_tune_source(configs: list) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'D void mma{n}(float* a,uint64_t w,uint64_t x) {{asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;":{outs}:"l"(w),"l"(x));}}')
    cases = '\n'.join((f"case {i}:return launch_t<{','.join(map(str, c))}>(x,w,y,m,n,k,stream,query);" for i, c in enumerate(configs)))
    return tt_v10_tune_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CALL@@', '\n'.join((f'if constexpr(BN=={n}) mma{n}(a,aw,ax);' for n in ns))).replace('@@CASES@@', cases)

def tt_v10_tune_build(configs: list):
    for c in configs:
        bn, wg, st, pack, split, push, pw, px, hint, order, sw, prod = c
        assert bn in (128, 192, 256) and wg in (1, 2) and (st >= 2) and (split in (1, 2, 3, 4))
        assert st * ((64 * wg + bn) * 128 + 16) + 16 <= 227328
        assert prod in (0, 1) and (prod or wg == 1)
        assert pack in (0, 1) and push in (0, 1) and (pw in range(4)) and (px in range(4))
        assert hint in range(4) and order in (0, 1) and (sw in (0, 5))
        if push and split > 1:
            assert split * ((bn // 8 + split - 1) // split) * (wg * 128) * 16 <= st * (64 * wg + bn) * 128
    src = tt_v10_tune_source(configs)
    name = 'm256tune_' + hashlib.sha256(src.encode()).hexdigest()[:16]
    return load_inline(name, tt_v10_tune_CPP, cuda_sources=src, functions=['launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=True, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def tt_v10_tune_pack_weight(w: torch.Tensor, rows: int) -> torch.Tensor:
    n, k = w.shape
    col = torch.arange(64, device=w.device)[None, :] ^ (torch.arange(rows, device=w.device)[:, None] & 7) * 8
    return w.reshape(n // rows, rows, k // 64, 64).permute(0, 2, 1, 3).gather(-1, col.expand(n // rows, k // 64, rows, 64)).reshape(n, k).contiguous()

def tt_v10_tune_run_candidate(x: torch.Tensor, w: torch.Tensor, ext, index: int, n: int) -> torch.Tensor:
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = ext.launch(index, x.data_ptr(), w.data_ptr(), out.data_ptr(), x.shape[0], n, x.shape[1], torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'm256tune launch failed {rc}')
    return out
tt_v10_tune_CONFIGS = []
tt_v10_tune_ROUTES = {}

def tt_v10_tune_prepare_m256tune(weights: dict) -> dict:
    state = {'routes': {}}
    if not tt_v10_tune_CONFIGS:
        return state
    device = next(iter(weights.values())).device
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
        return state
    state.update(ext=tt_v10_tune_build(tt_v10_tune_CONFIGS), weights={}, routes=tt_v10_tune_ROUTES)
    for layer, index in tt_v10_tune_ROUTES.items():
        w = weights[layer].contiguous()
        cfg = tt_v10_tune_CONFIGS[index]
        state['weights'][layer] = tt_v10_tune_pack_weight(w, 64 * cfg[1]) if cfg[3] else w
    return state

def tt_v10_tune_run_m256tune(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    index = st['routes'].get(layer)
    if index is None or x.ndim != 2 or (not 129 <= x.shape[0] <= 256):
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda or (not x.is_contiguous()) or x.data_ptr() % 16:
        return None
    w = st['weights'][layer]
    if x.device != w.device or tuple(w.shape) != (n, x.shape[1]) or w.dtype != torch.bfloat16:
        return None
    return tt_v10_tune_run_candidate(x, w, st['ext'], index, n)
tt_v10_tune_RL_CUDA = _zsrc("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")
tt_v10_tune_RL_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint rl_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def tt_v10_tune_rl_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void rl_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return rl_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) rl_mma{n}(a,aw,ax);' for n in ns))
    return tt_v10_tune_RL_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def tt_v10_tune_build_rl(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (2, 3, 6, 7, 14, 15)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported rl configuration: {cfg}')
    tt_v10_tune_source = tt_v10_tune_rl_source(configs)
    name = 'rl_' + hashlib.sha256(tt_v10_tune_source.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_tune_RL_CPP, cuda_sources=tt_v10_tune_source, functions=['rl_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

"""Lossless BF16 projection refinements; only confirmed exact shapes are claimed."""
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_warps_W0_DC2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v10_warps_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_warps_W0_DC2_ESC_CAP = 32

def tt_v10_warps_wp0_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_warps_wp0_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_warps_wp0_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_warps_wp0_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v10_warps_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_warps_wp0_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_v10_warps_wp0_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_v10_warps_wp0_c12_unpack(data, base, esc, decode=0):
    """Inverse of wp0_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_warps_W0_DC2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_v10_warps_wp0_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_warps_wp0_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v10_warps_wp0_units(hi), tt_v10_warps_wp0_units(lo)]
    else:
        mats = [tt_v10_warps_wp0_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([tt_v10_warps_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v10_warps_wp0_c12_bytes(v, decode)
        if not torch.equal(tt_v10_warps_wp0_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(tt_v10_warps_wp0_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_warps_wp0_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v10_warps_wp0_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v10_warps_W0_DC2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v10_warps_W0_DC2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v10_warps_wp0_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
tt_v10_warps_W0_CUDA = _zsrc('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')
tt_v10_warps_W0_CPP = '\n\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint wp_bde7122f9b4f_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n\n'
tt_v10_warps_W0_INSTANCES = [(1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (0, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1)]
tt_v10_warps_WARPS_FAMILY_SPECS = {0: {'name': 'wp_bde7122f9b4f', 'launch': 'wp_bde7122f9b4f_launch', 'family': 'dc5', 'decode': 5}}
tt_v10_warps_WARPS_ROUTES = {('qkv_gate', 16): {'family': 0, 'inst': (1, 2, 1, 1, 3, 0, 128, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), 'cfg': (1, 4, 8, 4), 'rows': 2304, 'k': 8192}}
tt_v10_warps_WARPS_PACKERS = {0: (tt_v10_warps_wp0_pack_format, tt_v10_warps_wp0_stream)}
tt_v10_warps_WARPS_SOURCES = {0: (tt_v10_warps_W0_CPP, tt_v10_warps_W0_CUDA, tt_v10_warps_W0_INSTANCES)}

def tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0) -> int:
    return ext[sp['family']](inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), x.shape[0], n, x.shape[1], *cfg, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_warps_warps_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8) -> bool:
    n, k = exact.shape
    x = torch.zeros((rows_per_launch, k), device=exact.device, dtype=torch.bfloat16)
    out = torch.empty((rows_per_launch, n), device=exact.device, dtype=torch.bfloat16)
    for column in range(0, k, rows_per_launch):
        count = min(rows_per_launch, k - column)
        x.zero_()
        x[torch.arange(count, device=x.device), torch.arange(column, column + count, device=x.device)] = 1
        if tt_v10_warps_warps_launch_raw(ext, inst_id, sp, x, out, n, cfg):
            return False
        if not torch.equal(out[:count], exact[:, column:column + count].t()):
            return False
    return True

def tt_v10_warps_prepare_warps(weights) -> dict:
    state = {'plans': {}, 'ext': {}}
    entries = []
    for (layer, m), route in tt_v10_warps_WARPS_ROUTES.items():
        weight = weights.get(layer)
        if weight is None or weight.dtype != torch.bfloat16 or (not weight.is_cuda):
            continue
        if weight.ndim != 2 or weight.shape[0] < route['rows'] or weight.shape[1] != route['k']:
            continue
        if torch.cuda.get_device_capability(weight.device) != (9, 0):
            continue
        entries.append(((layer, m), route))
    for family in sorted({r['family'] for _, r in entries}):
        spec = tt_v10_warps_WARPS_FAMILY_SPECS[family]
        cpp, cuda, instances = tt_v10_warps_WARPS_SOURCES[family]
        ext = load_inline(spec['name'], cpp, cuda_sources=cuda, functions=[spec['launch']], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        state['ext'][family] = getattr(ext, spec['launch'])
    packs = {}
    streams = {}
    for key, route in entries:
        layer, m = key
        family = route['family']
        spec = tt_v10_warps_WARPS_FAMILY_SPECS[family]
        weight = weights[layer][:route['rows']].contiguous()
        n, k = weight.shape
        pack_format, stream = tt_v10_warps_WARPS_PACKERS[family]
        cfg = route['cfg']
        original = route['inst']
        for fmt in (1, 0):
            inst = (fmt, *original[1:])
            pack_key = (family, layer, fmt, n)
            if pack_key not in packs:
                packs[pack_key] = pack_format(weight, fmt, decode=spec['decode']) if spec['decode'] is not None else pack_format(weight, fmt)
            packed = packs[pack_key]
            if packed is None:
                continue
            stream_key = pack_key + (inst[2], *cfg)
            if stream_key not in streams:
                if spec['family'].startswith('rag_'):
                    rows, columns, size = packed['data'].shape
                    target = cfg[1] * cfg[2] * cfg[3]
                    padded = torch.zeros((rows, target, size), device=weight.device, dtype=packed['data'].dtype)
                    padded[:, :columns].copy_(packed['data'])
                    packed = dict(packed, data=padded)
                streams[stream_key] = stream(packed, inst[2], *cfg)
            sp = streams[stream_key]
            if sp is None:
                continue
            index = tt_v10_warps_WARPS_SOURCES[family][2].index(inst)
            plan = dict(sp, family=family, inst_id=index, cfg=cfg, rows=n, k=k, inst=inst)
            x = torch.empty((m, k), device=weight.device, dtype=torch.bfloat16)
            out = torch.empty((m, n), device=weight.device, dtype=torch.bfloat16)
            attr = tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 2)
            occupancy = tt_v10_warps_warps_launch_raw(state['ext'], index, plan, x, out, n, cfg, 1)
            required = n // (16 * inst[2] * cfg[0]) if cfg[3] > 1 else (n // (16 * inst[2] * cfg[0]) + 131) // 132
            if attr < 0 or attr % 1000 or occupancy < required:
                continue
            if not tt_v10_warps_warps_verify_onehot(state['ext'], index, plan, weight, cfg, inst[1] * 8):
                continue
            state['plans'][key] = plan
            break
    return state

def tt_v10_warps_run_warps(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    plan = st['plans'].get((layer, x.shape[0]))
    if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    rc = tt_v10_warps_warps_launch_raw(st['ext'], plan['inst_id'], plan, x, out, n, plan['cfg'])
    return out if rc == 0 else None

"""Lossless m32 pooler and qkv projections, with an optional unpooler route."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
tt_v10_m32x_M32X_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v10_m32x_M32X_ESC_CAP = 32
tt_v10_m32x__CFG = (1, 8, 16, 1)
tt_v10_m32x__CUDA = _zsrc('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')
tt_v10_m32x__CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def tt_v10_m32x_m32x_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v10_m32x_m32x_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v10_m32x_m32x_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v10_m32x_m32x_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v10_m32x_m32x_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
    """Inverse of m32x_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else tt_v10_m32x_M32X_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def tt_v10_m32x_m32x_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v10_m32x_m32x_units(hi), tt_v10_m32x_m32x_units(lo)]
    else:
        mats = [tt_v10_m32x_m32x_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v10_m32x_m32x_c12_bytes(v, decode)
        if not torch.equal(tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(tt_v10_m32x_m32x_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v10_m32x_M32X_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v10_m32x_M32X_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v10_m32x_m32x_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32x_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_m32x_m32x_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_m32x_m32x_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v10_m32x__build():
    name = 'm32x_pooler_' + hashlib.sha256(tt_v10_m32x__CUDA.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_m32x__CPP, cuda_sources=tt_v10_m32x__CUDA, functions=['m32x_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__make_plan(ext, weights, fmt):
    pack = tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = tt_v10_m32x_m32x_stream(pack, 1, *tt_v10_m32x__CFG)
    if stream is None:
        return None
    return dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=4096, cfg=tt_v10_m32x__CFG)

def tt_v10_m32x__run(x, plan):
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    rc = tt_v10_m32x_m32x_launch_raw(plan['ext'], plan['idx'], plan, x, out, 4096, tt_v10_m32x__CFG)
    if rc:
        raise RuntimeError(f'm32x launch failed: {rc}')
    return out
tt_v10_m32x__W_CUDA = _zsrc('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')
tt_v10_m32x__W_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_w_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint m32x_w_info(int);\n'

def tt_v10_m32x__w_c12_bytes(v):
    return tt_v10_m32x_m32x_c12_bytes(v, 5)

def tt_v10_m32x__w_c12_unpack(data, base, esc):
    return tt_v10_m32x_m32x_c12_unpack(data, base, esc, 5)

def tt_v10_m32x__w_table(base):
    return base << 8 | base << 24

def tt_v10_m32x__w_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = tt_v10_m32x__w_c12_bytes(v)
        if not torch.equal(tt_v10_m32x__w_c12_unpack(data, b, entries), v):
            return None
        table = tt_v10_m32x__w_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = tt_v10_m32x_m32x_to_i32(entry)
    else:
        data = tt_v10_m32x_m32x_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def tt_v10_m32x__w_launch_raw(ext, idx, pk, x, out):
    return ext.m32x_w_launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def tt_v10_m32x__w_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if tt_v10_m32x__w_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))

def tt_v10_m32x__w_build():
    name = 'm32x_unpooler_' + hashlib.sha256(tt_v10_m32x__W_CUDA.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_m32x__W_CPP, cuda_sources=tt_v10_m32x__W_CUDA, functions=['m32x_w_launch', 'm32x_w_info'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__w_make_plan(ext, weights, fmt):
    pack = tt_v10_m32x__w_pack_weight(weights, 64, 4, 1, fmt == 1)
    if pack is None:
        return None
    return dict(pack, ext=ext, idx=0 if fmt == 1 else 1, fmt=fmt)

def tt_v10_m32x__w_run(x, plan):
    out = torch.empty((x.shape[0], 8192), dtype=torch.bfloat16, device=x.device)
    rc = tt_v10_m32x__w_launch_raw(plan['ext'], plan['idx'], plan, x, out)
    if rc:
        raise RuntimeError(f'm32x unpooler launch failed: {rc}')
    return out

def tt_v10_m32x__w_verify_random(plan, weights):
    for m in (9, 16, 17, 24, 32):
        x = torch.randn((m, 4096), device=weights.device, dtype=torch.bfloat16)
        x[-1].zero_()
        x[-1, m * 131 % 4096] = 16
        ref = (x.double() @ weights.double().T).to(torch.bfloat16).double()
        out = tt_v10_m32x__w_run(x, plan).double()
        diff = out - ref
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 0.0025:
            return False
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if float(rows.max()) > 0.01 or float(rows[-1]) > 0.00025:
            return False
    return True
tt_v10_m32x__Q_CFG = (1, 4, 8, 4)
tt_v10_m32x__Q_CUDA = _zsrc('eNrNff1X27jS8O/9K9TuoY+duGmchI8CoYel3X16KLQvocu+h5ebYxwHUhw72M4H7PK/vzMjyZZtOQnd3Xtve0JkWRqNRjOjGX1Mfhp4w1HgsQ9Hrf6Hj0dfPnxkmy9e/DQKXH868Ni+Ox04/WgaJKOx17g9KL65Htpb+ew4GYyCJJ/n+DdhNEpux2pm8jDx+knkjJI4BzYMJ17kJKOZ17+JwukkRlgvfhJ4fvrlA+v3B95s5Hr9PiSHYeR6o8CHl/3+ixf4HtrvsHG7tejf9/3B3HDDIE4ot8YmJvvjBeNFZnuQcuIxm4U+NOh7xit/0Ljxw2vHb7g3jVmnEbdb7I+NprVhWxsta6P9ZLHLjc7V3iuomfu3y151o1fGrLEwLZl8yJKPWXJuYmH/lTExTUQg8pJpFCAyTxz9aR7/hcB/Fo4GKf7T1R0IXOzAtKoDfzfG7VY/yZD2BNLyTQ5xKroKd0Qc8AZsbRXb56IyicaJkWY6VlbgWknHBeQGAjnjFdZvXAtcgIgMqMg22hlGA8IIEg4SDL6vxXecw3CQYojjKLEbjx1j6IdOwozXrnnZubJYkW45lJvqg82RJjQBUCN+ALI5/ugm8AaNsb0V7NyBcEbhvOGGfmPYbjVIWunPsMwWf2x0rI1Na2PL2timxx1r4x0llGIp40Of68NXhnvZvML+8rStpFtKun1lZtWQVKIWJe0s2cqSbZm8bsqEjQQFIr59y3bY3Bvd3CYxC4csBLUAPWVx4k12mR/O2fVD4sUg4+xzkxkzx5/CU/NN22ROMGCf7TSv82YbgHfeXI8SoPuA1zlqlIZp4Ln90B9kfPRZHYjPtvJwpKTP1VLnSikYVT52+cFm7ph12RF7zZqLbfFvulcuduuHUE5lcGwJG3DHpq787aiyPDs4YPYW1cJRKZTDfkJrFiC0aXeaU5P9yQzjiO3vM7tlEqI7zWYTP1MBw14CY7u91VJh7OhBtEogAFXohA6Njh5EewmIDIujUt0KLTKe+nolEruO71UqD6jWiAKSuIWqQZZoDw5QrzlKWDnxXf965MQKbpW4+OGknVNk0PHt4RA/mO4AAfCDac/RIrhanYGc/N0y8tNoqNom3S5rAhYFucSmsAUALDnbAWx/8vxSbXtpbaBDu9PqII/Rw3Znq7PZ6UwFvEnk3IwdNg2i0PcBENgezAB82T1wW3MPvoAh91i9fg+q5vJe4UFkIMxBBDlmsVfs20GXddSxux2lgxffDhtRYx45k/xcBJ9ONla3IzlYR2baSFkboDJA6QHJF9LT3Ob/tYLF/08JYDAYDas1xVEmoEt0wVEmg8vEfVQE1q4uJqE9a4B0vNVif/5ZyOrQzFUezZIMYhEzN75KRXrLCSqEDmhdh8IKVZ9kEiQrZ+IGobRvc9IWeDMvMjKzsGBNgWk92Xtl7jE+ZZ5iaZY4d15gsetpwibJwomZ6wRBCA9ROPPYKNllgELMkluPXTvxyGXXfujesTiEedWBF6M4nnpsfutF8BdM+sQLytNlGA28qD8BrBMi/tgqG6/EiknZBhyHs9T029hwsfWMwROug2DYDCMBaqbKxhybpBoWQFj6B9TNUWlSrdvjsTfuQ5MlM1vquqyVft+dJU7/xgu8aOT2k7Af3zqRN0DwrAJ85AR3RqHf0ep+T8Gaifpu4mD9jAJRTgtHS6aGiaOfsRBexSShogP1G7xzu7sCmwy/VZMYtbGG9XvtRP1RMFIMdMhRcHVDcDxTK1dFD8pFI0AJa2d4Jk7jeqsDLkPzCrFEDHfJdHQigZmACOjCqIfRwyuzGjNvMfHcStySxVLE8GvmNTiMfrIoIdm3luGJ0FchOfQCUBBEv7L008uGRIdKNSLP95zYa4jxpGbVRoCFi224fp/3RNOE7KpkD9FlaMVZgAciPJE1W5k72m4U28BiyyCXQMdJ30HHqGMoEmAxcrrAgVYFA6mvGdIYGiTXqigOkriQE44nvpd4MM67u+R8oOuNPhYf4j8yxxE+HfLBN3WLCLtcepAJhqlrPkw982HqmA+5Xy7EjTDXsct1GPo5nk6ihxxDFzRBeFdWBX/8v4DBoN6wxgS6zyZ78JyyOcCjoWtMHJgNHhqOez8dRSmLlbh+wh16mLQBTKn/r1js+ZNUz6CRalGDT6muCe+ksil3OlU34R17ifO9qh23Opl2vMHVLEUp07uqyQiw5kqZr01g1UjoPj+dkkTDiXRQfwY9M4COM4ddRyHMuQym4yB23GQUBlyvsfg2nMdsOoFGoRif79nQmfq4VgXUcwbo3eKsNB2PghuwaoYgxvEtDMLAAXqWeB0Hog9k0YwwTplFPsCXpsA87zVyejQzg0cQDEvNb4E0zHiphUYDqrQlCf0GoR2wVpP/m/q+CXYNWiiGXrc5g0GHr4x0XuP0xXGTGdekKBoLVu+y68ZiD9IPPP2A6UeefsT0nKfnqXIQki/Vw+3Q7y/CqGPkWwDFgJZLyKknRncMxlOflzD6adU+KgejuRhK0wOsncbCgroWW1HqgUoVpYD/W1H1ca0G5lhKT+E4CSMPuhHM+tfUJ3urhuLJl6NmSOHaBDigz3vckoX6UWDMtFqcQ6RiWkgzGoPEA13pJB7bRwL/cnJ+wOIkmroJOwEw/+eXMchQVqb0cr95APDiBCTU5VwBU2xEg3X6GRrpWOz0EL5taGw5IHs5oPbagFrLAe0IQK2VgNrLAW3lAGkIqfPHw3QpllaMjddz8zJtECvt7p5+vlKmQeP1YbnEIasxe+tqid+uY+L0/Te18Dc7VUlZJw1oiCx4dMBEukU+WtGlU5w6nzt1Pjh1pS6hk+fzhvDf4WUHuuCDy0Wea2bSzy/9K9AgpWK2pthDuVhLU+yxXKytKTbnxZ7Q82PoL67o7S3v7a2mt4fY29ust+qGxxSqzREFQOV2L1cgs2VR1d+y9zBQMK/h0Ca2zLExx5YV1YWey+YV6jn6fsDvFj0nVN1ih0jFLWwVqW5WQXgUEOYCwkMFhI4egi1wsDMcHisg7FRCeBQQJA7zCgh2y1QGTUz1H2PXmXjgN8MsuAsGzD2tju6wP5nvBB49tOEBLKgRmkS07gnPt44/xPQ2pLdpBfr3L2c44aPvfTu6uaVV7P+JGchHGAB0BmXiRkHsT8/KUj9xEvfWyAv06ZkqvJ529RltPOBSDxeGaGW2betWnxcRL0Wr1L+Aow2dMHaAOgavuo1Vbciv8YXl6kWZiHM0kAT6gSwcZQYEIaPKDBgyh5fRFftXFzCQZtbh4LvjIm2Q1DH0AGwmR+aJmT72wyS22HyU3DLPcW/5sIBNRUZVcgtzFBYBSscTf5S8OWZgyyYjx88R2/WdOGYfx6OkTHCwcKeul7MgaiyOXG5DnH7i31/FV+/8jKeS0YAngsQiyMyDP3xoUhqNBgugEhTdoyRQ6hNPgVkjfWOWujNosSlGSrOBa7vqH8G/KfgJH4IJAP6KIzAxcxYYmEHQj0sjHj2Cc2NOYEwRfxhsQOFKAEOsDXgGg4NypGCAFg92WMKtXzAaYRDOzoEn3kThnGewCcjD3InAWOidA4MAg+MOzNDHvR16KYs22IdPh78yA+xJHLUw8B/MXWazIGQnJ4cWNtfChwVfp7JYB59oYNDitkDk+AIW1rRQomFkkwd250WB51sMPA6DOAQYH0YckAinyWQKrIFGDcADLyAt4AD045+xzRj3rwHXN7g8R1tIYsEMmArl2AtgiO3WjqjbMRGrHORGeR7nTCG/z8R3Tz5//MATSBDBYecnh7/z5NGJ+D4V38fi++JMJmTO8TeR6PHv30XVcwH16OvBi36fOz1y5bHf951p4N72r9G/ift9g7fNv7pd6Or7zd22mXLRjTeeGS+Kc1IfRAZIN3ITADyfSPO+YIzmSi3kGmIuF0lpvSjPaIViXixkcdx3ohtBtCx5nCUvzpS0kn/8jaeX2z/LDZ6UAoy2+U5gdoU/uxypvVyBUyxwigVOoYBA9RgzjzHzGDKPy7UuzrAE/H1Pf3fT3lxQ1Quqe4GVZdeOvxHMbwT0G0L9lgObN0BLRpYwSEvmiAXce44vzkFhnP+aB0drEqdfLuA9yfRr1trc0hQ5sls40Qhj0FYMwzbMMK/ZS4Chqfa/nz5/gXpY9ABtZV3boDKy1mGSP/3yu4INPoOUZTngU5x9yR7BoP948vX8/yogdPiffen3zgFQVqzdQkB9vrMgM7fQY/lSKAlypIH49Vvvf5EkiBz2DamA6X3wDSRJ8BlTgDB+EaIIahqj3vwyTXBc8OhN4IxhFkxwbSYMBiPUko6/j9QTnpuVl0cYWKyGsBBKDWUPQEUesIcXTcBF7rtOnOzTywMDJZNmA28BrwNyVHEpiDY0aAEPlAfYB6D1Y1rNYy68Z7gYf3mF9bQVQPmm6xMxLnXkBWDBN/6BRL+fpNs5PNnOkpsmsDsyaA1mhRozjtlbZhz1UAZ6IAI30WjwYTRuPJhkM0KJFuQ2OU45BRUvdATIFzowsEtmHs/ImUtMj75CAyCsNRTSGk6QNRS2Gtu0W/lKaZW0dl32syMkvkbiLaGg09ZqdnYQe75vQso3nuuwpnccWYDLCSk3WxQ3jfhJWh3Z0lHqwOQ0PZotyG23aJd9GizQUp/MuTmDluKmRZM/ZEDuW5p4khv+tEFPZKnx4miN7hXAB0lO38nen/8Kf9o4aDQX41guilVPUf0IsgMWaIkSKhsMJ8r5nXx+C8/Fuj1stcwuxWK//q4oceAwao1GBtk+q1jC7VeoB5Vr0BCM4kXx9Q1S0KCuIU0foBCUrrM0Z2FSPcjCLhSrR+TsZfUXnP+gcGQSgsUKbrPUHpAZyt9h+eNvxfLnv6EVOgoMdH5BzCxmIEvBJIhHTcw9zXSN65TZYmO6IfeaRFzUyG1gkdUqfQWFPVFDZm6wKJnbiJMuoFpPKtTNrCoWMIh9oSWzvGFFe0F2Co3U1UPgclaPs2aesuaMtBlURRw3Ag56OmspQ6BAInRKYqn9pW4T6Q5qtMxVMowee4MERzZAVtO+ujhT0C+RYx+psbrbK8nIdFtqBMbovbHNGlbqdjffn17sAkK1s/OaATzztmfWQAFOlZaEiqmCxskjBEzSqJMq+g6uYvxGPmlrRwWr2Usrjl15K0wuAmg4cMXEU1qhYXc4qDgF9WS7qXM2k57fDNd80gkLarxlO5ib8wELqhcXXFHt8NIm2NiUs5Hm7BVqcfMa3D0ot8C5RTh9KMpgDIwl3xyruuaB41MH6IBcBpIDG8SIRozQEhwWKrljlovraKgnnEYyYkejO6BppYfFjR13Ivb03BuxRSVOnooNO75HBWYdbjHyc1m0KedDInJL2045Dq2V5tapmFwRpwxVPNmLwDSqYgUtqvoSjsET54emS3ujAjhhqNVVT3ktO37o49TbXPwi/k11+hZtdG5tmrKGcgQY3C6Fj2Deomm5TtM6tFpyC+d4Smc+KdYxMqOo3TJFfXUa4YVZjP7eAHGQ1X8t1VarFay5RbPA+DDx1dDvhxwyRF6j41GjIQEt9w50C5pWtMbFSUOdAOssQlP7OTYWsOc0CXHp/8bry5OtXXb5+or0AC5PcTc0HmWSUBiJr+yANaVflJ/IsEegQY6/VSyoF9TO925zj33fB0uA1evfzeoa6lI81vH3S+vuFSILktAtSqwkHC4kGtjl2ulFHS0YnBUAj9rpZ4Bca7fqgn90LQgTN3K78/u6HMh4VBO8UTe+p2CKAP6qhkDpXqUiVDlXU39BojmQp72lewf3ODz3+0dfxdm3HKcZ99a9AEd7RPPL3vnV5Rl8Tj9fZcZXhy1m9OYEPq0rhW9xZU3PrpUMlzEbX4D8Dvy5iuF0+z4pw5WkgTaTUnFAIQC5fYk2AajS5EaYW/NLxPbq8vvVpX+laq+5Adoo0wVoWkj9ArnfuUoBVWSqeqUolC9xYSHzxoUyb2bKfCtLvsuSdtNcIqkpGQJOhoCMAyRDoJsmi1oO3S1Qdfn53QCCoLEO2YFJZoZUemCzt8zC7B+TXlS5GjiD0zG4yh0NxTluMTGritqlosKnFfqVnLgdU51jn8P1MSdQDATqEYGAH5FXjdhiUvFqjiyOLZauXjzHNKkQYjqVQSIMyFQIccWMnK3tM8d1L08PuViiAHau9tbdLZTbg/+AJGpYcI0JxuW1XHkYF7Qldu+WxBDYwkW2aDaGBcdvGPqDzPFLs4NFojVUoCFvUuUoqoYLByBOUpRPUaBdY9EKSqFrdJZc3arh3eJTLX6Ded47X60CdWyqM+ljbGCE8pcznI0eXxTF/kAREB2c6K+Ble6qDGzs9VqWwPossUpBFyZbGst/qft7BqhioYcbC/Yvpjw+5B8f84/znHoR1k4dN5S4yZNKPM/NlV4p/Ypdr6rWbOGChHHnqoCCZg4QqmM7Ux07isa3zaUj8gyt/89q/lFJ9eetuw57bHbzyt9ij3Y+q75UxfN/13wq6T428XzXNZ8t8OlBPLXo6VE8telproXSgXd2CmWTniSULXqSULbpaV5lr63NMflK6xrq680hG01hcgavANYbu2BprsNCzzDwn2Hil/1ysMIrjPy6gTb+aOPoq/k8K5+bqK/ZrJtpgHKZ0rVSYcHPOnR9ZvV92Ow6bHYbNncZNlpl5edZ4EmnOw65jS3OVu39mD4uDEOe1wSDrTe+Doyvs396WAOnxqnX8bgFkNe56jYX7eFOEz/TgmRwxVhlW+cOodHeoNTxFsHOLmvhduk3+wcFKO/LEFNZ8agOrLUOSVer0yJJf1+fpB4H7gHwHYTtmVzPeGTgrCar0u4zfQetMkX9P8MRoCdFrxZeZCq28CLTtoUXVYpXLWZfKTq48CJTx4UXmWYuvFiipJ/+WXNAY0quO3evlt+SEp3igbJuttGHmwU3dLod1/xTq/fjNL/MTxP1Gc7q30uKifYi0OjF/ZgdujiFreg0+iDUZHIIfG+uSx0zM3htcXAsOwJCx9H2hZo7MLjkQ3GNbmcoIxNdPrfRPTynBAr8/Spr3Zugai75BHkpEyfZdaRYpsuXMlvJVexo1ImRqsh1FckzDfH82k929ZEGQdx/1BA5nl8aBm17prwj95FxFfTeVBdwQSC1B9fp/BlO0jTSl/e1jlD3mKzb6kNLfWhfmcvsror9tRxNX/6naaoVFuX0NW7oPI/IOllI6dqdoUZVaAsZD3sqfSHjcU+lMWTMNbLww+y9pXdnmutqxPK93DVH71krD2uMvIaRsxAduDrRFKsTnGVxpHDELDFT1vBqt0zCSORtGd2Eriwf/Rih1mb0QBzyDFB7GuTqgdfwFu8RANkm1ZY87SOGuIUunUTuHwYT0uXKxsg2bssY8qQWzEs4G0kvskN+pUbdK2cMQlwyMOZ3dKKAHE1Tbr7cp42lEGmSIad0Z2+VF+IMBrqzBXxTMryTB4B66c5k6GtQVUOwXNvWdcu6bmt8k9wIC3LoeKvooIydJBotChFcdoKdxqIlfZdroAF3XMhlaSkuCw+R0uUxUuRVXOi2qelIejH9RxHqlBGq8KRyaMlUK021i6j+BfHVKl69ACPvtspSDB7Ita01SUqDiqdYlgGn2yOlBlrwaZurdO9SXbzklsjftHC3zuJSJgqLwsKXdpI4QI0AVisk94UB+1I4Uau0zjPWp6rUQGkNbJXQLj+/oWEPWlaL1X1jnVbRq7bcvjI/RFqWVs3CHW9hyYIcSxfjcPXtr/VZK1uPuFlePumgLATWDoo7P2lde626uRs7GimoQKqw88Rbe9YeU5VkMuT4bElyoSxILpTlyEXVYiTVzxYjF8pS5EJZiFzolyEpPsha873WVFjOAXQkeqVFsXKDSecPyJV+MkKz21zwfX/F/qXL5rfvql614JX+xqxqHpeqtbWmdLoNsVRTvacB5hjvCiuPnkzAct2qvFdqdcp5Hoevskmpt9YPd8f6y71ZYwbFI+al+RP8H1Djxe6Q2fff2Kenv3N9oNK1LWx2644bFhdDf3gPWrdBqN86pPsLxQNG/Bx34QSrYivgET0UNjCCRApEx57yU4nisQWPg+yxnVu0KVj1d7n7tGP2J8O7iqsNI0U3bukWQopRcAYNP8zi3uAHbLmNTYqBobwUsbj0L1si9od8+apafb3KVWzLgCFUkezquoyyU5eRJjHhykQaiedOhAa5z3Ov2EJ3gLzX8HHhM6jiN83I0y2V4tArRgwxQHk9lQcGwPsidTp+nnMehRnX7+No9p2Y3100CNPXtEguIzplR56KChLR4F3DZfV3Xnt7+931uywGVIqAxez0DmTWz+weQwk2P3wRrTp28d9yjCJST1DQTaj3TJ0o8EWdlRQuZu8WC+qPNKWWwkraRwisKfYdbHmaI9Vjqrl/9rO8jSHuR+yVikThHK273LWFs58VHA1+9wkMV3n6m9QiXn1DJ3ydA1XP8pp+dCVq2fD98WK5Q6RzhnCiUkQQxMqlxZHK85F0pxZgoSSK1ceiOKLpjWBd9cpGfrlVXll7LwOEoOiiCBsZD5oFngAew37ss3HZVNSoiry3c4rKI5xbCo9XLhirAl2lzvhdOZXmb9+yX6a+zy+M74qb1wbOBDmip4uzDXYUjieOy088xrQ2YJvFiue/4UKTHK2bRnbmE49O4t3p/hBbzc5P0lVqeaNbd6SPduOR7bAl3G0Bonh01B4z+MWqgA4gQ+YGv4ND2IgsvJp9Xp5cRbyWGd+3+4OC2FCMGoo2M2vMn/Z+TCvl2brI1EGOkYOlbFxkYtIKdSZILdg4WIOJacdA8OIzeA+IU2A75TTvnmZoXcEhzxzdm3Rwt4tjCwZBko3t+W/FoT3/7b9mZHFEk78wlDf/wSEs2D7am03KdcJPNIHh9SUQrcLh8qzYlwu82H366W0vZ5SJiBGR58501wbEe35xoM6vDdTn6u2BIteFc7xQuxbP5XVKF0q+lvoEHw4ONquGDWZix+8HXSy6IS5tcWakrLciK1891XcG6UekVl0AqvXqFDeSzv4E1iw/Rppt+jRqYNXevLhbtnqLJT0VRYeivtfrq3eh+NQPVYJ9mPUDPC+zYkHTC/pEnOQGel0PLLwEQZcSu9nLjd6yXRIYj65hzCNxUkoQuJ5Vx2tz4uRUGY5kM7x7gaxWpzCdtYvj+hyIDqxZB/hVzUvm6arhReRM3LyyZNLOkq0s2dYtc4yGhEC3K+kAcyTiNqtYBi0FdFRDnequf1kpYGtmqYXxwmD6bvm5Ld1SjCQkXjnklOwh+fb+HUxWIg0icTm/q51+qiusgWIl+CzliKu/aeyW7sTnI+IgctbpJ+viGP9ivJsgsVSlVKWT9FpJUUqWIlCkgNDIyXRPwb7JA1xL7rLipEz7JHuksqrFLigCKYkcSRyKWgq0bFuvKRX/Rpl4qjyZt4QXijEps5vcOk5B6lhAGKuX3gi+ON61TcoTjJNNbeXD2eXpWh6WE2k7z2KqDK+cc9PLenjrvTzvlswA6cfyK7toDPxbd+xeaDQEHbclc6h09ASxrLOCZ7nimE+FJmGKKmGKLmEVymQJ++gHtKgnNOqG4m1hUAf8phBbGF4rtTt0BzP06t2NpxinhzgksyZPL8qj+0MYVdlm1O4lhtcSITNLdnPx95HSiMRihddFZ1JXLLkdxX1Rtqi4XZ82+41qK50WIPCwHxSlZZhCRIViBQzfBTVAJupMBB1QLrkXS3u22PBFIhnQWJ0f8AAohRYKodGiO16KfIc0Sppna6KkFceYzmPVoDugAWW0GeoTDX2dD02zZCDkFpNtvpjcExttheBpq4Dfl82PzEJWQqpV7TRoR+1p2c0VVVmhQ6dESigrLgohJTPPfyuet8lHG5BrUBZ5r+li0g97Oiu07ltwdW895oPq8KL/iXGaTgP34SFOEcaNU14EW9uj8G/Q7jhMPHZ0fhizyTS+5bEB0zjfq7RQIfKGViNpfRPus6YUx7PywDoYlmLFIeAoFx8lna+JN//5w47rHAopsEOgWazEAw9JxfZgnihBBVU0/lCeg9ECfsbchofUijX4cl26unqzp9/Uzgw74mNg52yKIAMuHyUHZY1Efm9JlyqCOP6F+TVPYyXkTI0jPtMX/kHTkuF5L1B6UdHUqz5Q8bREt6Uo82O9+TXd9V2RSgO0Qro7ps40xRG0KCATn8jdlbZFnozPgiUWDdc3VVIrNseHPc5462qg/4giqVAHWt3BMdVyLLdxpYV7l0qcXry10FOJrzrY/jfavc863L58RnnGTLHaTHUtDUuutYi4QiUKjVipBtdUT3+/YnrSTND/rNKpVgjr64LKEyIi+K6IIU+hRhWChJMEDGw+dEoRnHDe2PKHlTDWVOpn0XKJN8tu0eOPvv7qJR/oxyqM1/DOzF7wXHh9mCTR6HqaQAnonXiFmSfOokcG2QkdPvnqRT+jJ/EFEbOYhPZU/GFPsPXup170AB49jz+LppwtysQglQuGv6wx85jwbmJmiJSoYDJQQuS18KDDvZM9sOMkgMi7iWt2s9nka9H89yP/uyPzlodYROaln9yiAJkYWFemF1mSjkHIhywyrkBYhLuV0XBlJFwZBdeqUo1EiAxu0rSzh6n6ECcgOKI1GlTxmzu4eYHRmJUfHfHGs32iOn6Q2khpojKnsKAuUhapihQlahIliYpIQaQeUg6olg+oOQn9kfsA7XHeesuAA7AEf9zoymeUCp4nf5RAMvwvQPGU12M2dNKzVIZ8rUpDbLweOhb10kQFg2V6U9f14liqGhBEIWnieeg0gun4DBgUd8WARVGrohdOoVvBYd7nKvbAgJLEvT14/hkZ2GLv3r1LBYqm1aMTPBMxxsaPTkw6LHF0ilmnlHUqso4x65iyjk0FtVb662sYbxNPV5xRoYszUfHiWJy5oNxjCe4bwfvGAX4TuT3M7FFer9RIVaROCvlJazAiyqelPzyCSELePo8bDAiJFNQ/4HyjtCgrnLINgJa9yLormNfsyTit0Bk6ww3YH2cV2mmF3jm+aot+b1AcEGge0vuUljU6ssZYYjgGBEXcZLo2kxbdlEVBAg66mxRT4uS8dv6rudFTOtPJyEdBg2k1CC2+l0KJI4iXkCupb2eV38n3/LXRA6TI2+0x+pXYQslCzE/JkSLwJ3UhC/75lvEoSTJY25ip+Bgy+PJOen1ZsVRhPqLyQPoDul+kW42TJNu003i9LLugLiBtCkgZAL5nC2SsiYi75UpbhUqGXQfersn93qV17WJdCsyKAqNDXA+jsOCQg8djxOoDX0rAueCXShN5xPCuAopsrbjmIypJJbLitgAArWdQFU4u3ImosVS80pOw+bqFqMi636TB8MbEVLmKFFNZYY1i2PKM1uos6iz6g4cgZwvJPMUiWqL2V6l8EXAqbUdnmr3hyn0of3wxVeb5JnqqjYXArTJiYGt9eAic8chVTS4EaEkkTGW+8s1nQT8Ng69hlDjXvnfELS2EfOj74dwbyBixT0X9GaN2k42nyoRHT4cWPpMJcxQGw9EN6AgfFyj/eMq/TTFgTiLC0PluQwRThvKD0bhtYMDls5/JpuaYQBEZDFqWQRPbVt4PBLFgOH4WkbbjsXjHjRaUFpqz6Qm7Q4lMaUomAcyaVw0Kfq1BXNALUPGCeBQK94LXmTm+/O08ig1NK8zV79F66S15/5jVh144gAF2y0nSKvZKNHtgbQ+m/ii4+Ur20tfIG3oRhstVgSjtVlegOQIFTlz4gTYrGjkwuHFmpriDEXQo0G+prJUaZm3cOyF3BuPUIOyPURRGaON28emL604nTuA+gFwckrNArkcMPsjJ1E9GkyhEQyyM0GchpgcGicfmnmRTr9tVDLb3s903Nv42WBGPTsrX3Ewbl0xIW/WuCr5VijOOiQ9aXI+6IFtM/hWX0Ne+a1YZ57vs2RQQUFFI4rGZc/4+w+ARntL5lQTCzikkAuxn0DSSSQxYxmXH9LsrHxcGoC3bqrgRJi6EzTGSVEWRYsj9RdWP6pXqU5xi+rWCaujpj4ocYERbvHJooZ8E1ig6G2iVVvpFSkQO8IjoPLuMVYEXU5Sw3PJnDUwsh8f8WvkC01L1lS1mMKcpTPXHKsvDWR5AvMj9Zoec8JKryVeFgEKZc/fvcDn/qqcZz0eJe4vSGdPOIIy0x5q78jce8870vm11LBv+b1pNCwwh+NtqNzvWjv2uZeG7HfhsQ6qJHGotLOwq9nBsnVrHFvQGOgJ96FmIJ6InsOII7fHW7arWm/9w6xTjxaPfHN3NbPt38lec/j/uDftt')
tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v10_m32x__q_build():
    name = 'm32x_qkv_' + hashlib.sha256(tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
    return load_inline(name, tt_v10_m32x__Q_CPP, cuda_sources=tt_v10_m32x__Q_CUDA, functions=['m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def tt_v10_m32x__q_make_plan(ext, weights, fmt):
    pack = tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = tt_v10_m32x_m32x_stream(pack, 1, *tt_v10_m32x__Q_CFG)
    if stream is None:
        return None
    plan = dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=2304, cfg=tt_v10_m32x__Q_CFG)
    x = weights[:32]
    out = torch.empty((32, 2304), dtype=torch.bfloat16, device=weights.device)
    capacity = tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG, 1)
    attr = tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG, 2)
    if capacity < 144 or attr < 0 or attr % 1000:
        return None
    return plan

def tt_v10_m32x__q_run(x, plan):
    out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
    rc = tt_v10_m32x__q_launch_raw(plan['ext'], plan['idx'], plan, x, out, 2304, tt_v10_m32x__Q_CFG)
    if rc:
        raise RuntimeError(f'm32x qkv launch failed: {rc}')
    return out

def tt_v10_m32x_prepare_m32x(weights, *, include_unpooler: bool=False) -> dict:
    state = {'plans': {}, 'device': None}
    configs = [('pooler', (4096, 8192), tt_v10_m32x__build, tt_v10_m32x__make_plan), ('qkv_gate', (2304, 8192), tt_v10_m32x__q_build, tt_v10_m32x__q_make_plan), ('unpooler', (8192, 4096), tt_v10_m32x__w_build, tt_v10_m32x__w_make_plan)]
    for layer, shape, build, make_plan in configs:
        if layer == 'unpooler' and (not include_unpooler):
            continue
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != shape or (not w.is_cuda):
            continue
        if torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        if state['device'] is not None and state['device'] != w.device:
            continue
        state['device'] = w.device
        ext = build()
        for fmt in (1, 0):
            plan = make_plan(ext, w, fmt)
            if plan is None:
                continue
            if layer == 'pooler':
                ok = tt_v10_m32x_m32x_verify_onehot(ext, plan['idx'], plan, w, tt_v10_m32x__CFG, 32) and tt_v10_m32x_m32x_verify_random(ext, plan['idx'], plan, w, tt_v10_m32x__CFG, [9, 16, 17, 24, 32])
            elif layer == 'qkv_gate':
                ok = tt_v10_m32x__q_verify_onehot(ext, plan['idx'], plan, w, tt_v10_m32x__Q_CFG, 32) and tt_v10_m32x__q_verify_random(ext, plan['idx'], plan, w, tt_v10_m32x__Q_CFG, [9, 16, 17, 24, 32])
            else:
                ok = tt_v10_m32x__w_verify_onehot(ext, plan['idx'], plan, w, 32) and tt_v10_m32x__w_verify_random(plan, w)
            if ok:
                state['plans'][layer] = plan
                break
    return state

def tt_v10_m32x_run_m32x(x, layer, n, st) -> torch.Tensor | None:
    expected = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'unpooler': (8192, 4096)}.get(layer)
    if expected is None or x.ndim != 2 or x.shape[0] != 32:
        return None
    if (n, x.shape[1]) != expected or x.dtype != torch.bfloat16 or (not x.is_cuda):
        return None
    if x.device != st.get('device') or not x.is_contiguous():
        return None
    plan = st.get('plans', {}).get(layer)
    if plan is None:
        return None
    if layer == 'pooler':
        return tt_v10_m32x__run(x, plan)
    if layer == 'qkv_gate':
        return tt_v10_m32x__q_run(x, plan)
    return tt_v10_m32x__w_run(x, plan)

V10_ROUTES = [{'m': 8, 'layer': 'pooler', 'name': 'smallm2', 'module': 'v10_sm2'}, {'m': 16, 'layer': 'qkv_gate', 'name': 'warps', 'module': 'v10_warps'}, {'m': 256, 'layer': 'o', 'name': 'm256tune', 'module': 'v10_tune', 'config': [256, 64, 5, 7, 1, 1, 1]}, {'m': 32, 'layer': 'qkv_gate', 'name': 'm32x', 'module': 'v10_m32x'}, {'m': 32, 'layer': 'pooler', 'name': 'm32x', 'module': 'v10_m32x'}]
V10_SELECTED_ROUTES = [r for r in V9_SELECTED_ROUTES if (r["m"], r["layer"]) not in {(v["m"], v["layer"]) for v in V10_ROUTES}] + V10_ROUTES



def prepare_v10_tune(weights: Weights) -> dict:
    weight = weights.get('o')
    if weight is None or not weight.is_cuda or weight.dtype != torch.bfloat16 or tuple(weight.shape) != (8192, 1024):
        return {}
    if torch.cuda.get_device_capability(weight.device) != (9, 0):
        return {}
    return {'ext': tt_v10_tune_build_rl([(256, 64, 5, 7, 1, 1, 1)]),
            'weight': tt_v10_tune_pack_weight(weight.contiguous(), 64)}


def run_v10_tune(x: torch.Tensor, layer: str, n: int, state: dict) -> torch.Tensor | None:
    if layer != 'o' or n != 8192 or not state or x.ndim != 2 or tuple(x.shape) != (256, 1024):
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.data_ptr() % 16 or x.device != state['weight'].device:
        return None
    output = torch.empty((256, 8192), device=x.device, dtype=torch.bfloat16)
    rc = state['ext'].rl_launch(0, x.data_ptr(), state['weight'].data_ptr(), output.data_ptr(),
                                256, torch.cuda.current_stream(x.device).cuda_stream, 0)
    if rc:
        raise RuntimeError(f'm256tune O launch failed: {rc}')
    return output


def prepare_v10(weights: Weights) -> dict:
    state = prepare_v9(weights)
    state['v10_sm2'] = tt_v10_sm2_prepare_smallm2(weights)
    state['v10_warps'] = tt_v10_warps_prepare_warps(weights)
    state['v10_tune'] = prepare_v10_tune(weights)
    state['v10_m32x'] = tt_v10_m32x_prepare_m32x(weights, include_unpooler=False)
    for route in V10_ROUTES:
        m, layer = route['m'], route['layer']
        weight = weights[layer]
        data = {'x': torch.zeros((m, weight.shape[1]), dtype=torch.bfloat16, device=weight.device),
                'config': {'m': m, 'n': weight.shape[0], 'k': weight.shape[1], 'layer': layer}}
        candidate_v10(data, state)
    torch.cuda.synchronize(weights['qkv_gate'].device)
    return state


def candidate_v10(data: Inputs, state: dict) -> torch.Tensor:
    x, layer, n = data['x'], data['config']['layer'], data['config']['n']
    m = x.shape[0]
    output = None
    if layer == 'pooler' and m == 8:
        output = tt_v10_sm2_run_smallm2(x, layer, n, state['v10_sm2'])
    elif layer == 'qkv_gate' and m == 16:
        output = tt_v10_warps_run_warps(x, layer, n, state['v10_warps'])
    elif layer == 'o' and m == 256:
        output = run_v10_tune(x, layer, n, state['v10_tune'])
    elif layer in ('qkv_gate', 'pooler') and m == 32:
        output = tt_v10_m32x_run_m32x(x, layer, n, state['v10_m32x'])
    return candidate_v9(data, state) if output is None else output


V11_ROUTES = [{'m': 32, 'layer': 'unpooler', 'name': 'm32x_unpooler', 'module': 'v11_m32x'}]
V11_SELECTED_ROUTES = [r for r in V10_SELECTED_ROUTES if (r['m'],r['layer']) != (32,'unpooler')] + V11_ROUTES


def prepare_v11(weights: Weights) -> dict:
    state = prepare_v10(weights)
    state['v11_m32x'] = tt_v10_m32x_prepare_m32x({'unpooler':weights['unpooler']}, include_unpooler=True)
    weight = weights['unpooler']
    data = {'x': torch.zeros((32,weight.shape[1]),dtype=torch.bfloat16,device=weight.device),
            'config': {'m': 32,'n': weight.shape[0],'k':weight.shape[1],'layer':'unpooler'}}
    candidate_v11(data,state)
    torch.cuda.synchronize(weight.device)
    return state


def candidate_v11(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'unpooler' and x.shape[0] == 32:
        output = tt_v10_m32x_run_m32x(x,layer,n,state['v11_m32x'])
        if output is not None:
            return output
    return candidate_v10(data,state)


"""Lossless BF16 small-token projection routes validated on H200."""
import torch
from torch.utils.cpp_extension import load_inline
tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE = _zsrc('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')
tt_v12_smtrace1_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v12_smtrace1_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
tt_v12_smtrace1_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v12_smtrace1_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
tt_v12_smtrace1_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v12_smtrace1_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v12_smtrace1_SMTRACE_ESC_CAP = 32

def tt_v12_smtrace1_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def tt_v12_smtrace1_smtrace_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return tt_v12_smtrace1_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v12_smtrace1_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, tt_v12_smtrace1_SMTRACE_CPP + extra_cpp, cuda_sources=tt_v12_smtrace1_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v12_smtrace1_smtrace_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v12_smtrace1__smtrace_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v12_smtrace1__smtrace_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v12_smtrace1_smtrace_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v12_smtrace1__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v12_smtrace1_smtrace_c12_bytes(v):
    return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def tt_v12_smtrace1_smtrace_c12_unpack(data,base,esc):
    return b2q_tt_v10_m32x_m32x_c12_unpack(data,base,esc)

def tt_v12_smtrace1_smtrace_table(base):
    return b2q_tt_v10_m32x_m32x_table(base)

def tt_v12_smtrace1_smtrace_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v12_smtrace1_smtrace_units(hi), tt_v12_smtrace1_smtrace_units(lo)]
    else:
        mats = [tt_v12_smtrace1_smtrace_units(w)]
    if fmt in (0, 2):
        data = torch.cat([tt_v12_smtrace1_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v12_smtrace1_smtrace_c12_bytes(v)
        if not torch.equal(tt_v12_smtrace1_smtrace_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(tt_v12_smtrace1_smtrace_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v12_smtrace1_smtrace_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v12_smtrace1_smtrace_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v12_smtrace1_smtrace_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v12_smtrace1_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v12_smtrace1_SMTRACE_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v12_smtrace1__smtrace_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
tt_v12_smtrace1_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
tt_v12_smtrace1_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
tt_v12_smtrace1_SMTRACE_MAX_M = 32
tt_v12_smtrace1_SMTRACE_TABLE_M.update({('router', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 1]})
tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
tt_v12_smtrace1_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
tt_v12_smtrace1_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
tt_v12_smtrace1_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def tt_v12_smtrace1_smtrace_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v12_smtrace1_smtrace_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v12_smtrace1_smtrace_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v12_smtrace1_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v12_smtrace1_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v12_smtrace1_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v12_smtrace1_smtrace_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = tt_v12_smtrace1_smtrace_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = tt_v12_smtrace1_smtrace_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = tt_v12_smtrace1_smtrace_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = tt_v12_smtrace1_smtrace_verify_onehot(ext, inst_id[inst], sp, tt_v12_smtrace1_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v12_smtrace1_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def tt_v12_smtrace1_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(tt_v12_smtrace1_SMTRACE_TABLE_M if table_m is None else table_m)
    table_mt = dict(tt_v12_smtrace1_SMTRACE_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = tt_v12_smtrace1_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, tt_v12_smtrace1_SMTRACE_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if tt_v12_smtrace1_smtrace_bf16_twin(inst):
            instances.append(tt_v12_smtrace1_smtrace_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = tt_v12_smtrace1_smtrace_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and tt_v12_smtrace1_smtrace_bf16_twin(inst):
            plan = tt_v12_smtrace1_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, tt_v12_smtrace1_smtrace_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def tt_v12_smtrace1_smtrace_run_core(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= tt_v12_smtrace1_SMTRACE_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', tt_v12_smtrace1_smtrace_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != tt_v12_smtrace1_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    if tt_v12_smtrace1_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
tt_v12_smtrace1_SMTRACE_ROUTES = {('unpooler', 1): ((1, 1, 1, 2, 2, 0, 256, 1, 1024, 4096, 1, 8, 2, 4), (1, 8, 2, 4)), ('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def tt_v12_smtrace1_prepare_smtrace(weights):
    usable = {}
    for (layer, m), (inst, cfg) in tt_v12_smtrace1_SMTRACE_ROUTES.items():
        w = weights.get(layer)
        if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
            usable[layer] = w
    routes = {key: value for key, value in tt_v12_smtrace1_SMTRACE_ROUTES.items() if key[0] in usable}
    return tt_v12_smtrace1_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_3c2c50fe46c8')

def tt_v12_smtrace1_run_smtrace(x, layer, n, st):
    if (layer, x.shape[0]) not in tt_v12_smtrace1_SMTRACE_ROUTES:
        return None
    return tt_v12_smtrace1_smtrace_run_core(x, layer, n, st)

V12_SM2_O1 = {('o', 1): ((1, 1, 1, 2, 0, 0, 1024, 0, 8192, 1024, 4, 8, 2, 1, 7), (4, 8, 2, 1))}

def prepare_v12_o1(weights) -> dict:
    st = {'plans': {}}
    entries = []
    for (layer, m), (inst, cfg) in V12_SM2_O1.items():
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or (not w.is_cuda):
            continue
        if tuple(w.shape) != (inst[8], inst[9]) or torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        entries.append(((layer, 'm', m), inst, cfg))
    if not entries:
        return st
    instances = list(dict.fromkeys((v for _, inst, _ in entries for v in (inst, tt_v10_sm2_smallm2_bf16_twin(inst)))))
    ext = tt_v10_sm2_smallm2_build(instances, 'smallm2_o1_v12')
    st['ext'] = ext
    ids = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, inst, cfg in entries:
        layer = key[0]
        w = weights[layer]
        plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, inst, cfg)
        if plan is None:
            plan = tt_v10_sm2_smallm2_plan(ext, ids, packs, streams, w, layer, key, tt_v10_sm2_smallm2_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][layer, key[2]] = plan
    return st

V12_ROUTES = [{'m': 1, 'layer': 'unpooler', 'name': 'smtrace1', 'module': 'v12_smtrace1'}, {'m': 1, 'layer': 'o', 'name': 'smallm2_o1', 'module': 'v12_o1'}]
V12_SELECTED_ROUTES = [r for r in V11_SELECTED_ROUTES if (r['m'],r['layer']) not in {(n['m'],n['layer']) for n in V12_ROUTES}] + V12_ROUTES


def prepare_v12(weights: Weights) -> dict:
    state = prepare_v11(weights)
    state['v12_smtrace1'] = tt_v12_smtrace1_prepare_smtrace(weights)
    state['v12_o1'] = prepare_v12_o1(weights)
    for route in V12_ROUTES:
        m,layer = route['m'],route['layer']
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': torch.zeros((m,weight.shape[1]),dtype=torch.bfloat16,device=weight.device),
                'config': {'m': m,'n': n,'k':weight.shape[1],'layer':layer}}
        candidate_v12(data,state)
    torch.cuda.synchronize()
    return state


def candidate_v12(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'unpooler' and x.shape[0] == 1:
        output = tt_v12_smtrace1_run_smtrace(x,layer,n,state['v12_smtrace1'])
        if output is not None:
            return output
    if layer == 'o' and x.shape[0] == 1:
        output = tt_v10_sm2_run_smallm2(x,layer,n,state['v12_o1'])
        if output is not None:
            return output
    return candidate_v11(data,state)


"""Lossless BF16 small-token projection routes validated on H200."""
import torch
from torch.utils.cpp_extension import load_inline
tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE = _zsrc('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')
tt_v13_smtrace8_SMTRACE_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint smtrace_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
tt_v13_smtrace8_SMTRACE_FIELDS = ('fmt', 'mt', 'rt', 'st', 'red', 'diag', 'ntmax')
tt_v13_smtrace8_SMTRACE_NL = {0: 4, 1: 3, 2: 8, 3: 6}
tt_v13_smtrace8_SMTRACE_FMT_NAME = {0: 'bf', 1: 'c12', 2: 'hl', 3: 'hlc'}
tt_v13_smtrace8_SMTRACE_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
tt_v13_smtrace8_SMTRACE_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
tt_v13_smtrace8_SMTRACE_ESC_CAP = 32

def tt_v13_smtrace8_smtrace_inst(fmt, rt, st, red, mt=1, diag=0, ntmax=512):
    return (fmt, mt, rt, st, red, diag, ntmax)

def tt_v13_smtrace8_smtrace_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return smtrace_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return tt_v13_smtrace8_SMTRACE_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def tt_v13_smtrace8_smtrace_build(instances, name='small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, tt_v13_smtrace8_SMTRACE_CPP + extra_cpp, cuda_sources=tt_v13_smtrace8_smtrace_cuda_source(instances) + extra_cuda, functions=['smtrace_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def tt_v13_smtrace8_smtrace_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def tt_v13_smtrace8__smtrace_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def tt_v13_smtrace8__smtrace_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def tt_v13_smtrace8_smtrace_bf16_bytes(v):
    r, c = v.shape[:2]
    words = tt_v13_smtrace8__smtrace_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def tt_v13_smtrace8_smtrace_c12_bytes(v):
    return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def tt_v13_smtrace8_smtrace_c12_unpack(data,base,esc):
    return b2q_tt_v10_m32x_m32x_c12_unpack(data,base,esc)

def tt_v13_smtrace8_smtrace_table(base):
    return b2q_tt_v10_m32x_m32x_table(base)

def tt_v13_smtrace8_smtrace_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [tt_v13_smtrace8_smtrace_units(hi), tt_v13_smtrace8_smtrace_units(lo)]
    else:
        mats = [tt_v13_smtrace8_smtrace_units(w)]
    if fmt in (0, 2):
        data = torch.cat([tt_v13_smtrace8_smtrace_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = tt_v13_smtrace8_smtrace_c12_bytes(v)
        if not torch.equal(tt_v13_smtrace8_smtrace_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(tt_v13_smtrace8_smtrace_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def tt_v13_smtrace8_smtrace_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = tt_v13_smtrace8_smtrace_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, tt_v13_smtrace8_SMTRACE_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > tt_v13_smtrace8_SMTRACE_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = tt_v13_smtrace8__smtrace_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.smtrace_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
tt_v13_smtrace8_SMTRACE_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
tt_v13_smtrace8_SMTRACE_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
tt_v13_smtrace8_SMTRACE_MAX_M = 32
tt_v13_smtrace8_SMTRACE_TABLE_M.update({('router', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['router', 1], ('qkv_gate', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 1], ('pooler', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 1], ('o', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['o', 1], ('unpooler', 1): tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1], ('unpooler', 8): tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 1]})
tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
tt_v13_smtrace8_SMTRACE_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
tt_v13_smtrace8_SMTRACE_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
tt_v13_smtrace8_SMTRACE_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def tt_v13_smtrace8_smtrace_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def tt_v13_smtrace8_smtrace_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def tt_v13_smtrace8_smtrace_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def tt_v13_smtrace8_smtrace_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if tt_v13_smtrace8_smtrace_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def tt_v13_smtrace8_smtrace_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = tt_v13_smtrace8_smtrace_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = tt_v13_smtrace8_smtrace_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = tt_v13_smtrace8_smtrace_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = tt_v13_smtrace8_smtrace_verify_onehot(ext, inst_id[inst], sp, tt_v13_smtrace8_smtrace_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and tt_v13_smtrace8_smtrace_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def tt_v13_smtrace8_smtrace_prepare_core(weights, table_m=None, table_mt=None, name='small_m_gemv'):
    table_m = dict(tt_v13_smtrace8_SMTRACE_TABLE_M if table_m is None else table_m)
    table_mt = dict(tt_v13_smtrace8_SMTRACE_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = tt_v13_smtrace8_smtrace_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, tt_v13_smtrace8_SMTRACE_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if tt_v13_smtrace8_smtrace_bf16_twin(inst):
            instances.append(tt_v13_smtrace8_smtrace_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = tt_v13_smtrace8_smtrace_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and tt_v13_smtrace8_smtrace_bf16_twin(inst):
            plan = tt_v13_smtrace8_smtrace_plan(ext, inst_id, packs, streams, weights[layer], layer, key, tt_v13_smtrace8_smtrace_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def tt_v13_smtrace8_smtrace_run_core(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= tt_v13_smtrace8_SMTRACE_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', tt_v13_smtrace8_smtrace_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != tt_v13_smtrace8_SMTRACE_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    if tt_v13_smtrace8_smtrace_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out
tt_v13_smtrace8_SMTRACE_ROUTES = {('unpooler', 8): ((1, 1, 1, 2, 2, 0, 512, 8, 1024, 4096, 1, 16, 2, 2), (1, 16, 2, 2))}

def tt_v13_smtrace8_prepare_smtrace(weights):
    usable = {}
    for (layer, m), (inst, cfg) in tt_v13_smtrace8_SMTRACE_ROUTES.items():
        w = weights.get(layer)
        if w is not None and w.is_cuda and (w.dtype == torch.bfloat16) and (w.ndim == 2) and (w.shape[1] == inst[9]) and (w.shape[0] >= inst[8]) and (torch.cuda.get_device_capability(w.device) == (9, 0)):
            usable[layer] = w
    routes = {key: value for key, value in tt_v13_smtrace8_SMTRACE_ROUTES.items() if key[0] in usable}
    return tt_v13_smtrace8_smtrace_prepare_core(usable, table_m=routes, table_mt={}, name='smtrace_final_d46e5decdceb')

def tt_v13_smtrace8_run_smtrace(x, layer, n, st):
    if (layer, x.shape[0]) not in tt_v13_smtrace8_SMTRACE_ROUTES:
        return None
    return tt_v13_smtrace8_smtrace_run_core(x, layer, n, st)

V13_ROUTES = [{'m': 8, 'layer': 'unpooler', 'name': 'smtrace8', 'module': 'v13_smtrace8'}]
V13_SELECTED_ROUTES = [r for r in V12_SELECTED_ROUTES if (r['m'],r['layer']) not in {(n['m'],n['layer']) for n in V13_ROUTES}] + V13_ROUTES


def prepare_v13(weights: Weights) -> dict:
    state = prepare_v12(weights)
    state['v13_smtrace8'] = tt_v13_smtrace8_prepare_smtrace(weights)
    for route in V13_ROUTES:
        m,layer = route['m'],route['layer']
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': torch.zeros((m,weight.shape[1]),dtype=torch.bfloat16,device=weight.device),
                'config': {'m': m,'n': n,'k':weight.shape[1],'layer':layer}}
        candidate_v13(data,state)
    torch.cuda.synchronize()
    return state


def candidate_v13(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'unpooler' and x.shape[0] == 8:
        output = tt_v13_smtrace8_run_smtrace(x,layer,n,state['v13_smtrace8'])
        if output is not None:
            return output
    return candidate_v12(data,state)


"""QKV candidates. Production routes are enabled only after measured validation."""
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def tt_v14_qkv_qbig_mma(n):
    regs = ','.join((f'%{i}' for i in range(n // 2)))
    outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
    return f'__device__ __forceinline__ void q_mma_{n}(float* a,uint64_t x,uint64_t w) {{\n      asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(x),"l"(w));\n    }}'
tt_v14_qkv_QKVBIG_CUDA = _zsrc("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")

def tt_v14_qkv_build_qkv(configs, profile=False):
    ns = sorted({c[0] for c in configs} | {c[2] for c in configs if c[5] & 8})
    source = tt_v14_qkv_QKVBIG_CUDA.replace('@@MMA@@', '\n'.join((tt_v14_qkv_qbig_mma(n) for n in ns)))
    source = source.replace('@@CALL@@', ' '.join((f'if constexpr(NR=={n})q_mma_{n}(a+h*(NR/2),dx,dw);' for n in ns)))
    source = source.replace('@@SWAP@@', ' '.join((f'if constexpr(TM=={n})q_mma_{n}(a,dw,dx);' for n in ns)))
    source = source.replace('@@CASES@@', '\n'.join((f"case {i}: return qb_launch_t<{','.join(map(str, c))},{str(profile).lower()}>(w,x,out,m,stream,query,stats);" for i, c in enumerate(configs))))
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    namespace = 'qb_' + digest
    begin = source.index('#define QD')
    args = 'int id,int64_t w,int64_t x,int64_t out,int m,int64_t stream,int query,int64_t stats'
    source = source[:begin] + f'namespace {namespace} {{\n' + source[begin:] + f'\n}}\nint qb_launch({args}) {{return {namespace}::qb_launch(id,w,x,out,m,stream,query,stats);}}'
    cpp = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint qb_launch(int,int64_t,int64_t,int64_t,int,int64_t,int,int64_t);'
    return load_inline('qbig_' + hashlib.sha256(source.encode()).hexdigest()[:16], cpp, cuda_sources=source, functions=['qb_launch'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False, extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'])

def tt_v14_qkv_qbig_pack(w, groups, kt=1):
    rows = (2304 + groups - 1) // groups
    if kt == 2:
        return w.view(groups, rows, 128, 64).permute(0, 2, 1, 3).contiguous().reshape(-1, 64)
    if 2304 % groups == 0:
        return w.contiguous()
    out = torch.zeros((groups, rows, 8192), dtype=torch.bfloat16, device=w.device)
    for g in range(groups):
        lo, hi = (g * 2304 // groups, (g + 1) * 2304 // groups)
        out[g, :hi - lo].copy_(w[lo:hi])
    return out.reshape(-1, 8192)

def tt_v14_qkv_prepare_qbig_candidates(weights, configs, profile=False):
    w = weights['qkv_gate']
    return {'ext': tt_v14_qkv_build_qkv(configs, profile), 'configs': list(configs), 'packed': {(g, kt): tt_v14_qkv_qbig_pack(w, g, kt) for g, kt in {(c[1], 2 if c[5] & 32 else 1) for c in configs}}, 'device': w.device}

def tt_v14_qkv_run_qbig_candidate(x, st, index, stats=None, query=0):
    out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
    cfg = st['configs'][index]
    rc = st['ext'].qb_launch(index, st['packed'][cfg[1], 2 if cfg[5] & 32 else 1].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, query, stats.data_ptr() if stats is not None else 0)
    if query:
        return rc
    if rc:
        raise RuntimeError(f'qkv launch {cfg} failed: {rc}')
    return out
tt_v14_qkv_EXPERIMENT_CONFIGS = [(64, 36, 128, 3, 8, 539, 5), (64, 36, 128, 3, 8, 6683, 5), (64, 36, 256, 3, 4, 219, 5)]
tt_v14_qkv_ENABLED_ROUTES = {128: 0}

def tt_v14_qkv_prepare_qkvbig(weights: dict) -> dict:
    w = weights.get('qkv_gate')
    if w is None or w.shape != (2304, 8192) or w.dtype != torch.bfloat16 or (not w.is_cuda):
        return {'routes': {}}
    if torch.cuda.get_device_capability(w.device) != (9, 0) or 'H200' not in torch.cuda.get_device_name(w.device):
        return {'routes': {}}
    if not tt_v14_qkv_ENABLED_ROUTES:
        return {'routes': {}}
    state = tt_v14_qkv_prepare_qbig_candidates(weights, tt_v14_qkv_EXPERIMENT_CONFIGS)
    state['routes'] = dict(tt_v14_qkv_ENABLED_ROUTES)
    for m, index in state['routes'].items():
        x = torch.zeros((m, 8192), device=w.device, dtype=torch.bfloat16)
        tt_v14_qkv_run_qbig_candidate(x, state, index)
    torch.cuda.synchronize(w.device)
    return state

def tt_v14_qkv_run_qkvbig(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    if layer != 'qkv_gate' or n != 2304 or x.ndim != 2 or (x.shape[1] != 8192):
        return None
    index = st.get('routes', {}).get(x.shape[0])
    if index is None or not x.is_cuda or x.dtype != torch.bfloat16 or (not x.is_contiguous()):
        return None
    if x.device != st['device'] or x.data_ptr() % 16:
        return None
    return tt_v14_qkv_run_qbig_candidate(x, st, index)

V14_ROUTES = [{'m': 128, 'layer': 'qkv_gate', 'name': 'qkvbig', 'module': 'v14_qkv'}]
V14_SELECTED_ROUTES = [r for r in V13_SELECTED_ROUTES if (r['m'],r['layer']) not in {(n['m'],n['layer']) for n in V14_ROUTES}] + V14_ROUTES


def prepare_v14(weights: Weights) -> dict:
    state = prepare_v13(weights)
    state['v14_qkv'] = tt_v14_qkv_prepare_qkvbig(weights)
    for route in V14_ROUTES:
        m,layer = route['m'],route['layer']
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': torch.zeros((m,weight.shape[1]),dtype=torch.bfloat16,device=weight.device),
                'config': {'m': m,'n': n,'k':weight.shape[1],'layer':layer}}
        candidate_v14(data,state)
    torch.cuda.synchronize()
    return state


def candidate_v14(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'qkv_gate' and x.shape[0] == 128:
        output = tt_v14_qkv_run_qkvbig(x,layer,n,state['v14_qkv'])
        if output is not None:
            return output
    return candidate_v13(data,state)

V15_ROUTES = [{'m': 256, 'layer': 'unpooler', 'name': 'ltsearch_transposed', 'module': 'lt', 'descriptor': [66, 20, 1, 0, 0, 4, 35, 0, 2], 'layout': 3, 'epilogue': 0, 'algorithm_workspace_bytes': 0}]
V15_SELECTED_ROUTES = [r for r in V14_SELECTED_ROUTES if (r['m'],r['layer']) not in {(n['m'],n['layer']) for n in V15_ROUTES}] + V15_ROUTES


def prepare_v15(weights: Weights) -> dict:
    state = prepare_v14(weights)
    
    for route in V15_ROUTES:
        m,layer = route['m'],route['layer']
        weight = weights[layer]
        n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
        data = {'x': torch.zeros((m,weight.shape[1]),dtype=torch.bfloat16,device=weight.device),
                'config': {'m': m,'n': n,'k':weight.shape[1],'layer':layer}}
        candidate_v15(data,state)
    torch.cuda.synchronize()
    return state


def candidate_v15(data: Inputs, state: dict) -> torch.Tensor:
    x,layer,n = data['x'],data['config']['layer'],data['config']['n']
    if layer == 'unpooler' and x.shape[0] == 256:
        output = tt_lt_run_cublaslt(x,layer,n,state['new_lt'])
        if output is not None:
            return output
    return candidate_v14(data,state)


_bundle_base_prepare = prepare_v15
_bundle_base_kernel = candidate_v15
'Experimental lossless nine-instruction BF16 decode. No route is promoted. See FINDINGS.txt.'
import hashlib
import torch
from torch.utils.cpp_extension import load_inline












































dd_decodediet_CPS32_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}











dd_decodediet_CPS32_TABLE_MT = {('qkv_gate', 4): ((1, 4, 1, 1, 4, 0, 256, 0, 2304, 8192, 1, 4, 8, 4, 7, 1), (1, 4, 8, 4)), ('o', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 1024, 1, 4, 4, 1, 8, 1), (1, 4, 4, 1)), ('pooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 4096, 8192, 1, 8, 16, 1, 7, 1), (1, 8, 16, 1)), ('unpooler', 4): ((1, 4, 1, 1, 0, 0, 256, 0, 8192, 4096, 1, 4, 16, 1, 8, 1), (1, 4, 16, 1)), ('o', 2): ((1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), (2, 4, 4, 1))}

def dd_decodediet_dd_cap2_b_cps32_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0






dd_decodediet_CPSSM_CUDA_TEMPLATE = _zsrc('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')
dd_decodediet_CPSSM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint dd_cap2_b_cpssm_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'
dd_decodediet_CPSSM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}
dd_decodediet_CPSSM_ESC_CAP = 32

def dd_decodediet_dd_cap2_b_cpssm_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = dd_decodediet_cpstage___to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = dd_decodediet_cpstage___to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_cpssm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def dd_decodediet_dd_cap2_b_cpssm_table(base):
    return base * 16843009

def dd_decodediet_dd_cap2_b_cpssm_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [dd_decodediet_dd_cap2_b_cpssm_units(hi), dd_decodediet_dd_cap2_b_cpssm_units(lo)]
    else:
        mats = [dd_decodediet_dd_cap2_b_cpssm_units(w)]
    if fmt in (0, 2):
        data = torch.cat([dd_decodediet_dd_cap2_b_cpssm_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = dd_decodediet_dd_cap2_b_cpssm_c12_bytes(v)
        if not torch.equal(dd_decodediet_dd_cap2_b_cpssm_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(dd_decodediet_dd_cap2_b_cpssm_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def dd_decodediet_dd_cap2_b_cpssm_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = dd_decodediet_dd_cap2_b_cpssm_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, dd_decodediet_CPSSM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > dd_decodediet_CPSSM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = dd_decodediet_cpstage___to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.dd_cap2_b_cpssm_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)
dd_decodediet_CPSSM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}
dd_decodediet_CPSSM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}
dd_decodediet_CPSSM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))
dd_decodediet_CPSSM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))
dd_decodediet_CPSSM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))
dd_decodediet_CPSSM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def dd_decodediet_dd_cap2_b_cpssm_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def dd_decodediet_dd_cap2_b_cpssm_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if dd_decodediet_dd_cap2_b_cpssm_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def dd_decodediet_dd_cap2_b_cpssm_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = dd_decodediet_dd_cap2_b_cpssm_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = dd_decodediet_dd_cap2_b_cpssm_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = dd_decodediet_dd_cap2_b_cpssm_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = dd_decodediet_dd_cap2_b_cpssm_verify_onehot(ext, inst_id[inst], sp, dd_decodediet_dd_cap2_b_cpssm_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and dd_decodediet_dd_cap2_b_cpssm_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def dd_decodediet_cpstage___to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def dd_decodediet_cpstage___to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)
dd_decodediet_CPS_LAYERS = ('qkv_gate', 'pooler', 'unpooler')

def dd_decodediet_cps_build(family, instances, depth):
    is32 = False
    prefix = 'dd_cap2_b_cpssm_'
    template = dd_decodediet_CPSSM_CUDA_TEMPLATE
    cpp = dd_decodediet_CPSSM_CPP
    cases = []
    for i, inst in enumerate(instances):
        full = inst if len(inst) == 14 else inst + (0,) * 7
        args = ', '.join(map(str, (*full, depth)))
        cases.append(f'case {i}: return {prefix}launch_t<{args}>(wp,x,out,esc,m,N,K,WR,WK,KU,S,t01,u01,stream,query);')
    source = template.replace('@@CASES@@', '\n'.join(cases))
    name = 'cps_' + family + '_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name, cpp, cuda_sources=source, functions=[prefix + 'launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-Xptxas=-v', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def dd_decodediet_prepare_cps(weights, depth=None, layers=dd_decodediet_CPS_LAYERS, buckets=(2, 4), include_policy=False):
    if depth is None:
        st = {'plans': {}, 'depth': 'selected', 'policy': 0}
        selected_layers = tuple((layer for layer in layers if layer in ('pooler', 'unpooler')))
        for bucket, selected_depth in ((2, 2), (4, 1)):
            if bucket in buckets and selected_layers:
                part = dd_decodediet_prepare_cps(weights, selected_depth, selected_layers, (bucket,))
                st['plans'].update(part['plans'])
        return st
    if depth not in (0, 1, 2, 3):
        raise ValueError('ring depth must be 0, 1, 2 or 3')
    st = {'plans': {}, 'depth': depth, 'policy': 0}
    for family in ('sm',):
        table = dd_decodediet_CPSSM_TABLE_MT
        entries = []
        for (layer, bucket), (inst, cfg) in table.items():
            is_policy = include_policy and family == 'sm' and (layer in dd_decodediet_CPS_LAYERS) and (bucket == 1) and (inst[4] in (2, 3))
            if layer not in weights or (not is_policy and (layer not in layers or bucket not in buckets)):
                continue
            if not is_policy and family == 'sm' and ((layer, bucket) in dd_decodediet_CPS32_TABLE_MT):
                continue
            if len(inst) == 7:
                rows = 1024 if layer == 'unpooler' and bucket == 1 else weights[layer].shape[0]
                inst = inst + (0, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
            entries.append(((layer, 'mt', bucket), (inst, cfg)))
        if include_policy and family == 'sm':
            for (layer, m), (inst, cfg) in dd_decodediet_CPSSM_TABLE_M.items():
                if layer not in dd_decodediet_CPS_LAYERS or layer not in weights or inst[4] not in (2, 3):
                    continue
                rows = 1024 if layer == 'unpooler' and m <= 8 else weights[layer].shape[0]
                if len(inst) == 7:
                    inst = inst + (m, rows, dd_decodediet_CPSSM_LAYER_K[layer], *cfg)
                entries.append(((layer, 'm', m), (inst, cfg)))
        if not entries:
            continue
        twin = dd_decodediet_dd_cap2_b_cpssm_bf16_twin
        instances = list(dict.fromkeys((inst for _, (inst, _) in entries for inst in (inst, twin(inst)) if inst is not None)))
        ext = dd_decodediet_cps_build(family, instances, depth)
        ids = {inst: i for i, inst in enumerate(instances)}
        plan_fn = dd_decodediet_dd_cap2_b_cpssm_plan
        packs, streams = ({}, {})
        for key, (inst, cfg) in entries:
            sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, inst, cfg)
            chosen = inst
            if sp is None and twin(inst) is not None:
                chosen = twin(inst)
                sp = plan_fn(ext, ids, packs, streams, weights[key[0]], key[0], key, chosen, cfg)
            if sp is not None:
                st['plans'][key] = dict(sp, ext=ext, family=family, inst=chosen, depth=depth)
    return st

def dd_decodediet_cps_launch(sp, x, out, n, policy=0, query=0):
    fn = dd_decodediet_dd_cap2_b_cpssm_launch_raw
    return fn(sp['ext'], sp['inst_id'], sp, x, out, n, sp['cfg'], query + policy * 100)

def dd_decodediet_run_cps(x, layer, n, st):
    if x.ndim != 2 or not 1 <= x.shape[0] <= 32:
        return None
    m = x.shape[0]
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', dd_decodediet_dd_cap2_b_cps32_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != dd_decodediet_CPS32_LAYER_K[layer]) or (not x.is_cuda):
        return None
    if not x.is_contiguous() or x.data_ptr() % 16 or x.device != sp['wp'].device:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    rc = dd_decodediet_cps_launch(sp, x, out, n, st.get('policy', 0))
    if rc:
        raise RuntimeError(f'cpstage launch failed: {rc}')
    return out
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline

def dd_decodediet_tt_mw_mma(n):
    nr = n // 2
    regs = ','.join((f'%{i}' for i in range(nr)))
    outs = ','.join((f'"+f"(d[{i}])' for i in range(nr)))
    inputs = ','.join([f'"r"(a[{i}])' for i in range(16)] + [f'"l"(b+{2 * i})' for i in range(4)])
    ops = ['wgmma.fence.sync.aligned;']
    for kk in range(4):
        ar = ','.join((f'%{nr + 4 * kk + i}' for i in range(4)))
        ops.append(f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{{ar}}}, %{nr + 16 + kk}, 1, 1, 1, 0;')
    ops.append('wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;')
    return f'D void mma{n}(float* d,uint32_t* a,uint64_t b) {{asm volatile("' + ' '.join(ops) + f'" : {outs} : {inputs} : "memory");}}'
dd_decodediet_tt_mw_CUDA_PREFIX = '// dd_cap2_b_\n' + '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.cg.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {\n  uint32_t rotated;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));\n  const uint32_t low_high = (C & 0x87878787u) + T0;\n  const uint32_t high_high = (rotated & 0x87878787u) + T0;\n  a[0] = prmt(L0, low_high, 0x5140u);\n  a[1] = prmt(L0, low_high, 0x7362u);\n  a[2] = prmt(L1, high_high, 0x5140u);\n  a[3] = prmt(L1, high_high, 0x7362u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
dd_decodediet_tt_mw_CUDA_CORE = _zsrc("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")
dd_decodediet_tt_mw_CUDA_MAP = '\nusing EncodeFn=decltype(&cuTensorMapEncodeTiled);\nint map(CUtensorMap* tm,const void* ptr,int rows,int k,int box) {\n  static EncodeFn encode=nullptr;\n  if(!encode) {cudaDriverEntryPointQueryResult q;cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled",(void**)&encode,12000,cudaEnableDefault,&q);}\n  uint64_t dims[]={uint64_t(k),uint64_t(rows)},stride[]={uint64_t(k)*2};uint32_t b[]={64,uint32_t(box)},es[]={1,1};\n  return int(encode(tm,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)ptr,dims,stride,b,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_L2_256B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));\n}\n\n'
dd_decodediet_tt_mw_CUDA_HOST = '\ntemplate<int BN,int R,int WK,int PF,int Q,bool C12,int CAP>\nint launch_t(const void* x,const void* w,const void* e,void* out,int m,int n,int k,int split,uint64_t tab,cudaStream_t st) {\n  CUtensorMap tx;int rc=map(&tx,x,m,k,BN);if(rc)return rc;\n  constexpr int a=WK*((Q*(BN*128+8)+1023)/1024*1024),b=WK*BN*(R+4)*4,smem=a>b?a:b;\n  auto kern=m32w_gemm<BN,R,WK,PF,Q,C12,CAP>;\n  static bool ready=false;\n  if(!ready) {\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));if(rc)return rc;\n    rc=int(cudaFuncSetAttribute(kern,cudaFuncAttributeNonPortableClusterSizeAllowed,1));if(rc)return rc;\n    ready=true;\n  }\n  cudaLaunchConfig_t cfg={};cfg.gridDim=dim3((n/R)*split,(m+BN-1)/BN);cfg.blockDim=dim3(128*WK);cfg.dynamicSmemBytes=smem;cfg.stream=st;\n  cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeClusterDimension;attr.val.clusterDim={unsigned(split),1,1};\n  if(split>1){cfg.attrs=&attr;cfg.numAttrs=1;}\n  return int(cudaLaunchKernelEx(&cfg,kern,tx,(const uint4*)w,(const uint32_t*)e,(__nv_bfloat16*)out,m,n,k,split,uint32_t(tab),uint32_t(tab>>32)));\n}\nint launch(int id,uint64_t x,uint64_t w,uint64_t e,uint64_t out,int m,int n,int k,int split,uint64_t tab,uint64_t st) {\n  switch(id) {DISPATCH}\n  return -1;\n}\nint info(int id){cudaFuncAttributes fa={};switch(id){INFO}return fa.numRegs*10000+fa.localSizeBytes;}\n'
dd_decodediet_tt_mw_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint info(int);\n'

def dd_decodediet_tt_mw_cuda_source(instances):
    cases = []
    attrs = []
    for i, inst in enumerate(instances):
        bn, r, wk, pf, q, c, cap = inst
        params = f'{bn},{r},{wk},{pf},{q},{str(c).lower()},{cap}'
        cases.append(f'case {i}: return launch_t<{params}>((void*)x,(void*)w,(void*)e,(void*)out,m,n,k,split,tab,(cudaStream_t)st);')
        attrs.append(f'case {i}:cudaFuncGetAttributes(&fa,m32w_gemm<{params}>);break;')
    return dd_decodediet_tt_mw_CUDA_PREFIX + '\n'.join((dd_decodediet_tt_mw_mma(n) for n in (16, 32, 64))) + dd_decodediet_tt_mw_CUDA_CORE + dd_decodediet_tt_mw_CUDA_MAP + dd_decodediet_tt_mw_CUDA_HOST.replace('DISPATCH', '\n'.join(cases)).replace('INFO', '\n'.join(attrs))

def dd_decodediet_tt_mw_build(instances):
    src = dd_decodediet_tt_mw_cuda_source(instances)
    name = 'dd_cap2_b_m32w_' + hashlib.sha256(src.encode()).hexdigest()[:12]
    return load_inline(name, dd_decodediet_tt_mw_CPP, cuda_sources=src, functions=['launch', 'info'], no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'])

def dd_decodediet_tt_mw__to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def dd_decodediet_tt_mw__to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def dd_decodediet_tt_mw_sm_bf16_bytes(v):
    r, c = v.shape[:2]
    words = dd_decodediet_tt_mw__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def dd_decodediet_tt_mw_sm_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = dd_decodediet_tt_mw__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def dd_decodediet_tt_mw_sm_c12_unpack(data, base, esc):
    """Inverse of dd_cap2_b_sm_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def dd_decodediet_tt_mw_sm_table(base):
    return base * 16843009

def dd_decodediet_tt_mw_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = dd_decodediet_tt_mw_sm_c12_bytes(v)
        if not torch.equal(dd_decodediet_tt_mw_sm_c12_unpack(data, b, entries), v):
            return None
        table = dd_decodediet_tt_mw_sm_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = dd_decodediet_tt_mw__to_i32(entry)
    else:
        data = dd_decodediet_tt_mw_sm_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def dd_decodediet_tt_mw_launch_raw(ext, idx, pk, x, out):
    return ext.launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if dd_decodediet_tt_mw_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))
dd_decodediet_tt_mw_CONFIGS = {(16, 'pooler'): ((16, 64, 1, 2, 2, True, 1), 8), (16, 'unpooler'): ((16, 64, 4, 2, 2, True, 1), 1), (16, 'qkv_gate'): ((16, 64, 1, 2, 2, True, 1), 8), (32, 'pooler'): ((32, 64, 1, 2, 2, True, 1), 8), (32, 'unpooler'): ((32, 64, 4, 2, 2, True, 1), 1), (32, 'qkv_gate'): ((32, 64, 1, 2, 2, True, 1), 8)}

def dd_decodediet_tt_mw_prepare_m32w(weights, configs=None) -> dict:
    configs = dd_decodediet_tt_mw_CONFIGS if configs is None else configs
    device = next(iter(weights.values())).device
    state = {'plans': {}, 'device': device}
    if device.type != 'cuda' or torch.cuda.get_device_capability(device) != (9, 0):
        return state
    instances = []
    for inst, split in configs.values():
        for c in (inst[5], False):
            twin = inst[:5] + (c,) + inst[6:]
            if twin not in instances:
                instances.append(twin)
    ext = dd_decodediet_tt_mw_build(instances)
    state['ext'] = ext
    packs = {}
    for (m, layer), (inst, split) in configs.items():
        if layer not in weights:
            continue
        w = weights[layer]
        for compressed in (inst[5], False):
            actual = inst[:5] + (compressed,) + inst[6:]
            idx = instances.index(actual)
            if ext.info(idx) % 10000:
                continue
            key = (layer, inst[1], inst[2], split, compressed)
            if key not in packs:
                packs[key] = dd_decodediet_tt_mw_pack_weight(w, inst[1], inst[2], split, compressed)
            pk = packs[key]
            if pk is None or not dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, m):
                continue
            if m == 32 and (not dd_decodediet_tt_mw_verify_onehot(ext, idx, pk, w, 33)):
                continue
            state['plans'][m, layer] = (idx, pk)
            break
    return state

def dd_decodediet_tt_mw_run_m32w(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (x.device != st['device']):
        return None
    m = x.shape[0]
    if not 16 <= m <= 33 or not x.is_contiguous() or x.data_ptr() % 16:
        return None
    key = (16 if m <= 16 else 32, layer)
    plan = st['plans'].get(key)
    if plan is None:
        return None
    idx, pk = plan
    if n != pk['n'] or x.shape[1] != pk['k']:
        return None
    out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
    rc = dd_decodediet_tt_mw_launch_raw(st['ext'], idx, pk, x, out)
    if rc:
        raise RuntimeError(f'm32w launch failed: {rc}')
    return out
import torch
from torch.utils.cpp_extension import load_inline





















import hashlib
import torch
from torch.utils.cpp_extension import load_inline
import torch
from torch.utils.cpp_extension import load_inline














import hashlib
import torch
from torch.utils.cpp_extension import load_inline


































dd_DECODEDIET_ROUTES = (('pooler',16),('unpooler',16))

def dd_prepare_decodediet(weights) -> dict:
    routes = dd_DECODEDIET_ROUTES
    state = {'routes': frozenset(routes)}
    if not routes:
        return state
    if any((m == 16 and layer in ('pooler', 'unpooler') for layer, m in routes)):
        layers = tuple((layer for layer, m in routes if m == 16 and layer in ('pooler', 'unpooler')))
        state['cps'] = dd_decodediet_prepare_cps(weights, layers=layers, buckets=(2,))
    return state

def dd_run_decodediet(x, layer, n, state) -> torch.Tensor | None:
    m = x.shape[0] if x.ndim == 2 else 0
    if (layer, m) not in state.get('routes', ()):
        return None
    return dd_decodediet_run_cps(x, layer, n, state['cps'])
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
rw_SELECTED = ()
rw_SOURCES = {128: _zsrc("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"), 256: _zsrc("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")}

def rw__prepare(weights: dict, arms: tuple) -> dict:
    w = weights.get('router')
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
        return {'enabled': False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'enabled': False}
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    packed = torch.stack([p.view(4, 64, 128, 64).permute(0, 2, 1, 3) for p in (hi, lo)], dim=2).contiguous().view(-1, 64)
    state = {'enabled': True, 'weight': w, 'packed': packed, 'arms': {}}
    for arm in arms:
        label = 'rwdl_' + str(arm)
        source = rw_SOURCES[arm]
        cpp = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint ' + label + '_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);'
        ext = load_inline('routerwide_' + hashlib.sha256(source.encode()).hexdigest()[:14], cpp, cuda_sources=source, functions=[label + '_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        state['arms'][arm] = (ext, getattr(ext, label + '_launch'))
        x = torch.zeros((arm, 8192), device=w.device, dtype=torch.bfloat16)
        rw_run_routerwide_candidate(x, state, arm)
    torch.cuda.synchronize(w.device)
    return state

def rw_prepare_routerwide(weights: dict) -> dict:
    return rw__prepare(weights, rw_SELECTED) if rw_SELECTED else {'enabled': False}

def rw_prepare_routerwide_candidates(weights: dict) -> dict:
    return rw__prepare(weights, (128, 256))

def rw_run_routerwide_candidate(x: torch.Tensor, state: dict, arm: int) -> torch.Tensor:
    out = torch.empty((x.shape[0], 256), device=x.device, dtype=torch.float32)
    rc = state['arms'][arm][1](0, state['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream(x.device).cuda_stream, 0)
    if rc:
        raise RuntimeError(f'routerwide launch failed: {rc}')
    return out

def rw_run_routerwide(x: torch.Tensor, layer: str, n: int, st: dict) -> torch.Tensor | None:
    if not st.get('enabled') or layer != 'router' or n != 256 or (x.ndim != 2):
        return None
    if x.shape[1] != 8192 or x.shape[0] not in rw_SELECTED or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
        return None
    if x.device != st['weight'].device or x.data_ptr() % 16:
        return None
    return rw_run_routerwide_candidate(x, st, x.shape[0])
'Router completion experiment. No route passed H200 confirmation.'
import hashlib
rt_SELECTED = ()
rt__CUDA = _zsrc('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')
rt__CPP = '#include <stdint.h>\n#include <pybind11/pybind11.h>\nint routertail_small(int64_t,int64_t,int64_t,int,int,int,int64_t,int);\n'

def rt__pack(w, s=16, wk=8):
    import torch
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)

    def units(v):
        v = v.view(16, 2, 8, 128, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7).reshape(16, 128, 32, 4, 8)
        return v.permute(0, 1, 3, 2, 4)
    data = torch.cat([units(hi), units(lo)], dim=2).contiguous()
    ku = 128 // s // wk
    return data.view(16, s, wk, ku, 8, 32, 8).permute(3, 1, 0, 2, 4, 5, 6).contiguous()

def rt_prepare_routertail_candidates(weights) -> dict:
    import torch
    from torch.utils.cpp_extension import load_inline
    w = weights.get('router')
    if w is None or tuple(w.shape) != (256, 8192) or w.dtype != torch.float32 or (not w.is_cuda):
        return {'enabled': False}
    if torch.cuda.get_device_capability(w.device) != (9, 0):
        return {'enabled': False}
    name = 'routertail_' + hashlib.sha256(rt__CUDA.encode()).hexdigest()[:12]
    ext = load_inline(name, rt__CPP, cuda_sources=rt__CUDA, functions=['routertail_small'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
    st = {'enabled': True, 'ext': ext, 'weight': w.contiguous(), 'packed': rt__pack(w)}
    for m in (8, 16, 17):
        x = torch.zeros((m, 8192), device=w.device, dtype=torch.bfloat16)
        rt_run_routertail_candidate(x, st)
    torch.cuda.synchronize(w.device)
    return st

def rt_run_routertail_candidate(x, st):
    import torch
    if not st.get('enabled') or x.ndim != 2 or x.shape[1] != 8192 or (not 2 <= len(x) <= 32):
        return None
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.device != st['weight'].device:
        return None
    if not x.is_contiguous() or x.data_ptr() % 16:
        return None
    out = torch.empty((len(x), 256), device=x.device, dtype=torch.float32)
    rc = st['ext'].routertail_small(st['packed'].data_ptr(), x.data_ptr(), out.data_ptr(), len(x), 16, 8, torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f'routertail launch failed: {rc}')
    return out

def rt_prepare_routertail(weights) -> dict:
    return {'enabled': False}

def rt_run_routertail(x, layer, n, st):
    return None
'Lossless BF16 projection refinements; only confirmed exact shapes are claimed.'
import torch
from torch.utils.cpp_extension import load_inline
mt_W0_DC2_NL = {0: 4, 1: 3, 2: 8, 3: 6}
mt_W0_DC2_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)
mt_W0_DC2_ESC_CAP = 32

def mt_wp0_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def mt_wp0_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def mt_wp0_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def mt_wp0_bf16_bytes(v):
    r, c = v.shape[:2]
    words = mt_wp0_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def mt_wp0_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else mt_W0_DC2_SIGN_POS, device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = mt_wp0_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def mt_wp0_c12_unpack(data, base, esc, decode=0):
    """Inverse of wp0_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long() if decode >= 2 else (torch.arange(8, device=dev) * 4).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3) if decode >= 4 else (7, 15, 23, 31, 3, 11, 19, 27) if decode >= 2 else mt_W0_DC2_SIGN_POS, device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def mt_wp0_table(base):
    t0 = base | base + 1 << 8 | base + 2 << 16 | base + 3 << 24
    t1 = base + 4 | base + 5 << 8 | base + 6 << 16 | base + 7 << 24
    v = t0 | t1 << 32
    return v - 2 ** 64 if v >= 2 ** 63 else v

def mt_wp0_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [mt_wp0_units(hi), mt_wp0_units(lo)]
    else:
        mats = [mt_wp0_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([mt_wp0_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = mt_wp0_c12_bytes(v, decode)
        if not torch.equal(mt_wp0_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base << 8 | base << 24)
        else:
            tabs.append(mt_wp0_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def mt_wp0_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def mt_wp0_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = mt_wp0_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, mt_W0_DC2_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > mt_W0_DC2_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = mt_wp0_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
mt_M16TRACE_ROUTES = {'o': {'cfg': (2, 4, 4, 1), 'inst': (1, 2, 1, 1, 0, 0, 256, 0, 8192, 1024, 2, 4, 4, 1, 7, 1), 'decode': 0, 'rows': 8192, 'k': 1024, 'variant': 'next_early'}}
mt_M16TRACE_SOURCES = {'o': ('\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint m16trace_59fcb5b72994_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n', _zsrc('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'), 'm16trace_59fcb5b72994_launch')}

def mt__launch(fn, index, plan, x, out, query=0) -> int:
    return fn(index, plan['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), plan['esc'].data_ptr(), x.shape[0], plan['rows'], plan['k'], *plan['cfg'], plan['t01'], plan['u01'], torch.cuda.current_stream().cuda_stream, query)

def mt__onehot(fn, index, plan, weight) -> bool:
    x = torch.zeros((16, plan['k']), device=weight.device, dtype=torch.bfloat16)
    out = torch.empty((16, plan['rows']), device=weight.device, dtype=torch.bfloat16)
    ids = torch.arange(16, device=weight.device)
    for column in range(0, plan['k'], 16):
        x.zero_()
        x[ids, column + ids] = 1
        if mt__launch(fn, index, plan, x, out):
            return False
        if not torch.equal(out, weight[:, column:column + 16].t()):
            return False
    return True

def mt_prepare_m16trace(weights) -> dict:
    state = {'plans': {}}
    for layer, route in mt_M16TRACE_ROUTES.items():
        weight = weights.get(layer)
        if weight is None or not weight.is_cuda or weight.dtype != torch.bfloat16:
            continue
        if weight.ndim != 2 or tuple(weight.shape) != (route['rows'], route['k']):
            continue
        if torch.cuda.get_device_capability(weight.device) != (9, 0):
            continue
        cpp, cuda, name = mt_M16TRACE_SOURCES[layer]
        ext = load_inline(name, cpp, cuda_sources=cuda, functions=[name], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        fn = getattr(ext, name)
        for index, fmt in enumerate((1, 0)):
            pk = mt_wp0_pack_format(weight, fmt, decode=route['decode'])
            if pk is None:
                continue
            sp = mt_wp0_stream(pk, 1, *route['cfg'])
            if sp is None:
                continue
            plan = dict(sp, **route, fn=fn, index=index)
            x = torch.empty((16, route['k']), device=weight.device, dtype=torch.bfloat16)
            out = torch.empty((16, route['rows']), device=weight.device, dtype=torch.bfloat16)
            attr = mt__launch(fn, index, plan, x, out, 2)
            if attr < 0 or attr % 1000:
                continue
            if not mt__onehot(fn, index, plan, weight):
                continue
            state['plans'][layer] = plan
            break
    return state

def mt_run_m16trace(x, layer, n, state) -> torch.Tensor | None:
    if x.ndim != 2 or x.shape[0] != 16 or (not x.is_cuda) or (x.dtype != torch.bfloat16) or (not x.is_contiguous()):
        return None
    plan = state['plans'].get(layer)
    if plan is None or n != plan['rows'] or x.shape[1] != plan['k']:
        return None
    if x.device != plan['wp'].device or x.data_ptr() % 16:
        return None
    out = torch.empty((16, n), device=x.device, dtype=torch.bfloat16)
    rc = mt__launch(plan['fn'], plan['index'], plan, x, out)
    if rc:
        raise RuntimeError(f'm16trace launch failed: {rc}')
    return out
BUNDLE_QKV_CUDA = _zsrc('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')
BUNDLE_ROUTES = {'router128': (128, 'router'), 'router256': (256, 'router'), 'qkv8': (8, 'qkv_gate'), 'router8': (8, 'router'), 'router16': (16, 'router'), 'o16': (16, 'o'), 'pooler16': (16, 'pooler'), 'unpooler16': (16, 'unpooler')}
dd_DECODEDIET_ROUTES = (('pooler',16),('unpooler',16))

def prepare_bundle(weights) -> dict:
    state = {'dd': dd_prepare_decodediet(weights), 'rw': rw_prepare_routerwide_candidates(weights), 'rt': rt_prepare_routertail_candidates(weights), 'mt': mt_prepare_m16trace(weights)}
    symbol = 'qs_literal_owner5_decode5_launch'
    cpp = SM_CPP.replace('sm_launch', symbol)
    ext = load_inline('bundle_owner_' + hashlib.sha256(BUNDLE_QKV_CUDA.encode()).hexdigest()[:12], cpp, cuda_sources=BUNDLE_QKV_CUDA, functions=[symbol], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)
    pk = tt_v10_m32x_m32x_pack_format(weights['qkv_gate'], 1, 0, 5)
    sp = None if pk is None else sm_stream(pk, 1, 1, 4, 8, 4)
    state['qkv'] = None if sp is None else dict(sp, fn=getattr(ext, symbol))
    return state

def run_bundle(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2:
        return None
    m = x.shape[0]
    if layer in ('pooler', 'unpooler') and m == 16:
        return dd_run_decodediet(x, layer, n, st['dd'])
    if layer == 'router' and m in (128, 256):
        if n != 256 or x.shape[1] != 8192 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()):
            return None
        if not st['rw'].get('enabled') or x.device != st['rw']['weight'].device or x.data_ptr() % 16:
            return None
        return rw_run_routerwide_candidate(x, st['rw'], m)
    if layer == 'router' and m in (8, 16):
        return rt_run_routertail_candidate(x, st['rt']) if n == 256 else None
    if layer == 'o' and m == 16:
        return mt_run_m16trace(x, layer, n, st['mt'])
    if layer == 'qkv_gate' and m == 8:
        sp = st['qkv']
        if sp is None or n != 2304 or x.shape[1] != 8192 or (x.dtype != torch.bfloat16) or (not x.is_cuda) or (not x.is_contiguous()):
            return None
        if x.device != sp['wp'].device or x.data_ptr() % 16:
            return None
        out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
        rc = sp['fn'](1, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, 8192, 1, 4, 8, 4, sp['t01'], sp['u01'], torch.cuda.current_stream(x.device).cuda_stream, 0)
        if rc:
            raise RuntimeError(f'bundle qkv launch failed: {rc}')
        return out
    return None

def prepare(weights: Weights) -> dict:
    state = _bundle_base_prepare(weights)
    state['bundle'] = prepare_bundle(weights)
    return state

def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    output = None
    if 'bundle' in state:
        output = run_bundle(data['x'], data['config']['layer'], data['config']['n'], state['bundle'])
    return _bundle_base_kernel(data, state) if output is None else output

V16_ROUTES = [{'m': 128, 'layer': 'router', 'name': 'router128', 'module': 'v16_bundle', 'owner_family': 'routerwide'}, {'m': 256, 'layer': 'router', 'name': 'router256', 'module': 'v16_bundle', 'owner_family': 'routerwide'}, {'m': 8, 'layer': 'qkv_gate', 'name': 'qkv8', 'module': 'v16_bundle', 'owner_family': 'owner5_decode5'}, {'m': 8, 'layer': 'router', 'name': 'router8', 'module': 'v16_bundle', 'owner_family': 'routertail_local4'}, {'m': 16, 'layer': 'router', 'name': 'router16', 'module': 'v16_bundle', 'owner_family': 'routertail_local4'}, {'m': 16, 'layer': 'o', 'name': 'o16', 'module': 'v16_bundle', 'owner_family': 'm16trace_early'}, {'m': 16, 'layer': 'pooler', 'name': 'pooler16', 'module': 'v16_bundle', 'owner_family': 'decodediet'}, {'m': 16, 'layer': 'unpooler', 'name': 'unpooler16', 'module': 'v16_bundle', 'owner_family': 'm32x_round3_decoder9'}]
SELECTED_ROUTES = [r for r in V15_SELECTED_ROUTES if (r['m'],r['layer']) not in {(q['m'],q['layer']) for q in V16_ROUTES}] + V16_ROUTES


_b2_base_prepare = prepare
_b2_base_kernel = kernel_fn

b2p_tt_v10_m32x_M32X_ESC_CAP = 32
b2p_tt_v10_m32x__CFG = (1, 8, 16, 1)
b2p_tt_v10_m32x__CUDA = _zsrc('eNrNPW1X27jS3/sr1O6h107cNE5SoATYQ2l37x4K7SF0u8/hYXMcxyEpjh1s5wW6/Pc7M5Js+SWJA713tz0hsiyNRqOZ0czoJT/1ncHIc9j740b3/YfjT+8/sDfPnv008mx32nfYvj3tW91g6kWjsVMbHmbf9Abmdjo7jPojL0rnWe61H4yi4VjNjO4mTjcKrFEUpsD6/sQJrGg0c7rXgT+dhAjr2U8Cz99+ec+63b4zG9lOtwvJgR/Yzshz4WW3++wZvof2Wyxodie+7zYb9OUE3f7brm1NGl2rO242Fl23P9ds3wsjKl5hE519f8Z43VkbUlY4ZjPfBUxcR3vh9mvXrt+z3Jp9XZu1amGzwb5v1Y0t09hqGFvNB4NdbrWu2i+gZurfHntxELzQZrWFbsjkXZK8T5JzHQu7L7SJriMCgRNNAw+ReeD9mpbs2EJ0bOaP+nHHput75tnYs+mynv3orkAPohK9cURvZJVUjwjGuk5hj6BD0A1T7cYPw3ESjCMtLm0ZSc2ekg4zWPcF1toLrF/rCSSB7AzozraaCap9QhUSFpIYvnviO0yh3o9Rx5Ffi/Z4bGkD17cipr209cvWlcGylE71pa4+mLw3hD8AqoV3QGjLHV17Tr82Nre93RvQDIE/r9m+Wxs0GzVSFfRnkOew71stY+uNsbVtbO3Q466x9ZYSSrFYuIAY1cELzb6sXyEheNpU0g0l3bzSk2pIQ1GLkmaSbCTJpkz26jJhIqWBuq9fs102d0bXwyhk/oD5oJOgpyyMnMkec/05691FTgh6hH2sM21muVN4qr9q6szy+uyjGee1Xu0A8Nar3igCuvd5neNa+fHrO3bXd/sJ531UR+ijqTwcK+kLtdSFUgqGmw9qmguYPWYH7Ji9ZPXFjvg3beeLDV0fypWSFUQBW7bHehGg4WhzQOzwkJnbBA4HuCwAJBkgbkDf3pit+lRnfzFNO2b7+8xs6NTn3Xq9jp+pAG4+BvhOc7uhAt8tht0oDxt6DYQqQrxVDLv5GNgJ3sc5oJuqyfHULdaSoW25zlLtCNVqgUeaY6GqyBXqkQMsVo3l0bXCm25vZIUK0kuRdP1JM6XCgVQ7gwF+MN0CkuEH045ViPkTFDkogh+vBBI6+ZEVOUlXw+GgFtTmgTVJz1nwaSU9E7Vk/471IkkHddkdgioFxpT8tcP/A89VAcki7QDl40qilWVVN9UEEp9Eqh4j8ioULj+PEm7ZzywyzceDkdgAc6VsaM+XBnQ5hvOcmRNoiXmZMb7AqJ+0X+htxufLMyzNIuvG8QzWm0ZsEi2skNmW5/nwEPgzh42iPeZ4/ZBFQ4f1rHBks57r2zcs9IGeFrwYheHUYfOhE8BfcCYix9tgrvSDPmROoJ+RBh82NvLWMbFXlLclx/4sNiG3tmxEK+HziDP2aADaNwI+jAVRH+vs4ACJPuD/gC/L0XXyCN0ajp1xF5DMWf5SpSR4dbv2LLK6147nBCO7G/ndcGgFTh/bZZu2G1jejZZVGOtJOAVzKejakYX1FaWR0oLBY3T2xCqeY7ChJdpbxRPq1zg59vYEmgni66YdauMpBnnPCrojb6Q4E5CjdML2wRGPDW8VbygXjABXrJ10ILJqve0W+D31K0QfUd8ja9YKBMoCIvQDGMgP7l7oj0DZWUwceynS0WIlxvg1c2ocRjda5LDvGqs6gNAfjf3A8UADEsXzWoxe1iSeVKoWOK5jhU5NsAbho7YO8lO6cdvt8r4XtC2JI1lQEAmatxbgXwk/66nNz63Cjmcbx2KrmizfZhh1LXQUW5oilwYjJ7TFZqq44ggXsE0ImJCrmRVSOU6Q448nrhM5wEt7e+SMYVQDfU7ORt8TDxs+LQpvvCkK3OxxmUZGG8RRj0Ec9BjEMY8BD3kIJUCYF7FkD2hSTqCi4C4lTRnF5d/kNdf3//cY8Mc1q02ALmzShudYxgAeDXZtYsHMeVez7NvpKIjZOCdyEx4rAZMBwOQI84KFjjuJ1SIauwY1+BCrRv9G6sY8NWLt6N+w5zBBprT8dquElr/GYKQy61ClZRM3dIfPOjwehFUDocPdePoWGEXSxX8HarEPFGEW6wU+GC4MbBovtOxo5HtcDbNw6M9DNp1Ao1CMW1NsYE1djCgCWa0+xgdwPp6OR941m3oD0CHhEEanbwGhy4sNDl0XCFnAE2h3lGYprKWLvqYta07BegnDUtAeq8+HQGWmPd+sfeKmMmjLUX6FiB2yRp3/m7quDiYr2pjahure6vdbPOLVeonGAe+/zOiRJqwtWPWA9WqLNqTvePoO0/c8fY/pOU/PY+0nNNha/TccuN2FH7S0dNOg+dAo9fmYCmYcg8Hc5SW0bly1i9pPqy8G0qoEQ7a2MKCuwdaUuqNSWWnm/9ZUvS/VwBxLbTgmYeQHDvTPm3V71Flzu4L6hwcmZzgmlQnwZZeToiELdQNPm2024/GmqH5hEzMazsiB6QOcSraPQ/LL6cUhC6NgakfsFID8MgYdkRTJvtuvHwK0EJxScGBojMGuCWhwzz5CEy2DnR3BtwlNrYRjrobTLAunsRrOroDTWAenuRrOdgpOAQ03Cmb4ccyf1iy0l3P9UiKCwPb2zj5eKcaC9vIoV+CIVZi5fbUi5FEkCPH7L2rhL2asbJO+a9AQOXjsr7+YSDew3E+TwLoeW6DvA991qZWBD+WRFC7DCQ++9lm2Q21Wrbq8Hfx3dNmCHrisyiiEkbhv80v3CrRTrphZUOwuX6xRUOw+X6xZUGzOiz3A3wfmuKGzprND3tlhvrNH2Nlh0ll1HW4KteaIAWAybKcKJC4FzlRD9jMME0zkOLCRKXNMzDFlxVKhs8v6FapQ+r7D7wY9RwTXYEdI3W1EB0dD3xj0vQA9F6DvloBubQjaFFibCdb3S0Dvbg76XoCWWM+XgDYbusIXwn76EAJMhzkeTPt7YC7eUkB9l/3FXMtz6KEJD2CvjtAApYg4PA8td4DpHUjv0MLIH5/O0YrCqBDF/NCe/1fIQAJ9D6AzKBPWMvrm7HwDdTOxInuopVXJ2bmqN5zCRRE0tUFCHFx3oGB+0yxaFFkEvBQtnvwy1bF32i6QTeNVd7CqCfkVvnqRk6dYmgIuTUAr6CDKT6CYUoiMKq9gHR5dBlfszwPAQBq1R/1vlo1EwzEIoQdgoVoyT1gwoetHocHmo2jIHMse8vECC5ZM2GgIEyYWgSEIJ+4oenXCwKWIRpabGgXbtcKQfRiPog1GAjyQqe2kLKMKCwOb20Znv/Hvz+Krc3HOU9GozxNeZFCTzIE/fMxi4o36CyAfFG1TEkj4G0+BHScjKiz2Q6GoanzVaxjDV/8Ijo/BT/jYTADwZxyaiV7OFgW7Dzp4qYWje3BX9QlwAXYM2ANwuxKtYHc0eAYLi3KkjMGU4+2yiHsnYInDsJ1fABe9Cvw5z2ATEK25FYCt07kAlgJZwTXGgYurl/RSFq2x978d/co0MLlxnH3PvdP3mMk8n52eHhnYXAMfFjwYa7AWPtGIoUdkgPTyKC3WNFA5AC9Ed+zGCTzHNRi4ihrxFIgK8Agg4U+jyRSYCW0ygAdeWlzAAugn77DNELeHAK6vMDhNi6QiKgxsiCrB8WDszcauqNvSEasU5FreFuHcIr/PxXdHPn94zxNIEMF6F6dHf/Dk8an4PhPfJ+L767lMyJyTLyLR4d9/iKoXAurx58Nn3S53SmXcvdt1ralnD7s99D/DblfjbTfW89O1M55pz7KTaRekCog4siNoYj6RLk/G3E6VWsjIeCoXiWo8y0/FmWJOKMR13LWCa0G+JHmSJL+eK2kl/+QLT68221bbaTEFGK1cn4JZAH/2OFLtVIEzLHCGBc6ggED1BDNPMPMEMk/ytb6eYwn4+zP93Yt785WqfqW6X7Gy7NrJF4L5hYB+QahfUmDT5nTWNhTWddaKMoCLLzD/AhTHxa9pYBRtOvv0Fd6TbL9kjTfbBUWOzQZOUcKCNRVrtglz00v2HGAUVPv3bx8/QT0seoh2f1HboDqS1sFuOPv0h4INPoO0JTngHJ1/Sh7BN/lw+vni/xQQRfiff+p2LgBQUqzZQEBd2ssWZ26j6/UpUxJURwHEz186/0aSEHLQNySJSDeVdEuSB58xBcjjFyGNYKch6tJP0wjHCHe7edYY5tIIA22+1x+h5rTcfaSkcEaNtGTCIGM1hIVQKiiFGKJxgFGcYBI4EaiAMNqnl4cayijNEM4CXnvkrWNcj5b4KH4LCgWsDJgJQgrmMhveM1w6urzCeoUVQCHHEaIQo1BpUVjw7S5Arj9OY2rxZDNJvtGB8ZFZKzBTVJh2wl4z7biD0tABYbgORv33o3HtTidbFUo0ILfOcUqpqnBRRIB0oUMNu6Sn8QysucT0+DM0AGJbQXGt4KRZQbGrsDdmI10prhLXrsp+toTsV0jQJRT0Ohv11i5iz9cFSQ2H8yKs6R1HFuByQsrFRMXPJH6SJkoSvItdsJTORxsHuW2I1t1v/QV6CJM5t33Q3nxjkEEAGZD7miaj6Jo/bdET2Xu8ONq07Qx4L0ppPtn7i1/hTxMHjeZnHMtFtuoZqiJBdsAC7VlCZYvh5Dm/kc+v4Tlbt4Ot5tklW+zXPxR1DhxGrdHIINsnFXO4/Qr1oHIFGoJR/Jp9fY0U1KhrSNM7KASlqyzOWehUD7KwC9nqAfmrSf0F5z8oHOiEYLaCXc+1B2SG8jdY/uRLtvzF72iyjjwN3XcQM4NpyFIwHeKuKL1dMHFjCLlEuDdeV35Jsi9ApVZVyfaVrojCt6hGExdflCy3niydVfJtqCLGWUC5EndDOoGb6xf6I+ES9d1CNZR4SZrWYa+QSjh2yB+Fr76ex/iwDZZwaRHU3LSqWEqlyrwngoFld1qxIm1hoON38hwbu9MNGlJXPWWth5LhAWXRUnr9BUO/ZirIhX3YDY4YTgodiVDsW82k4zbDOFI8hUCN12wXc1MuXEYZYlQXFQEvrYP9SzlbcU47U4ubvuCUQbkFanvhmqFwwfQ8lkxxokr/HcenCtABuQQkB9YPEY0QoUU4kFRyV88XL6JhMeEK2D60NpFmwEnpenbdzJ6IRVb7WiwNis3UYgWVrw2CNYaLwXzDHa2SupAI7Nxyn+AwPl9VctPgVMyDiFOJPuD2dmwly7nrqbesk/4YPGx+1iC3vC2AE+pdWvTgU2uoKVEvRSGO77o4fdYXv4h/0yLViDY3txh1WaPMPnjwrhSWhEmJ5twqzdmATs77m9/inDrJ1tESi6fZ0EV9dY7ghVmIbl0fkZPVf83VVqtlTLVFPSNDMKtV0NGHHLIyXqKHUaGxAhfiLSg2tJsoDMZpRp0A0ytAO3oTAwoYehr5uF5x7XTlZu0DdvnyilQKRrC4txmOEqHKDNFndsjnHHSAVNHDSSkcgTI6+bIk3J/RYN8O6m32bR+meVatftOX11AXCrCOu59bFlgi/SA7B6WFX1IUg5Aa0qJy9rWKdoteOb8ABCtnH6HJSrNRFYxV1LQwbAP7YH5blSMcjiqCaaratxhMFsBTlQ0qinXaRtUMauoJOoADeWivXPO4xXG73T+mMOCtnmZB7da4FeBoYWt+2bm4ujyHz9nHq8SyarHFjN6cwqdxpTA0xtiK+XgpJyZcyGOU34Bx13Fi0XJVzIk5MZG2GZcTlA4Q6OdoqYDyja6FvTa/RGyvLr9dXbpXpfTdXAP9lWgPtISkRoLcb1wJgfLSVU2UFePnGHNInHMxL9STeWE7Sb5NkmZdXyHbMX08Th+PLBOkj1c0R2f1InpfoBzTxoUGlELbHbI9nWwcqSbBhG/oGdMjJE2qsjuwDCewd1Vu0zPOo4uJvgyGWR6GcIqFDicvcFdXZ/5NBCjkJA2BpB0iKbA2sr0WGkwq9002+44NFsdFNjGxligK2opDagKwXKIoltgJyRIDs2z78uyIiz4KeeuqXXYlVa6d/hekvYCbS8xuNq9lQ60WVgKNjN0bkqgDI9nISPXaIOM5Dny3n3iOcba3iArNJ2jImSzzNFVzigMQG1Xym1TQ2jIoNpPpGjPTJB/xbvF5Hr/BzehcrFezRfxb5JqE2MAIRTnlAGgdHnrF/kARkCm0MnrASjfLHAXsdSkzpDxLrJsEMhM6jeWf6vqjBupe6Pragv3JlMe79ON9+nGeUkjC1Kri8hW3t2JVwHNTpR+vFhT/RFXfSayEpHT3KoNbwTwjdMpOolN2lVnF1FcO1QYzy393dhnlppe0zdli9/WDkhOMwe7NkmWrK6cR/q/HJ7iD+zpu0evxqQqf7sRTg57uxVOTnuaFUFrwzoyhvKEnCWWbniSUHXqaLzMvn858aWhlPZFy89RWXZjO3guA9crMWMxluHEDD2YDHyYfwwBvYlMvpqqhEzPaOv6sb+bGcBv8JZsdJOonXyZ3qly4KLMWHX5bf04+OSafnJJPHZIP1rkxad4ozydp14NIZISjKhCqXaTjjrgjIvbNtR83oWTGOI2g4N5yzGMB81j7Z0cV8PysahX3s8AQWVcH9UVzsFvHzzQjj1yBL3NAyu07pLVVOXsZ1KjB5CljXGz+YmbEtgSh1ivzLKH+KE8ohwN3APguwnZ0rrMcsrvWE0tpd0PvqFAx4+wzQ/LRk6KjMy8SdZ15kWjuzItlSlwtZl4p+jzzIlHtmReJls+8WKHwH/4mK6XA9C1rUqwX15xCnuLOvYNkyRNXYK7p0AYupMRW+odpeu2E7IdzNDa+5XQZHZREIx1XpnbpiCS2UjQ79P2CTA6Br1IeUMf0BF5TbMTT9XL7/vaFujvUuKADnIJ5gqFUTYryubPh4L4vmAx+Xud2OBNU8znnJi2X4lRFEY1WzQtlZwkxlAUKSItVZVnVs6FHkQ6U8bq35NLiahYPmBUQOZxfahqtDMdMJZfaMZZ8q6thcBDhwgMOtJ8PJ3wa6cvbSktod0xWTfWhoT40r/RVVl+hq5+h6fO/m6aFUqTsr8cVts2IXCQLMV0PZqiDFdpCxl1bpS9k3LdVGkPGvEAWHs3e28XuV72sqlQIKfmy3OhtFEIpMfJ5Mpe51AfjL3URf+G8jEOIQ2mISbdye5UkYYjSNk2RbaAEyB5HwdIS4IndtB6qVY18VvBZXuPZEqDnZLkfQSu+Pu4ykN4ud3S9CSl5Zd1pB1e9NLnjDWYynL+kO9wiB7lgHlD2Z/gYFNHmN7QbgzxmXa5t3caNxRBpWiLvere9zgey+v2N9mXwdWX/Ru6q6sSLy75b0Af1NqeeafQaRq9Z4Bmlhl7QqYgbs+7R2IqC0SJzGdSut1tbNKTn1APicLeJHKaG4jDx25YO+HVL8mw90EMv6Aiy6JMQauURWuLHpdCSqUacamZRfYLAF6rqDUUeub2Rl3vwXXpmoXWTG23cM/SoVukYUq7lBnya+jr9vlLfrzhu9IOinGUCbonwLDLBwMKJ6BCVC5jMkNwX1vNz4dqtU2AbxOyWaZRcXHCdmK/etFPANxRqDNUV/iI9VKwlUzsA+L7evHwXBDN5CyuClCwOUGLg8Wl9LpTGe9zWkN/FogRHK4fZhbW4rlmqburAVoEULEGq7IofR+PHrO0tk2WGMpKEaRdKkHahhGgXywK0VD8J0C6U8OxCCc4uikOziE85Y6PQTlnNM7Svfa05s3b9rshLkQspZBon5wLh+/aK/VmUzQ9+LnvVgFfF571Voz1XrVlo4MerPCt12880wBzjPWFi0pMOWJatynulVqeczWTi0ZYykcF4dD+NJ3ezxCyNhwbKz9Hgx8FUke0nWan/xM4+/Mg4x1IXPbMtoWgfazak/uhNAUUrtsVruXRsJbvdjG/Zz+x7VuwR3PuJ4gmGlkiBsJlTvt1VPDbgsZ88NlPBp4wTcpM6/D1mfzE83Lre+FK06XZRQCd72Va/5vrJ9Vr4AXtx6w3dXaO8FLcLFr9siMt85MsXyxXei1TFprwBiCqStV+Vl3lV5R27mLBlIr7w60bc9XOb5l6xp8EC8vbgY8Onv4zfCkaeDidlh14xlIgBNogk8/su8MxQlY4gpJxgYUN2uzjMXSvkh1016sJLWjeQt9YlG+Cyuhbx433GlYa3TnNn523vbYl77mLMDGbGp2mVTejxIZdco3z/TLBu58w/ZSdMoG6CoSNzPzN1zsEXVZZT0Zi9ly1YvMEttkbWDwoCq4s1GlNuyIk1n+qEnL+TR3XE4Zl2rkjgz9HmTJ1pOX+nHtzgB+PAnJZHF0iR4glJDCaU2V63kS/32BjcquH7/my1m1bkouHUpggtyJtN0Z+l22jpEDbeZRrVZdw1K6foECBYWz3Pkw40y7ONP8ubclCmUba1hAf1DE8Aj2E/9tk4b45uolzSztkZqht/bijMvzSGrkr6Ms3IT1iqg/H6Nftl6rr8soI9cbhfw0klNRpxvLrGjv3xxLL5jtmQYhymnq148TtG0uQwXteSPcO49RZP4XcH2Gqy/5YO5ctLA4p2ftIuCORHbAlXpoAoDh0HwQx+HM+jne2QucVPbhE2IgsP+V/k52lxcdGML35+p/uf6HonuqhpVps/tB+nrtL8nuV2L8Xh3kr+znI3qYsqE6QW/O2V4G5aRBFM+iOYEqiW4Udlm3i7YMxtwTobDvt1POo72UEHoyNKBv3i9+yYX/z+jxlyHOroCWN8/U8c24zhJSen5JigmSaLvIsEz+MVHUMR7zMHUfC4pnoiJUuzs99Scyw/2YaHJf6nMc6M3scuXtKGKqJvbkEQsayyzKy3ZvFVvUpFTgl4sEOmTSXdUNKZpdcVS63FA5rl7ZL3zyAJDLpuBo8p4zfdMIO3y8RzQNFyWboxyTN2OMU7KIh1Er49+5of9h+L6jJdRQhd4u0y4sK7nExnf30nvmJVeLg2zoBFxaLhKOyKstmlcNulJRhtuQYhcwp3c0BRMiozR4GzFfD2GqgBUlRl4qiucno0W9oxRVAdiaRBY1W+HgdQMi1krgwKbngp0mvx7UGOWXB7UHbwaV29At0ZWxN5sQL1iXiiyoemnjPmUs60yZ3pjghNlr1UaF2rt/moS8zb6lVDy0IwhcP5sGortar3cBZSTjPndSBdqCIzwSzLrJumz/dKU9ugKTe2mdsFCjxw7NlTFTjYnhdDh7mghZzgXyHz5158BRbu0hHXG3HKi0uI2nQtErQ79iOHHV8chWwyDYf8lq34RuN1Ci1zkH0z5Rbf9pyfgeOhwN2SwGx4pnzNvrCg3OUBdBs7HWWvr1sW/SH7X8qs4WUYyCvw4nB9KloSgk1Ty1tCrmwzXJcoPK9p82CDiRW3J2RrcHcldjuv28UrCoL1+SF0EoBkGuJ3K6TulkDpJCXRXtGlJfekPWFyT9M4vvIBMOKIz4oLk5opf+l4KX7d6Aw9craB1/cGyeUY6xbGHlYo1rj3fNNY2j0useWs9E3SeXQLdU5LL2uLIGuhDWIbdIcKN0jstcZTyTHcqBHhsZU30mLDPiUdHS4OZRXm36LeliipQo3GMS2UI272S6P/JtYDxUqnEHqsh5btwPyBrsBGuzAfOTP+iBnvCSa9bRRwd4kAxFqdL1T+Uj1fUv/+ozXvQ4GV8zdp1UcotvI6belSprgkVNzXTf5VGdr6kwhcJM5QSl2c51+Z8leO8Bqe2LfG131nlhzMxR+F/dWJ3tOvIWgv4Z2evOC58PooioJRbxpBCei2eIWZp9aiQ5bzKS2ffnaCd+gLfkLEDCahPWR/rROM8tupE9yxOuMXaKLNbYoyIaidBcOfbpg5TPinIdNESlTQGWhZ8jv5ramd0zYY3BJA4FyHFbNer1ehhOXyn3j8Z18tusHYiztH6Yex6Jo/vChUphdJktbr5ENy06foibi+U97uKW/2lLd6GssmBaJQAjeqm8nDVH0IIxBO0RqNtvh9GIzU4j2zZX68whnP9mmc8IPjg2ND48LHRIwHjgWOA44B0Z9oT3RHmiO9kdZA5/R9gRPfHdl3gAjnxtcMeAZL8MetA/mMcsTz5KXxUkR+gaGIpSNkAyveP6DJ16r8hNrLgWVQ93XUblimM7VtJwylngPRFbIpngdWzZuOz4GlMUoLTI2zA0Ze6GbK8cjb51PFoQYlid878PwOWd5gb9++jUWQLI3jU1zVG2Pjx6c6Lfcdn2HWGWWdiawTzDqhrBNdQa0R/3gaXieI64PnVOjruaj49USsGlLuiQT3heB94QC/iNwOZnYor5NrZNlFhHSjIQXkxCWGRvHyJyIJefv8ilRASKSg/iHnG6VFWeGMbQG05EXSXcHVekdeQwmdob2RgP1JUqEZV+hc4Kum6PcWXUYAzUN6n9KyRkvWGEsMx4CguCKWdrbHRd8kdKGLTym0h9btc6HPEcxzyJVkNZO6b+V7/lrrQGsUoegw+iHWTMnMXYWS1cSFhYRbcmnha8avc5H3UI2Zio8mL5DdjU8hKlY5TE1UHmh6SHv7i2KukhZvzPieUZacHk1WAQhS7C3T/Y7ImEVwimFkYjYpePyqyeL7/STg1B1/ShNpxHCvLYpGJRs2E5WksK7Z7QpAqwlUhWMye3orLGbjeJdVum7mctWCn+bAS1JpiFP16GZWZaAy1yAnlFZmN5jcu/07L2WjyDzFUlmhXNcpVl4qaWcjW+oV160D+bOCsS5Nt91RjSJs1chjDMbR+zvPGo9s1UZCgIbETlemC1ffCPqZ7332g8jquc4xN40Q8pHr+nOnL2+yfMiqrxCVi2w8Fnl+TzO08JFMi2PfG4yu8fdeMfT7/SH9NsaAWZG47sq1a+KqVijfH42bGl7nev6OrGOOCRSRV83KMmgsm8r7viAWDMc7cY9vOBbvuDGBQkRTJj1hdyiRqDbJPYBZ/apGV+sWIC7oBag4XjjyhWvB68wsV/7MGt08S0H95e/ReOiseH+f1IdeWIABdsuK4irmWjQ7YB73p+7Iu/5M5srnwBk4Ad4JqgJR2l1egTQ5SqLYxw5tLmnkUOO2kR7jDjbIkUC/obJWbBc1cbmK/A+8RwJhfwgCP0Db8wCfPtn2dGJ59h3IxRFZ9+QrhOA0nE7daDQJfLSD/ACdDGJ6YJBwrLclmzoHB4q99PNs75WJv6eUxaMV8zW3ksY5C85U3aGMMxTjjGPignIvRl2QLSSHiEvoS9fWlxnNe2xjCgioKCThWE95ax9h8AhP6fhKAmHnFBIB9jNoGskkBizhshP6pYcPCw3Qlm0tOeggzjnM8W6YJUWyF3ovlv1uWa4+3blKd6Evhx7/eMEhXqmJJ2kM9F/AGERbH43Cpf6KcsodPBXaKSmPeeNeaOX+YHlpuo7lcNdHI11gmqu+tsUE5jSGqf58YX448wOIRx1f7ZLXXN435FEnIF3ijf0vfMSnuobhfBTZQxTbkFZpgQUcVt+TPy9c0i3eN42WYcL/OvxvvNmGv636221j13zbgNxdw9yGrx0sgkxtLAwkAvZ9bJwZJwb0E7oIvesY2ANEXODLUW1zvMyN8ar/t/GiSxgc+knLvcScfyt/hOY/HoACKg==')
b2p_tt_v10_m32x__CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def b2p_tt_v10_m32x_m32x_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def b2p_tt_v10_m32x_m32x_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def b2p_tt_v10_m32x_m32x_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def b2p_tt_v10_m32x_m32x_bf16_bytes(v):
    r, c = v.shape[:2]
    words = b2p_tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def b2p_tt_v10_m32x_m32x_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = b2p_tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
    """Inverse of r3_pool32_pooler_d9_cap2_a_m32x_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def b2p_tt_v10_m32x_m32x_table(base):
    return base * 16843009

def b2p_tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [b2p_tt_v10_m32x_m32x_units(hi), b2p_tt_v10_m32x_m32x_units(lo)]
    else:
        mats = [b2p_tt_v10_m32x_m32x_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([b2p_tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = b2p_tt_v10_m32x_m32x_c12_bytes(v, decode)
        if not torch.equal(b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base * 16843009)
        else:
            tabs.append(b2p_tt_v10_m32x_m32x_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def b2p_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def b2p_tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = b2p_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, b2p_tt_v10_m32x_M32X_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > b2p_tt_v10_m32x_M32X_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = b2p_tt_v10_m32x_m32x_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.r3_pool32_pooler_d9_cap2_a_m32x_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def b2p_tt_v10_m32x_m32x_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def b2p_tt_v10_m32x_m32x_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if b2p_tt_v10_m32x_m32x_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def b2p_tt_v10_m32x__build():
    name = 'r3_pool32_pooler_d9_cap2_a_m32x_pooler_' + hashlib.sha256(b2p_tt_v10_m32x__CUDA.encode()).hexdigest()[:12]
    return load_inline(name, b2p_tt_v10_m32x__CPP, cuda_sources=b2p_tt_v10_m32x__CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2p_tt_v10_m32x__make_plan(ext, weights, fmt):
    pack = b2p_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = b2p_tt_v10_m32x_m32x_stream(pack, 1, *b2p_tt_v10_m32x__CFG)
    if stream is None:
        return None
    return dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=4096, cfg=b2p_tt_v10_m32x__CFG)

def b2p_tt_v10_m32x__run(x, plan):
    out = torch.empty((x.shape[0], 4096), dtype=torch.bfloat16, device=x.device)
    rc = b2p_tt_v10_m32x_m32x_launch_raw(plan['ext'], plan['idx'], plan, x, out, 4096, b2p_tt_v10_m32x__CFG)
    if rc:
        raise RuntimeError(f'm32x launch failed: {rc}')
    return out
b2p_tt_v10_m32x__W_CUDA = _zsrc('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')
b2p_tt_v10_m32x__W_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_w_launch(int,uint64_t,uint64_t,uint64_t,uint64_t,int,int,int,int,uint64_t,uint64_t);\nint r3_pool32_pooler_d9_cap2_a_m32x_w_info(int);\n'

def b2p_tt_v10_m32x__w_c12_bytes(v):
    return b2p_tt_v10_m32x_m32x_c12_bytes(v, 5)

def b2p_tt_v10_m32x__w_c12_unpack(data, base, esc):
    return b2p_tt_v10_m32x_m32x_c12_unpack(data, base, esc, 5)

def b2p_tt_v10_m32x__w_table(base):
    return base << 8 | base << 24

def b2p_tt_v10_m32x__w_pack_weight(w, r, wk, split, compressed=True):
    n, k = w.shape
    total = wk * split
    rw = r // 16
    if w.dtype != torch.bfloat16 or r not in (32, 64) or n % r or k % 64 or (k // 64 < total):
        return None
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 64, 32, 32)
    esc = torch.full((n // r * total * rw, 64), -1, dtype=torch.int32, device=w.device)
    table, max_esc, escapes = (0, 0, 0)
    if compressed:
        data, b, entries = b2p_tt_v10_m32x__w_c12_bytes(v)
        if not torch.equal(b2p_tt_v10_m32x__w_c12_unpack(data, b, entries), v):
            return None
        table = b2p_tt_v10_m32x__w_table(b) & 18446744073709551615
        er, ec, el, ev, xor7 = entries
        escapes = er.numel()
        if escapes:
            boundaries = torch.tensor([k // 64 * s // total for s in range(1, total)], device=w.device)
            rank = torch.bucketize(ec, boundaries, right=True)
            seq = ec - k // 64 * rank // total
            group = (er // rw * total + rank) * rw + er % rw
            max_esc = int(torch.bincount(group, minlength=esc.shape[0]).max())
            if max_esc > 64 or int(seq.max()) >= (1 << 14) - 1:
                return None
            order = torch.argsort(group * (1 << 14) + seq)
            gs = group[order]
            ss = seq[order]
            pos = torch.arange(len(gs), device=w.device) - torch.searchsorted(gs, gs)
            entry = ss.long() << 18 | el[order].long() << 13 | (ev[order] // 2).long() << 8 | (ev[order] % 2).long() << 7 | xor7[order].long()
            esc[gs, pos] = b2p_tt_v10_m32x_m32x_to_i32(entry)
    else:
        data = b2p_tt_v10_m32x_m32x_bf16_bytes(v)
    payload = data.view(n // r, rw, k // 64, -1).permute(0, 2, 1, 3).contiguous().view(-1)
    return dict(w=payload, esc=esc, table=table, r=r, wk=wk, split=split, compressed=compressed, n=n, k=k, escapes=escapes, max_esc=max_esc, pack_exact=True)

def b2p_tt_v10_m32x__w_launch_raw(ext, idx, pk, x, out):
    return ext.r3_pool32_pooler_d9_cap2_a_m32x_w_launch(idx, x.data_ptr(), pk['w'].data_ptr(), pk['esc'].data_ptr(), out.data_ptr(), x.shape[0], pk['n'], pk['k'], pk['split'], pk['table'], torch.cuda.current_stream().cuda_stream)

def b2p_tt_v10_m32x__w_verify_onehot(ext, idx, pk, w, m=32):
    n, k = w.shape
    eye = torch.eye(k, device=w.device, dtype=torch.bfloat16)
    out = torch.empty((k, n), device=w.device, dtype=torch.bfloat16)
    for start in range(0, k, m):
        if b2p_tt_v10_m32x__w_launch_raw(ext, idx, pk, eye[start:start + m], out[start:start + m]):
            return False
    return bool(torch.equal(out.view(torch.int16), w.T.contiguous().view(torch.int16)))

def b2p_tt_v10_m32x__w_build():
    name = 'r3_pool32_pooler_d9_cap2_a_m32x_unpooler_' + hashlib.sha256(b2p_tt_v10_m32x__W_CUDA.encode()).hexdigest()[:12]
    return load_inline(name, b2p_tt_v10_m32x__W_CPP, cuda_sources=b2p_tt_v10_m32x__W_CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_w_launch', 'r3_pool32_pooler_d9_cap2_a_m32x_w_info'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2p_tt_v10_m32x__w_make_plan(ext, weights, fmt):
    pack = b2p_tt_v10_m32x__w_pack_weight(weights, 64, 4, 1, fmt == 1)
    if pack is None:
        return None
    return dict(pack, ext=ext, idx=0 if fmt == 1 else 1, fmt=fmt)

def b2p_tt_v10_m32x__w_run(x, plan):
    out = torch.empty((x.shape[0], 8192), dtype=torch.bfloat16, device=x.device)
    rc = b2p_tt_v10_m32x__w_launch_raw(plan['ext'], plan['idx'], plan, x, out)
    if rc:
        raise RuntimeError(f'm32x unpooler launch failed: {rc}')
    return out

def b2p_tt_v10_m32x__w_verify_random(plan, weights):
    for m in (9, 16, 17, 24, 32):
        x = torch.randn((m, 4096), device=weights.device, dtype=torch.bfloat16)
        x[-1].zero_()
        x[-1, m * 131 % 4096] = 16
        ref = (x.double() @ weights.double().T).to(torch.bfloat16).double()
        out = b2p_tt_v10_m32x__w_run(x, plan).double()
        diff = out - ref
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > 0.0025:
            return False
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if float(rows.max()) > 0.01 or float(rows[-1]) > 0.00025:
            return False
    return True
b2p_tt_v10_m32x__Q_CFG = (1, 4, 8, 4)
b2p_tt_v10_m32x__Q_CUDA = _zsrc('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')
b2p_tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_pool32_pooler_d9_cap2_a_m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.r3_pool32_pooler_d9_cap2_a_m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def b2p_tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def b2p_tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if b2p_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def b2p_tt_v10_m32x__q_build():
    name = 'r3_pool32_pooler_d9_cap2_a_m32x_qkv_' + hashlib.sha256(b2p_tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
    return load_inline(name, b2p_tt_v10_m32x__Q_CPP, cuda_sources=b2p_tt_v10_m32x__Q_CUDA, functions=['r3_pool32_pooler_d9_cap2_a_m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2p_tt_v10_m32x__q_make_plan(ext, weights, fmt):
    pack = b2p_tt_v10_m32x_m32x_pack_format(weights, fmt, 0, 5)
    if pack is None:
        return None
    stream = b2p_tt_v10_m32x_m32x_stream(pack, 1, *b2p_tt_v10_m32x__Q_CFG)
    if stream is None:
        return None
    plan = dict(stream, ext=ext, idx=0 if fmt == 1 else 1, n=2304, cfg=b2p_tt_v10_m32x__Q_CFG)
    x = weights[:32]
    out = torch.empty((32, 2304), dtype=torch.bfloat16, device=weights.device)
    capacity = b2p_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, b2p_tt_v10_m32x__Q_CFG, 1)
    attr = b2p_tt_v10_m32x__q_launch_raw(ext, plan['idx'], plan, x, out, 2304, b2p_tt_v10_m32x__Q_CFG, 2)
    if capacity < 144 or attr < 0 or attr % 1000:
        return None
    return plan

def b2p_tt_v10_m32x__q_run(x, plan):
    out = torch.empty((x.shape[0], 2304), dtype=torch.bfloat16, device=x.device)
    rc = b2p_tt_v10_m32x__q_launch_raw(plan['ext'], plan['idx'], plan, x, out, 2304, b2p_tt_v10_m32x__Q_CFG)
    if rc:
        raise RuntimeError(f'm32x qkv launch failed: {rc}')
    return out

def b2p_tt_v10_m32x_prepare_m32x(weights, *, include_unpooler: bool=False) -> dict:
    state = {'plans': {}, 'device': None}
    configs = [('pooler', (4096, 8192), b2p_tt_v10_m32x__build, b2p_tt_v10_m32x__make_plan), ('qkv_gate', (2304, 8192), b2p_tt_v10_m32x__q_build, b2p_tt_v10_m32x__q_make_plan), ('unpooler', (8192, 4096), b2p_tt_v10_m32x__w_build, b2p_tt_v10_m32x__w_make_plan)]
    for layer, shape, build, make_plan in configs:
        if layer == 'unpooler' and (not include_unpooler):
            continue
        w = weights.get(layer)
        if w is None or w.dtype != torch.bfloat16 or tuple(w.shape) != shape or (not w.is_cuda):
            continue
        if torch.cuda.get_device_capability(w.device) != (9, 0):
            continue
        if state['device'] is not None and state['device'] != w.device:
            continue
        state['device'] = w.device
        ext = build()
        for fmt in (1, 0):
            plan = make_plan(ext, w, fmt)
            if plan is None:
                continue
            if layer == 'pooler':
                ok = b2p_tt_v10_m32x_m32x_verify_onehot(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__CFG, 32) and b2p_tt_v10_m32x_m32x_verify_random(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__CFG, [9, 16, 17, 24, 32])
            elif layer == 'qkv_gate':
                ok = b2p_tt_v10_m32x__q_verify_onehot(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__Q_CFG, 32) and b2p_tt_v10_m32x__q_verify_random(ext, plan['idx'], plan, w, b2p_tt_v10_m32x__Q_CFG, [9, 16, 17, 24, 32])
            else:
                ok = b2p_tt_v10_m32x__w_verify_onehot(ext, plan['idx'], plan, w, 32) and b2p_tt_v10_m32x__w_verify_random(plan, w)
            if ok:
                state['plans'][layer] = plan
                break
    return state

def b2p_tt_v10_m32x_run_m32x(x, layer, n, st) -> torch.Tensor | None:
    expected = {'pooler': (4096, 8192), 'qkv_gate': (2304, 8192), 'unpooler': (8192, 4096)}.get(layer)
    if expected is None or x.ndim != 2 or x.shape[0] != 32:
        return None
    if (n, x.shape[1]) != expected or x.dtype != torch.bfloat16 or (not x.is_cuda):
        return None
    if x.device != st.get('device') or not x.is_contiguous():
        return None
    plan = st.get('plans', {}).get(layer)
    if plan is None:
        return None
    if layer == 'pooler':
        return b2p_tt_v10_m32x__run(x, plan)
    if layer == 'qkv_gate':
        return b2p_tt_v10_m32x__q_run(x, plan)
    return b2p_tt_v10_m32x__w_run(x, plan)




































b2q_tt_v10_m32x_M32X_ESC_CAP = 32

def b2q_tt_v10_m32x_m32x_units(w, layout=0):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    if layout == 9:
        v = bits.view(n // 16, 2, 8, k // 64, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    elif layout == 5:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 2, 4, 2).permute(0, 3, 2, 6, 4, 5, 1, 7)
    else:
        v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def b2q_tt_v10_m32x_m32x_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def b2q_tt_v10_m32x_m32x_to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def b2q_tt_v10_m32x_m32x_bf16_bytes(v):
    r, c = v.shape[:2]
    words = b2q_tt_v10_m32x_m32x_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def b2q_tt_v10_m32x_m32x_c12_bytes(v, decode=0):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = b2q_tt_v10_m32x_m32x_to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode=0):
    """Inverse of m32x_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def b2q_tt_v10_m32x_m32x_table(base):
    return base * 16843009

def b2q_tt_v10_m32x_m32x_pack_format(w, fmt, layout=0, decode=2):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [b2q_tt_v10_m32x_m32x_units(hi), b2q_tt_v10_m32x_m32x_units(lo)]
    else:
        mats = [b2q_tt_v10_m32x_m32x_units(w, layout)]
    if fmt in (0, 2):
        data = torch.cat([b2q_tt_v10_m32x_m32x_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = b2q_tt_v10_m32x_m32x_c12_bytes(v, decode)
        if not torch.equal(b2q_tt_v10_m32x_m32x_c12_unpack(data, base, esc, decode), v):
            return None
        if decode in (1, 2, 4) and (not 1 <= base <= 119):
            return None
        parts.append(data)
        if decode in (1, 2, 4):
            scale = 127 + 2 * (base - 64) << 7
            tabs.append(scale | scale << 16)
        elif decode in (3, 5):
            tabs.append(base * 16843009)
        else:
            tabs.append(b2q_tt_v10_m32x_m32x_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def b2q_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def b2q_tt_v10_m32x_m32x_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = b2q_tt_v10_m32x_m32x_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, b2q_tt_v10_m32x_M32X_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > b2q_tt_v10_m32x_M32X_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = b2q_tt_v10_m32x_m32x_to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}
b2q_tt_v10_m32x__Q_CUDA = _zsrc('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')
b2q_tt_v10_m32x__Q_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

def b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

def b2q_tt_v10_m32x__q_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def b2q_tt_v10_m32x__q_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if b2q_tt_v10_m32x__q_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def b2q_tt_v10_m32x__q_build():
    name = 'm32x_qkv_' + hashlib.sha256(b2q_tt_v10_m32x__Q_CUDA.encode()).hexdigest()[:12]
    return load_inline(name, b2q_tt_v10_m32x__Q_CPP, cuda_sources=b2q_tt_v10_m32x__Q_CUDA, functions=['r3_qkv16_qkv_gate_push_d9_cap6_a_m32x_q_launch'], extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'], no_implicit_headers=True, with_pytorch_error_handling=False)

b2o_OLAYER_CUDA = _zsrc("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")
b2o_OLAYER_CPP = '#include <pybind11/pybind11.h>\n#include <stdint.h>\nint r3_o32_o_direct_a_ol_launch(int,int64_t,int64_t,int64_t,int,int64_t,int);\n'

def b2o_olayer_source(configs) -> str:
    ns = sorted({c[0] for c in configs})
    functions = []
    for n in ns:
        regs = ','.join((f'%{i}' for i in range(n // 2)))
        outs = ','.join((f'"+f"(a[{i}])' for i in range(n // 2)))
        functions.append(f'OD void r3_o32_o_direct_a_ol_mma{n}(float* a,uint64_t w,uint64_t x) {{\n          asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{n // 2}, %{n // 2 + 1}, 1, 1, 1, 0, 0;" : {outs} : "l"(w),"l"(x));\n        }}')
    cases = '\n'.join((f"case {i}: return r3_o32_o_direct_a_ol_launch_t<{','.join(map(str, tuple(c) + (1,) if len(c) == 6 else c))}>(x,w,out,m,stream,query);" for i, c in enumerate(configs)))
    calls = '\n'.join((f'if constexpr(BN=={n}) r3_o32_o_direct_a_ol_mma{n}(a,aw,ax);' for n in ns))
    return b2o_OLAYER_CUDA.replace('@@MMA@@', '\n'.join(functions)).replace('@@CASES@@', cases).replace('@@CALL@@', calls)

def b2o_build_olayer(configs):
    for cfg in configs:
        bn, bk, st, pipe, pack, hint, *groups = cfg
        tw = groups[0] if groups else 1
        if bn < 16 or bn > 256 or bn % 8 or (bk not in (64, 128, 256)) or (st < 1) or (st * ((64 + bn * tw) * bk * 2 + 16) > 227000) or (pipe not in (0, 1, 2, 3)) or (pipe & 1 and st < 2) or (pack not in (0, 1, 2)) or (hint not in (0, 1)) or (tw not in (1, 2)) or (bn * tw > 256) or (tw > 1 and (not pipe & 2)):
            raise ValueError(f'unsupported olayer configuration: {cfg}')
    source = b2o_olayer_source(configs)
    name = 'olayer_' + hashlib.sha256(source.encode()).hexdigest()[:12]
    return load_inline(name, b2o_OLAYER_CPP, cuda_sources=source, functions=['r3_o32_o_direct_a_ol_launch'], extra_cuda_cflags=['-O3', '-std=c++20', '-gencode=arch=compute_90a,code=sm_90a', '--ptxas-options=-v'], no_implicit_headers=True, with_pytorch_error_handling=False)

def b2o_pack_olayer(weight) -> torch.Tensor:
    columns = torch.arange(64, device=weight.device)[None, :] ^ (torch.arange(64, device=weight.device)[:, None] & 7) * 8
    tiled = weight.reshape(128, 64, 16, 64).permute(0, 2, 1, 3)
    return tiled.gather(-1, columns.expand(128, 16, 64, 64)).contiguous()

def b2o_prepare_olayer_candidates(weights, configs) -> dict:
    weight = weights['o'].contiguous()
    return {'enabled': True, 'ext': b2o_build_olayer(configs), 'configs': configs, 'weight': weight, 'packed': b2o_pack_olayer(weight) if any((c[4] == 1 for c in configs)) else None, 'tiled32': weight.reshape(256, 32, 16, 64).permute(0, 2, 1, 3).contiguous() if any((c[4] == 2 for c in configs)) else None}

def b2o_run_olayer_candidate(x, st, index) -> torch.Tensor:
    pack = st['configs'][index][4]
    weight = st['tiled32'] if pack == 2 else st['packed'] if pack == 1 else st['weight']
    out = torch.empty((x.shape[0], 8192), device=x.device, dtype=torch.bfloat16)
    rc = st['ext'].r3_o32_o_direct_a_ol_launch(index, x.data_ptr(), weight.data_ptr(), out.data_ptr(), x.shape[0], torch.cuda.current_stream().cuda_stream, 0)
    if rc:
        raise RuntimeError(f"olayer launch failed: {rc}, config={st['configs'][index]}")
    return out














BUNDLE2_ROUTES = {'qkv16': (16, 'qkv_gate'), 'pooler32': (32, 'pooler'), 'o32': (32, 'o'), 'unpooler32': (32, 'unpooler')}
BUNDLE2_UNPOOL32 = 'decodediet'
BUNDLE2_UNPOOL1 = 'v16'

def b2_prepare_qkv(weights) -> dict:
    w = weights['qkv_gate']
    ext = b2q_tt_v10_m32x__q_build()
    cfg = (1, 4, 8, 4)
    for fmt in (1, 0):
        pk = b2q_tt_v10_m32x_m32x_pack_format(w, fmt, 0, 5)
        if pk is None:
            continue
        plan = b2q_tt_v10_m32x_m32x_stream(pk, 1, *cfg)
        if plan is None:
            continue
        idx = 0 if fmt else 1
        plan.update(ext=ext, idx=idx, cfg=cfg, n=2304)
        x = w[:16]
        out = torch.empty((16, 2304), device=w.device, dtype=w.dtype)
        capacity = b2q_tt_v10_m32x__q_launch_raw(ext, idx, plan, x, out, 2304, cfg, 1)
        attr = b2q_tt_v10_m32x__q_launch_raw(ext, idx, plan, x, out, 2304, cfg, 2)
        if capacity < 144 or attr < 0 or attr % 1000:
            continue
        if b2q_tt_v10_m32x__q_verify_onehot(ext, idx, plan, w, cfg, 16) and b2q_tt_v10_m32x__q_verify_random(ext, idx, plan, w, cfg, [9, 15, 16]):
            return plan
    return {}

def prepare_bundle2(weights) -> dict:
    state = {'qkv': b2_prepare_qkv(weights), 'pooler': b2p_tt_v10_m32x_prepare_m32x({'pooler': weights['pooler']}), 'o': b2o_prepare_olayer_candidates(weights, [(32, 128, 8, 1, 1, 1)])}
    state['diet'] = dd_decodediet_tt_mw_prepare_m32w(weights, configs={(32, 'unpooler'): dd_decodediet_tt_mw_CONFIGS[32, 'unpooler']})
    return state

def run_bundle2(x, layer, n, st) -> torch.Tensor | None:
    if x.ndim != 2 or x.dtype != torch.bfloat16 or (not x.is_cuda) or (not x.is_contiguous()) or x.data_ptr() % 16:
        return None
    m, k = x.shape
    if layer == 'qkv_gate' and m == 16 and (n == 2304) and (k == 8192):
        p = st['qkv']
        if not p or x.device != p['wp'].device:
            return None
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        rc = b2q_tt_v10_m32x__q_launch_raw(p['ext'], p['idx'], p, x, out, n, p['cfg'])
        if rc:
            raise RuntimeError(f'bundle2 qkv launch failed: {rc}')
        return out
    if layer == 'pooler' and m == 32:
        return b2p_tt_v10_m32x_run_m32x(x, layer, n, st['pooler'])
    if layer == 'o' and m == 32 and (n == 8192) and (k == 1024) and (x.device == st['o']['weight'].device):
        return b2o_run_olayer_candidate(x, st['o'], 0)
    if layer == 'unpooler' and m == 32:
        return dd_decodediet_tt_mw_run_m32w(x, layer, n, st['diet'])
    return None

def prepare(weights: Weights) -> dict:
    state = _b2_base_prepare(weights)
    state['bundle2'] = prepare_bundle2(weights)
    return state

def kernel_fn(data: Inputs, state: dict) -> torch.Tensor:
    out = None
    if 'bundle2' in state:
        out = run_bundle2(data['x'], data['config']['layer'], data['config']['n'], state['bundle2'])
    return _b2_base_kernel(data, state) if out is None else out

V17_ROUTES = [{'name': 'qkv16', 'm': 16, 'layer': 'qkv_gate', 'module': 'v17_bundle2', 'owner_variant': 'push_d9_cap6'}, {'name': 'o32', 'm': 32, 'layer': 'o', 'module': 'v17_bundle2', 'owner_variant': 'direct'}, {'name': 'pooler32', 'm': 32, 'layer': 'pooler', 'module': 'v17_bundle2', 'owner_variant': 'd9_cap2'}, {'name': 'unpooler32', 'm': 32, 'layer': 'unpooler', 'module': 'v17_bundle2', 'owner_variant': 'decodediet'}]
SELECTED_ROUTES = [r for r in SELECTED_ROUTES if (r['m'],r['layer']) not in {(q['m'],q['layer']) for q in V17_ROUTES}] + V17_ROUTES


from concurrent.futures import ThreadPoolExecutor as _PreparePool

def prepare(weights: Weights) -> dict:
    state = {name: w.contiguous() for name, w in weights.items()}
    device = weights['qkv_gate'].device
    hi = weights['router'].to(torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = (weights['router']-hi.float()).to(torch.bfloat16).t().contiguous()
    state['router_weight'] = weights['router']
    torch.cuda.synchronize(device)
    jobs = [
        ('sm', prepare_small_m, weights, {}),
        ('m32', prepare_m32, weights, {}),
        ('wide2', prepare_wide2, weights, {}),
        ('router', prepare_router, weights, {}),
        ('qkv', prepare_qkv, weights, {}),
        ('router2', prepare_router2, weights, {}),
        ('olayer', prepare_olayer, weights, {}),
        ('cps', prepare_cps, weights, {}),
        ('new_lt', tt_lt_prepare_cublaslt, weights, {}),
        ('new_mw', tt_mw_prepare_m32w, weights, {}),
        ('v10_sm2', tt_v10_sm2_prepare_smallm2, weights, {}),
        ('v10_warps', tt_v10_warps_prepare_warps, weights, {}),
        ('v10_tune', prepare_v10_tune, weights, {}),
        ('v10_m32x', tt_v10_m32x_prepare_m32x, weights, {'include_unpooler':False}),
        ('v11_m32x', tt_v10_m32x_prepare_m32x, {'unpooler':weights['unpooler']}, {'include_unpooler':True}),
        ('v12_smtrace1', tt_v12_smtrace1_prepare_smtrace, weights, {}),
        ('v12_o1', prepare_v12_o1, weights, {}),
        ('v13_smtrace8', tt_v13_smtrace8_prepare_smtrace, weights, {}),
        ('v14_qkv', tt_v14_qkv_prepare_qkvbig, weights, {}),
        ('bundle', prepare_bundle, weights, {}),
        ('bundle2', prepare_bundle2, weights, {}),
    ]
    def run_job(fn, selected_weights, options):
        with torch.cuda.device(device):
            return fn(selected_weights, **options)
    with _PreparePool(max_workers=4) as pool:
        pending = [(name, pool.submit(run_job, fn, selected_weights, options))
                   for name, fn, selected_weights, options in jobs]
        for name, future in pending:
            state[name] = future.result()
    if state['qkv'].get('large') is not None:
        state['qkv']['is_enabled'] = True
    for routes, fn in ((V9_SELECTED_ROUTES,candidate_v9),(V10_ROUTES,candidate_v10),
                       (V11_ROUTES,candidate_v11),(V12_ROUTES,candidate_v12),
                       (V13_ROUTES,candidate_v13),(V14_ROUTES,candidate_v14),
                       (V15_ROUTES,candidate_v15)):
        for route in routes:
            m, layer = route['m'], route['layer']
            weight = weights[layer]
            n = 1024 if layer == 'unpooler' and m <= 8 else weight.shape[0]
            data = {'x':torch.zeros((m,weight.shape[1]),dtype=torch.bfloat16,device=weight.device),
                    'config':{'m':m,'n':n,'k':weight.shape[1],'layer':layer}}
            fn(data, state)
    torch.cuda.synchronize(device)
    return state

v18d_SM_CUDA_TEMPLATE = _zsrc('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')

v18d_SM_CPP = '\n#include <pybind11/pybind11.h>\n#include <stdint.h>\nint v18d_native_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,\n              int S, int64_t t01, int64_t u01, int64_t stream, int query);\n'

v18d_SM_LAYER_K = {'qkv_gate': 8192, 'o': 1024, 'pooler': 8192, 'unpooler': 4096, 'router': 8192}

v18d_SM_ESC_CAP = 32

def v18d_v18d_native_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ', '.join((str(v) for v in (inst if len(inst) > 7 else inst + (0,) * 7)))
        cases.append(f'    case {i}: return v18d_native_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01, stream, query);')
    return v18d_SM_CUDA_TEMPLATE.replace('@@CASES@@', '\n'.join(cases))

def v18d_v18d_native_build(instances, name='dd_cap2_b_small_m_gemv', extra_cuda='', extra_cpp='', extra_funcs=()):
    return load_inline(name, v18d_SM_CPP + extra_cpp, cuda_sources=v18d_v18d_native_cuda_source(instances) + extra_cuda, functions=['v18d_native_launch', *extra_funcs], extra_cuda_cflags=['-O3', '-lineinfo'], no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def v18d_v18d_native_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 65535
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def v18d__to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def v18d__to_i32(v):
    return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

def v18d_v18d_native_bf16_bytes(v):
    r, c = v.shape[:2]
    words = v18d__to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def v18d_v18d_native_c12_bytes(v):
    """Packs units into the 12-bit format. Returns (bytes [r, c, 1536], base, escapes (r, c, lane, value, xor7))."""
    r, c = v.shape[:2]
    dev = v.device
    lo = v & 255
    sign = v >> 15
    eh = v >> 8 & 127
    hist = torch.bincount(eh.flatten(), minlength=128).cpu()
    window = torch.stack([hist[b:b + 8].sum() for b in range(121)])
    base = int(window.argmax())
    d = eh - base
    inwin = (d >= 0) & (d < 8)
    d = torch.where(inwin, d, torch.zeros_like(d))
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    code = ((d.view(r, c, 32, 4, 8).long() << nib) + (sign.view(r, c, 32, 4, 8).long() << pos)).sum(-1)
    code = v18d__to_i32(code).contiguous().view(torch.uint8).view(r, c, 32, 16)
    lo8 = lo.to(torch.uint8)
    data = torch.stack([lo8[..., :16], lo8[..., 16:], code], dim=2).reshape(r, c, 1536).contiguous()
    er, ec, el, ev = (~inwin).nonzero().unbind(1)
    xor7 = eh[er, ec, el, ev] ^ base
    return (data, base, (er, ec, el, ev, xor7))

def v18d_v18d_native_c12_unpack(data, base, esc):
    """Inverse of v18d_native_c12_bytes, used to prove the packing is exact."""
    r, c = data.shape[:2]
    dev = data.device
    p = data.view(r, c, 3, 32, 16)
    lo = torch.cat([p[:, :, 0], p[:, :, 1]], dim=-1).to(torch.int32)
    code = p[:, :, 2].contiguous().view(torch.int32).view(r, c, 32, 4).long() & 4294967295
    nib = torch.tensor((0, 8, 16, 24, 4, 12, 20, 28), device=dev).long()
    pos = torch.tensor((7, 15, 23, 31, 11, 19, 27, 3), device=dev).long()
    d = (code[..., None] >> nib & 7).reshape(r, c, 32, 32)
    sg = (code[..., None] >> pos & 1).reshape(r, c, 32, 32)
    val = (sg << 15 | base + d << 8).to(torch.int32) | lo
    er, ec, el, ev, xor7 = esc
    val[er, ec, el, ev] ^= (xor7 << 8).to(torch.int32)
    return val

def v18d_v18d_native_table(base):
    return base * 16843009

def v18d_v18d_native_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [v18d_v18d_native_units(hi), v18d_v18d_native_units(lo)]
    else:
        mats = [v18d_v18d_native_units(w)]
    if fmt in (0, 2):
        data = torch.cat([v18d_v18d_native_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {'data': data, 't01': 0, 'u01': 0, 'esc': None, 'fmt': fmt}
    parts, tabs, escs = ([], [], [])
    for h, v in enumerate(mats):
        data, base, esc = v18d_v18d_native_c12_bytes(v)
        if not torch.equal(v18d_v18d_native_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(v18d_v18d_native_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple((torch.cat([e[i] for e in escs]) for i in range(6)))
    return {'data': torch.cat(parts, dim=-1).contiguous(), 't01': tabs[0], 'u01': tabs[-1] if fmt == 3 else 0, 'esc': esc, 'fmt': fmt, 'n_esc': int(esc[0].numel())}

def v18d_v18d_native_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return (gx, wr * wk, gx * s * wr * wk)

def v18d_v18d_native_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk['data']
    R, C, B = data.shape
    geo = v18d_v18d_native_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = (gw // nw, gw % nw)
    by, bx = (cta // gx, cta % gx)
    wri, wki = (warp % wr, warp // wr)
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, v18d_SM_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk['esc'] is not None and pk['esc'][0].numel():
        er, ec, el, ev, x7, half = pk['esc']
        bx_e, rem = (er // (wr * rt), er % (wr * rt))
        wr_e, j_e = (rem // rt, rem % rt)
        by_e, rem2 = (ec // (wk * ku), ec % (wk * ku))
        wk_e, s_e = (rem2 // ku, rem2 % ku)
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > v18d_SM_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = (g_e[order], seq[order])
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = seq_s.long() << 18 | el[order].long() << 13 | reg.long() << 8 | (ev[order] % 2).long() << 7 | x7[order].long()
        esc[g_s, rank] = v18d__to_i32(entry)
    return {'wp': wp, 'esc': esc, 't01': pk['t01'], 'u01': pk['u01'], 'fmt': pk['fmt'], 'max_esc': int((esc != -1).sum(1).max())}

def v18d_v18d_native_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.v18d_native_launch(inst_id, sp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(), sp['esc'].data_ptr(), m, n, k, wr, wk, ku, s, sp['t01'], sp['u01'], torch.cuda.current_stream().cuda_stream, query)

v18d_SM_TABLE_M = {('router', 1): ((2, 1, 1, 1, 2, 0, 256), (1, 4, 4, 8))}

v18d_SM_TABLE_MT = {('pooler', 1): ((1, 1, 1, 2, 0, 0, 256), (1, 8, 16, 1)), ('qkv_gate', 1): ((1, 1, 1, 2, 2, 0, 256), (1, 4, 8, 4)), ('o', 1): ((1, 1, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('o', 2): ((1, 2, 1, 2, 0, 0, 256), (2, 4, 4, 1)), ('unpooler', 1): ((0, 1, 1, 2, 2, 0, 256), (1, 8, 4, 2)), ('router', 1): ((2, 1, 1, 1, 3, 0, 256), (1, 4, 4, 8))}

v18d_SM_MAX_M = 32

v18d_SM_TABLE_MT['pooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (1, 8, 16, 1))

v18d_SM_TABLE_MT['pooler', 4] = ((1, 4, 1, 1, 0, 0, 256), (1, 8, 16, 1))

v18d_SM_TABLE_MT['qkv_gate', 2] = ((1, 2, 1, 1, 2, 0, 256), (1, 4, 8, 4))

v18d_SM_TABLE_MT['unpooler', 2] = ((1, 2, 1, 2, 0, 0, 256), (2, 4, 16, 1))

def v18d_v18d_native_mt(m):
    return 1 if m <= 8 else 2 if m <= 16 else 4 if m <= 32 else 0

def v18d_v18d_native_rows(layer, m, n_full):
    return 1024 if layer == 'unpooler' and m <= 8 else n_full

def v18d_v18d_native_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def v18d_v18d_native_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if v18d_v18d_native_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def v18d_v18d_native_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 0.0001 if is_router else 0.01
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if v18d_v18d_native_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or float(rows.max()) > tol or (float(rows[m - 1]) > 0.00025):
            return False
    return True

def v18d_v18d_native_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def v18d_v18d_native_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == 'm' else 8 * inst[1]
    rows = v18d_v18d_native_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = v18d_v18d_native_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = v18d_v18d_native_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ('verified', inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = v18d_v18d_native_verify_onehot(ext, inst_id[inst], sp, v18d_v18d_native_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and v18d_v18d_native_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)

def v18d_prepare_small_m(weights, table_m=None, table_mt=None, name='dd_cap2_b_small_m_gemv'):
    table_m = dict(v18d_SM_TABLE_M if table_m is None else table_m)
    table_mt = dict(v18d_SM_TABLE_MT if table_mt is None else table_mt)
    entries = [((layer, 'm', m), v) for (layer, m), v in table_m.items()] + [((layer, 'mt', mt), v) for (layer, mt), v in table_mt.items()]
    specialized = []
    for key, (inst, cfg) in entries:
        if len(inst) == 7:
            layer, kind, m = key
            fixed_m = m if kind == 'm' else 0
            rows = v18d_v18d_native_rows(layer, m if fixed_m else inst[1] * 8, weights[layer].shape[0])
            inst = inst + (fixed_m, rows, v18d_SM_LAYER_K[layer], *cfg)
        specialized.append((key, (inst, cfg)))
    entries = specialized
    instances = []
    for _, (inst, _) in entries:
        instances.append(inst)
        if v18d_v18d_native_bf16_twin(inst):
            instances.append(v18d_v18d_native_bf16_twin(inst))
    instances = list(dict.fromkeys(instances))
    st = {'plans': {}}
    if not instances:
        return st
    ext = v18d_v18d_native_build(instances, name)
    st['ext'] = ext
    inst_id = {inst: i for i, inst in enumerate(instances)}
    packs, streams = ({}, {})
    for key, (inst, cfg) in sorted(entries):
        layer = key[0]
        plan = v18d_v18d_native_plan(ext, inst_id, packs, streams, weights[layer], layer, key, inst, cfg)
        if plan is None and v18d_v18d_native_bf16_twin(inst):
            plan = v18d_v18d_native_plan(ext, inst_id, packs, streams, weights[layer], layer, key, v18d_v18d_native_bf16_twin(inst), cfg)
        if plan is not None:
            st['plans'][key] = plan
    del packs
    return st

def v18d_run_small_m(x, layer, n, st):
    m = x.shape[0]
    if not 1 <= m <= v18d_SM_MAX_M:
        return None
    sp = st['plans'].get((layer, 'm', m)) or st['plans'].get((layer, 'mt', v18d_v18d_native_mt(m)))
    if sp is None or n != sp['rows'] or x.dtype != torch.bfloat16 or (x.shape[1] != v18d_SM_LAYER_K[layer]) or (not x.is_cuda):
        return None
    x = x.contiguous()
    if x.data_ptr() % 16:
        return None
    out = torch.empty((m, n), dtype=torch.float32 if layer == 'router' else torch.bfloat16, device=x.device)
    if v18d_v18d_native_launch_raw(st['ext'], sp['inst_id'], sp, x, out, n, sp['cfg']) != 0:
        return None
    return out

V18S_CUDA_TEMPLATE = _zsrc('eNrtPP1T20iyv+evmOQKTsKKY9kOEIzJEpLdSxFICsNmX/FYlSzL2EGWHEn+Isv/ft09M9LoyyZ7udt7VS8pYDTq6enp6a/pmdGTv419x5sNXHbozAa2Fc78eDxx66OjJ7k3/aG5m62O4sHYj7N1tncbhON4NFEr49XUteLQHsdRBm0QTN3Qjsdz17oNg9k0QlxP/jZwh2PfZb2z1ltmWQN3PnZcy4LiMAgdd+x78NaynjwhAKCgzebmfmR5g4XmBH4UU90Om+rs2xMmADpQsqMJmwcedOi52jNvUL/1gr7t1Z3b+rxdj1pN9m2rYWyZxlbT2Go9GOx6q33TeQYtM/8O2LNu+Eyb15e6IYurtHifFhc6AnvPtKmuIwGhG89CH4l5ENTPVPKXgvx5MB4k5M820+87SP+siv4fTnCracWSZlfQLOszdHPADaQj3UA2EGuqxH4nJdNwEmtJlW2kr/tKOcrRNhC0ac+wfb0vSAEWMuAh22qlBA2IICjYyC742xd/owyBg5RAnEZO3GRia0MvsGOmbTv6dfvGYHmuZShuqA8mp5moBET1aAVMs73xre8O6hNz19+/A8UMg0XdCbz6sNWsk6bSr2FRJr5ttY2tl8bWrrG1R4/7xtYrKihgidDDkGvDZ5pz3bjB4fKyqZSbSrl1o6fNkFOiFRXNtNhMiy1Z7DdkwUR+Ag9fvGD7bOGOb0dxxIIhC8AkwEhZFLvTA+YFC9ZfxW4E+s0+NJg2t70ZPDWet3Rm+wP2wUzq2s/3AHn7eX8cA98HvM1JPT9LA9dJJeiDOgcfTOXhRClfqlCXChRMaE7UwiC2YzcVuGg0rIf1RWhPs2IHP+1U6kQrKXsnXNaywoO8sEbAp652wrZZY7m/x//P9NplowQeYUUDgb+8Gc5fN9UuYInsyWgsX5rtxkznYGY12F5rtynBmhkw00joyKFrrYGT+EBAMq7BD6RfUGbUd+duqKXmNGeGwCNNO8/0DuPSdo7QLLbvXN9g/VnMpvHSjphj+34AD2Ewd9k4PmCuP4hYPHJZ347GDut7gXPHogCmwYYX4yiauWwxckP4DZ4wdv2CpAXhwA2tKZAca/DDJkbR5tNMxUXbOQnmicnc2nKw81RcYi4f4yHTtBgmNZFnfaKzbhcmeW/I/810lUVTvcquRhN3YkF/BdckDV7ahWU589i2bl3fDceOFQdWNLJDd4DYWTn20PbvtLyibB7zDGxAaDmxje0VZclY4rBqSBN7ape7CkRX4SFUaqB9nQ/t4EAQk5JHrmON26A+NnuMvh1aY3+s+DSoUUh1AgjUEs+gUgdw4RgowtYpmbFd7++2wck2bpBIJPCAzK0dCsIERqAWpjwIV8/0SsLc5dR1KkmLl2vpwj9zt85xWPGyQKNlrCMTsW+gcej6YBWIe0Wlp5d1SQ1B1UPXc+3IrYvJpF7VPkB6c104nsXHUdKDHKgUDTFg6MRegssWrvtxnSzs0kHku0CwdYjzmKPYsjGOaGuK7BuMYhQINlWVQM6XTGcE/VEkklcEyVmoCSZTz40h8l8eHJCvxjAVQxI+vd/SMAt+2hSvviyLtw+43qAADJMwdphEscMkiB3yGFYoGlFeKir9IPAUcY7DVUaWcyYguCvagG//6zOY0VtWn8Lo2bQDz4mEAz6auPrUBgewqtvO19k4TMSrIPBTHvwarAFoCsN/xiLXmyYGBngGv7HDh8TIBHfSyhTHnNiZ4I49BQ+QtYq7bWkVb3HZp9hielPlf4Bobot5GI9NQ2HzvMQLiX5jGc29AQMzgHEzm/XDALwsAwfsR7YTjwOfGzQWjYJFxGZT6BTAFmHg3wKeaObhog6YZw8wFLTZyPZvC24VeW4BB0omEz1idsLxlS5ozMZIfOQN1s0wBmEWI2AB056WYKJZS3qRzHyOeI5Ys8H/zTxPh1gF4w6t1HLZg0GbrxTa2+iZOFGyok+GoL5ktS7r15cdKK94eYXle16+x/KClxep8kvdJvUfDT1rGYRtLdsBKD6GIwHnmZi/CQREFofQrKSphcqvNZZDGU9ACFNfGtDWYBugVgSVF3P+b0PT+0d1sECoUv5GcRC6MAp/bvVpSObuDqofX5zNkb87U5h4iw+4KYGs0IcVaYmJ5ggJqhTRnCYgdsESQpzNDpG9P59dHsEqJpw5MetNWj9PQENSiNyrw8YR4IogSodQk2YLHGdI03T+ATpoG+z8GP6a0NE6NOZ6NK1HommuR7Mv0DQ3oGmtR7ObQVPCvpKlW5CkIihhom0v9GvRG7Y4ODj/cKP4NW37OP/+mO0wc/dmzfKuTGaT91cq8JWZ2J10fBp0RFE4++MPJspNhPvbNLRvJzab+WHgedTLMAB4HK7H0GjDn0OWG06H1Woe7wb/HV+3YQAeq7HGDbRJ4/LFtXcD1qIAZpaArYpgzRKw+yJYqwRswcEe4PcDc73I3TDWER/rqDDWYxzrKB2rmuabQaMFEgCEjDoZgDQkRWs+Yq9hksBJ4bTGpqwxscaUDdNMACx+0aDR3xX+bdJzTI0Ndowc3MU+keN6eft70X4h2q8q2rfL2puifzPt/76i/X5F+3vRXva/qGhvNnVlqoS/fhc59tSF5S44uQMIQr6yw0Nm7rM/mGf7Lj204AGioDGGNViBL0e2N8TyHpT3KOfy28cL9Nq4ZMZVPOVt/h4x0InAB+wMYKJ6TsvPLwpKPrVjZ6RlNfj8QtVXl0tIbvIxTAPRdNnRETAKVsUtc1aSFVmGHApTIXs/w/oYxqDtA3M03nQPm5pQj1YC2FUQ5ESMQy7GwBEYBkpumIYHRIyqKBCgHF+HN+z3LlAgQ6XjwRfbQdYgpyMYAcQ9tqwTnjzygjgy2GIcj5hrOyM+K9FsMhlD0BSPwA8hCDA6mnrj+Pkpg3A0HttehteOZ0cRezcZxwV+Q4w6c9xMiLDDotDhQcL5e/73k/jTu7zgpXg84AU/Nggxc+EXn5mERePBEpgEoB0qAqPe8xKELXJdy5KYBUDVKKRRH0IQrP4S0pugn/IZmALiTzgBU12JryDKgVFca9H4HtYm+hQmFKmHmQYCbgQqpFmDZ4gnqEYqBVhtf5/FPHyFaBBm4OISBOJ5GCx4BZuCLizsEKKB3iVIBwg35huHHmYy6aUErbO3749/YRoEizhlge+t9ANmMj9gZ2fHBnbXxIclTy0ZrI1PNC0YMhugbjznhC0N1GaY1njF7tzQdz2DwYpBI/EAqYfpBiKCWTydgVxg1AL4IIxPAGzAfvoG+4xwpwZofY75NEqYihwXSBTqsOvDBJvNfdG2rSNVGcz1osvmIiH/Xoi/Pfn87i0vIEOEfF2eHf/Giydn4u+5+Hsq/n6+kAVZc3olCj3+9+MnCBYsiy9TZG7Qsjx75jsjq48rksiyNOpMyMitO5lrT/JexgJ1AMaMnRiwLKYyNM+FkhmopczqZWqRUcaToo/KgbmR0LOJZYe3giVp8TQtfr5Qykr96RUvr49m1ocvCQfYBDTq5Az8Jfw64ER1MgDnCHCOAOcAIEg9xcpTrDyFytNiq88XCAG/X9Pvg2Q0n6npZ2r7GRvLoZ1eEc4rQnqFWK8yaLORZC5kEnFlPrjItKXkwInZBDgRo5lKvNYqAf7H+w8fBfQRBq1FiPOPoNAAQvq+jWv484+/pc9NfAYdSGsgrL/4mD5CXP3u7NPl/ygodku6ufho9S4BUQrWaiIi6+TjW6VyFxcNH3OQoNAlGD9d9f6BbpGIg7EhJ0S5Bf5wmz3lhGMJCMY/RCiimkVo1T7O4ks09aup69sTcFAxZj4CfzBGG2Z7h8g9sXAysvoEk4PNEBdi2UHdAVShC9PrhlNYn1qOHcWH9PJIQ80iW+0u4bVPq0RMtND+ACXHQNPBdYNNjihTxhx4zzDDfX2D7UobgGlMkgIRZhc6hbCeRix9VppRSELejC1Bp4f8GKFTfz9YYoyH3oK7Qww0XhrclfMKDFcM8IkZPdkhpdhBC3/AjfPb8UTG+IpOfoZWBI4GcoFxCXW1xdByLu7k8wt4zrftYYc97K8HfdyG4wF2scqD/fKbovcvmEa9gUekIElpWKDtF2gHjXegox0gNP/6doFiR0NDJq0ACKBrLKlZ6tQOqnAI+eYhRfxp+yXyAF38ItSJwHwDp1HoDzgM8HcIf3qVh7/8FeHBvaB26jB8XAeBAQDbOPY1/mAwbYLBtY6zauqdEquPGSqZaEq2WrZJzgR8ZnMCwxqDTVYWSkdj+bP4N+vI+FKRyqeCOlOXopl9j6qdrqgEdmVXRq4oKGqlt7hyBQ0nAYZy2rgwKow0I0CXmI3XSsSraT32HLmC84cyUvrq80VCAcvvidB+g1n6XmxNEASnQogjTNgOGrlZvpm6WyBfPairKSXXLxdJJexEj8GtoC4nKDkiAc4c1xEi4ATRJs2tkZoDThWpGnMsvqKCTvNNNRoPuDfEogs0qnBxYBZhMDFAQmTzXwqt1Wa5WGaJGrFUOgcV2cGQEWrIPm2jC9ghT8T5gXmXxXXv8ub6An7OP9ykEtxmyzm9OYOfpnhhz2K+awodXW/fUOSOSxYev0QVmRElzP/Cw/wvEOaDRkOc/0Uvb1GeT0lSKItr7Pbm+svNtXeTTtxCgxlIx4+iJHkKtV84G4H9usrLghqip1fco9DKpl45PIVYnxPrA7FnNEBfVbqqiUNfArOXUo4myQci95O5A4PU1MGSSG1DUxfR9HYU9DBnnDH+zXVDYcxSW071KkAzB5hJT6Sqxf8+dNYmhCI+/ggTQjR+EAoUGA0WL5G+zuq1db241z4xWBIk5BtKY/n/tvL/jK3kyXbbca7Pj7nNQevSvuk8Nsko84r/BjNTorlVbZRWDm/lQKs2NnJ0Gt6ITBPoloO61agPc6HBMPAG1DJb7S/j0jABOnKnKXz13HAEYreluNOCU2dQ7jM3NGZmWT7mwxrDsCCSgr81WINdbrbvZeqf1yOCww7GaMJSeUVV7PElG44HQGBtdnqlsz4E3ncZOGX0OOpvj5io7xCJTd4nt6lAc/m7miHUwD0J3wRx7O9MeVxlH++zj4uMjUaGRMgk4MohcSKxpLw2D/3nrCrpp7ryyW2T9ElJ98uV9DvdX55McLVrpy+D3OXIXUC+j7hdEA3UMZd0bNka7jfwZ9bJIBBDK/MdzTxxP9o/jwsOOg2vZjnvbFDWtNoN83997tm7M9xN7nPvDQ8r8dDEh3vx0MKHRSmG9k13nmB4iQ8Swy4+SAx7+LAo5ee3csKQUXMUaHpSiMy9SAnOvUiJz72oGogKZt4og8q9SAeYe5EONvciP/An5eVEUY55FC12JTt/ziip26OU75LGxAD8+BuTgnhwFdN/V+ZfZTBKnNAGRV476HI1TDg7w72sjIUl1YLQp8a+FHSEDlOiJ8RF/D4do0QExV4YGwQllRwDz+h0iWY9xdcS+1GSTbTHdSgm/UjjcwTABarwH9isaVk9d9su7n60MJzc4MDdKQaWhTAhq6Xi6EsZI/LgD2vk/N/nWb8r4Mq0/MpbfpVBV+nU5h3N2dnxRlezMewsOwEj/T9O/XW6QQx/v96Aey+p5lv5Va/AJP5eftaGyS38smatm86TauK4sZU0pU9ESpn4lBr5dDUysTWMcxsizuXGSVJjqN0Z+d5KNSM7W5hh1rM9gYaBZ813B1z4s30+VCpBbsFbth7iUWd5PEq5e1Ui0ZrwhHFuXan4DhuEDecKKBclmCYTNN5JH5vwOEgfWxm1z1nMu4y5nLA/GG6hb1ZmRbV2hW5l+JQ/WD2oe0F6lBp/2vDzkk5XKi/FPYnyl01xqFS+fFYt/c8yDVvyJCo1pNOMNXlwuyYv/GDBkYXkcPedOHP6Vc+IhliX2agj8OPAz6AqE1Iy87RDk596JfdGAvAkp0v8RBrulNQoC77DpToTVloWzqVlR3xHXSM6tynolbcD0uV9XneRCD4wDJNfua29vVf9V/I+QdK9wcxkXz4dY7o7UsDMV/PhpnX8f8u6PFSX5LQD+JqpNgxf1FjB1mD1QR6wPHOYOJmNnA8RWUMEzKZMDyTGRl0qX7yRW0Jik6ZTAAmDBaZ+M3snF2/UFBff8/vjj2RfkHJduGXbFdsNm6bhu7z8n/Xx66avPESk/bkArZ1cjqHRV1QOFMmhNGGnsj2e7gBribqH4WSJAuJSDNE66r5QNlyU27Ov5VFUVFZUWi2VOz0nByBXSPshmxQji4JpkEtPbLHDztFYBAtDkWq9armiqnCV8eK7wiqXX7xgP888j59aOhDnfzS0+3zfrMZ8XU311dlJMJnaDt8RiJgGMKaeb3j5K4DHupir23rSHW0t4Bkea4i9pvsLdKRHnisqS6ZTlgYFDXvim694LENU8O1Zn3ahoXKLdvmIGFGDJ4SKjlQcC57zBeY3OilNB6HpSDMsih86f84KZcV4vRD7a0U4L8BkBWpMMFqIsP8IAUbxlXL4aLkD1uRETgmgOiXT6gjpeNTMlmVr9vGMdiIhyVEvKd6ZgE7Zlk4EYy8vFxA6xKlcXP6al4vLX/9r5ALlIf4XBOH2LxOAXIwkvU96KsXMjl0eW8TtjbKjI+L9kYab73qndM7P36db/Xio4bXiP2mnZJ/OLcmqs0uyY53/8Fq36EOeKjQDg9AY5DZvD4H+fEvk1HVmtHLGFjRh7+lE7CJMEybSDhc70Is5TdzCShDiKiblGuAq9wWgGyUxvHoiVLotvFYmy6ZSbirlVlbMLMqL8OM4kbYmJyWFLa9c6llZZJ5BR2PxwA3+pdOweBK2ykyVb0MC41Uzl5onCplLzNZDVnfKFtxSF5xohqf5SCVSfTz/XBDgHzlK7PQaz9eKSzEFo5P/FEhyo5A/Mwe9eBlYPBrjviPBanm0Hn0uQas2cbSyxZwcgFLEm9vRzTfA87vQgtRA7L++YL0qaBdvGWB2H5mkQWc1ns4HLLkeciejwzsORYY3OSTtmiWHpPMTTIn/HRjOxJ7KA200Jpr3Gp+aRiHqzKzZTb5m74l0WObs9CbUuTWwPEydldcn1bmT0jl7WLe3pJpqJ7O84ea53KyjP5CWyiDHnlipTokHCV1n/n0eBELey5HLPDAsbvj3iAULPzmCj5lTcSab81GcnO7QWW7oZRLELju5PI7YdBaN+Cn/5M7tJhvVLLVRyXXiEnfNXcGmxHmonFajW/R0dKChdx6Zaf1Of/ede4GFCfZFCIxOKa5IE2qxXzH2PDqx8lDP/BX8YInzIodX5i5jP1m3lOYwU9EbRDx2AQFM7TU/Z5I5pYjST0rYWTOUijsV/4IHzfIzOf4CFHHC5+XApMa52+ipZGVPRAIiHQXNwCuqYXo4cn0Odt1WhELsUyI2u6zdGBuw3JXjIk2qx8S5Q0/pGHTclbtNR/GbqrN/vCdPotqMZPS4KKzT8r9EXSsUMNHQ0nBUDUqzcSgdE/ph2phQkSpledj5Q4PPvExuFrgKI/5oQ70ppnOMEul8xBJ7o70S5qrSRj3SdvxIq/FQ4gn/CovweGNQuakkLqiJK9U4VykrgmkMESifLhUA2PvclJ8LmoP2JUsbfD1w5+kRMfwA4C9u/Ja+t6Rtwzs9fcFr4fVxHIfj/iwGCBibeIWVZ/ayR1HOGX2W4pMbvsFQ+yMSZjCJ7SH/mTcIoL7O3HAFC3x+YQvjI1PARKCDS4Yfj5i7TIT/EdNESTTQGdgjCuv5tbzeWYc1EwShexvtmI1GowYQtse/J/ZfdnctP6Pi5hp9N4rupOBdNFlepkXagZEP6WUyQZ+4ISYvkMnLY/LimFFl92jcKd64YaYPM/UhikFDRG80h+IbMpi/w+uJycc13Mn8kFiMP8haZCuxlLNTsBLZiCxE9hHriG3EMmQXskpKMhcZNQpGSfwZ2JbIZ8SGdnL2UZOvVQmOtO2hbRCpOpoDhOnNHMeNImkYQHmEdojnoV33Z5MLECrMk4FYofXDpSVdeYJV4CE3hUcaQJLE9eD5DQqdwV69epUoAW3znZzhnsoEOz85owSOdnKOVedUdS6qTrHqlKpOdYW0ZvIdMLwvhLszFwT0+UI0/Hwq9myo9lSiuyJ8VxzhlajtYWWP6nqFTqquG9GVJcoq0CW0sn0npA/qDvkNO6BFlKDpERN3MpPOZINztgXY0hfpSIXw6TzEoms7/MYCEH6aNmglDXqX+KolhrxFZ1KheygfUlm2aMsWE0nhBAikPMl+CvUy5QbdnaOMBa7Ingo7Skk5qJXMNNO2r+R7/lrrQUd02a4HHZm7eg7yMfeQpNQVLiO9YPyugrwmMmEqfZq8k7iv3htSsl8ED+wFupTkYC599NLEm2lCP+SJSS3N0RKSZClD17hQMnMoyps3c835BbL82XWlbbazHvZ1yo/NlzV6yLpJ8DLWYOVnnKWsU1zmGhuzyb5wqLSfoud+zu3IUH52LrEb2Q56qgtG1EaRLHDFb1e+PRk7qkdGhIYkQVdMo6d/F/bzwP8UhLHd99wT7ogR87HnBQt3IK8hPOT1NUJtkp0ngs4vuEIPH8jlnQT+cHyLX7vEFM+3h+zbhAJmx9f8CJLn1MXlQ4AfjCctDS8oXryhgItTAiDy8qSEwfjLVN4PBLNgOojjABdNxDvu5FC2yD3QEw6HCqlCSxEByho3dbr9WUK44BeQ4vrROBBxJ28zx48UJ+/rS8rQVb9f4RXnNe/v0/YwChsowGHZcVIFbuxY1JrqjCWu1VQjxVyc+C4Mg9Di56vBmr2myo+OM5vavrMC+TummE2MN6JYkYvTtufoVZHHQQUaCiQjiCjPZl48noYBuuggVLDijEYTPRPIfrCjmOiUIbwQOjqiqfh6oH4OXT83EzakM3dKX1l4t9SAbNlXITuoXOc70hZ4LLsCJHtG/EhbVn0hq9CevjFAt5+rsScfGTjC+zV4sNzAIBC8NMZS6K0rgz7llCyEe3RKSJ4gBTzqdSR5TVpHONwQbWYBZoXmG3tMcc4SnOq35YrTWZxA/OLP831aUOSiaL6mBf6kcet/Ipr+V4PoaDGOnRG6hIg2AX766eS4967300/Uz8Ad2qAKB2nA8Cr5fskP+dZ7sj6i5dGZXPHIRVHxcxt82QssdEMrgqgX1sZKKjL/aY3HfVlj7YnWJBeRa7YS06Z82yL5WsRpd998xUNJfke/m7mhHwaLbnogil/Q7wLUNp4PwX1MfDg6ohAwvZZ2RlcjuK8SaZMFXZDtiI8sNtlyya/FUhW/EDvoZi7DkpBKe7sQV1W7FYqe7EYsakDyzql+PRany8qvxFE3cbfRiQ/POnGtpp5D1eJDYBUQyK92VndKI0GbVYvVHikrgcWyLyXx8UHH0WHvshNhzx7eLQA21qKdc363oORWmWzatyOagA4WDk9ftKlQ6/YusbEYxZoxZ7vOnEFss3kX+Zwcy+MzZYtJqkBcwU4b+YY8RAamZ+g0olaLarDMAIJxAIIB2Rf69/coJy7NzmVUqcnuG92dwkxmYWAyt5Hy+lLJP+XRmI9Hs1LQ8C8rsmGja6WQTV6r3Td0Y2iWvlGPmaByxXRzaTixh/hJWGPYgF9JfRHWTGBXCLsykvoibDOBvQdqUrzNEthWArtA2ASvkulVk3SZr4xwSxHNJhGaivPLFy1+Nb5SYdZMtDhK1NUSHtTSEda0ZAS1PH1rlGTQhXX6oAOGrQsR37zWFTdJBsHCz18nMebGIFk50IGPbhdXizg46A5t5E133kmZkM+jQisypN3GI9R3rcRLViARh8TV15IMOk5y0EiPyK7Vqz85/iwHVtfxTvPlLtpjwQAlYZtEJMI/poEJD0eMNBhRYxEefeTihKpcW7fogA9NA5bXRlMkzXhg3+0+LmO2JmFG69mn3e/LlvEU7GMzZdlEmTZ52jVzKRgk4vDwEHhOg2wYZcuzo6MjLROG6AtDK4s89KUhvu+qI+cn68PPrjr2140DHnj+E69Bugk=')

V18S_CPP = r"""
int v18s_router_launch(long,long,long,int,long,int);
#include <pybind11/pybind11.h>
#include <stdint.h>
int v18s_launch(int inst, int64_t wp, int64_t x, int64_t out, int64_t esc, int m, int N, int K, int WR, int WK, int KU,
              int S, int64_t t01, int64_t u01, int64_t stream, int query);
"""

V18S_SIGN_POS = (3, 19, 7, 23, 11, 27, 15, 31)

V18S_ESC_CAP = 32

def v18s_cuda_source(instances):
    cases = []
    for i, inst in enumerate(instances):
        args = ", ".join(str(v) for v in (inst + (0,) * (15 - len(inst))))
        cases.append(f"    case {i}: return v18s_launch_t<{args}>(wp, x, out, esc, m, N, K, WR, WK, KU, S, t01, u01,"
                     " stream, query);")
    return V18S_CUDA_TEMPLATE.replace("@@CASES@@", "\n".join(cases))

def v18s_build(instances, name="v18s_gemv", extra_cuda="", extra_cpp="", extra_funcs=()):
    return load_inline(name, V18S_CPP + extra_cpp, cuda_sources=v18s_cuda_source(instances) + extra_cuda,
                       functions=["v18s_launch", "v18s_router_launch", *extra_funcs], extra_cuda_cflags=["-O3", "-lineinfo"],
                       no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)

def v18s_units(w):
    """BF16 [n, k] -> int32 [n/16, k/64, 32 lanes, 32 values] of 16-bit patterns in A-fragment order."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    v = bits.view(n // 16, 2, 8, k // 64, 4, 4, 2, 2).permute(0, 3, 2, 4, 5, 6, 1, 7)
    return v.reshape(n // 16, k // 64, 32, 32)

def v18s_to_i16(v):
    return torch.where(v >= 32768, v - 65536, v).to(torch.int16)

def v18s_to_i32(v):
    return torch.where(v >= 2**31, v - 2**32, v).to(torch.int32)

def v18s_bf16_bytes(v):
    r, c = v.shape[:2]
    words = v18s_to_i16(v.view(r, c, 32, 4, 8).permute(0, 1, 3, 2, 4)).contiguous()
    return words.view(torch.uint8).view(r, c, 2048)

def v18s_c12_bytes(v):
    return b2q_tt_v10_m32x_m32x_c12_bytes(v)

def v18s_c12_unpack(data,base,esc):
    return b2q_tt_v10_m32x_m32x_c12_unpack(data,base,esc)

def v18s_table(base):
    return b2q_tt_v10_m32x_m32x_table(base)

def v18s_pack_format(w, fmt):
    """Units of one format for a whole matrix, or None if the 12-bit packing is not exact.

    Returns dict: data uint8 [R, C, NL*512], t01/u01 decode tables, escapes (r, c, lane, value, xor7, half) where
    half 1 marks the lo matrix of a HILO pair."""
    if fmt in (2, 3):
        hi = w.to(torch.bfloat16)
        lo = (w - hi.float()).to(torch.bfloat16)
        mats = [v18s_units(hi), v18s_units(lo)]
    else:
        mats = [v18s_units(w)]
    if fmt in (0, 2):
        data = torch.cat([v18s_bf16_bytes(v) for v in mats], dim=-1).contiguous()
        return {"data": data, "t01": 0, "u01": 0, "esc": None, "fmt": fmt}
    parts, tabs, escs = [], [], []
    for h, v in enumerate(mats):
        data, base, esc = v18s_c12_bytes(v)
        if not torch.equal(v18s_c12_unpack(data, base, esc), v):
            return None
        parts.append(data)
        tabs.append(v18s_table(base))
        escs.append(esc + (torch.full_like(esc[0], h),))
    esc = tuple(torch.cat([e[i] for e in escs]) for i in range(6))
    return {"data": torch.cat(parts, dim=-1).contiguous(), "t01": tabs[0], "u01": tabs[-1] if fmt == 3 else 0,
            "esc": esc, "fmt": fmt, "n_esc": int(esc[0].numel())}

def v18s_geometry(R, C, rt, wr, wk, ku, s):
    if R % (wr * rt) or C != s * wk * ku:
        return None
    gx = R // (wr * rt)
    return gx, wr * wk, gx * s * wr * wk

def v18s_stream(pk, rt, wr, wk, ku, s):
    """Stream-ordered weights and per-warp escape lists for one launch geometry, or None if a list overflows."""
    data = pk["data"]
    R, C, B = data.shape
    geo = v18s_geometry(R, C, rt, wr, wk, ku, s)
    if geo is None:
        return None
    gx, nw, G = geo
    dev = data.device
    si = torch.arange(ku, device=dev).view(ku, 1, 1)
    gw = torch.arange(G, device=dev).view(1, G, 1)
    j = torch.arange(rt, device=dev).view(1, 1, rt)
    cta, warp = gw // nw, gw % nw
    by, bx = cta // gx, cta % gx
    wri, wki = warp % wr, warp // wr
    tile = (bx * wr + wri) * rt + j
    chunk = (by * wk + wki) * ku + si
    idx = (tile * C + chunk).reshape(-1)
    wp = data.view(R * C, B)[idx].contiguous().view(-1)
    esc = torch.full((G, V18S_ESC_CAP), -1, dtype=torch.int32, device=dev)
    if pk["esc"] is not None and pk["esc"][0].numel():
        er, ec, el, ev, x7, half = pk["esc"]
        bx_e, rem = er // (wr * rt), er % (wr * rt)
        wr_e, j_e = rem // rt, rem % rt
        by_e, rem2 = ec // (wk * ku), ec % (wk * ku)
        wk_e, s_e = rem2 // ku, rem2 % ku
        g_e = (by_e * gx + bx_e) * nw + wk_e * wr + wr_e
        seq = s_e * rt + j_e
        if int(seq.max()) >= (1 << 14) - 1:
            return None
        counts = torch.bincount(g_e, minlength=G)
        if int(counts.max()) > V18S_ESC_CAP:
            return None
        order = torch.argsort(g_e * (1 << 14) + seq)
        g_s, seq_s = g_e[order], seq[order]
        rank = torch.arange(len(g_s), device=dev) - torch.searchsorted(g_s, g_s)
        reg = ev[order] // 2 + 16 * half[order]
        entry = (seq_s.long() << 18) | (el[order].long() << 13) | (reg.long() << 8) | (
            (ev[order] % 2).long() << 7) | x7[order].long()
        esc[g_s, rank] = v18s_to_i32(entry)
    return {"wp": wp, "esc": esc, "t01": pk["t01"], "u01": pk["u01"], "fmt": pk["fmt"],
            "max_esc": int((esc != -1).sum(1).max())}

def v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg, query=0):
    """cfg = (WR, WK, KU, S)."""
    if sp["fmt"] == 4:
        return ext.v18s_router_launch(sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), x.shape[0],
                                     torch.cuda.current_stream().cuda_stream, query)
    wr, wk, ku, s = cfg
    m, k = x.shape
    return ext.v18s_launch(inst_id, sp["wp"].data_ptr(), x.data_ptr(), out.data_ptr(), sp["esc"].data_ptr(), m, n, k,
                         wr, wk, ku, s, sp["t01"], sp["u01"], torch.cuda.current_stream().cuda_stream, query)

def v18s_rows(layer, m, n_full):
    return 1024 if layer == "unpooler" and m <= 8 else n_full

def v18s_bf16_twin(inst):
    """The same launch shape on uncompressed weights: the fallback when a 12-bit pack does not fit."""
    return (inst[0] - 1,) + inst[1:] if inst[0] in (1, 3) else None

def v18s_verify_onehot(ext, inst_id, sp, exact, cfg, rows_per_launch=8):
    """Runs every column as a one-hot input: each output must equal its weight column exactly.

    exact: [n, k] values the kernel must reproduce (BF16 weights, or hi + lo in FP32 for the router)."""
    n, k = exact.shape
    mm = rows_per_launch
    dev = exact.device
    out_dtype = torch.float32 if exact.dtype == torch.float32 else torch.bfloat16
    got = torch.empty(k // mm, mm, n, dtype=out_dtype, device=dev)
    eye = torch.eye(k, dtype=torch.bfloat16, device=dev).view(k // mm, mm, k)
    for i in range(k // mm):
        if v18s_launch_raw(ext, inst_id, sp, eye[i], got[i], n, cfg) != 0:
            return False
    return bool(torch.equal(got.view(k, n), exact.t().to(out_dtype)))

def v18s_verify_random(ext, inst_id, sp, w_rows, cfg, ms, seed=11):
    """Random inputs with a spike row against an FP64 reference, at a quarter of the judge's tolerances."""
    n, k = w_rows.shape
    is_router = w_rows.dtype == torch.float32
    tol = 1e-4 if is_router else 1e-2
    gen = torch.Generator(device=w_rows.device).manual_seed(seed)
    for m in ms:
        x = torch.randn((m, k), generator=gen, device=w_rows.device)
        x[m - 1].zero_()
        x[m - 1, (seed * 131 + m) % k] = 16.0
        x = x.to(torch.bfloat16)
        ref = (x.double() @ w_rows.double().T).to(torch.float32 if is_router else torch.bfloat16).double()
        out = torch.empty((m, n), dtype=torch.float32 if is_router else torch.bfloat16, device=w_rows.device)
        if v18s_launch_raw(ext, inst_id, sp, x, out, n, cfg) != 0:
            return False
        diff = out.double() - ref
        rows = diff.norm(dim=1) / torch.maximum(ref.norm(dim=1), 0.1 * ref.norm(dim=1).mean())
        if not bool(torch.isfinite(out).all()) or float(diff.norm() / ref.norm()) > tol / 4 or \
                float(rows.max()) > tol or float(rows[m - 1]) > 2.5e-4:
            return False
    return True

def v18s_exact(w_rows, fmt):
    if fmt in (2, 3):
        hi = w_rows.to(torch.bfloat16)
        return hi.float() + (w_rows - hi.float()).to(torch.bfloat16).float()
    return w_rows

def v18s_plan(ext, inst_id, packs, streams, w, layer, key, inst, cfg):
    """Packs, stream-orders and verifies one table entry; None if any step fails."""
    m_hi = key[2] if key[1] == "m" else 8 * inst[1]
    rows = v18s_rows(layer, m_hi, w.shape[0])
    fmt = inst[0]
    pkey = (layer, rows, fmt)
    if pkey not in packs:
        packs[pkey] = v18s_pack_format(w[:rows].contiguous(), fmt)
    if packs[pkey] is None:
        return None
    skey = pkey + (inst[2],) + tuple(cfg)
    if skey not in streams:
        streams[skey] = v18s_stream(packs[pkey], inst[2], *cfg)
    sp = streams[skey]
    if sp is None:
        return None
    vkey = ("verified", inst) + skey
    if vkey not in streams:
        w_rows = w[:rows].contiguous()
        lo = 1 if inst[1] == 1 else 4 * inst[1] + 1
        fixed_m = inst[7] if len(inst) > 7 else 0
        ms = [fixed_m] if fixed_m else sorted({lo, (lo + 8 * inst[1]) // 2, 8 * inst[1]})
        streams[vkey] = v18s_verify_onehot(ext, inst_id[inst], sp, v18s_exact(w_rows, fmt), cfg, fixed_m or 8 * inst[1]) and \
            v18s_verify_random(ext, inst_id[inst], sp, w_rows, cfg, ms)
    if not streams[vkey]:
        return None
    return dict(sp, inst_id=inst_id[inst], cfg=cfg, rows=rows)


V18_ROUTES = [
    {'m':1,'layer':'qkv_gate','name':'decodediet_qkv1','module':'v18'},
    {'m':1,'layer':'pooler','name':'p32_s1_o6','module':'v18'},
    {'m':8,'layer':'o','name':'warps_finalist_o8','module':'v18'},
]
V18_SMALL = {
    ('pooler',1):((1,1,1,1,0,0,1024,0,4096,8192,2,16,8,1,6),(2,16,8,1)),
    ('o',8):((1,1,1,2,0,0,256,0,8192,1024,1,8,2,1,6),(1,8,2,1)),
}

def prepare_v18_extra(weights):
    qkv = v18d_prepare_small_m({'qkv_gate':weights['qkv_gate']},table_m={},
        table_mt={('qkv_gate',1):v18d_SM_TABLE_MT['qkv_gate',1]},name='v18_decoder_qkv1')
    instances=list(dict.fromkeys(inst for original,cfg in V18_SMALL.values()
                                 for inst in (original,v18s_bf16_twin(original))))
    ext=v18s_build(instances,'v18_small_selected')
    ids={inst:i for i,inst in enumerate(instances)}
    packs,streams,plans={},{},{}
    for (layer,m),(inst,cfg) in V18_SMALL.items():
        key=(layer,'m',m)
        plan=v18s_plan(ext,ids,packs,streams,weights[layer],layer,key,inst,cfg)
        if plan is None:
            plan=v18s_plan(ext,ids,packs,streams,weights[layer],layer,key,v18s_bf16_twin(inst),cfg)
        if plan is not None:plans[layer,m]=plan
    return {'qkv':qkv,'ext':ext,'plans':plans}

def run_v18(x,layer,n,st):
    if x.ndim!=2 or x.dtype!=torch.bfloat16 or not x.is_cuda or not x.is_contiguous() or x.data_ptr()%16:
        return None
    m,k=x.shape
    if (layer,m)==('qkv_gate',1):
        return v18d_run_small_m(x,layer,n,st['qkv'])
    plan=st['plans'].get((layer,m))
    if plan is None or n!=plan['rows'] or k!=(8192 if layer=='pooler' else 1024) or x.device!=plan['wp'].device:
        return None
    out=torch.empty((m,n),device=x.device,dtype=torch.bfloat16)
    rc=v18s_launch_raw(st['ext'],plan['inst_id'],plan,x,out,n,plan['cfg'])
    if rc:raise RuntimeError(f'v18 selected launch failed: {rc}')
    return out

_v18_base_prepare=prepare
_v18_base_kernel=kernel_fn

def prepare(weights: Weights) -> dict:
    state=_v18_base_prepare(weights)
    state['v18']=prepare_v18_extra(weights)
    torch.cuda.synchronize(weights['qkv_gate'].device)
    return state

def kernel_fn(data: Inputs,state: dict) -> torch.Tensor:
    out=run_v18(data['x'],data['config']['layer'],data['config']['n'],state['v18'])
    return _v18_base_kernel(data,state) if out is None else out

SELECTED_ROUTES=[r for r in SELECTED_ROUTES if (r['m'],r['layer']) not in {(x['m'],x['layer']) for x in V18_ROUTES}]+V18_ROUTES

