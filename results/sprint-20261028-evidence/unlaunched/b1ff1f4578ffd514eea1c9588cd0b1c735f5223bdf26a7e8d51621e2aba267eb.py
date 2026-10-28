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


QKV_SM_CUDA_TEMPLATE = _zsrc('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')

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


CPS32_CUDA_TEMPLATE = _zsrc('eNrNPWtz27ay3/MrkJxxDmkpiigpjmPZbh037ek4djKW0/SOr6uhKMpiTJEyST3s1P/97i4AEiRBSU7ae+oZWSAILBaLfeKlJ//yAsefDV2278yGdj+aBYk3cRvjwyeFN4ORtZPPjpOhFyT5PNu/DiMvGU/UzORu6vaTyPaSOAc2DKduZCfe3O1fR+FsGiOsJ/8auiMvcNlpu/UT6/eH7txz3H4fkqMwclwv8OFtv//kCRUADDrMmcbtVt8fLgwnDOKEMrfZ1GRfnzBeYt6FlB1P2Dz0oUXfNZ75w8a1Hw5svxE4jffW3l4Q9m3fDx07cRvvW3t7rVc7bxvzTgNgs69bzfqWVd9q1bfaD3V2udW56j4DkLm/PfbsIHpmzBtLsy6Td1nyPksuTCzsPzOmpomYRW4yiwLE8kH0a5br2FJ0bB56w7Rjs416Bh2YVXXgL8cYkE1SpF2BtHyRQ5xKrsMdEQe8AVtLxfaxqEyjSWKkeXY9ez9Q0nEBuaFAzniG9RsDgQsQkQEV2VY7w2hIGEHCRoLB90B8xzkMhxmGOJACu8nENkZ+aCfMeO6Yl52rOivSLYdyU32wONKEJgBqxHdANtv3rgN32JhYO8HuDYhtFC4aTug3Ru1Wg+SY/o3KbPF1q1PfelXf2qlvvabH3frWG0ooxVLGhz7XRs8M57J5hf3laUtJt5R0+8rMqiGpRC1KWlmylSXbMjloyoSFBAUivnzJdtnC9a7HSczCEQtBYUBPWZy40z3mhws2uEvcGISfvW8yY277M3hqvmibzA6G7L2V5nVevAbgnRcDD9glHPI6x43SMA1dJ+Oh9+ogvLeUh2MlfaGWulBKwYjyccsPNHMm7IAds+esuXwt/mbdcrGxH0I5hbexIYTvTExd8bFXVZwdHjJrhyrheOSLYSehqTpg88rqNGcm+5MZxjHb32dWyyQsd5vNJn5mAoRVDeJ1e6elgtjVQ2gVIQCe0AEdEh09hHY1hAyH41JVYKqcsQlCaWlULgjcuRsZmRouaC8wctPuM7PLOIueYWmW2DduUGeDWcKmydKOmWMHQQgPUTh3mZfsMTcYxiwZu2xgx57DBmCEblgcAh/b8MKL45nLFmM3gv9gXBM3KLNnGA3dqD8FpBMDPmxSLxsL4oakrHMn4TxVtVtbDrae6baEqzBvBGRPgGqpEJgTkx0cIFVH/A8oqxJpalYq5HjiTvrQYsmqSU2ZNdLvO/PE7l+7gRt5Tj8J+/HYjtwhgmcV4CM7uDEK3Y7Wd3sGyiPqO4mN9TMCRDkdHlV2amJPbb2VQXgVxkVFB+o3eOf29gQ2GX5kdVZYHGpjA2MzsKO+F3iKPYQcBVcnBA8wNSoqelAu8gAlrJ3hmdiNwU4HLHTzCrFEDPdIU9uRwExABHRh1MPo7plZjZm7nLpOJW7JciVi+DV3GxxGP1mWkOzXV+GJ0NchOXIDUA9Ev7Lw08uGRIdKNSLXd+3YbYjxpGbVRoCFi204fp/3RNOE7KpkD9FlaMVegsEXhn/DVha2thvFNrDYKsgl0HHSt9EP6RiKBNQZ+Tjgr6qCgdTXDGkMDZInUxQHSVzICSdT300grlju7ZGtR08XXRo+xF8zPw0+HXJ5X+l89j0uPcgEo9QTHqWO8Cj1g0fcDRbiRphr2WUQhr7K00l0l2PogiYIb8qq4Ov/BgwG9Zo1ptB9Nu3Cc8rmAI+GrjG1wRjcNWznduZFKYuVuH7K/WewfwCm1P9nLHb9aapngGjwHxt8SHVNeCOVTbnTqboJb9hTMAZ57bjTSbXjNYaVilKmV1W2CLDmSpmHAlg1ErrPTy2SaDiR/uBb0DND6Diz2SAKweQysMZBbDuJFwZcr7F4HC5iNptCo1CMW3s2smc+xoxAPXuIziRapdnEC67ZLIBQ1IvHMAhDG+hZ4nUciD6QRTPCaDELbIDvTIF43lHj5GimnosgFxZajIEwzHiqg0WjmTUkifwCQR2yVpP/zXzfBI8GnRNDr9fs4bDDg5DOczRdHDGZMSAl0Viy2gEbNJZdSN/x9B2m73n6HtMLnl5kikGIvVAN45HfX4ZRx8i3AEoBnZaQU06M7AT8pj4vYfTTqn1UDEZzOZJeBzg6jWUd6tbZmlJ3VKooAfxvTdX7jRpYYCk9heMkjFzoRjDvD6hP1s42iiaP/OZI4e0pDH+f97glC/WjAAJenQbnEKmYFtKcxiBxQU/aicv2kcA/n14cQowUzZwE51d+noD0ZCUKr/abhwArTkAyHc4RYFojGqiz99BAp87OjuDbgoZWgbFWg2lvCKa1GsyuANNaA6a9GsxODoyGfLrAMEznOmhKxni+MC9Fc1hlb+/s/ZVi9oznR8X3R2wbArGrFcGjjm3T95/Uwp+sVAFlHTSgIXLX2Z9/MpFuYbl/TSP7emKDqotC36dWRiGUx/76DFU6fO2zQne6rFbzeTP4d3TZgQ74rMYoiMzc98WlfwUao1TM0hS7KxdraYrdl4u1NcUWvNgD/H9grh+7a/o65n0dl/p6hH0dZ31VpxhnUGmBCAAi426uQOa1olYfsx9gkMCC4bAmlsyxMMdKK2YTDRCMo1Kj7zv8btFzQrXr7AhJuIONIsnNCgD3AsBCALirANDRArAEBlaGwX0FgN0qAPcCgMRgUQHAapnKcAmD/i527KkLwTGYuz1wU25p5mEXwnffDlx6aMMD+EkeOj40qQDPY9sfYfo1pF/TtM7vH87RrGOAPfauxzQ19O+YgVyEAUBnUCZuFET97Lws6VM7ccZGXozPzlWhdbXTOujJAX+6ONtC0x5tSzets4x4KZr++RmiaeiEsQvUMXjV11jVgvxtPmdT4uaUlyPOy0AS6Aeyb5R5CoSMKi3grhxdRlfsjwPAQDpTR8MvtoO0QVLH0APwjGyZJ0x67IdJXGcLLxkz13bGfFjAcyLXKRmDNcIiQOl46nvJixMGHmvi2X6O2I5vxzF7N/GSMsHBj505bs5X2GZx5HBv4exX/v1RfPUuznkq8YY8ESR1gsxc+MeHJqWRN1wClaBol5JAqV95ChwYGQGzNGiBoqo70myMwFFW/wn+TcFP+RBMAfBHHIGpqbpa4O9ANy6N2LuHCMacwpAi+jDWgMGVgIVIG/AMngXlSLkA5R3ssoS7uOAbwhicXwBLvIjCBc9gUxCHhR2BV9C7AP4A/sZZzZGP86X0UhZtsJ9+PfqFGeA44qCFgX9n7jGLBSE7PT2qY3MtfFjyuag66+ATjQu61XWQOD5JhTXrKNAwsMkdu3GjwPXrDMIKgxgE+B4GHJAIZ8l0BpyB3gvAA1c/LWAD9JO32GaMq0WA6wucgaNpWTEpBjyFYuwGMMJWa1fU7ZiIVQ5yo2y6OU/I73Px3ZPP737iCSSIYLCL06PfefL4VHyfie8T8f35XCZkzsknkejx799F1QsB9fjj4ZN+n0c2cnKx3/ftWeCM+wMMYuJ+36C2Jc9cu5O58aRofPogH0Aoz0kAzGIqvfaCj5krtZSzgrlcJFz9Sdl0FYq5sRC8Sd+OrgWJsuRJlvx8rqSV/JNPPL3ayVnt1aQUYDRZfgpmFP7tcaS6uQJnWOAMC5xBAYHqCWaeYOYJZJ6Ua30+xxLw/wf6v5f25jNV/Ux1P2Nl2bWTTwTzEwH9hFA/5cDmPcyCJyX8zYLPUQc+vcDsC1ANF7/kQdEMw7HVgvfCk7MUr66tKfyfX99/EKUP0bctlzj7APIORUgdPMd5gLMPv2fPLXwGEclywPs//5A9gvv97vTjxf8oIHY0zZx/6PcuAFBWrN1CQP3jDz8pmTsYW3wolAR510D8+Kn3H7SbhBz0DSkh0m0ljYsFz9lT3glMAfL4RUgj2FmMCvDDLEGy44p1YE/AmiU4kxIGQw/Vne3vIyVFrFXPixqMG1ZDWAhlG8UKQEUujLwbTSGo7Tt2nOzTy0MDhY7UuruE1wGFljhxQ4sPNN0GWgDsPKjvmObemAPvGU6dX15hPW0F0KLpdEKMExN53l7yVTEg1++nKbV4sp0lX5nAych/26Det5lxwl4y47iH7N0D7r6OvOFP3qRxZ5LrByVakNvkOOV0T7zUESBf6NDALpl5PCN7ITE9/ggNgBxuo/xto6XbRjnaZq+sVr5SWiWtXZP97Ahh3ibJlVAw6Go1O7uIPV/kIL0aL3RY0zuOLMDlhJQrI0qYRfwkvYdspicNQXJKHN0P5LYx+le/DpfocE8X3C1Bj+9Vnaw4ZEDuS7IgyTV/2qIn8rh4cfQquwXwQZJTZbL3F7/AvzYOGhlVHMtlserZZ6gqyA5YoEdJqGwxtHiLG/n8Ep6LdXvYapldisV++V3Rz8Bh1BqNDLJ9VrGE2y9QDypvQ0Mwip+Lr6+RggZ1DWl6B4WgdI2lOUuT6kEWdqFYPaKALau/5PwHhSOTECxWcJql9oDMUP4Gy598Kpa/+A29SS8wMHwFMaszA1kK7BuuxppdjSXGWcV0bjBdPXtOEi4q5FabyPmULr/Cnagss0hWlFQXzWQcR5ECvcYpA1CUxKmQzmqXcETvPq5QxR1UKVnMYRg99gJ7jOOAY6199fk8xYeV1q9oacjSFxDLSFSEYyUYS6LWSRVcByPw3yimau3OSuDUBR/57iEX6yrLNTKE1RB+jbotzS2wG6QkKt6ebDYNLeYybpnjXEWqpqHGS7aLubkIpqBwcGIQhY2XNsFppJytNKdbqMX9RYhWoNwSNaqIWZCBwQRO5GCdqBJ2x/GpAXRALgPJgQ1jRCNGaAkOCpXcNcvFdTTUE07DjrFdlhhoWelgce3BmYplJ+darKKIvUhiTYkvo4Bfg6tgfH8PrRv5kIic0sqI4Bau+bdLBmUmLArilGKKW70QVpHX1lOiqishhOMJ319XWr0TwAnBPs1oc1MUG8p0jKJaJnd9NDfN5c/ib6ZTMuiZcg/LlDWyLWEQRChMBKqaLFGNLBk0WgpyFrdoaabFOkbmB7Rbpqivak5emMUYvQwRBVn9l1JttVrBgVk2C1wPun4bY1bIIdsLVheZH0cEHOs3oFbQm6DpGU4Z6gQ4JBF6l49xK4A5Z0mIM9XXbl/udDpgl8+vSAngzAoPqmIvE4PCQHxkh3mNjWo99kBtnHyqmPst6JovB80u+7IPRo/Val/M6hrqrDHW8fdLk8QVcgr8f1AQU0kvnPoysKfbZ59raKvN7fMLQGP77D0A3m63aoJtdA0IZy5yDha3NTl+sbctWKJmfEnBFAF8r1pAkV6nF1TpVlPfIcccyEN35Tz3LY7O7f4xzUrdmnkGM27rtwIcrWQsLnsXV5fn8Dl7f5X5GR22nNObU/i0rhR2xckgPZdW8lvGa3zK7Auw5zp+061QpPxWEgLutyDvg5g+ResPijO5FvmLS8Ty6vLL1aV/peiqhQG6J5N89CGkNoHcL1yBgOIxVS1SFMGnGEVn4abQ3M1Mc+9kyTdZ0mqaK+Qz7X3Aex+QH4C9D3QWsajTMJ4AxZY35QbQA71RyA5M8iikigOntGUWDH1MWlBlZmAITsbgSt1IiPZsOTWrSlrFkiJkE7qUYpRdU7Wmj2H1mJMnBvL0iDzAhMigRlxnUsmWd89N6iyNzR/jglQILu0QILEFXCoEt8L2ZjPQzHacy7MjLooodJ2r7qarWXIB62+QPg3/bWBTHF7LgVodrAQaErs3JhEEpnCQKZqNUSGuGYX+MItr0uxgmWhdEmjInVbFQaqLwgGIlf3yqj56MHWaHyh0jVl5knu8W9y64je44b2L9WpPx6U61z3GBjwUvpyDbPT4rB/2B4qA5KBtHwAr3VQ50tjrjYz/5iyxTikXDCyN5R/qKpQBaljoYIi4/2DK413+8T7/uMgpF+Hg1HDdg3s5qcDz3FzpdcKv+O+qWs2icpLF3asCBiWF8TpTGLuKkrdWDsMj9Pzfq+u9krLPe3Eddt88yKn7Oru3cjm1lUqd/w246Ti4b+IWowE3D/h0J55a9HQvntr0tNBC6cA7K4Xyip4klB16klBe09Oiyi3blEnyddZ54ZtZi62mcCiDZwDjhVXwIzfhm0d474/w38uRNvjYeg++ZqAD720dfzQf58Jz//M5mx9kol4uUzo6JNzzeYeO66w/85QdecpOPOUOPEXrXPj8yD/otMQRd6DFFp/utyneSvdV3YREy0dSk9ahYfwvT3vgatony+yuZdR8RECjV4+9GozhJriv11X5VsFBXqMGFeAuB+4C8F2E7ZpcjF1yGZbt0W4TP7OCPHMVnm/3ka64Vlehcp0jtelJUVuFF5kGK7zIlFnhRZVeU4tZV4qKK7zItF3hRab4Ci9W6MCHv9XAanyzTe2iTk7WzDbMcB/RQbYuhFPb17R1GWeoUzfy3Sw/KU1G8Bwt5peSAqCjMehF4vT9Lh2KwVZ0inMYajI5BL6Uc0AdMzN4bbFfyDTVTUj7QpscGlzGobRGhTKUkKkun/u8Lu5OAT35wzrv152iBiz52HkZE/uUdZRYpTJXslrJk+polImR6sJN1cgjHdv8/Amve0uRFS4g8HkUDZHjxaVh0CJZyjpy1REnEG9Nde4TxFG7MZl2HaEtpJG+vN3uCMWOyZqlPrTUh/aVucqp0UacBZo+/W/TVCsryj5bXAh5HJF1spDS9WCO+lShLWTcdVX6QsZ9V6UxZCw0svDN7L2jDxSamypEhZCSLzcbvUdF8huMvGbc0sPOGOw3RbDPORYHCgesLszk9u1VloSByHvaOmuuzMZ8G5025vNA7OwLUHkaFESBa/4SN4wD1abV7jItv4W44CrDLx55BVNS5cqSwmtc0DDkzh6wSmiLZHzWoYhNo+2VBekQI3BjcUPLzxTCmXLZ4jZtLIVIJobCvd3uOlffHg41C9F8KS+8kZtFeul6XuhrMFXPsg+s+qBVH7Q1AUBugAU1dJxVjAImdhJ5y8JR+N1gt7FsyQBhACTg0QHFBS0lLuBnzQ/4YXN5yBJ6bWo6goz4XQh1yghVhCs5tGSqlabaRVS/Q3j1alcnvsi5rbIMQ6AxsLT+SGlMccPDCth0UqAEvwWftrlO767UwytOBPxFk2CbzNlkgrAszCJpDcQhqgNwWCG5L3zXpyJ8WqdyHjHtU6UDSlNL60R29Z4HDXfQbFWsLrfqdIper+WWY/nWw7KsaubDeAsr5rlYOseFs1rf12etZN3jGnN5e4Ayv7Z9WFxDSetaG9XNHdHQSEEFUvklHN7YYxZrquSSIb9nM31LZZ5vqczyLavm+Kh+Nse3VGb4lsr83lI/u4f4bGbqtV7C6vGnbbRrnYm1SzW6SEDOmZP7mZ3dge/bK/aHLpufs6p61YJX+rOQqmNcqtbWOtHphP5KPfUDDTDHeE84ePRkApabVuW9UqtTzuP4e403Sp2tf3Nv6t/dmQ2sJ+5ELtpOiHtAhRd7Q/7eP7FLD3/lvEBlSFtYM9Ztz5NTn9+8hqtbYNMvvdFO9+KeHL7L96C4TJC53aAjUMTA8REpEBhrxnfviccWPA6zx3Zukqbgxt/kzktO2J8MT6Std4YUjbijm/go3mgybPhhdocJfsB/23pF9xkoLy1xwYn2ZUvc4yBfPqtWWs9yFdvy8geqSJ50Td6YUpOXdGHCkYn0VpUbcc3DbZ5rxRK0DeQdwMeBz7CKzzQjT+cZikOvOC7EAKXZU37OGw8T1Ghvci5YFJ5bv4+D2bdjfkDNIESf04y4vJsn2yVU1IqIBe8ZzqG/cduvX78ZvElv80nbrzMrPeeW9TLb414CzbcuROs2LfxTNiFE6v4DOjHzA1OtA76osZKaxey9YkH9bqDUO1hL+giBNcUagyX3QqTaS3Xwz9/Knfpi73y3VCQKF+jP5ba0n79V93rzczHgqsod0rTpG088YdC9yV6kR8VJ3zrvtGr4vj5ZHQLpwh80T4oAglQ5NBdSuaOQDk4CLBREMddYlEZ0thGso27nz0+uyqNNP8jrHlByUYKNjAfNAk8Aj2E/9tmk7B6WFUU+vDlD1REu6gqLV84Oq/Jcpcv4MSqV5C9fsp9nvs/PBO+Jw7UGmoEczdOZ2AY7DidT2+FbBGOaC7DMYsWL33BeSQ7WdSPbJIl7DfF8bH+ErWYbDum4rDy0q9sMRwvcyHXYEq6sAFFc2o+OGfzMTUAbdSFzix/PIGxEFh6/vShbVnH5xpwv0X2lG0nowhG6OmTeWDx0v00p5bm6yNNBjo+DlVxc5GFSCjUmSC24ONiAh2l5QLDi5qwHtClwnbL7tasZWUcwyCMH9zod29fFoQVnIMmG9uK34she/PaPGVgc0OQ7RvL6vzeCBbdHmpPsLJCV77w8148HdXQ75MX7wh55PF+lbpYvUubs15xV5MdkcKf3/+uMX0FTYxcvaUcM0be0bIVY1ljBTq1ZIlSvJZDqHXely7SlpFtKurBAuGJBUD+gJQ5Wb2zAjtbpggY8PYjfdCcD3seQam3dok4epOQMJ57hWW9ikIw7zz6XB/dbEKrSLtTsJd7HIC5TKklh8frq9J46ESs6aJl0xZKxF/dF2eLiq+PTQoFRLfPkzOAuAShKLl3h7F6xAt73ADVAImpMnK1TjpUVS7uWmC5GIhnQWI2vDQGUQguFqzSiG16KNFF6q4ZraW7VKA4xreRuQ3cm9lSeaqY+0cjX+NA0S65ULiy1eFjaExN1+cs21sG+Lc9OpHyqXsFRuUdQN2gPq/aQqpoKrYNyMLGstegSApkJTlFhpS5/vE+6s3Uyhalf2tWo3Mh15t+rcsHzuxi7zAe94Ub/jlm4CNJ7XnD3h7j2g1NeXM7RpetCoN1JmLjs+OIoZtNZPOZXyaSXP65TQYXzpTp1lN5yWbZ/KcFxMx0wDh4CXbN3KFJP4dJ1r3TCtLluue0v2SOxyXpSgRkCTdSDayVJxeRiniZBBVGKzXDpV/jXMBbRI8waLm4Xa3DHPw3Trrv6GXHBxvw8KTFzZh748eXcUWyUNBL47oouVdz48x2mNU/j9FQ1YMQRn+sLk8oo3qiq8J7maCvyYh0vU4yyc+HrlmIeVqi1FF++FSgfGm6wkahwR2YZKa1cd8y87caBRZvt1OnAPzfgzlqXIkfBR4ESkcfmDkrquuY4sMdZblPV819RIRWKQKs1OKZaXuWOrXRrb1JZ0wu2Fnoq61U74f5CZ/dRu+FWWpLNLcRa19Spa/hxg9B3rSYUirBS+22olf5iffSgscl/q66pVASb64DKJSVxOZu4VpQi/owa4TQBd5oPmlICLcwLS96tj3c5pDEVvh668+z0Gv4Czy9u8hNdWGw8h3dm9oLnwuujJIm8wSyBEtA58QozT+1lj/yvU1q0+uhGbzFu+ICI1ZmE9lD8LRVw7W5nbnQH0Tu/ngw9N0uUiUEYlwxvV54DW/BYJmaGSIkKJgPdQzEKv5Oud9oFt00CiNzreNtqNps1/NUdn/9kxz/74rbSCIt72+hHF+jWJbyITaaXWZLWT+RDdpOawFdcjyZvT5M3p8lb0+pVCpHokMFNmlb2MFMf4gSERrRGYyquXcfZNryrL7t42p3M94nm+EFaI52Jxpy+grZIV6Qp0pNoSXQkGiL9kHZIN6BZ/pKmaeh7zh00xznrJYPxxxL8cetAPqNM8Dx5Ua1k95+B4Cmnx2xkd7NbsMVrVRZi4/nIrlMnTdQtWKY3cxw3jqWWATEUciaeR3YjmE3OgT1xPg0YFLUpRtx0HRgEx/tctR4aUJJ4twfPb5F96+zNmzepOJEtPT7FtZQJNn58atIiy/EZZp1R1pnIOsGsE8o6MRXUWunPb+AdTrgqc06FPp+Lip9PxFoN5Z5IcJ8I3icO8JPI7WFmj/J6pUaqbn+ia6RoukXcHFXXLzohkpC3z6+jA4RECuofMnGhYdqirHDGtgBa9iLrruBdsyfv/oLO0G4vwP4kq9BOK/Qu8FVb9HuLTt9C85Dep7Ss0ZE1JhLDCSAortqj3bVp0VcZXfi9/r6IdZ4K3YxgnkKuJKuV1X0j3/PXRg9ao5i1x+hXdwolCxdESVYTt0QRbtlNUS8Zv3FAXnMyYSo+hry1bzc9u6T4nXjEC8sDTQ9pf7FuRk3S4pWVXu7G0o3XRjaTS5DSmIsu1ULG1MHRwyhE8Tl4/H4v/eVNEnDuAieliTxiuHsQRWO7OJEiKklhXbN/D4DWMqgKxxR2KW6zlI3T/Sn5uoUb7cq3gePFdDTCuWp0G54yTvm7JDM6K3YKzHR/eBfkvA2Zp/gcK1TrOrXKS2XtaHyfF1x/juQv3KT6Mt9CT3ViEHa9jBc4Mz/dBfbEc1SfBgHWJQ6mYhJ881HQz8LgYxgl9sB3j7krg5CPfD9cuEN54dhDUUXFqEBk46lY8wswoYX35CQch8HIuwZp9XHC7+tD/m2KAbMTcaeK7zTEHXhQfuhN2gbek3f+lnxWjgkUkXf4yTLowlrK+6EgFgzHW3FBYjwR77hbgIJCZpGesDuUyNSX5BHArHnVoDsLNYgLegEqbhB7oXDeeZ257csfKKEr/WjCtvo9Ogi9Fe/vs/rQCxswwG7ZSVrFWotmD9zZ4cz3guuP5JJ8jNyRG+F1byoQpd3qCqStUd7E7ltos6KRQ4P7P2aKO/gZRwL9lspaqe/TxqUIihfwWDbCfhdFYYRe5AE+fXCc2dQOnDuQiyPyxsm3j8HJP535iTeNQvR1wgiDAmJ6YJB4YnYlm7oHB4pP9MN874WFP8VQxKOT8jX3hCYlL81Sw5dC8JLijGPigwLXoy7IFlMAwyX0ue+YVe7vHns0BQRUFJJ4Yuaiq/cweISnDC0lgbBzCokA+zk0jWQSA5Zx2Qnde/1uaQDasq2K7dlid/YCr0uoKFK8KXVZ9eslpfp00R5dMlsNPb3m+RBvZcP9/3WMRMDhQ38eHb/KyEM5GQsxB+00k4dGcaeocpmjvI3WxHK4Bt/KF5iVqq9tMYM5S2GqvwhUHs7yAOKRqhe7FOUWYzk+3wIEyqKn/4+Y7ntDuXjhJc4YhTOmdbYffzw+6r3r/fgjtTN06VeO9jKX8428Uv7/AJImeHM=')

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



CPSSM_CUDA_TEMPLATE = _zsrc('eNrNPPtX27jSv/NXqOyBa5OQxklKKQF2Ke3u3cOjHALLfoeP9XEch7g4drCdB+3yv9+ZkWTLr4R29z44J0SWR6PRaF4aSVn7wfVtbzpw2L49HVhmOPVjd+w0RodruTf9obGTrY7igevH2TrLuw9CNx6N1cr4aeKYcWi5cZRBGwQTJ7Rid+aY92EwnUSIa+2HgTN0fYcdX/R6H5hpDpyZazumCcVhENqO63vw2jTX1jgE0NBh9iSKxqY3mGt24EcxVW6xic6+rjEOMetCyYrGbBZ40KfnaOveoHHvBX3La/h249TY2/MD0/K8wLZip3Ha2ttrvdl535h1GlG7xb5uNOsbRn2jVd9oP9fZ7UbnrrsOKDN/e2z9IFzXZo2FXpfFp7T4JS3OdQT21rWJriNloRNPQx+pfJYDm2ZGthAjmwXuIBnZ9EVDgxFMq0bw95PcbplxQrUjqJYvMpQT5CrikXIgHMg1VHK/mZZJOI61pM6qp+/7SjnKUTcQ1Gnr2L7RF8QAGxnwkW20U5IGRBIULGQZfPfFd5QhcaCQiHMpyBuPLW3oBVbMtE1bv+3c1Vmecxmam+qDwakmOgFRI3oCxlmee+87g8bY2PF3H0B5w2DesAOvMWy3GqTN9G9YlIyvG536xpv6xk594y097tY33lFBAUuEHwZdG65r9m3zDgfMy4ZSbinl9p2eNkNeiVZUNNJiKy22ZbHflAUDOQpcfP2a7bK5496P4ogFQxaA2YCRsih2JnvMC+as/xQ7ERgAdtpk2szypvDU3G7rzPIH7NRI6jrbbwF5Z7vvgrwEA97muFGcp4Fjp1J0qs7CqaE8HCvlKxXqSoGCKeUTl51pZo/ZATtmm6y5eCv+pt0i2MgLAE6RbuwI8dtjvQx85FaBs8NDZuxQI5yQLBgOErqqAzVvjE5zqrM/maYds/19ZrR0onK32WziZypQGNUo3rZ3WiqK3XIMrTwGoBMGUEZEpxxDuxpDSsNxoSlIVcbl+IH0N6oU+M7MCbXUFOcMGPi6SXdd7zIuo+cIzWLrwfHrrD+N2SReWBGzLd8P4CEMZg5z4z3m+IOIxSOH9a3ItVkfPNEDiwIQZAteuFE0ddh85ITwH3xs7Pgl8hmEAyc0J0B1rMGHjetFj0HiEBft7jiYJeZ2Y8PG7lPzFnMr5g6B7zGwLdECfayzgwNk65D/AWtVLk30aqMcjZ2xCV0WfJu0lmkvpmnPYsu8d3wndG0zDsxoZIXOAPGzKvyh5T9ouYGHqwc+BQMSmnZsYfuUBWHGkIfVwxpbE6vc1yDCChej0gPtG3x4e3uCnJRA8j1L/A718RKX07dC0/VdxS1CjUKsHUA0mLgWlT6AC12gCVunhMZWo7/TAU/dvEMykcQ9stdWKEgTGIFemPggfFrXl5DmLCaOXUlcvFhKGX7NnAbHYcaLApVmfRmhiH0llUPHBytBHCzaAHrZkPQQVCN0PMeKnIaYUupX7QXkuNCJ7Zl8LCV9yMFKERGDhm6sBXh+EQG8tJu5VTqQfCcItgx1EXcUmxaGJB1N0YM6o3AHoldVPXAGSqY1gh4pqMkrheQv1ATjiefEsNBY7O2R28e4F6MbPs1f05gNPh0KgN+UhfB7XIdQEIZJXDxMwuJhEhUPeVAslI4oLxeZfhB4qmDH4VNGqnMGIXgoWoSv/+8zmNd71pjA+NmkC8+JrAM+mrzGxAK/8NSw7MepGyZiVhD9CY+mwRUCmgID1lnkeJPE3ADX4D92+JyYnOBB2pziqBOrEzywV+AWclZyp5NYyXtcaSrWmV5VuSUgm1tnvjLApqGwgV7inETPsQwO34O5GcDImcX6YQDul4Fn9iPLjt3A5+aNRaNgHrHpBDoFMO752dCaeriIBPZZA4ws0T9Nx65/z6Y+rE7daASzMLCAoUVxx6kwgTElc4zeMycI+E4XlGejNs6PZhLGCH4h0HwEnGHaqzJcNJ9pR5LL24jqkLWa/G/qeTqENxipaBXmzRoMOnxN0tlEJ8YpkxV9shSNBasdsH5j0YXyEy8/YfkLL3/B8pyX54p1ELov7MNo6JmLIOxo2S7AMmAIE3DeickdQxhlcgjNTJqaaB205mIoYxAIexqLOrStsxVQTwSV1wL+t6Lplxd1MEeoCh5HcRA6MA5/ZvZpUMbOFuonXwrOkMdbE5AAkw+5JYHM0Ic1cKkl5ygJrhTVjKYhdsBcWrHD9pHFP59dHcKqKZzasHSZRL3xz2NQohSm8HK/eQj4ohhU1OaCAZ42pOk6P4VOOnV2fgTfBnS2HJGxHFH7xYhayxHtCkStlYjayxHtZBCVMLJ00RgkqRBK2Wibc/026RFb7e2dn94prlDbPCpCHLEtWKjdLVlclslx8v5aBb42EpuUjlKDjiiaZ3/+yUS5hXA/TELrfmyB+QsDz6NehgHA46A9hnYevvZZYUhdVqt5vCP8O7rtwBA8VmO0zEzD+/mtdwdWpABmlIA9FcFaJWBfimDtErA5B3uG/8/M8SJnxWhHfLSjktEe4WhH6WjVXOQUms2RBCBl1M0ApDEtGvsR+xEmCjwbTm1syBoDa4ykYZqMgAU7Wjr6fsLvFj3H1LrOjpCJO9gpMl2vQPBFIJgLBE8VCDqlCAxBgZFS8KUCwW4Vgi8CgaRgXoHAaOnKhAlH/zGyrYkDC2jwgnsQvzxSdmIXlvie5Tv00IYHCKBcjIgo8QDPI8sbYvktlN9S7uf3T5fo7nERPnLvR5Q/+kfEQDcCH7AzgIkaOZ0/vyxR+YkV2yMtq83nl6rmOqW5H4zxQEQdTMlQbqRtlOV+FiGHohzRz7DihlFou8AejTd9i00NqN/iiZ2CQCfiHHJxBp7AQFB+wzSCIGJUhYEw5ug2vGN/HAAFMso6Gny2bGQO8jqCEUDIZMk64egjL4ijOpu78Yg5lj3i8wIhFcVU8Qg8FIIAq6OJ58bbJwxi2di1vAy3bc+KIvZx7MYlHIcQd2o7mRBii0WhzYOI81/594X46l1d8lLsDnjBj+uEmjnwj89NwiR3sAA2AWiXisCqX3kJAhu5RGbJggZA1Sil2RhCDK3+ExKcoJ/wOZgA4gucgomuhmAQBsEwbrXI/QKrG30Cc4rkw2QDBXcCFxKtwTMEHFQjNQNMuL/LYh78QtAIk3B5BTKxHQZzXsEmoBBzK4RQoXcFAgISjsnPoYdpVXopQRvsw69HvzANIkqctcD3nvQ9ZjA/YGdnR3XsroUPC56xqrMOPtG8YMBdB53jqSxsWUeVhpmNn9iDE/qOV2ew4tBIQkDwYcaBiGAaT6YgGhjSAD5YBCQAFmA/eY99Rri1BLRuY56OsrcidQZChYrs+DDDRmtXtO3oSFUGc6PoxblMyO9L8d2Tzx8/8AIyRAjY1dnR77x4fCa+z8X3ifi+uZQFWXNyLQo98X1xuGaafI0jU46m6VlT3x6ZfVzORKapUV9SRu6d8Uxby7sbE/QBGOPaMaCZT2T0nos0M1ALmSrM1CKj6mtFZ5UDcyKhaCC04b1gSVo8SYs3l0pZqT+55uXloc3yWCbhAKMU+hk4Tvi3x4nqZgDOEeAcAc4BQJB6gpUnWHkClSfFVjeXCAH/f6T/e8lobqjpDbW9wcZyaCfXhPOakF4j1usM2mxwWYifRLBZjDQy7SnDcGy0AFIEbYYSwLVLgP/56+knAX2IsWwR4vwTKDWAkM5vYh7g/NPv6XMLn0EP0hqI+S8/pY8QcH88u7j6PwXFTkk3l5/M3hUgSsHaLURkHn/6oFTu4IriUw4SlLoE48V175/oHYk4GBtyQpTb4BY32StOOJaAYPwiQhHVNELL9mkaX6G9f5o4vjUGPxVj9iTwBy7aMcvbR+6JlVU9q1MwOdgMcSGWLdQfQBU6MMVOOIFFrGlbUbxPLw811C6y184CXvu0lMRkDe09UJIN1B08ONjliDJuzIb3DPPmt3fYrrQBmMckgRBhKiIrxKE15/tiIFcX4E1AkLdQgLfQNWyhIG6xN0arW1gcEJukt0tTFknYnDFC6C6RiSMMCH4dLDBERDfDHSkGKW/qPAzgFRjq1MGbZhRsi7RpC13DHrfqH9yxXCcoynwDrcQwoBeMaairDYYmd/4gn1/Dc75tDzvsYX896OM+dAfYxVMe7JffFYPxmmnUG/ELpydtWKDtF2gHjbegI+DtTf71/RxllYaGTHoCIICusaRmoVM7qMIh5JuHtGZI2y/4fAJwqBOB+QZ2s9AfcBjgHxD+5DoPf/UbhjOur+EaCv2hNsYoXKc9Q71b4hkw3ZUkrZIdnk0SRNEgsx9CwY+MORVpQz1O11ICUt3XkSsJClXpNS5cQZ9J8qCcti7QiOElKkBiJH5UwlxN67FtHCMyHie39NXNZUIBK2yp0GaFUQ4gNjYIhNMhJAl4vYVGbVpop241yHfPmYWUsk8g10dZVo+fTFS15uJn8TctYzo6EW4YddkiPcsBTh6XGCIQBcklxayRFkOvhSBk/oiKN8m30VIz027por0qSRyYRRhdDJAE2fyXQmu1WS64WaCkL5TOQfS3MIaEGrI7m+gPtsgtcU4Q0WAdQ3QCJUabAA41lGnRsTWNA8wV3TumPIdwwG437yisxxUNj20iNxXFHL8v2GFWUFGaIxeWAifXFYkX/pesHj4fwNrh8z6oOiwdPuvVLdSUDbbx9gv5mQptGUTxQU6fJZtwyanhSLfOb2poofStyysgY+v8FBBvtVs1IR1lHXCOw+rmYP5Yk9MUuVti5mva5wRNHkF2j8CeiA0i+15sd4gjRGL3h+93wEzjhhXtXMCQcOvGW9eg+8IeRvr3vFYsVfUdwHI25mfZCjtjGaV97i5NMT3i7DzuH9Ni8FHPCpj2WH8U6CiVOL/tXd3dXsLn/PQuNa8dtpjRmzP4tO4UccU1WLmUVspbKmt8pfoZxHOVvJWlBxN5KygBN9fzW6Tm7vbz3a13p5ieuQamJFVkNJjSOEDtZ24PwI7oqlHIq9orjF+lbfu6kmyfk+0D2Wc0VD+rJuU2B8MbMDwpreg6fSBrNzE74DhbIHpj6UfQ+0ZkmVTJg9njvPDv1CM5eBBwMdGrII08ZCbllsrwi8Qw4hyIgAM94gAICAqPBkvxSFrewqmTcZ2JqDZNijDLtm/Pj7iYokB27rovTbLKrOq/QTJLpvgF9tbmrWxo1cFGYD1weCMSW5gDG+eg2RjmQp1h4A3SUCep9hdxqVeGjpxJVWikemmOQOxBFfef0InXKfGbGxozsix3+bC458Hv2gFM+2qTUCYl+bCL+0AAdFHWuxlf1+NrVBwPgMBiFP1eHxYNDxk4ZfRL9ff7RGKVwco5H5rLP9TMqAamS9gtiMH/YMrjU/bxS/ZxntFl4fxrmIrjEUCicLw2A71M+RKNVxdnL4tAlrg42tsmB8c2msKZ+uuAY9vI+dCXMPobIpdviF0K0QvGF+XRS03D4MXdOL7Qvy184b53k80O0qkswhSONYvQZNahk8SrD2Sn57HT49iZ09jhqvBFDVuy5YQ9Rzx4EHuL3e9TrMxOJ2WppEbUoQP8L4+bYuLu2shLfYlAZqMemqV65NZgrrplw+iTY9l90QBW+fY8SRA5LDU5GeQOR+4A8l3E7YA5Q7/gkF9YtIe7TfxMc3FmUU/xr5/4f3TzIhrA4xb9xN2rL57Ei1b+xRfxop1/Me+W9djJgBlpj2/yL2SPO/kXsse3+RfzqgA7W/77TWGJF10xq0slfsWaaYrbkBkXAXTTQr/GPhe0lM7VoivHtMounahFBGXWbRCUVHIMPJ12QDTrKb622EjUdXV3cl+o/KHGFRSgS+wcQwmelNXzwMPBXas25k9WhCDOBM1UIdDJ6oA42FTGiWV2LS9F/77Y4JtCxuyijrd8lGFj6dzmzc7Z2dFKw7MycC472SQjGJz623R7H74f7yBAKanmhzGqXoHB+aP8DBWThzDKmrXvumvVxHFTJmlKn4iUMvEpNaFK9m1saRiqN0Wozl2TJKeu9lfPd1eqGtnpwl0BPdcV6Bgs8/L9AR++t9PnSjXIre3W1qpd6+rorlmewCgLxMtDdNq/yee1eP7/IB9upkflQXpx8oERogTzboAJsdPHFjwO0sd2xo7kbPBDxgCP2Z8MT1Ostg6Kru4IZV0SHY+tQcML0gP6+OnA5w2d0lVeGuL0funLljieLF+uV6vTeqZhW55ppoYUQNbkdYCavIeGBVsWkisDD+L08qOekTSxVLVQ6eBjw2dQlUMomXnapctPvZJyJQEo+G5+bhG3y2q0q7HFtSSTMzFNnEzTivjZCo0I3aSgSt49STNteWuAVPCRYRj2zmm/ffuu/y65rZL0X2dGckQjHWW63VVAzVMc4arkxv9KsiJU8xS0D/wjU60ivqixgvHC6r08YHmmLfFbK1kfIrKmiHANmTNJrJeaP7h8L/f4xK5btwASBnPM+Wc2wy7fq9tEfOf3zz+T3WHaL8LNe+hef0la8JsCh+8NG5ZN39eqTDqe8gFwmWxEL6IoHWiSjVvm1Zl4OucD9hKVD0PUEg3EbCKitdXdv2wIKjfpf5RnllFbUWu1VO70nByAXCHt+2xcDFaKxkFmVrHJFjtHcxHM64pYV2bwVR2usl/8cIDK5tev2c9Tz+Nn2PbEWTANTT/tiEL/vq7uhDXYcTCeWDZPrUdMAxhDzze8+g3AY11M1n0j3VzAHD0e5zKH2GuaqKfTXfKMWVkumpIjKGnYE99OxxM6ooJvuPu0rwWVG7SZS8SIGjwsVvSl4vj4jC8Jv9Kpejo0T8ffZ435c/f7zFBWjpdLsb9UhvMSTGagxgSjhQz7L5BglF8piC8XPOBNTuaUmKxbMq+2EI9vnNr7ZGbf5icW3H+cTuzVb/mJvfrtf2ZicULjvzCT9/+9GcwFOtKBpMeLjOzg5SFU3OMv204W7/mGMtIizuXo3dL5P/814wDPrsjodP/Di92sdcah3WqSrRTep+SBEym3kCBwJcGtemZWGnPcu5VlQym3lHI7O3UmZSD4saNIW5L+kRNYkFj1ODEOsE6nh/FkEX7TgWE8LJzY6KzklC1CpSTY0RQPJpJApNJ4flOY0++hp8qYUK+3eFZY3P4pKF3+d1iS+5ViMWijGyoDi0duZApYLY/Wo9+h0KpVnFZnmKkCUIrZcud68g3wLDK0AD2oMXEK5zXrVUE7hji4hEzSoLMa324FLLkecse8wwcORYYnOfHtGCUnvvMzTGe/t2A4Y2siD+bRmGjia3xqmoW4KbPuNPi6sydyRNmD4Ktw5xZy8mQ4ial6PLwqn1A6ac/LNpNU+2RnDBSdZtott2VoD2VsWifPlgSZ3RILGjr27LstKMRvVyOHeWAPnPAfEQvmfnK7AFOL4qw5Z6k4Ed6lM+rQ4TiIHXZ8dRSxyTQa8QsMyW3kVaalVW5akrvWJb4LZv7qt5W55VA9Yke/NkCHyZrL992+3yN84+ZJYbZ9EdHh2au4IpGmxX7F4PPoRCCtzcM0n55E1H4SUnfLE59CpAYR98kgWKkl5scIMyciUapJubpLqKq4+fEXvFiWNcnpRqCIEz4rByb1zN+6V8Qktw3KD0Hh7+qAbQnT85nLU47Lcu8Kva+I3uyaa6WLZvlL1EWiMv4QJxD9oF2n87XcKdp5Ly0i7pc76iRky4hHj8vDMrX9r6hfhUIlGsepKxUYHsDJ+O0hEfhy7UowJkpWHsr9rQFdXsBeID0VFvblVnRl/GXXS2TtBcu5lSZIWKBKs/NCc/A3G4LnEl/131Hyl+t35caIuB0nrnjTKjblRjCJIWbkk6ZAoGnfNuQvIM1Ao5J1A74eOLP0rBb+XuIvTvyBfktC24R3evqC18LrozgO3f40BggYnHiFlWfWokexyBltvVw44XsMjj8hYXUmsT3nf/UOwpzHqRM+wYqU3xfDKMYQMBEo44LhD1/MQCx4wB7BGpqXRAOdgY2hQJxfCuyddWG5LhGEzn20ZTSbzRr+QqLHf1rtf+vmXGFGxcU5+iksug2DN+FkeZEWKesvH9KrbII+cT9NXl+TV9fktbV6lQGkcad446aRPkzVhygGJRG90RyK38DBjBFejkx/AsQZz/aJx/hB3iJfiaecn4KXyEfkIfKPeEd8I54hv4BX2RtBk8Bz7SfohkvQawbzjBD8ceNAPqPs8zo1ukXZ/RkYnUh0xIZWN/0dEvFalflI2xxadRqcjjYEYXpT23aiSFoTUDehT+J5aDX86fgSxBBzQSCIaDVx+UjXs2Clt89N6KEGkCSjPXh+j2JaZ+/evUvUhrajjs8w8z/Gzo/PdNoSOD7HqnOqOhdVJ1h1QlUnukJaK/kxNLymhHsIlwR0cyka3pyInQWqPZHorgnfNUd4LWp7WNmjul6hk6pbTnRTilIHdGGubHcE6YO6fX4bEGgRJWh6yMQl0qQz2eCcbQC29EU6UiGuek9eF4Nx0IUKIPwkbdBOGvSu8FVbDHmDjpNC91Dep7Js0ZEtxpLCMRBIyZDdFOpNyg3+i0qeuL/zSlhexPAKaiUzjbTtO/mev9Z60BFdDOwx+unDHGTu+pMUsMIdqNeMHzaXF1bGTCVFk1cld/XEVwzVvBPBAyeBBLx/VJ4OemPg3TehCuTWFSSGQJIsYOiiGAphDkV581auOb+ilr+OpLTNdtbDvk50eYWw0AiVDVrU0iZ03zDtoXDdNSVWseXguszBk5/xwLJO8cNLzNAqE8Sh0n5K4oFtbmuG8of5EtuS7aGnOnbEXS/SBQ7+w5NvjV1b9fOIsC5p0BXz6enfhP088C+CMLb6nnPM3TtiPvK8YO4M5J2057xOR6hxsvNEGfiFXejhlBzpceAP3XsQdA8zPV+fs28TCpgVixsnnt0Q9yIBfuCO2xrenbx8T3EcpwRA5L1OCYNhnaG8HwhmwXS8F/dWo7F4x10nCiW5EHrC4VAhVXopI0BZ865BF1NLCBf8AlIcP3IDEdDyNjPLkz+oRtc8KVNX/R6daW/J+y9pexiFBRTgsKw4aWKsJLMHId5g6rn+/QW574vQGTohXhRUkSj9VjcgG4f6xtNq2GdFJ4cajxX0hHbwyUeC/JYqWkmc0MYcNMXQeHAbcX8MwyDESOsAnz7Z9nRi+fYT6MURRagU70YQ+J5NvdidhAHGBUGIgTIJPQhINNa7UkydgwMlfvhxtrdt4M9F5enoJHLNo4ZxIaIx1JA+F9AnNOOceGCuy0kXbIsoqOcauunZelWIuMe+mQMCKypJNNYzK45TmDyiUy63JINwcAqLgPoZdI1sEhOWStkJ/RjHx4UGZMu+CnlX5fbeoTaf6JUg2etRh9qi6pfWCu3ppyjognw19uS3KA7xaireqapjtA7BEca8GCRVRufKsVyIy+kMkTyxCnjU+77yJr2OcLjX2soCTAvNV/aY4pwmONUfMCxOZ3EC8Reitndp5Zdf7/AcBDAoXWH8J9Y9f3W5E83d2B6hcka0wfLTT8dHvY+9n36ifgYO/SjjXhqovZO/c/Mv1XD/CQ==')

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
tt_mw_CUDA_PREFIX = '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t l0,uint32_t l1,uint32_t c,uint32_t t0,uint32_t t1,uint32_t* a) {\n  uint32_t cm=c&0x77777777u,lo=prmt(t0,t1,cm),hi=prmt(t0,t1,cm>>16);\n  a[0]=prmt(l0,lo,0x5140u)|((c<<12)&0x80008000u);\n  a[1]=prmt(l0,lo,0x7362u)|((c<<8)&0x80008000u);\n  a[2]=prmt(l1,hi,0x5140u)|((c<<4)&0x80008000u);\n  a[3]=prmt(l1,hi,0x7362u)|(c&0x80008000u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
tt_mw_CUDA_CORE = _zsrc("eNq1WG1v4jgQ/s6viDhRObEPSIhQRXBPx0ornbaH+rKnfkAoMsGUFOelSSDQVf/7jZ0EAqXd3p2uUhN76pnxzDzzkjYaGQ9iwTI+9MNMG42JfN2p58M39br5ql63ZBZFQvtiWmr75febq4brPopoxoTraq4r2Dr0lu4sWofz1HWRaV0aIAIO6tom8uda0LNy95EHAWpo8ONFYZoB32Piz121YWEGkr78lfEwjZI/WaxlW1IcW4NO24DTCU+zxPfkwZzU5MgDPcvNTs7w1COuG27c2UJELDP7hhatM2VBoJ6heq7UM42Fn5FKkpZ1a2tT136AOr7NeBKCjnTJEj5XhjPhP4bS4K5l6+oil8CRBpOp0yivx7dxoinPPtC7jtkn4++09M/dDb3DNrkf0dHYABp5uKfo1kD3I3ypY5DZ0ztSsiEfZHxNIQK/9Qa2lC0lZv6cZsuEs/kf8217S/JHCqSOlJSzJJabFmw6PYsIFnK171mSe2/cjKW8S9l8nqA00IncUkXD+aPxcE8WayEUAd8a96NKccLCFZ2JyFspxa3Ce2FWo3UKWlCj7Sr21axL0arTt3UDSVHgC1Cnd5DigZ1OwtXszBFs1g/9CnIqY/o2GBPnJN5KCksDwB1A2xccNT3wUMbjSPjerr1ImJf5UchE+9oaDPhGgmXhJ2nWnvVtrdUlZrvrNAdNKpooznX938gT7Ly4bSHOX6AyNJR2C3Bp2iJKkEIi7Trp8NZJMdaP9QYzliQ+T9p+6GftAoWDgZcxpWrS6k6J6TS1wUBrJk0kI4dT41LXYB/wIEp2TaX91JwFDz3eroS7SnjCBYegtz2xBgQnSuqJmFf4hVzYhV4BwRQVrlpnkQb5Nt/SycVUWeRXJh7ZrV1caP4Q4lz9tcBGSv3WrVMS3rFfvja8DZnFwdnZ9o0vXKK80TrnDiJ396MzbtG0LGBIgT0FsJMLKEEIMIZ93ejbAGUDamQlB5BWsklHvMrlL3HCHgOmrcMkEqJxPqTKNSjVj7IQUrC7XZQ/a8LNo224zep7R4rkMe2WUIKiIH0pM35491B587Q0ckGhIGKEwsyoMgjXks+4e8BShDS1sAsuxcVEFo6pAzcq17hnTR15I1kHF0IBAB1dvku6e4Co2g3lIp/cfJ1OxtdT+uP1CCV7kKQiKmpzHS0KH2eMq5sHrSFNPJqDaSIKH3VpYFE5cBm7vWXja6NnYWlFFfA3IatnogAXi+H42hEQN2mDvON0IqZUzHMESrEAeQSKxGeBIIMWD2++OnGJBBSTWHGrFqUxz5uMxh1r+qGUJ5DyNJTnnKe3RSLatGc9S1aeou4smkiKfZoWtedYqmbW5PozEOzPpM/hhenN18rfZ9x01qSjTPYpCIkdGcYrKvN8BmViVXl+j302MfvTiuovDk1TAvsgUdPm3EMyCvF00p1Ctzusd9XaknTo3ZlJ2D6pTzlfapx5jXNXcWL7PK9Z02oea33Z816+y/tS463rzfe8YG/F/AoJm/Ka9WeReopVW0G1xgWZNrENQGypToAJjiJhs07clUSrTnwpib06MT+Y99o4fpeA9jEA5zMBzZcSsQiKydWVealTWmEC+fqxEYAhNcBQOFAx9PSLnrnW9Zhl3hIxWSVrrtc0jHlc3x+K7YbyeNizfuPdATfrRz4saxvCY9B4pAPuxeMrCrXmtEa/66Q3PS5nfgaNbd+gCPI7t/rFQdGHNerDWgDpjw/pPxi86XdH0RmNKTX7ehAwsy95CCNzaBmHhqi/z9ezJF/P+qd84Dng69s/4fsfXVD0Yx/f1ov4+dHmv49tG5gSz81tn5ja3ruTnMajnMoGZ5h91dw6Nnw5CWoZKmlatv65bnKp3Pfzir8EhuXQcpZHBX+B5BWuKDRpCHHmh+t9k0VIdTdDhy+MCYLvitEYwwXxpfEEzfkGw/1hvZxSGSXbeMIWbD7BDF8D77FDdftMONUgdGXq5ZhbZH3Rj0tjn+VHUzEnjb87z+AnA911wKXPmI6/F5PUfmCRn2TRiof0uaMOERkZ9NxSG90o5yrQq0ZJrM5e0eDEYcpeW0vXAQ3YirvFHkE/J9VoVWu/qv/Gw2qkg/VpH05pHHfgL6eNNwF0ZdXXHkJx3Hr4tr9W5dn9rY/8dQJwFrMDrAtXttfFDNIySctScwiAu9CoD5qHtZrG00Oml8Zvzn8BiPkbPRu7vQBVPypdpNV7henfnlbDz6a9BS3FardfvexX+dF93hYIiEN7iymIceRyJ5c7tXyRyxe1zOUyP0BOAq72nwcLygx0FrVLrT3VTUKkxBMlWSdL/4NTL+pUNWwaSAbSMnQUrTNcjr41XOlGiGESvlNBLICkGFDBqD4K9AsRkaP90v+w2LyfMa+Nxt/5qGwT")
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
tt_v10_sm2_SMALLM2_CUDA_TEMPLATE = _zsrc('eNrtPGtT28iy3/MrJjkFR8KKY9kOEIzJEpLdk+KRFIbN3uKyKlmWsYMsOZL8Ist/v909M9LoZSBn9+65VTcpYDTq6enp6df0zOjZP8a+480GLtt3ZgPbCmd+PJ649dHBs9yb/tDczlZH8WDsx9k627sJwnE8mqiV8WrqWnFoj+MogzYIpm5ox+O5a92EwWwaIa5n/xi4w7Hvst7p4cnJafM9s6yBOx87rmVBcRiEjjv2PYCwrGfPEiCgpM2iie15k6blDRaaE/hRTNVbbKqz788Yh5l3oGRHEzYPPOjbc7UX3qB+4wV926v7Tv3E3NvzAwsQBY4du/WT5t5e8/X2u/q8XY9aTfZ9o2FsmMZG09ho3RvsaqN93XkBKDP/9tiLbvhCm9eXuiGLq7R4lxYXOgJ7L7SpriNloRvPQh+pvFdGN8sNbymGNw/Gg2R4s0eND4YxqxrGX0J3q2nFCumuIF2+ypBPsA+NAMkH6oFmU6X5xwiahpNYS2ptI4XoK+UoR+JAkKi9wPb1vqAIGMqAo2yjldI1ILqgYCPz4G9f/I0ydA6ydOLEJjROJrY29AI7Ztqmo1+1rw2W52GG8Ib6YHLSiVhAVI9WwELbG9/47qA+Mbf93VtQ7DBY1J3Aqw9bzTppOv0aFgXl+0bb2HhtbGwbGzv0uGtsvKGCApYoBIy8NnyhOVeNaxw1L5tKuamUW9d62gwZJlpR0UyLzbTYksV+QxZMZCuw8tUrtssW7vhmFEcsGLIATAqMlEWxO91jXrBg/VXsRmAU2EmDaXPbm8FT42VLZ7Y/YCdmUtd+uQPI2y/74xj4PuBtjuoVkzVwnVSeTtSpODGVhyOlfKFCXShQMK85wQuDGIxSKn7RaFgP64vQnmaFEH7aqQyKVlISj7jkZWUIWWKNgF1d7YhtssZyd4f/n+m1i0YJPMKKBgJ/eTOcxm5G3YArsjOjsXxtthsznUOaayF3WttNCdnMQ5pGQlAOaWs9qMQKMpPxNH4g3Ux2gn137oZaanlzpgr83LTzQu8wLoNnCM1i+9b1DdafxWwaL+2IObbvB/AQBnOXjeM95vqDiMUjl/XtaOywPnifWxYFMCs2vBhH0cxli5Ebwm/wr7HrV8lfEA7c0JoC8bEGP2xiFL0EzV9cNLOTYJ5Y140NB2lIhSjmUjMeMk2LYaoTKdcnOut2Yep3hvzfTM8xa6o/YIWjiTuxoOeCW5PmMe3Mspx5bFs3ru+GY8eKAysa2aE7wE7Y2k5C27/V8vr0MBNmYDFCy4ltbK/oVMZ8hw8McGJP7XI3g1grvItKFLSv84Hu7QmaUirJ7axxOdTHo71N3w6tsT9W3CLUKBQ7AQSJiVdRiQS4cAyEYeuU2tiu97fb4K4b10gr0rlHptoOBX0CIxANchCEqxf6Q/S5y6nrVFIYL9eSh3/mbp3jsOJlgVTLWEctYn8cqUPXB2NCvCwaCnpZl0QRVD10PdeO3LqYYepc7QoEvLwnx7P4qEo6ksOWYiOGD33ZS4gBRCzwpL4WdumQ8j0h2Dr8FR1EsWVjmNLWFC0xGIVAEOCqyoOzUjLVEXRLgU5eZSS7oSaYTD03hoXJcm+PQgEMjTHi4VP/PQ3m4KdNMfLrslB/j2sYCscwCZ2HSeQ8TALnIY+bhUoS5ZVi1A8CLyvxcbjKiHvOZgS3RaPx/b99BtN8w+pTYAKbduA5UQLAR9NYn9rgUFZ12/k2G4eJ6BV0YsoDboM1AE2BCy9Y5HrTxCIB6+A3dnifWKXgVpql4tATwxTcsufgSorWdLutWNMbXKMqppxeVvkzoJ2bcr6CwKahsJVe4tVE97EMHd+BRRrA8JnN+mEAzpuBX/cj24nHgc8tIItGwSJisyl0CmA8cmBDe+bhshN4aA8w7kSPNpuM/Rs282FdO45GMBUDG7haIf04KRawqGS20fcWhALf6mIA2TCNs6XBunnGIdhiBCxi2vNyfDS/aneS5S8R4QFrNvi/mefpECth0KOts4T2YNDma5j2Jro/TqSs6JMpqS9Zrcv69WUHyiteXmH5jpfvsLzg5UXWfAjTkBiQ0dCzlkHY1rL9gOnAkCjg3BRzPoHYzOIQmpU0tdB8aI3lUMY0EEbVlwa0NdgDUCuCymsI//dA07tHdbBAqHXcjuIgdGEw/tzq08jM7S1UYL6InCO3t6YgFhYfd1MCWaEP6+hqy8/xEnApvjnNSuyCZYX1ANtHZv98enEAi65w5sSsx/H8PAE9S6FKXu83DgBnBKsKiIVpDsFZhzR5ZyfQUdtgZ4fw14QOH0JlrkfVegKq5npUuwJV8xGoWutRbWdQlbC0evUZJCkWygdpmwv9SukYG+/tnZ1cK25U2zwsgzlkW8zcvl6zUi0T8uT9pQp8aSYmLB2uBh3R0oH98QcT5SbC/WMa2jcTG0xmGHge9TIMAB5H7zF0EPBnn5UMq8NqNY93hf8Or9owCI/VWOMa2qXLiMWVdw2WpgBmloCtimDNErC7IlirBGzBwe7h9z1zvch9YLwjPt5R6XgPcbyjdLxq1nMGDRdIBBAz6mQA0kgZvcOIvYXJAo+I0xubssbEGlM2zOQ3YEmPxpD+rvBvk55jam+wQ2TkNnaLjNcrUdwJFAuBYlWBol2BwhRUmCkVdxUodqtR3AkUkopFBQqzqStTJwKFD5FjT11YvoPr3IMg6Bvb32fmLvuDebbv0kMLHiAKG2NYhRX4cmR7QyzvQHmHMku/fTrHcAFTAJiYoOzUPyMGehL4gJ0BTFTPGYKz8yo7MLVjZ6RlFfzsXFVllwtNTh4wWgSJddnBAbAMVvktc1aS+1mGHAoTPjs/w3ofhqLtAo803nQHm5pQjwYEuFaQ70S6Qy7dwBgYDQpzmMYeRIyqPxAGHV6F1+z3LlAgQ7XDwVfbQQ4hwyMYAcRdtqyTgYEXxJHBFuN4xFzbGfHJgbiMArN4BF4MQYDf0dQbxy+PGUTF8dj2Mix3PDuK2IfJOK5iO0TMM8fNRB1bLAodHnecfeR/P4s/vYtzXorHA17wY4PwMxd+8QlKODUeLIFXANqhIvDrIy9BQCTX4yxZJAGoGtg06kMIydVfQpYT9FM+EVNA/BnnYapngzeInWAgV1o0voM1kz6FqcUBwJwDDdcCG5KtwTNEKVQjtQRMu7/LYh5IQ9wJc3F+AaLxMgwWvIJNQTkWdghRRe8C5ASkHdOsQw8TuPRSgtbZ+4+HvzANAlKcvMD3VvoeM5kfsNPTQwO7a+LDkufODNbGJ5oZDN4N0D+eVMOWBqo3THC8Yrdu6LuewWAJo5GggPzDxAMRwSyezkBCMPoBfLCgSABswH78DvuMcIMLaH2JOUPKE4skHsgWKrXrwxybzV3Rtq0jVRnM9aKb51Ih/56Lvz35/OE9LyBDhIhdnB7+xotHp+Lvmfh7LP5+OZcFWXN8KQo9/vfTZwgwLIsvmGT+07I8e+Y7I6uPa6PIsjTqLBWTG3cy157lvZAFSgG8GTsxIFpMZeifC04zUEuZsszUIq+MZ0UflgNzI6FtE8sObwRX0uJxWvxyrpSV+uNLXl4f9awPcxIOsAno1dEp+FP4tceJ6mQAzhDgDAHOAECQeoyVx1h5DJXHxVZfzhECfr+l33vJaL5Q0y/U9gs2lkM7viScl4T0ErFeZtBmA9CS0EqEpGVBSAYH5S6OzCbAipjOVOK7Vgnwvz6efBLQBxjzFiHOPoFuAwip/ibmF84+/ZY+N/EZ1CGtgRXC+af0EULzD6efL/5LQbFd0s35J6t3AYhSsFYTEVlHn94rldu4/viUgwTdLsH4+bL3L/SVRByMDTkhyi1wkpvsOSccS0Aw/iFCEdUsQgP3aRZfoOFfTV3fnoDXijErE/iDMZoz29tH7om1mJHVK5gcbIa4EMsW6hCgCl2YZjecwgLYcuwo3qeXBxpqGJltdwmvfVqGYhKItkMojQdKD/4czHNEOT3mwHuGyfura2xX2gCsZJrAwcRGp7AMoBFLD5ZmMpLwOGNT0AUiP0bo6T8OlhgCouPgzhGjj9cG9++8AmMYAzxkRl+2SDm20NjvcTv9fjyR6wFFN79AKwJHW7nAYIW62mBoRBe38vkVPOfb9rDDHvbXgz5uwvEAu1jlwX75TdH/V0yj3sA5UuSkNCzQ9gu0g8Zb0NEWEJp/fbNAsaOhIZNWAATQNZbULHVqB1U4hHzzkFYGafsl8gC9/SLUicB8A6dR6A84DPC3CH98mYe/+BXhwdOgduowfFwzgQEAGzn2Nf5gMG2CgbeOs2rqnRLrj8kxJcGVbCRtkqiJJpnNFoxzDDZZWSggjeXP4t+sI+NORTCfCwJNXUpn9j1qd7r4Etiz+01y2UEBLQHgehf0nMQYymn7wtgwCI0AY2I83irBsKb12EvkDc4iSkrpqy/nCQWsZIuH9k3MKhCxy0JAnBYhmjB5W2jwZiUt1U0P+fY+t/ZStivkkqqEu+hDuF3U5Xyp50jA0+OKQwSkIO+kzjXSfUCr4lUDksU31NppvqlGAwPfh1h0gUaVOA7MIow0BkiLbP5LobXaLBfoLFFNlkrnoDdbGFJCDRmtTfQLW+SeOEswibO46l1cX53Dz9nJdSrTbbac05tT+GmKF/Ys5tvG0NHV5jUF97i44cFNVJFeUVYCX/lK4CusBEDNYSnwVS9vUZ6USXIwiyvs9vrq6/WVd52Zu4UGk5CyAMVKshVqv3JOwgzoKjsLuokRgOI2hao29coRKvT6nF4f6D2lMfqqGlbNHfoYmMCUcjRVPhC5m0wfGKqmDuZF6h+awIhmuKOgh2njvPGvrxpZ3iy15VSvgjWLsJncRqpp/O99Z21qKeJciDC1RFwA6UDJ0WCVE+nrDGJb10tPHEwMlkQR+bbSlP6/Jf0/bEl5yt92nKuzQ26R0Pa0rzuPzWHKlOVfYIRKlLqqjdLK4a0caNXGRo5OwxuR4QKdc1DnGvVhLpQYBt6AWmar/WVcGlZAR+40ha+eHo5A7AAVd39w9gzKqeaGxswsy8d8WGMYFgRf8LcGS7eLh61/mU3IqxXBYQdjtG6p4KJm9vgqD8cDILCcO77UWR9i9dsMnDJ6HPX3R0zUE0TiId+U27egufxdzTRq4LyE54LQ93emPK6yj3fZx0XGdtMOKjIJuLJPnEjMK6/NQ/+wqSUVVddLuc2YPunpbrmePtE55ikFR7x2BjPIXY7cBeS7iNsF6UA1c0nNlq3hbgN/Zp0MAjG0MofSzBP3Z3vvccF9p/HXrOi7DUq+rnXS/F+fe//uDHe8+9y9w8NKPDTx4U48tPBhUYqhfd2dJxhe44PEsI0PEsMOPixKufq9nDBk1xwlm54UInMvUoJzL1Licy+qBqKCmdfKoHIv0gHmXqSDzb3ID/xZeTlRl0MebIsd0M6PWafcriyly6RhMaAL/I25RTzyi1nES/NvNB4lPukBpV479HKVTPg7w52yjMElNYOQqMa+FjSFzpyiY8Q0wC6dNkUExV4YGwQllRwDzwl1iWY9xdcS21wKp2j3bF/M/oHGZwrgC4ThPzBh07J67shd3FBpYbz5gEt3pxh5FgKHrLqKQztlvMiD368R+L/O1z4pBMu0/MZbfpNhWOns5v3O6enhg57nwUC07JyOjAhw6q/SfWj4++0aHH5JNT87UPUKbOPv5SeCmDwzUNasdd15Vk0ct7qSpvSJSCkTn1Jrn1mrTGwNg9+GCH65lZIEGWqPRr7DUuXIThhmqvVCZ6Bn4G7zPQIvfrTb+0pVyK2Oy5ZKPBotj1NpG0CVSzQrPPecW34qrsQGkcMZA8pFCSbLBL130scmPA7Sx1ZG+XOm8zZjNyfsD4Zb9A+rtKJg20LDMnzKHzgf1L0gPWKOP234eU2HSJWX4rZJ+cumOEIrX76o1oEXmYYtee6WGtJpzZo80F6Tl6iw4MhCcuj9Vpyw/aZnREOs12zUFPhx4GdQlTYpmXna7MlPvZKxIwF4VtQofmQO911qlFPf4oKdCTctC6fTsiO+W68RqZsUDMsLFen6P6/ESAcfG4bPb9zWzs6b/hvlCkZCgcHMZNs/HWm63VJAztf64UOr/P+UVXuoLthpS/EtU40ZvqixgsXB6r08YHnKMXE4DzI/RGQNEUWbMnmQmBx1IX3+Tu4xiV2fTgEkDBaYNs5sxpy/U/NhfBPxjz+SjUZKjOFecFdsXjw0DU/y+D/q79dNX3nESBt+Ado8uVJD068oHuiSQ2nFTmV7PDkCNhPVD6PLEh3E9RmiddSNpmz0KPd738pTs6ivqLdaKnd6Tg5ArpD2fTYpRhllBkIuTLHRFjtDkxEsDEWw9apljKrFVVaM7zSrjH71iv088zx+PGpPnDDS0AHwvbga83U1F1hnR8Fkajt8QyFiGsCYer7hxa8AHutium7qSXe0M4FHhKwh9ppuT9CJIXlyqSwRT2kclDXsiW/o4pEPUcG3fH3a2YbKDdo5JGJEDR5AKnpUcXp5zhee3+l4N53epnPYsFi+7/yYIcpK8no59tdKcV6GyRDUmGC0kGL/ETKMEixF8SmiB9zJSZ0STHVKZtYRAvKoyS1L5+ziafJESJLDZFLCM8GdstudyMZOXjQgjIhT0bj4NS8aF7/+x4gGikT8b8jCzd8pA7mQSbqh9LyLmR2+PB6J2yFlh1LE+wMN9/T1Tum0n31MDxHgcYm3iiOlnZVdOhklq04vyJp1/pcXwEVn8lyhGRiEJiG3A7wP9OdbIqeuMqOVM7agCftIB3AXYZpIkda42IFezHvixleCEBc1KdcAV7lHAPUoCenVk6fSeeGdOlk2lXJTKbeyYmZRsoQf9Im0NbkqKWx5/cody0X+GXQKF0/z4F86eIuHbquMVfkWJvBeNXapkaLwucR43WfVp2whLtXBiWZ4ZJC0IlXJsy8FGf6TB4r9XuFRXnGPp2B98h9rSS5V8mfmoEcvA4tH48gSsFoerUcfpNCqbR0tdzFjB6AUAOd2g/MN8KgwtCBlEHu3r1ivCtrF2w64D4BM0qCzGk/8A5ZcD7lz2OEthyILnBzJds2SI9n5Oab9gS0YzsSeygNzNCaa+hqfmkYhCM0s5E2+kO+JTFn+pPZD2HNrY3l0Oyu1z6pzKqXTdr9uI0q12U5mwcPtdLl9R8cgTZZBTj4xV50SVxK6zvxprgQi4IuRyzywMG74z4gFCz85+o95VXECnPNRnNPu0Mlx6GUSxC47ujiM2HQWjfjtguTm8UPGqlllrJLr1SXem7uFh5LrYfZMHH17gE4eNPTOI7OxT3R/T9w+LEyzL+Ji9FFxRR5Ri/2K4efRieWIeriw4BZLfBn5vzLvGfvJYqY0yZkK4CDioQyIYWq7+WGVzHFI1AFSxc6aoVRc5fg3HGqWn8kZGqCIEz4vByZlLl7Oz8hX4QAmYNRR4gy8dRumZzHXp2rX7VsoVD8nqrOL3gdjBla8Yl0kK+dJcSrRgzoGHbPl7tRR/KkaBzzeyScxb0ZQelwy1un936K9FfqYKGxpsKqGrNkolc4a/WnKmVCR6mh5UPqnhqZ5yXyU2FVY9qdY70cEfY5RIqaPWIw/aMeEGau0XY+0KX+JNbkv8ZZ/n6V4vJGo3JkSd+fEDXE6cKXyJJjGELTyCVRg0Fe8NOVnmuagmMmaCF8P3Hl6BA2/6viLG7+n71Zom/BOT1/wWnh9GMfhuD+LAQKGJ15h5am97FFUdEpf8fjshu8wOv+EhBlMYrvPf4kPAq5vMzdcsQbj18kwnjIFTATquWT4kY25y8SKIWKaKIkGOgNTRSsBfmmwd9phzQRB6N5EW2aj0ajh9xs9/pG3/7CbdSWTKq7W0Se76KYM3pST5WVapG0c+ZBedRMkivtr8nqbvNomr7UZVVaRhp7ijRtm+jBTH6IYtEX0RtMoPr6D6T+8P6l+asSdzPeJ0fiDDEbmEmM5UwVDkZnISGQiMZCYR4xDpiHDpDxzwVFjZ5THn4FziZRGbGgnJyw1+VqV40jbHNoGUaujdUCY3sxx3CiSdgJUSOiIeB7adX82OQfRwkwbCBdaRVyT0l0sWD7ucxN5oAEkyV0Pnt+h6BnszZs3iSrQpuHRKe7NTLDzo1PK/2hHZ1h1RlVnouoYq46p6lhXSGsmX2HDi0y4y3NOQF/ORcMvx2Lvh2qPJbpLwnfJEV6K2h5W9qiuV+ik6h4U3aWijATdjivbv0L6oG6fX/0DWkQJmh4wcW806Uw2OGMbgC19kY5UyJ/OYzC6T8RvTQDhx2mDVtKgd4GvWmLIG3TyFbqH8j6VZYu2bDGRFE6AQMqx7KZQr1Nu0KU+SnXgOu65sKaU04NayUwzbftGvuevtR50RLcAe9CRua3nIB9zQUpKXeGW1CvGL0vIqyoTptKnycuSu+ptJiV5RvDAXqBLyS3mUk+vTbwyJ/RDHsrU0hQvIUmWPnS/DCUzh6K8eTPXnN9syx+VV9pmO+thX8f8rH5Zo/usswRfYw1WfsZlyjrFca6xMQ/ZFw6V9lPqwl9yUzKU3/dLTEe2j57qixG7UaQMfPL7lW9Pxo7qmhGhIanQFevo6U/Cfhb4n4Mwtvuee8Q9MmI+9Lxg4Q7k3Yf7vMpGqFCy80TW+eVb6OGEHN9R4A/HN/gRUswNfb/Pvk0oYHZ8xU82eU5dXIwE+MF40tLw8uT5Owq+OCUAIi92ShiMxUzl/UAwC6aDOE5TJN5xV4fiRR6CnnA4VEh1WkoJUNa4rtPN1BLCBb+AFNePxoGIRHmbue3Jj67RPU/K7lW/X+EV7DXv79L2MAobKMBh2XFSBZ7sUNSa6owl3tVUQ8ZcwPghDIPQ4qe4waC9pcpPjjOb2r6zAvk7pOBNjDeioJGL06bn6FXxx14FGoooIwgtT2dePJ6GAXrpIFSw4oxGEz0T0Z7YUUx0yqBeCB0d/lTcPVA/h65fmgkb0pk7po9BfFhqQLbsq5BWVG4VHmgLPPldAZI9iX6gLas+D1ZoT99BoJvZ1diTDyEc4EUePL5uYCgIjhrDKXTYlaGfcv4Wgj46cyQPpgIe9RqUvMKtIxzuqjazALNC8wd7THHOEpzqx/iK01mcQPxq0ctdWlkUw2m+4gUWpQHs/0ZY/e9G09FiHDsj9AoR7SH89NPRYe9D76efqJ+BS58Y3EvDhjfJl1b+lI/5J2slWiqdytWPXCAVPwySLISBi25oRRD+woJZyWHmPwLyuG+ArD0umyQrcs1WYuaUr3Ak37U47u6ab3hMyb8i0M18QyAMFt30hBX/hEAXoDbxtAnuh+LDwQHFguktuFO6gME9lsirLOi2bkd8m7LJlkt+R5eq+O3cQTdzM5fkVFrdhbg3261Q92QzY1EDkreO9auxOK5WfgOPuom7jU68f9qJazX1eKsW7wOrgEB+w7S6UxoJWq5arPZIeQosln3giY8POo72exedCHv28PoCsLEWbZ3x6wsll9hk074d0QR0sLB//KpNhVq3d4GNxSjWjDnbdeZQY5vNu8jn5JwfnylbTFIF4gp22sg35CEyMD2UpxG1WlSD9QYQjAMQDMi+0J/eo5y4NH2XUaUmu2t0twozmYWBydxEyutLJSOVR2M+Hs1KQcO/KsmGja6VQjZ5rXbX0I2hWfpGPbGCyhXT/ajhxB7id3WNYQN+JfVFWDOBXSHsykjqi7DNBPYOqEnxNktgWwnsAmETvEpOWE3bZb6Dwi1FNJtEaCrOLl61+D39SoVZM9HiVFJXS3hQS0dY05IR1PL0rVGSQRcW7IMOGLYuxH3zWldcUxkECz9/V8WYG4Nk/UAHR7pdXDbi4KA7tJHX3XknZUI+vwqtyJB2G49Q37USL1mBROwTV99KMuhYyl4jPXO7Vq9+cPxZDqyu4q3m6220x4IBSgpXjUuEi0zDEx6UGGlIokYkPAbJRQtVqbduqRveNw1YbRtNkUPjQX63+7gE2pr8GS1vn3efljzjednHJs6yeTNt8rxr5jIySMT+/j5wngbZMMqWagcHB1omGNEXhlYWf+hLQ3zaVkfmT9aHol117G8bezwI/R+HcIND')
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
tt_v10_m32x__Q_CUDA = _zsrc('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')
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
tt_v14_qkv_QKVBIG_CUDA = _zsrc("eNrNWmlz20bS/q5fAWuLKoCAKV7WKgShy1ZULuuKyLXzrkqFAoGhCAkHBYCXFP737e7BSYKK7DipNymLwGCmp+fpe2a2/mV7pjOxmNA1J5ZRGx1sFVv0YOJFtstKvgyGjb1icxhZthcV2wzn3g/saORCo8WGtseE3z4Jum6xqW0yXYfHoR+YzPYc+KbrW/B1AlRaTT0SnnTDsgLR9L0wEqa+bVWFsSS8BCyaBJ4gJv0kXTenkaHfM48FtqlHvh6OjIBZ4lhSl0gRxwI127OjdJQwVuBJ8ICgEbrQxTEi22HitjswgsBmQQ271zilTseMjNpgry3cVup3SqWhbnc628E2zKDgjyd1tl3m+sFiuzglm4+Z+T2T4s+U1fg4PZqvMaAr38nCzFhf9XhkhAyY2BKEIh8vtYDdC7UxTCmMVeG8I6ScRcGCaNXGBkh0UTPMp4kdsBpIOozg+yqj45RR4ejdWBgEhnCuLotMcz5yjG/lOQ8XnimuQZXwk8wbIxYwhwGxmuHY9x6zVGG1H/GeMJ30AnZWYcupn2uMV5ELDO8RWEpbLRW4A/kZYyNDIJ5x0moKlTogoFSaMNO2Biu2pPz6iZqkxhptFeQ2noSj/OxDxzeitjBV0rZBmQRDWCQCt8ZNIkho8d2xwyIG6tXpDBYRC2vTdm0I3JLEXohhpdJSKu0lCPHDXV7VhtvitDZPHhbJw3PyMOMrG6xLFfneaxOwFgvN3Noyo076iOJ4pz5vDeE/6eCgLf0hpl/22lK322rmmxrQstfMyS+eBl2XGEuLmiJ1xez8aW0CugpSEiqVe8cfGA4OCkhczrYYSXn9iJmMCnKKXCMvpry3ilzSmUeuOTnBKSlHY9+xzUWZIM1xLMjBxHmsRcwL/aDWtNbkyrl+XbznTWg2zBHTR8gKifkWxPxCcl7iSxtM9UNe0Lh6lwvzMdbWRLT0MeZ8k/ECLK1/DpfWT8alrrwRmV1Qx7eCc3R0cXF8dLQVMWDEiFgX57y8IRTO6G//gn56/G+ffq4/X5/Sw83pJ2Xg+45wfXP16+fz04MtXeeLpDjqGBPPHOkDf+JZoa6LIg7caUqHzQ97nUZzX4olM9AfWeAxJ46rQCOwLZ1eDC8CUh//w0G9MMZCNFNAAILwpr5zRde9KWQG6Kkae1XBn0TEuatMvJAcruD43j39qQpAwx2HXMREEoJeIGD/L32Nc99C9jsNRTi/0cRmq96Wz943pN0zRfh2ol3eVGFZivD7ida/4I+9E+1Lvyp+O5F/PwHIV+lefsVB/YvdtiJcfbvUANEDrX0oipdfd2G83EPiPQlpdWL8GnvSIXyOP3Wwqwyfq/Qu0pNEA0qmuznRev1q70SGv409BQT48SvOqWmtw3rn5kRu7OEgNgd19QBcrsEkTIpPIMRGvdmWhBQ9E3oIoXt7l05GE0W2pUWjgBnWZ2temyuCY3hMg9adFmCHQUYbOL75SF8rPUW4D/zJONe221sh6OdGLAAxoOLPNBpWRTmgCKAl1ERqkxtS3PweWou0Hgca4rT7pS9VkZVdmN/LN1IUlBHe948DRfBnXplg3lOsTKTj2p4IElQuv1J7FZ4zAWQhfAAZgRankSGYLCYFoYatMklGEcAUo4WGzdgCKhQM+PebE6SX+qMBu7c9DcJEGAHOrs48S6tjD3uYSV2MLVOKuycRaK3f+U338oZrviBAAiwiUDZKTLW7wId4efP+/EaqtpqqLXNJfLJdCLvxEIGQBQO6Z5q9K2bdYYkPml3Jt6jxCDH1xlUJwJBpNGrnOdmRJN0+3PElCcKS/hb97pB5JoPM0J8vVjOMyFhJpGi4Tgkc18uQg7DMQzqeKeO5uu7foX/EuAOtDQPDjGzfA0eOvhqrhkgf2gFkOYM4bjdq9SRaj2eS9CMEHWMDvXlMbxL5ELQM7Xbn7ifFtFXFIV9db+9LceQcK0DwUQmUgRIPU5kTsjjfKPmK6K7rI5Jt13/Zk74XEs8PXGh4DWQEBTy9RaiQAidLy4w/VtFKr891Iq2IyOK4Cu4r6LO5FqLfjtWnsBCICFpTwqWTcSa6q+xAgKor3AnRePFxINtSFf0KRNbCLMA4p0xAltOKh++1FU7yvIzGv8aBce8a4JYD33EKNvwEFvTU/dJXn2R5nV0ZFgqx6an6O842V8SUW/mJJgW/uzLfPC9ZER29lkV29IR1CVF/hSUbWEKnAp5Ehq5xGUyz2DBDQ1JLNFGKu5F7pH5tTK/X5NI7aEhJZR0MkNjP8ByrUwg7O5j9dNvEf6xBMJ2IUbgKAQPCq6SWTZpkf8RhWiPGKeJa9fdmFEEUdheCmIT6L9qpbytxeRvWAuEtcwSb5cpd9+Y1xx3+6tJzTn+D196wjHdaSxLeWp8bc2ZtqLxLHVeziNCBRllsGgFj2KgxExOKCEWTqjvtf3A9FtERSaDMomjvwhMguQMqmwhxmVJe1fBkzpVvPJd7OwCFDYoSCayL8gc2MV5htSH88YeQMJ3sryQDktXzIjdTCfxHGb1g3IqQO0PBg/nJblO6K3GFiSBGYC+jLnVXRyCJ12zrAfo+dJGk+oBCW6/RB3wnReBBaLgtGrejmAn54Q4jxrr2AJ8rlGb3rmvUuHWQG9oAYJFVobG16gYy9frTcEfaV3DpXPVQ717D5C34bUawgMa7uJLBHG8FEUBTXkPzdXf8LkX358BbWPUjrOOx24aAqD5m+Jb5hv28UaYZljXX4s2lsrgL1Tro7u/4UGnzbNmalQ/gXb/lumZ2dXTU+3Z8DXV8bDc8n3gpl9Bmieb4//4VjKr7jV+af3EhuJSPx+fn6VIyJ7DcWAZwKZu+69qRTknSq8Iul16jXHfQWEpogunT9ou3LfIYeXCwL+38Gwrz/FunIa1OTIL5nonq36Gr/68sFGawfKj57ffJvgWi8zpY9VQ8Io490OoY1PERSmHKbl6xP4zNMJA2GrS69PppRvkZRrKlFsdmnJiH5+JJBueARJmE/DrPu2JWpWKs+kvy3lTP56r+fFFPNRADS0hroGmyT/+cJi+4n6JNK7RB9aCJU9zRkCoozX0o10dJyy5vyQQaBNoIigJZjHbBbCHdg6cKPOHulWlqD9V9bACjbvIxPEhPp7ftO+3luTZXnmsL+PcM/2bLV1XZBOU0u23VLPhc5DtQZqb6qhOOAg2qFjleRMaTLJoY42amts4/fANNru4XpZuQCoL0O4w2zZjSMpf3BV0XdXVmdnETSvIn0W0U0A6UDA3yzLzTdJ3waCa7kXrgiYCNeZdXl/UEHZKkP6mnftDu98nuUVdEkdvyPtgy7WbJoNSKazwyznK7YO3V9p1SfJcbay3NtZbWHa+jl4nlFHPBXPkBoSKyTd0IQxZEQEHTwEIB3ETEpSXdj+Slf5oBAqYcpYIOWmGkPVR6iuObhqOJD/EWIIEWOn6k4Qbge9pEPIC+UqauIry+0/iJXXyexs/0+Abf6cevsogUcPdQJvKopQoSUdKIWJRLu/qAUMMPlwI+NJMHwFzhE0CdhlQKhVWcC8Kn+veBAQuB9WvxQhJoYjcTTlwtz2Jdof/LZ8jmCGCOoNtTg2IikvguLeDTHX7f6ju51KIq8mF8s5GDLQYgpwN+6EmgF6Qp5StaAVdWm8saODIVHxf4uKDHZ3x8pscZPs5WS5j1yge0nUzvIVOdiZtOFidwaQlGC9WIATVZLr0u1GTR9PqsJkun1xU+SrZLiJcPGeJrRkXhSWgowOUbzOd7dQar+9wBTVNw/MRPhqmjbKKnXJG1pIzs13oWtCBXgfIohkE7DQH4UgFB72KwUCCO7cMobNvfVAOHkWtEgT0vhO6au+/t1+bNWgRLDOMMozZo7OXPsJeIYU4n6YAObb9ZFU2z+u8mxBqJn9tVc7vjO45f0jiypfUqe/kDgixK7ol2/Z+6oj/zeAiFuPIkF/c6hORUhnkaPy4Rn6gQ71HEfQIwK/sKQv1UQVzzmwAixlQa2nUlgS8JrRKCppx9k7LoCZhoWTexeKYngTHTgBi9dZVf5s3pb90u+VtD0l8K8yuCm2qlIR8j23QXj7F4bJtW4Fldd8Y/PzVQC6UzP83QclExOVPLomJBo5Bx7lrKvPz6YEl7zsbHxxi5cDzmIfe5NHiW+tGkBNlodsK63dG8Jbubq5E5L9aIjDPqgm2q0YpN/r0BuBRYQjX6eXGSQmIqrigXEzfFro07iMuNpVOuSowXmh5nh65tld3IiXf7KhXqEN+cwmcpVx7Z1lwTxdwpNd5NoFNSOTvYlkDfk+jTVpN81x2HtzAcojbSzxowzNPJbb4NY31a/eU/QNTP14LLreXWJLS9e+G3U8/0LaZZzHSixZiJO+akn1yT4N/6sFgLty09uqdEH0n5cxcqkjNFfnZI54q257GAnzBC3UMPA3/Ob4XA6BZHmGf0CRsC49x4E8cZR0F8kvSOt0oveJnzE3rc4NSLgsW1D0R/m7BgccPCiQPcqdjjjEWrnU4WX1kQ2r4nbpcvb1sR6WgUYiefDIJhvV5XkN6pZwwc9okNDZhD2XnCQ6iUp/ia1ftGva5yvRKzxeX3ymw3vAUpvOy1lfQymEQY7RaaEKylAlK0LXYL8nzBWnxlRLW5VLcKHhGQbSXkG0p23RTalwrjM0OAVxrxwJht6CbyhYggv4//0funl72rG/3i+Fr/dNw/1vv/d32qn/x6fnXch2qjFaMkgReE9cRcKjg5TLIy/vNl//Tm/PT466l+eXV5mlQpxU69b5//+9/zUx0WebIy/rypg11eXPU/X13iS/PD3moX4ku/ujrRwXzPaRpp/eyeoEckV1DcBHpjvSsCnmFNxBKhFWBucpgJ5B+DuLkB4jfi+w+gu/x5d8LIowyS22CRmEhslrqSefqU3c9KWrino8YndAK5D5D5l93VWr2Zlbtj9Q9csVq9Kbbegy764T2sL3StRu5f8ASM7mTJWJEUTjMP492RQ5i906MqFfrFe6W03Tf0tPQOHWR8ypkCEuopIBmSCkokEYaaOWMSEmYdC21oQCaSuGFq4l74NAggCEQC0/DtV5Bfj0XHESjrYBIxcegpSXvaeGHMPy08w7XNHhVAF5Ts9Oxn0G5cN7lV9o4I9iamycIw9a6oGRKLN8rfPuWl7137QYTu+yPPtHG6Y8fxZ8zitwn+ZEaOQhRMWOLdSdXwgsfL2nShMDTUpPUsx1wo7gwNZeilt3KHRs2buDfsPqxC4KjL8E4JKLJ3gmDQbEjqnGzjo+8N7XvA2xzeay9LFX5qcQqhgZtoiWfVniK6oDCoh6A21CO5i8W7rN2xpD5WLBJIPWlijWRBn7h9aSLBw7OKSOKNaoG7dJ2CAU/IH/7WbEsr6RMLAtiCMAwRmfedGk5SDCHDLz0eqwBvPNN+QXawX6jt4A+xBwAeU1NjRTTQnxcydbWoqkDqEFuuTHMyNjxzASp5bEaQKsRMgZymICZlB+hLnfKuJwhqeM2CC8gG7HHgo9pAosxH5mGvzRPFjqUOeWVO1Q6nnUTLkP+Vm6zRXMVF4I3AXMaF135mCl0wOtxrd/BcLX47Q0dRvbzpnOE9IN4GXgQa4DW7sUUanyg5U9fIz5W5QlRd9BQbxxXDWyblL+RpTuciIoh40EqU1bocfbkL4XP9oq3EfTeFmUJ04CfalvLzQ0Q4syOkb0kveMTYO+0dHQkWz/Y6iTv45RcVc+b/Aa2QHIQ=")

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
dd_decodediet_tt_mw_CUDA_PREFIX = '// dd_cap2_b_\n' + '\n\n#include <cuda.h>\n#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n#define D __device__ __forceinline__\nD uint32_t addr(void* p) { return uint32_t(__cvta_generic_to_shared(p)); }\nD void waitbar(uint32_t b,int p) {\n  asm volatile("{.reg .pred p; W: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1; @!p bra W;}" :: "r"(b),"r"(p) : "memory");\n}\nD void arrive(uint32_t b) { asm volatile("mbarrier.arrive.shared::cta.b64 _,[%0];" :: "r"(b) : "memory"); }\nD void cluster_sync() { asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;" ::: "memory"); }\nD uint64_t desc(uint32_t a) { return uint64_t((a&0x3ffff)>>4) | (uint64_t(64)<<32) | (uint64_t(1)<<62); }\nD void tma(uint32_t d,const void* map,int k,int row,uint32_t b,uint64_t pol) {\n  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0],[%1,{%2,%3}],[%4],%5;" :: "r"(d),"l"(map),"r"(k),"r"(row),"r"(b),"l"(pol) : "memory");\n}\nD uint4 ldw(const uint4* p,uint64_t policy) {\n  uint4 v;\n  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.L2::cache_hint.v4.u32 {%0,%1,%2,%3},[%4],%5;" : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"l"(policy));\n  return v;\n}\nD uint32_t prmt(uint32_t a,uint32_t b,uint32_t s) {\n  uint32_t d;asm("prmt.b32 %0,%1,%2,%3;":"=r"(d):"r"(a),"r"(b),"r"(s));return d;\n}\nD void dec(uint32_t L0,uint32_t L1,uint32_t C,uint32_t T0,uint32_t T1,uint32_t* a) {\n  uint32_t rotated;\n  asm("shf.r.wrap.b32 %0, %1, %1, 4;" : "=r"(rotated) : "r"(C));\n  const uint32_t low_high = (C & 0x87878787u) + T0;\n  const uint32_t high_high = (rotated & 0x87878787u) + T0;\n  a[0] = prmt(L0, low_high, 0x5140u);\n  a[1] = prmt(L0, low_high, 0x7362u);\n  a[2] = prmt(L1, high_high, 0x5140u);\n  a[3] = prmt(L1, high_high, 0x7362u);\n}\nD void patch(uint32_t* a,uint32_t e) {\n  uint32_t reg=(e>>8)&31u,xr=(e&127u)<<(8+((e>>7)&1u)*16);\n  #pragma unroll\n  for(int r=0;r<16;r++) if(reg==r)a[r]^=xr;\n}\n\n'
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
rt__CUDA = _zsrc('eNqtWG1z4jgS/p5foWWLlI21DnYYjsEodbM7d1tTmcxNDTOVDxTlErYAD34htsEQNv/9WpLfMJCZq72kEqSW1Hr6UbfUzdWvXuj4G5ehkbNxqb68uzqW2PEmTL2AnRmZzY3+sThJXS9Mj2XUX0Sxly4DELps7oUMfXmPbNtlW89htg3NeRQ7zAt9GLPtKxjdgJZb005RHG1SFqfU8+0kYIHiRGGSom3kuR20VtEBxSzdxCFSihWqbTvblNoLFrLYc+w0spMljZmrrFULvXDlfHVdsRd6abkezTCCJgq5cpoEMNunqeczpRXMaBx7LNb5Al1qHQ6dlOqzfg9N2t0pRm3DaqHhsBW3lJmK+UeoDluAPIr3rUsA2G7NnP8NAv/YMl2utNPdCRwb/x1EGT1DyXoJgK5QA9IB6TFbIH0N26O1hfjSISqBpvFeaNPXFJxgr1PnaePFTAfnSGC3E9zrGm70z1/WaBZToROQNuwAPDVDrs4akuxDRznlsUBXoMjpjJnPaMJ06nuLkLkWas4TlhQmFLM4rBNGz3lwQNcVqVSSGnNwFdMWBwrnTNe0oibffHNronaXMwN/ptUatoggQ3BCJSexqlp5SMwunO16kywrGC5Gcz+iaQ9tcQ3HuZNOwHTO5wmw4rBBEgVrn6UMPHI4nO1Tlujbnj4H4PJQD4C9beL2LW73XvhJv5mW3umCBfOWstV3RWNfNJ6LRiatnJ0efG7EkZ2+X7NTWlTYajVs890Tq3LgB6C8QC0g96aC+hIqKbGSEizJ0eaG1dDmh7O94K/UdRWJ8Ro8JEc7Ew6s7zQy03cW1fe8sYfGM288QyPjjazmev3eseu5LHFqvte8Ofl0VVHodXd3O4cf9e6up/6lVGP93mh0ax6JjNGob166QNKA1rhPUozqN3caSOdfYRECuPI7Gp/zPGede95s46/0lIVJFOvm6ZEt/GhG/Vf9MXfECQ+igzzUaXGqhSMmKRyh31LSQLrbqogt6XyA8dj9rlIG29CUjbg5D1+FVWPx//H+7sq2JS7x1Pl0EzpLexZtQjexbeXxvgO8nt5bAfV9e8XikPn5owfqeh2UrbHs2na4hReYe4jR76CddJYOAh1i50ASKSbDMxELwu+/EcMc3IxvHu8x+vSBPHyF3TH6THoY/efxE/n04eYzdN99+Tom445hFQrE4tRzSbqMGXU/uDt9h30aMgLC61sDZzRe8/bd3Rsc03BFZn7krMS89hjHUVYT3IDm/rHqRUYUvgwGNJh8Y/TVzuO9xrXyiWwHvIRgsjxwQaS4fYFAmIo2YSKuYuTAOEqCydQqY70Dju5sSR5VHTUJ6mPwKqWET9CE0R0goUJW3eEz0kxEkkBTyiUanOKnDyqgV/lqflIom9x/m04GU2sjurud6D98nU5MQPfrOqaLgALyOPJ9DiiKFRESpGutRvffrJWm8RM8mVhN/Q5Tv48G1nc5E4mfuq+sSbbWFGXVAZI6YyBU1RaZ2hlo31U4do2fn5UvO7kN81CCmPtoDIdhBHzDAYKP6x/N4dB80/+dX5DJKxckhEo2WU0n36fymqz1943+c6PPL06Iv7WqSnwvr9EQAg3h6OGrFZ7y0AiSNdlpgRcqIXCgcPPv7kwVB78ZQIrx1gSuhBPmnqd27r9pK7XT78nJ17eqcNwzWCo0S0CzHJnWkoN5hVUgb/M6eeAxwEY4nSwrAuuy/RnZ8xlZQaa27AwqQuWfNxcWjz6j62sR36QLFDZz43iGDdU6NmbOQodB0hft9s2sIKXNlIhvZYtcTN4fiSLj5Ee754kx7F+Gmoixc1CKS19ALjO5/GU4B+hnM0K6Y+6lXK+4SZADUS2Cuzclh5f/T3yfd+zqrYTLhRzKk+6Cl+B6b3/Uez7qZVXPOFpnHK0zjtZBL3v5gfs/AeanUc96umjdxWA5lMP1WwxRIm6FJ3Df3tSqzWlURgEkzNwN85PSA6MfDlZQoMJ7ojuRz5M5XVSs4t+Z1O7Q7uH2G9zu4/Y/RHeA229FozbNatUgHP0MWxpkfQ6EnCQZQrEuMJoCsym4neapvAh30dgXjeeiUWTAE7PzNK21NdCvVvS8XNU/Zbhf9MljR+NP4oTff/CkaWHxUExJQFfMlu8mQO7mVmoOWCabuJAaldSopGYlNSvpbSUF+63zF8XrofozRZm8beCOGcE7nbtanl0nm+DIti4Wv6r1WmzKMH68l2HcSOBBIxYsrjiFsOk0P5kq4YEck6dLN3CjYf6u+rzXzrMPdFKuNYrIZjIiEhjxdPFkROhTsUhjOZTm4pkYEpAgdz29hkegpOCo+Z0ALO7mxvx9+sSlfpFBbtREWnSWwi0pLYZx7Idk24ZkNo1WRNneQFItHvmQv9omzmLI4DLe50++VMUdIlqNAvUAm0+g2YG0RsviKYHtodA6lmoDKX+2XmqrNaNYL3vqkY69dTpS6MksGZU/rh9EjdQsDVKlqPIyXLR2ZassAkpJkkI8iS562rB4n5dZmzRC85BcqDzgTsBjDBisk0pCFFNEgVT+pgfpr9qBUuKNYWoys09SiFUHzaLI5+vm3mIDuQGZUz9heSz+UsnVA/868d9QGI1Z+i5NY28GcJR5iAt5KXygu/f7kAaeMxbpxoMI8LH3zLBApFo/repTFH6OwOaZz/6QNwlX8w7S3Iy5POGpAU/jDbPyhEnQR4hZoS5VJmhOSwB/1gAkyvUcyvmwrP/nVA83wRe2SDpGt9vVoC/ClkP4nRsiduOqPoqC8Q+BBc7RmS94ggEf+iL23PdeQFwvuBUZvirEotAq5bK+FANuThzcGGILIhgTQ9I7iMI3HIs2FPhSaB3hKC1CNLVoqkPSdmYw5xNAQKnuRSGfuYW81ynl5DDGBjakIRTWJeQaNPIe8PJOCET5WXxLAT6nVjvdC/f81065hhXAK67XyGpWdI/Tf3WHZRXYUXl8BKJ2PxdcPx9aKBH/s9WrcVZ+7T1+UCCcVO5FCSFjft9mK0JAcgBRMCIDNbf3NN5HBg/FOwVSNyzg43wruY0lFUB2fFmD+TMawFsua+i9ruHlips4wANV2Gr0a62eepXr/e3tW/GtyX8BYUXY3Q==')
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
b2p_tt_v10_m32x__CUDA = _zsrc('eNrNPW1X27jS3/sr1O6h107cNE5SoATYQ2l37x4K7SF0u8/hYXMcxyEpjh1s5wW6/Pc7M5Js+SWJA713tz0hsiyNRqOZ0czoJT/1ncHIc9j740b3/YfjT+8/sDfPnv008mx32nfYvj3tW91g6kWjsVMbHmbf9Abmdjo7jPojL0rnWe61H4yi4VjNjO4mTjcKrFEUpsD6/sQJrGg0c7rXgT+dhAjr2U8Cz99+ec+63b4zG9lOtwvJgR/Yzshz4WW3++wZvof2Wyxodie+7zYb9OUE3f7brm1NGl2rO242Fl23P9ds3wsjKl5hE519f8Z43VkbUlY4ZjPfBUxcR3vh9mvXrt+z3Jpn1z6ae3ue37Vc17etyKl9bOztNd5sv6vNWrWw2WDft+rGlmlsNYyt5oPBLrdaV+0XADL1b4+9OAheaLPaQjdk8i5J3ifJuY6F3RfaRNcRs8CJpoGHWD7wDk9L9nghejzzR/24x9NSXYaeTZf17Ed3BXoQleiNI3ojq6R6RDDWdQp7BB2CbphqN34YjpNgHGlxactIavaUdJjBui+w1l5g/VpPIAlkZ0B3ttVMUO0TqpCwkMTw3RPfYQr1fow6jvxatMdjSxu4vhUx7aWtX7auDJaldKovdfXB5L0h/AFQLbwDQlvu6Npz+rWxue3t3oDKCPx5zfbd2qDZqJEOoT+DPId932oZW2+MrW1ja4ced42tt5RQisXCBcSoDl5o9mX9CgnB06aSbijp5pWeVEMailqUNJNkI0k2ZbJXlwkTKQ3Uff2a7bK5M7oeRiHzB8wHZQU9ZWHkTPaY689Z7y5yQlAw7GOdaTPLncJT/VVTZ5bXZx/NOK/1ageAt171RhHQvc/rHNfKj1/fsbu+208476M6Qh9N5eFYSV+opS6UUjDcfFDTXMDsMTtgx+wlqy92xL9pO19s6PpQrpSsIArYsj3WiwANR5sDYoeHzNwmcDjAZQEgyQBxA/r2xmzVpzr7i2naMdvfZ2ZDpz7v1ut1/EwFcPMxwHea2w0V+G4x7EZ52NBrIFQR4q1i2M3HwE7wPs4B3VRNjqdusZYMbct1lmpHqFYLPNIcC1VFrlCPHGCxaiyPrhXedHsjK1SQXoqk60+aKRUOpNoZDPCD6RaQDD+YdqxCzJ+gyEER/HglkNDJj8D4SboaDge1oDYPrEl6zoJPK+mZqCX7d6wXSTqoy+4QVCkwpuSvHf4feK4KSBZpBygfVxKtLKu6qSaQ+CRS9RiRV6Fw+XmUcMt+ZpFpPh6MxAaYK2Vce760rMsxnOfMnEBLzMuM8QXW/qT9Qm8zPl+eYWkWWTeOZ7DeNGKTaGGFzLY8z4eHwJ85bBTtMcfrhywaOqxnhSOb9cDqvmGhD/S04MUoDKcOmw+dAP6ClxE53gZzpR/0IXMC/Yw0+LCxkbeOib2ivC059mexCbm1ZSNaCZ9HnLFHA9C+EfBhLIj6WGcHB0j0Af8HfFmOrpNH6NZw7Iy7gGTO8pcqJcGr27VnkdW9djwnGNndyO+GQytw+tgu27TdwPJutKzCWE/CKZhLQdeOLKyvKI2UFgweo7MnVvEcgw0t0d4qnlC/xsmxtyfQTBBfN+1QG08xyHtW0B15I8WZgBylE7YPHnpseKt4Q7lgBLhi7aQDkVXrbbfA76lfIfqI+h5Zs1YgUBYQoR/AQH5w90J/BMrOYuLYS5GOFisxxq+ZU+MwutEih33XWNUBhP5o7AeOBxqQKJ7XYvSyJvGkUrXAcR0rdGqCNQgftXWQn9KN226X972gbUkcyYKCSNC8tQD/SvhZT21+bhV2PNs4FlvVZPk2w6hroaPY0hS5NBg5oS02U8UVR7iAbULAhFzNrJDKcYIcfzxxncgBXtrbI2cMoxroc3I2+p542PBpUXjjTVHgZo/LNDLaII56DOKgxyCOeQx4yEMoAcK8iCV7QJNyAhUFdylpyigu/yavub7/v8eAP65ZbQJ0YZM2PMcyBvBosGsTC2bOu5pl305HQczGOZGb8FgJmAwAJkeYFyx03EmsFtHYNajBh1g1+jdSN+apEWtH/4Y9hwkypeW3WyW0/DVGKZVZhyotm7ihO3zW4fEgrBoIHe7G07fAKJIu/jtQi32gCLNYL/DBcGFg03ihZUcj3+NqmIVDfx6y6QQahWLcmmIDa+piqBHIavUxPoDz8XQ88q7Z1BuADgmHMDp9CwhdXmxw6LpAyAKeQLujNEthLV30NW1ZcwrWSxiWgvZYfT4EKjPt+WbtEzeVQVuO8itE7JA16vzf1HV1MFnRxtQ2VPdWv9/iEa/WSzQOeP9lRo80YW3BqgesV1u0IX3H03eYvufpe0zPeXoeaz+hwdbqv+HA7S78oKWlmwbNh0apz8dUMOMYDOYuL6F146pd1H5afTGQViUYsrWFAXUNtqbUHZXKSjP/t6bqfakG5lhqwzEJIz9woH/erNujzprbFdQ/PDA5wzGpTIAvu5wUDVmoG3jabLMZjzdF9QubmNFwRg5MH+BUsn0ckl9OLw5ZGAVTO2KnAOSXMeiIpEj23X79EKCF4JSCA0NjDHZNQIN79hGaaBns7Ai+TWhqJRxzNZxmWTiN1XB2BZzGOjjN1XC2U3AKaLhRMMOPY/60ZqG9nOuXEhEEtrd39vFKMRa0l0e5AkeswsztqxUhjyJBiN9/UQt/MWNlm/Rdg4bIwWN//cVEuoHlfpoE1vXYAn0f+K5LrQx8KI+kcBlOePC1z7IdarNq1eXt4L+jyxb0wGVVRiGMxH2bX7pXoJ1yxcyCYnf5Yo2CYvf5Ys2CYnNe7AH+PjDHDZ01nR3yzg7znT3Czg6TzqoLdFOoNUcMAJNhO1UgcSlwphqyn2GYYCLHgY1MmWNijikrlgqdXdavUIXS9x1+N+g5IrgGO0LqbiM6OBr6xqDvBei5AH23BHRrQ9CmwNpMsL5fAnp3c9D3ArTEer4EtNnQFb4Q9tOHEGA6zPFg2t8Dc/GWAuq77C/mWp5DD014AHt1hAYoRcTheWi5A0zvQHqHFkb++HSOVhRGhSjmh/b8v0IGEuh7AJ1BmbCW0Tdn5xuom4kV2UMtrUrOzlW94RQuiqCpDRLi4LoDBfObZtGiyCLgpWjx5Jepjr3TdoFsGq+6g1VNyK/w1YucPMXSFHBpAlpBB1F+AsWUQmRUeQXr8OgyuGJ/HgAG0qg96n+zbCQajkEIPQAL1ZJ5woIJXT8KDTYfRUPmWPaQjxdYsGTCRkOYMLEIDEE4cUfRqxMGLkU0stzUKNiuFYbsw3gUbTAS4IFMbSdlGVVYGNjcNjr7jX9/Fl+di3OeikZ9nvAig5pkDvzhYxYTb9RfAPmgaJuSQMLfeArsOBlRYbEfCkVV46tewxi++kdwfAx+wsdmAoA/49BM9HK2KNh90MFLLRzdg7uqT4ALsGPAHoDblWgFu6PBM1hYlCNlDKYcb5dF3DsBSxyG7fwCuOhV4M95BpuAaM2tAGydzgWwFMgKrjEOXFy9pJeyaI29/+3oV6aByY3j7Hvunb7HTOb57PT0yMDmGviw4MFYg7XwiUYMPSIDpJdHabGmgcoBeCG6YzdO4DmuwcBV1IinQFSARwAJfxpNpsBMaJMBPPDS4gIWQD95h22GuG8EcH2FwWlaJBVRYWBDVAmOB2NvNnZF3ZaOWKUg1/K2COcW+X0uvjvy+cN7nkCCCNa7OD36gyePT8X3mfg+Ed9fz2VC5px8EYkO//5DVL0QUI8/Hz7rdrlTKuPu3a5rTT172O2h/xl2uxpvu7Gen66d8Ux7lp1MuyBVQMSRHUET84l0eTLmdqrUQkbGU7lIVONZfirOFHNCIa7jrhVcC/IlyZMk+fVcSSv5J194erXZttpOiynAaOX6FMwC+LPHkWqnCpxhgTMscAYFBKonmHmCmSeQeZKv9fUcS8Dfn+nvXtybr1T1K9X9ipVl106+EMwvBPQLQv2SAps2p7O2obCus1aUAVx8gfkXoDgufk0Do2jT2aev8J5k+yVrvNkuKHJsNnCKEhasqVizTZibXrLnAKOg2r9/+/gJ6mHRQ7T7i9oG1ZG0DnbD2ac/FGzwGaQtyQHn6PxT8gi+yYfTzxf/p4Aowv/8U7dzAYCSYs0GAurSJrc4cxtdr0+ZkqA6CiB+/tL5N5KEkIO+IUlEuqmkW5I8+IwpQB6/CGkEOw1Rl36aRjhGuA3Os8Ywl0YYaPO9/gg1p+XuIyWFM2qkJRMGGashLIRSQSnEEI0DjOIEk8CJQAWE0T69PNRQRmmGcBbw2iNvHeN6tMRH8VtQKGBlwEwQUjCX2fCe4dLR5RXWK6wACjmOEIUYhUqLwoJvdwFy/XEaU4snm0nyjQ6Mj8xagZmiwrQT9pppxx2Uhg4Iw3Uw6r8fjWt3OtmqUKIBuXWOU0pVhYsiAqQLHWrYJT2NZ2DNJabHn6EBENsKimsFJ80Kil2FvTEb6Upxlbh2VfazJWS/QoIuoaDX2ai3dhF7vi5IajicF2FN7ziyAJcTUi4mKn4m8ZM0UZLgXeyCpXQ+2jjIbUO07n7rL9BDmMy57YP25huDDALIgNzXNBlF1/xpi57I3uPF0aZtZ8B7UUrzyd5f/Ap/mjhoND/jWC6yVc9QFQmyAxZozxIqWwwnz/mNfH4Nz9m6HWw1zy7ZYr/+oahz4DBqjUYG2T6pmMPtV6gHlSvQEIzi1+zra6SgRl1Dmt5BIShdZXHOQqd6kIVdyFYPyF9N6i84/0HhQCcEsxXseq49IDOUv8HyJ1+y5S9+R5N15GnovoOYGUxDloLpEHdF6e2CiRtDyCXCvfG68kuSfQEqtapKtq90RRS+RTWauPiiZLn1ZOmskm9DFTHOAsqVuBvSCdxcv9AfCZeo7xaqocRL0rQOe4VUwrFD/ih89fU8xodtsIRLi6DmplXFUipV5j0RDCy704oVaQsDHb+T59jYnW7QkLrqKWs9lAwPKIuW0usvGPo1U0Eu7MNucMRwUuhIhGLfaiYdtxnGkeIpBGq8ZruYm3LhMsoQo7qoCHhpHexfytmKc9qZWtz0BacMyi1Q2wvXDIULpuexZIoTVfrvOD5VgA7IJSA5sH6IaIQILcKBpJK7er54EQ2LCVfA9qG1iTQDTkrXs+tm9kQsstrXYmlQbKYWK6h8bRCsMVwM5hvuaJXUhURg55b7BIfx+aqSmwanYh5EnEr0Abe3YytZzl1PvWWd9MfgYfNDCLnlbQGcUO/SogefWkNNiXopCnF818Xps774RfybFqlGtLm5xajLGmX2wYN3pbAkTEo051ZpzgZ0ct7f/Bbn1Em2jpZYPM2GLuqrcwQvzEJ06/qInKz+a662Wi1jqi3qGRmCWa2Cjj7kkJXxEj2MCo0VuBBvQbGh3URhME4z6gSYXgHa0ZsYUMDQ08jH9Yprpys3ax+wy5dXpFIwgsW9zXCUCFVmiD6zQz7noAOkih5OSuEIlNHJlyXh/owG+3ZQb7Nv+zDNs2r1m768hrpQgHXc/dyywBLpB9k5KC38kqIYhNSQFpWzr1W0W/TK+QUgWDn7CE1Wmo2qYKyipoVhG9gH89uqHOFwVBFMU9W+xWCyAJ6qbFBRrNM2qmZQU0/QARzIQ3vlmsctjtvt/jGFAW/1NAtqt8atAEcLW/PLzsXV5Tl8zj5eJZZViy1m9OYUPo0rhaExxlbMx0s5MeFCHqP8Boy7jhOLlqtiTsyJibTNuJygdIBAP0dLBZRvdC3stfklYnt1+e3q0r0qpe/mGuivRHugJSQ1EuR+40oIlJeuaqKsGD/HmEPinIt5oZ7MC9tJ8m2SNOv6CtmO6eNx+nhkmSB9vKI5OqsX0fsC5Zg2LjSgFNrukO3pZONINQkmfEPPmB4haVKV3YFlOIG9q3KbnnEeXUz0ZTDM8jCEUyx0OHmBu7o6828iQCEnaQgk7RBJgbWR7bXQYFK5b7LZd2ywOC6yiYm1RFHQVhxSE4DlEkWxxE5IlhiYZduXZ0dc9FHIW1ftsiupcu30vyDtBdxcYnazeS0barWwEmhk7N6QRB0YyUZGqtcGGc9x4Lv9xHOMs71FVGg+QUPOZJmnqZpTHIDYqJLfpILWlkGxmUzXmJkm+Yh3i8/z+A1uRudivZot4t8i1yTEBkYoyikHQOvw0Cv2B4qATKGV0QNWulnmKGCvS5kh5Vli3SSQmdBpLP9U1x81UPdC19cW7E+mPN6lH+/Tj/OUQhKmVhWXr7i9FasCnpsq/Xi1oPgnqvpOYiUkpbtXGdwK5hmhU3YSnbKrzCqmvnKoNphZ/ruzyyg3vaRtzha7rx+UnGAMdm+WLFtdOY3wfz0+wR3c13GLXo9PVfh0J54a9HQvnpr0NC+E0oJ3ZgzlDT1JKNv0JKHs0NN8mXn5dOZLQyvriZSbp7bqwnT2XgCsV2bGYi7DjRt4MBv4MPkYBngTm3oxVQ2dmNHW8Wd9MzeG2+Av2ewgUT/5MrlT5cJFmbXo8Nv6c/LJMfnklHzqkHywzo1J80Z5Pkm7HkQiIxxVgVDtIh13xB0RsW+u/bgJJTPGaQQF95ZjHguYx9o/O6qA52dVq7ifBYbIujqoL5qD3Tp+phl55Ap8mQNSbt8hra3K2cugRg0mTxnjYvMXMyO2JQi1XplnCfVHeUI5HLgDwHcRtqNzneWQ3bWeWEq7G3pHhYoZZ58Zko+eFB2deZGo68yLRHNnXixT4mox80rR55kXiWrPvEi0fObFCoX/8DdZKQWmb1mTYr245hTyFHfuHSRLnrgCc02HNnAhJbbSP0zTaydkP5yjsfEtp8vooCQa6bgytUtHJLGVotmh7xdkcgh8lfKAOqYn8JpiI56ul9v3ty/U3aHGBR3gFMwTDKVqUpTPnQ0H933BZPDzOrfDmaCazzk3abkUpyqKaLRqXig7S4ihLFBAWqwqy6qeDT2KdKCM170llxZXs3jArIDI4fxS02hlOGYqudSOseRbXQ2DgwgXHnCg/Xw44dNIX95WWkK7Y7Jqqg8N9aF5pa+y+gpd/QxNn//dNC2UImV/Pa6wbUbkIlmI6XowQx2s0BYy7toqfSHjvq3SGDLmBbLwaPbeLna/6mVVpUJIyZflRm+jEEqJkc+TucylPhh/qYv4C+dlHEIcSkNMupXbqyQJQ5S2aYpsAyVA9jgKlpYAT+ym9VCtauSzgs/yGs+WAD0ny/0IWvH1cZeB9Ha5o+tNSMkr6047uOqlyR1vMJPh/CXd4RY5yAXzgLI/w8egiDa/od0Y5DHrcm3rNm4shkjTEnnXu+11PpDV72+0L4OvK/s3cldVJ15c9t2CPqi3OfVMo9cwes0Czyg19IJORdyYdY/GVhSMFpnLoHa93dqiIT2nHhCHu03kMDUUh4nftnTAr1uSZ+uBHnpBR5BFn4RQK4/QEj8uhZZMNeJUM4vqEwS+UFVvKPLI7Y283IPv0jMLrZvcaOOeoUe1SseQci034NPU1+n3lfp+xXGjHxTlLBNwS4RnkQkGFk5Eh6hcwGSG5L6wnp8L126dAtsgZrdMo+TiguvEfPWmnQK+oVBjqK7wF+mhYi2Z2gHA9/Xm5bsgmMlbWBGkZHGAEgOPT+tzoTTe47aG/C4WJThaOcwurMV1zVJ1Uwe2CqRgCVJlV/w4Gj9mbW+ZLDOUkSRMu1CCtAslRLtYFqCl+kmAdqGEZxdKcHZRHJpFfMoZG4V2ymqeoX3ta82Ztet3RV6KXEgh0zg5Fwjft1fsz6JsfvBz2asGvCo+760a7blqzUIDP17lWanbfqYB5hjvCROTnnTAsmxV3iu1OuVsJhOPtpSJDMaj+2k8uZslZmk8NFB+jgY/DqaKbD/JSv0ndvbhR8Y5lrromW0JRftYsyH1R28KKFqxLV7LpWMr2e1mfMt+Zt+zYo/g3k8UTzC0RAqEzZzy7a7isQGP/eSxmQo+ZZyQm9Th7zH7i+Hh1vXGl6JNt4sCOtnLtvo110+u18IP2Itbb+juGuWluF2w+GVDXOYjX75YrvBepCo25Q1AVJGs/aq8zKsq79jFhC0T8YVfN+Kun9s094o9DRaQtwcfGz79ZfxWMPJ0OCk79IqhRAywQSSZ33eBZ4aqdAQh5QQLG7LbxWHuWiE/7KpRF17SuoG8tS7ZAJfVtYgf7zOuNLx1mjs7b3tvS9xzF2NmMDM+TatsQo8PueQa5ftngnU7Z/4pO2ECdRMMHZn7malzDr6ospyKxuy9bMHiDW6xNbJ+UBBYXazRmHJDTqz5VCfk/J08qiMOz7RzRQJ/jjZn6kzL+Tv14AY/GAfmtDy6QIoUT0hiMKHM9rqNfLnHxuBWDd/3Z6vdtCIXDac2RWhB3myK/izdRkuHsPEu06gu465ZOUWHAMHa6nmedKBZnm38Wd6UgzKNsq0lPKhneAJ4DPuxz8Z5c3QT5ZJ2zs5Q3fhzQ2H+pTF0VdKXaUZ+wlIdjNev2S9T1+WXFeyJw/0aTiqp0Yjj1TV27I8nls13zIYU4zD1bMWL3zGSJofxupbsGcatt3gKvzvAVpP9t3QoX14aULTzk3ZBID9iS7gyBURx6DgIZvDjeB7tbIfMLX5yi7ARWXjI/yI/T4uLi2Z88fM73f9E1zvRRU2z2vyh/Th1leb3LLd7KQ73VvJ3lrtJXVSZILXgb68Ed9MiimDSH8GUQLUMPyrbxNsFY24L1tlw2K/jUd/JDjoYHVEy6Be/Z8f84vd/zJDjUEdPGOPrf+LYZgwvOTklxwTNNFnkXSR4Hq/oGIp4nzmIgsc11RMpWZqd/ZaaY/nJNjws8T+NcWb0PnbxkjZUEX1zC4KIZZVlZr01i6/qVSpySsCDHTJtKumGks4sva5Yai0e0Cxvl7x/Bklg0HUzeEwZv+mGGbxdJp4DipbL0o1JnrHDKd5BQayT8O3Z1/yw/1hUl+kqQugSb5cRF97lZDr7szzxFavCw7VxBiwqFg1HYVeUzS6F2y4twWjLNQiZU7ibA4qSUZk5CpytgLfXQA2QoioTR3WV06PZ0o4pgupIJA0aq/L1OICSaSFzZVBww0uRXotvD3LMgtuDsoNP6+oV6M7YmsiLFahPxBNVPjT1nDGXcqZN7kx3RGiy7KVC61q9zUddYt5WrxpaFoIpHM6HVVupVb2Hs5BymjmvA+lCFZkJZllm3TR9vlea2gZNubHN3C5Q4IFjz56qwMH2vBg6zAUt5AT/Cpk/9+IrsHCXjrjeiFNeXELUpmuRoN2xHzns+OIoZJNpOOS3bMU3Gq9TaJmD7Jspt/i25/wMHA8F7pYEZsMz5Wv2hQXlLg+g29jpKHt93bLoD9n/UmYNL8NAXoEXh+tT0ZIQbJpa3hJyZZvhukTheU2bBxtMrLg9IVuDuyux23ndLl5REKzPD6GTACTTEL9bIXW3BEonKYn2ii4tuSftCZN7msbxlQ+AEUd8VlyY1Ez5S8dL8etGZ+iRsw28vjdILsdYtzD2sEKxxr3nm8bS7nGJLWelb5LOo1uoc1p6WVsEWQttENugO1S4QWKvNZ5KjuFGjQiPrbyRFhv2KenocHEoqzD/FvW2REkVajSOaaEccbNfGv03sR4oVjqF0GM9tGwH5g90BTbahfnImfFHzHhPMOlto4C7SwQg1up8ofKX6vmS+vcfrXkfCqycv0mrPkKxlddpS5cyxSWh4r5u8q/K0NafROAicYZS6uI8/8qUv3KE1/DEvjW+7juz5GAu/lrsr070nn4NQXsJ7/TkBc+F10dRFIx60whKQLfFK8w8tRYdspxPafn0sxO8Q1/wEyJmMAntIftrnWCU306d4I7VGb9AE21uU5QJQe0sGP50w8xhwj8NmSZSooLOQMuS38lvTe2ctsHglgAC5zqsmPV6vYq/BOvyn3j8Z18tusHYiztH6Yex6Jo/vChUphdJktbr5ENy06foibi+U97uKW/2lLd6GssmBaJQAjeqm8nDVH0IIxBO0RqNtvh9GIzU4j2zZX68whnP9mmc8IPjg2ND48LHRIwHjgWOA44B0Z9oT3RHmiO9kdZA5/R9gRPfHdl3gAjnxtcMeAZL8MetA/mMcsTz5KXxUkR+gaGIpSNkAyveP6DJ16r8hNrLgWVQ93XUblimM7VtJwylngPRFbIpngdWzZuOz4GlMUoLTI2zA0Ze6GbK8cjb51PFoQYlid878PwOWd5gb9++jUWQLI3jU1zVG2Pjx6c6Lfcdn2HWGWWdiawTzDqhrBNdQa0R/3gaXieI64PnVOjruaj49USsGlLuiQT3heB94QC/iNwOZnYor5NrZNlFhHSjIQXkxCWGRvHyJyIJefv8ilRASKSg/iHnG6VFWeGMbQG05EXSXcHVekdeQwmdob2RgP1JUqEZV+hc4Kum6PcWXUYAzUN6n9KyRkvWGEsMx4CguCKWdrbHRd8kdKGLTym0h9btc6HPEcxzyJVkNZO6b+V7/lrrQGsUoegw+iHWTMnMXYWS1cSFhYRbcmnha8avc5H3UI2Zio8mL5DdjU8hKlY5TE1UHmh6SHv7i2KukhZvzPieUZacHk1WAQhS7C3T/Y7ImEVwimFkYjYpePyqyeL7/STg1B1/ShNpxHCvLYpGJRs2E5WksK7Z7QpAqwlUhWMye3orLGbjeJdVum7mctWCn+bAS1JpiFP16GZWZaAy1yAnlFZmN5jcu/07L2WjyDzFUlmhXNcpVl4qaWcjW+oV160D+bOCsS5Nt91RjSJs1chjDMbR+zvPGo9s1UZCgIbETlemC1ffCPqZ7332g8jquc4xN40Q8pHr+nOnL2+yfMiqrxCVi2w8Fnl+TzO08JFMi2PfG4yu8fdeMfT7/SH9NsaAWZG47sq1a+KqVijfH42bGl7nev6OrGOOCRSRV83KMmgsm8r7viAWDMc7cY9vOBbvuDGBQkRTJj1hdyiRqDbJPYBZ/apGV+sWIC7oBag4XjjyhWvB68wsV/7MGt08S0H95e/ReOiseH+f1IdeWIABdsuK4irmWjQ7YB73p+7Iu/5M5srnwBk4Ad4JqgJR2l1egTQ5SqLYxw5tLmnkUOO2kR7jDjbIkUC/obJWbBc1cbmK/A+8RwJhfwgCP0Db8wCfPtn2dGJ59h3IxRFZ9+QrhOA0nE7daDQJfLSD/ACdDGJ6YJBwrLclmzoHB4q99PNs75WJv6eUxaMV8zW3ksY5C85U3aGMMxTjjGPignIvRl2QLSSHiEvoS9fWlxnNe2xjCgioKCThWE95ax9h8AhP6fhKAmHnFBIB9jNoGskkBizhshP6pYcPCw3Qlm0tOeggzjnM8W6YJUWyF3ovlv1uWa4+3blKd6Evhx7/eMEhXqmJJ2kM9F/AGERbH43Cpf6KcsodPBXaKSmPeeNeaOX+YHlpuo7lcNdHI11gmqu+tsUE5jSGqf58YX448wOIRx1f7ZLXXN435FEnIF3ijf0vfMSnuobhfBTZQxTbkFZpgQUcVt+TPy9c0i3eN42WYcL/OvxvvNmGv636221j13zbgNxdw9yGrx0sgkxtLAwkAvZ9bJwZJwb0E7oIvesY2ANEXODLUW1zvMyN8ar/t/GiSxgc+knLvcScfyt/hOY/rb8J0A==')
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

V18S_CUDA_TEMPLATE = _zsrc('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')

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

