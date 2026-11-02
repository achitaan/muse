# okc: problem=muse_prefill_layer
"""DIAGNOSTIC (not a real submission): serving-op diagnostics run inside prepare().
kernel_fn returns correctly shaped zeros; the tests are expected to fail. Read the OD_* log lines.

S1: fused post-expert norm + weighted sum (H=4096, topk=8): stock (PyTorch fallback for T<32, Triton otherwise)
    vs patched min_tokens_for_fused=1 (Triton), parity vs FP64 with poisoned remote rows, CUDA-graph timing.
S2: local-only expert-id alignment: moe_align_block_size(ids, bs, 256) vs (ids, bs, 32) on an EP8 rank
    (local experts 32..63), set/offset comparison, timing, and both Marlin GEMMs under both alignments.
"""

import hashlib
import inspect
import statistics
import time
import types
from collections import defaultdict

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # local CPU dry run only
    class _TL:
        constexpr = object

    class _TR:
        @staticmethod
        def jit(f):
            return f

        @staticmethod
        def next_power_of_2(n):
            return 1 << (n - 1).bit_length()

    triton, tl = _TR(), _TL()

HIDDEN = 8192
BUDGET_S = 230.0
LOG_CAP = 9700
T0 = time.time()
_LOG_BYTES = 0
_LOG_DROPPED = 0


def over():
    return time.time() - T0 > BUDGET_S


def log(s, force=False):
    global _LOG_BYTES, _LOG_DROPPED
    n = len(s) + 1
    if not force and _LOG_BYTES + n > LOG_CAP:
        _LOG_DROPPED += 1
        return
    _LOG_BYTES += n
    print(s, flush=True)


def err(tag, e):
    log(f"OD_ERR {tag} {type(e).__name__}: {str(e)[-200:]!r}")


# ----------------------------------------------------------------------------------------------
# Pinned module text (commit a55ab7d5): Triton kernel + function, and the patched variant.
# ----------------------------------------------------------------------------------------------
_MIN_TOKENS_FOR_FUSED = 32


@triton.jit
def _post_expert_norm_weighted_sum_kernel(
    x_ptr,
    w_ptr,
    s_ptr,
    out_ptr,
    ids_ptr,
    TOPK: tl.constexpr,
    H: tl.constexpr,
    eps,
    APPLY_NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    WEIGHT_FIRST: tl.constexpr,
    HAS_IDS: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    tok = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < H

    if HAS_WEIGHT:
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.static_range(TOPK):
        base = (tok * TOPK + k) * H
        live = True
        if HAS_IDS:
            live = tl.load(ids_ptr + tok * TOPK + k) >= 0
        if live:
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

            if APPLY_NORM:
                if HAS_WEIGHT and WEIGHT_FIRST:
                    y = x * w
                    var = tl.sum(y * y, axis=0) / H
                    y = y * tl.rsqrt(var + eps)
                elif HAS_WEIGHT:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps) * w
                else:
                    var = tl.sum(x * x, axis=0) / H
                    y = x * tl.rsqrt(var + eps)
            else:
                y = x

            s = tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
            acc += y * s

    tl.store(out_ptr + tok * H + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _pn_core(x, topk_scores, norm_weight, eps, weight_first, apply_norm, slot_expert_ids, thresh):
    tokens, topk, hidden = x.shape
    out = torch.empty((tokens, hidden), dtype=x.dtype, device=x.device)
    if tokens == 0:
        return out
    if tokens < thresh:
        y = x
        if slot_expert_ids is not None:
            y = torch.where(
                (slot_expert_ids.reshape(tokens, topk) >= 0).unsqueeze(-1),
                y,
                torch.zeros((), dtype=y.dtype, device=y.device),
            )
        if apply_norm:
            f = y.float()
            if norm_weight is not None and weight_first:
                f = f * norm_weight.float()
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            elif norm_weight is not None:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
                f = f * norm_weight.float()
            else:
                f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)
            y = f
        else:
            y = y.float()
        y = y * topk_scores.float().unsqueeze(-1)
        return y.sum(dim=1).to(x.dtype)

    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        x,
        norm_weight if norm_weight is not None else x,
        topk_scores.contiguous(),
        out,
        slot_expert_ids if slot_expert_ids is not None else x,
        TOPK=topk,
        H=hidden,
        eps=eps,
        APPLY_NORM=apply_norm,
        HAS_WEIGHT=norm_weight is not None,
        WEIGHT_FIRST=weight_first,
        HAS_IDS=slot_expert_ids is not None,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return out


def post_norm_stock(expert_output, topk_scores, norm_weight=None, eps=1e-8, weight_first=True,
                    apply_norm=True, slot_expert_ids=None):
    """Pinned stock: tokens < _MIN_TOKENS_FOR_FUSED (32) -> PyTorch fallback, else Triton."""
    x = expert_output.contiguous()
    return _pn_core(x, topk_scores, norm_weight, eps, weight_first, apply_norm, slot_expert_ids,
                    _MIN_TOKENS_FOR_FUSED)


def post_norm_patched(expert_output, topk_scores, norm_weight=None, eps=1e-8, weight_first=True,
                      apply_norm=True, slot_expert_ids=None, min_tokens_for_fused=None):
    """s1_postnorm.patch.diff: threshold overridable (None / non-CUDA -> 32)."""
    x = expert_output.contiguous()
    if min_tokens_for_fused is None or not x.is_cuda:
        min_tokens_for_fused = _MIN_TOKENS_FOR_FUSED
    return _pn_core(x, topk_scores, norm_weight, eps, weight_first, apply_norm, slot_expert_ids,
                    min_tokens_for_fused)


# ----------------------------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------------------------
def bench_pair(fa, fb, flush, reps=25):
    """Capture both in CUDA graphs; replay with a READ L2 flush (amax over a 256 MB buffer) before each
    timed replay (flush outside the events); order alternates over 2 passes. Returns median us each."""
    for f in (fa, fb):
        f()
        f()
    torch.cuda.synchronize()
    graphs = []
    for f in (fa, fb):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            f()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        graphs.append(g)
    ts = ([], [])
    for p in range(2):
        order = (0, 1) if p == 0 else (1, 0)
        for _ in range(reps):
            for i in order:
                torch.amax(flush)
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                graphs[i].replay()
                e.record()
                e.synchronize()
                ts[i].append(s.elapsed_time(e))
    del graphs
    return statistics.median(ts[0]) * 1000.0, statistics.median(ts[1]) * 1000.0


def fp64_ref(x, s, w, ids, eps=1e-8):
    xd = x.double()
    live = (ids >= 0).unsqueeze(-1)
    xd = torch.where(live, xd, torch.zeros((), dtype=torch.float64, device=x.device))
    z = xd * w.double()
    var = (z * z).mean(-1, keepdim=True)
    y = z * torch.rsqrt(var + eps)
    return (y * s.double().unsqueeze(-1)).sum(1)


def rel_l2(o, ref):
    d = (o.double() - ref).norm()
    n = ref.norm()
    return (d / n.clamp_min(1e-300)).item()


def s1_slots(T, kind, dev):
    if kind == "p75":
        live = torch.rand(T, 8, device=dev) < 0.25
    elif kind == "rem":
        live = torch.zeros(T, 8, dtype=torch.bool, device=dev)
    elif kind == "loc":
        live = torch.ones(T, 8, dtype=torch.bool, device=dev)
    else:  # "one": exactly one live slot per position
        live = torch.zeros(T, 8, dtype=torch.bool, device=dev)
        live[torch.arange(T, device=dev), torch.randint(0, 8, (T,), device=dev)] = True
    ids = torch.randint(0, 32, (T, 8), dtype=torch.int32, device=dev)
    return torch.where(live, ids, torch.full_like(ids, -1)).contiguous()


def s1_inputs(T, kind, dev, w):
    ids = s1_slots(T, kind, dev)
    x = (torch.randn(T, 8, 4096, device=dev) * 3.0).to(torch.bfloat16)
    mask = ids < 0
    n = int(mask.sum().item())
    if n:
        pv = torch.tensor([float("nan"), float("inf"), -float("inf"), 1e30], device=dev)
        vals = pv[torch.arange(n, device=dev) % 4].to(torch.bfloat16)
        x[mask] = vals.unsqueeze(1).expand(n, 4096)
    s = (torch.rand(T, 8, device=dev) * 0.3 + 0.05).float()
    return x.contiguous(), s, ids


# ----------------------------------------------------------------------------------------------
# Image info
# ----------------------------------------------------------------------------------------------
_IMG_FN = None


def _sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()[:10]
    except Exception:
        return "?"


def info(dev):
    global _IMG_FN
    log(f"OD_INFO {torch.cuda.get_device_name(dev)} torch {torch.__version__} cuda {torch.version.cuda}")
    try:
        import sglang

        log(f"OD_INFO sglang {getattr(sglang, '__version__', '?')} file={getattr(sglang, '__file__', '?')}")
    except Exception as e:
        err("sglang", e)
    try:
        import importlib.util

        name = "sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton"
        spec = importlib.util.find_spec(name)
        if spec is None:
            log("OD_INFO post_norm_module exists=0")
        else:
            import importlib

            m = importlib.import_module(name)
            fn = getattr(m, "fused_post_expert_norm_weighted_sum", None)
            _IMG_FN = fn
            params = list(inspect.signature(fn).parameters) if fn else []
            log(f"OD_INFO post_norm_module exists=1 min_tokens_param={'min_tokens_for_fused' in params} "
                f"MIN={getattr(m, '_MIN_TOKENS_FOR_FUSED', '?')} sha={_sha(spec.origin)} file={spec.origin}")
    except Exception as e:
        err("postnorm_info", e)
    try:
        from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as mab

        src = inspect.getsourcefile(mab)
        log(f"OD_INFO moe_align_block_size file={src} sha={_sha(src)}")
    except Exception as e:
        err("align_info", e)


# ----------------------------------------------------------------------------------------------
# S1
# ----------------------------------------------------------------------------------------------
def run_s1(dev, flush):
    w = (1.0 + 0.1 * torch.randn(4096, device=dev)).to(torch.bfloat16)
    kw = dict(norm_weight=w, eps=1e-8, weight_first=True)
    for T in (1, 4, 8, 9, 16, 24, 31, 32, 33):
        if over():
            log(f"OD_S1 T={T} skipped(time)")
            continue
        try:
            fin_s = fin_f = r0_s = r0_f = True
            rel_s = rel_f = 0.0
            mx = 0.0
            img_eq = "na"
            for kind in ("p75", "rem", "loc", "one"):
                x, s, ids = s1_inputs(T, kind, dev, w)
                o_s = post_norm_stock(x, s, slot_expert_ids=ids, **kw)
                o_f = post_norm_patched(x, s, slot_expert_ids=ids, min_tokens_for_fused=1, **kw)
                torch.cuda.synchronize()
                fin_s &= bool(torch.isfinite(o_s).all().item())
                fin_f &= bool(torch.isfinite(o_f).all().item())
                mx = max(mx, (o_s.float() - o_f.float()).abs().max().item())
                if kind == "rem":
                    r0_s &= bool((o_s == 0).all().item())
                    r0_f &= bool((o_f == 0).all().item())
                else:
                    ref = fp64_ref(x, s, w, ids)
                    rel_s = max(rel_s, rel_l2(o_s, ref))
                    rel_f = max(rel_f, rel_l2(o_f, ref))
                if kind == "p75" and _IMG_FN is not None:
                    try:
                        o_i = _IMG_FN(x, s, slot_expert_ids=ids, **kw)
                        img_eq = int(torch.equal(o_i, o_s))
                    except Exception as e:
                        img_eq = "err"
            # timing: p75 and all-local
            tm = []
            for kind in ("p75", "loc"):
                x, s, ids = s1_inputs(T, kind, dev, w)
                a, b = bench_pair(lambda: post_norm_stock(x, s, slot_expert_ids=ids, **kw),
                                  lambda: post_norm_patched(x, s, slot_expert_ids=ids, min_tokens_for_fused=1, **kw),
                                  flush)
                tm += [a, b]
            log(f"OD_S1 T={T} stock_us={tm[0]:.1f} fused_us={tm[1]:.1f} ratio={tm[0] / tm[1]:.2f} "
                f"loc_stock_us={tm[2]:.1f} loc_fused_us={tm[3]:.1f} "
                f"relL2_stock={rel_s:.2e} relL2_fused={rel_f:.2e} maxabs_sf={mx:.3g} "
                f"finite={int(fin_s)}/{int(fin_f)} remote0={int(r0_s)}/{int(r0_f)} img_eq={img_eq}")
        except Exception as e:
            err(f"S1 T={T}", e)
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------
# S2
# ----------------------------------------------------------------------------------------------
BLOCK_CANDS = (8, 16, 32, 48, 64)


def pick_block_m(T, topk=8, nloc=32):
    b = BLOCK_CANDS[-1]
    for b in BLOCK_CANDS:
        if T * topk / nloc / b < 0.9:
            break
    return b


def make_expert_map(dev):
    m = torch.full((256,), -1, dtype=torch.int32, device=dev)
    m[32:64] = torch.arange(32, dtype=torch.int32, device=dev)
    return m


def make_routing(T, pat, dev, expert_map):
    r = torch.rand(T, 256, device=dev)
    if pat in ("rem", "hot", "one"):
        r[:, 32:64] = -1.0
    if pat == "hot":
        r[:, 37] = 2.0
    elif pat == "one":
        r[torch.arange(T, device=dev), 32 + torch.randint(0, 32, (T,), device=dev)] = 2.0
    gids = torch.topk(r, 8, dim=1).indices
    return expert_map[gids].contiguous()  # int32 [T, 8], -1 = non-local


def parse_align(sorted_ids, expert_ids, npp, bsm, nslots):
    s = sorted_ids[:npp].tolist()
    eids = expert_ids[: npp // bsm].tolist()
    blocks = []
    per = defaultdict(list)
    pads = set()
    for b, e in enumerate(eids):
        chunk = s[b * bsm:(b + 1) * bsm]
        valid = sorted(v for v in chunk if v < nslots)
        pads.update(v for v in chunk if v >= nslots)
        blocks.append((e, tuple(valid)))
        per[e].extend(valid)
    return eids, blocks, {e: sorted(v) for e, v in per.items() if v}, pads


def truth_sets(local_ids):
    per = defaultdict(list)
    for slot, e in enumerate(local_ids.reshape(-1).tolist()):
        if e >= 0:
            per[e].append(slot)
    return dict(per)


def build_bank(weights, dev):
    from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
    from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin

    src = "real"
    try:
        w13, w2 = weights["w13"].contiguous(), weights["w2"].contiguous()
        w13s, w2s = weights["w13_sf"].contiguous(), weights["w2_sf"].contiguous()
        g13, g2 = weights["w13_gs"].contiguous(), weights["w2_gs"].contiguous()
    except Exception:
        src = "random"
        w13 = torch.randint(0, 256, (32, 8192, 2048), dtype=torch.uint8, device=dev)
        w2 = torch.randint(0, 256, (32, 4096, 2048), dtype=torch.uint8, device=dev)
        w13s = (torch.rand(32, 8192, 256, device=dev) * 1.5 + 0.5).to(torch.float8_e4m3fn)
        w2s = (torch.rand(32, 4096, 256, device=dev) * 1.5 + 0.5).to(torch.float8_e4m3fn)
        g13 = torch.full((32,), 0.01, dtype=torch.float32, device=dev)
        g2 = torch.full((32,), 0.01, dtype=torch.float32, device=dev)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=w13.shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=w13, w2_weight=w2, w13_weight_scale=w13s, w2_weight_scale=w2s,
        w13_weight_scale_2=g13, w2_weight_scale_2=g2,
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(dev, max_blocks_per_sm=4),
        "src": src,
    }


def make_chain(bank, x, local_ids, topk_w, T):
    from sgl_kernel.scalar_type import scalar_types

    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm

    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    size_k = x.shape[1]
    bsm = pick_block_m(T)
    c1 = torch.empty((T * 8, gemm1_n), dtype=x.dtype, device=x.device)
    c2 = torch.empty((T * 8, size_n), dtype=x.dtype, device=x.device)
    c3 = torch.empty((T * 8, size_k), dtype=x.dtype, device=x.device)
    common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True, b_q_type=scalar_types.float4_e2m1f,
                  is_k_full=True, use_atomic_add=False, use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]

    def run(sorted_ids, expert_ids, npp, fill=True):
        if fill:
            c1.fill_(float("nan"))
            c2.fill_(float("nan"))
            c3.fill_(float("nan"))
        o1 = moe_wna16_marlin_gemm(
            x, c1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
            sorted_ids, expert_ids, npp, topk_w, top_k=8, size_m=T, size_n=gemm1_n, size_k=size_k, **common)
        silu_and_mul(o1.view(-1, gemm1_n), c2, expert_ids=local_ids.view(-1), expert_step=1)
        return moe_wna16_marlin_gemm(
            c2, c3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
            sorted_ids, expert_ids, npp, topk_w, top_k=1, size_m=T * 8, size_n=size_k, size_k=size_n, **common)

    return run


def gemm_compare(run, local_ids, al_s, al_c, T, bsm):
    """Returns (gemm_eq, tail_eq, nl_clean, finite) as ints, or None if there are no live slots."""
    live = (local_ids.view(-1) >= 0)
    ss, es, ns = al_s
    sc, ec, nc = al_c
    y_s = run(ss, es, ns).clone()
    y_c = run(sc, ec, nc).clone()
    # candidate data placed into stock-sized buffers, tail beyond npp = valid-looking token id 0 / expert 7
    npp_c = int(nc.item())
    sb = torch.zeros_like(ss)
    sb[:npp_c] = sc[:npp_c]
    eb = torch.full_like(es, 7)
    nb = npp_c // bsm
    eb[:nb] = ec[:nb]
    y_t = run(sb, eb, nc).clone()
    torch.cuda.synchronize()
    if int(live.sum().item()) == 0:
        return None
    def eq(a, b):
        return int(torch.equal(a[live].view(torch.int16), b[live].view(torch.int16)))
    nl = ~live
    clean = 1
    if int(nl.sum().item()):
        clean = int(bool(torch.isnan(y_s[nl]).all().item()) and bool(torch.isnan(y_c[nl]).all().item()))
    fin = int(bool(torch.isfinite(y_s[live]).all().item()))
    return eq(y_s, y_c), eq(y_s, y_t), clean, fin


def run_s2(dev, flush, weights):
    from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size as mab

    emap = make_expert_map(dev)
    bank = None
    try:
        t = time.time()
        bank = build_bank(weights, dev)
        log(f"OD_S2 bank={bank['src']} built {time.time() - t:.0f}s")
    except Exception as e:
        err("S2 bank", e)

    rem_gemm_jobs = []
    for T in (1, 8, 9, 16, 32, 33, 128, 256):
        pats = ["uni", "rem", "hot"] + (["one"] if T in (9, 33, 256) else [])
        for pat in pats:
            if over():
                log(f"OD_S2 T={T} pat={pat} skipped(time)")
                continue
            try:
                ids = make_routing(T, pat, dev, emap)
                bsm = pick_block_m(T)
                nslots = T * 8
                al_s = mab(ids, bsm, 256)
                al_c = mab(ids, bsm, 32)
                torch.cuda.synchronize()
                ns, nc = int(al_s[2].item()), int(al_c[2].item())
                es_, bs_, ps_, pad_s = parse_align(al_s[0], al_s[1], ns, bsm, nslots)
                ec_, bc_, pc_, pad_c = parse_align(al_c[0], al_c[1], nc, bsm, nslots)
                tr = truth_sets(ids)
                sets_eq = int(ps_ == pc_)
                blk_eq = int(sorted(bs_) == sorted(bc_))
                eid_eq = int(es_ == ec_)
                tr_s, tr_c = int(ps_ == tr), int(pc_ == tr)
                pad = lambda p: ",".join(map(str, sorted(p))) or "-"
                a_us = b_us = float("nan")
                try:
                    a_us, b_us = bench_pair(lambda: mab(ids, bsm, 256), lambda: mab(ids, bsm, 32), flush)
                except Exception as e:
                    err(f"S2 aligntime T={T} {pat}", e)
                g_eq = t_eq = nl = fin = "na"
                g_us = "na"
                if pat == "rem" or int(sum(len(v) for v in tr.values())) == 0:
                    rem_gemm_jobs.append((T, pat))
                elif bank is not None:
                    try:
                        x = torch.randn(T, 4096, device=dev).to(torch.bfloat16)
                        tw = torch.rand(T, 8, device=dev, dtype=torch.float32)
                        run = make_chain(bank, x, ids, tw, T)
                        res = gemm_compare(run, ids, al_s, al_c, T, bsm)
                        if res is not None:
                            g_eq, t_eq, nl, fin = res
                        try:
                            ga, gb = bench_pair(lambda: run(al_s[0], al_s[1], al_s[2], fill=False),
                                                lambda: run(al_c[0], al_c[1], al_c[2], fill=False), flush)
                            g_us = f"{ga:.0f}/{gb:.0f}"
                        except Exception as e:
                            err(f"S2 gemmtime T={T} {pat}", e)
                    except Exception as e:
                        err(f"S2 gemm T={T} {pat}", e)
                log(f"OD_S2 T={T} pat={pat} bs={bsm} npp={ns}/{nc} sets_equal={sets_eq} blk_eq={blk_eq} "
                    f"eid_eq={eid_eq} truth={tr_s}/{tr_c} pad={pad(pad_s)}/{pad(pad_c)} "
                    f"len={al_s[0].numel()}/{al_c[0].numel()} gemm_eq={g_eq} tail_eq={t_eq} nl_clean={nl} "
                    f"fin={fin} align_us={a_us:.1f}/{b_us:.1f} gemm_us={g_us}")
            except Exception as e:
                err(f"S2 T={T} {pat}", e)
            torch.cuda.empty_cache()

    # All-remote GEMM probes last (a launch with zero live blocks might fault; keep it away from the rest).
    if bank is not None:
        for T, pat in rem_gemm_jobs:
            if over() or T not in (8, 33, 256):
                continue
            try:
                ids = make_routing(T, pat, dev, emap)
                bsm = pick_block_m(T)
                al_s = mab(ids, bsm, 256)
                al_c = mab(ids, bsm, 32)
                torch.cuda.synchronize()
                x = torch.randn(T, 4096, device=dev).to(torch.bfloat16)
                tw = torch.rand(T, 8, device=dev, dtype=torch.float32)
                run = make_chain(bank, x, ids, tw, T)
                ys = run(al_s[0], al_s[1], al_s[2]).clone()
                yc = run(al_c[0], al_c[1], al_c[2]).clone()
                torch.cuda.synchronize()
                log(f"OD_S2G_REM T={T} npp={int(al_s[2].item())}/{int(al_c[2].item())} "
                    f"all_nan_untouched={int(bool(torch.isnan(ys).all().item()))}/{int(bool(torch.isnan(yc).all().item()))}")
            except Exception as e:
                err(f"S2 remgemm T={T}", e)
    del bank


def prepare(weights):
    global T0
    T0 = time.time()
    dev = torch.device("cuda")
    torch.manual_seed(1234)
    try:
        info(dev)
    except Exception as e:
        err("info", e)
    flush = None
    try:
        flush = torch.empty(64 << 20, dtype=torch.float32, device=dev)  # 256 MB
        flush.zero_()
        torch.amax(flush)
        torch.cuda.synchronize()
    except Exception as e:
        err("flush", e)
    if flush is not None:
        try:
            run_s1(dev, flush)
        except Exception as e:
            err("S1", e)
        try:
            run_s2(dev, flush, weights)
        except Exception as e:
            err("S2", e)
    try:
        torch.cuda.synchronize()
    except Exception as e:
        err("sync", e)
    log(f"OD_DONE prepare {time.time() - T0:.0f}s bytes={_LOG_BYTES} dropped={_LOG_DROPPED}", force=True)
    return weights


def kernel_fn(data, weights):
    """Pass-through: correctly shaped/dtyped zeros (tests fail; only the OD_* log matters)."""
    t = data["config"]["tokens"]
    dev = data["stream"].device
    hidden = torch.zeros((t, HIDDEN), dtype=torch.bfloat16, device=dev)
    stream = torch.zeros((t, HIDDEN), dtype=torch.float32, device=dev)
    ids = torch.zeros((t, 8), dtype=torch.int32, device=dev)
    return hidden, stream, ids
