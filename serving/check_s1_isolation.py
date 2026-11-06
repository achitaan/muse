"""Isolate the actual S1 helper on saved inputs; diagnostics are not acceptance.

The stock arithmetic authority is SGLang a55ab7d5, the helper's PyTorch branch.
The traced Triton kernel below mirrors its fused branch and is cross-checked
against the actual imported helper before its intermediates are interpreted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import triton
import triton.language as tl

from check_marlin_nvfp4_layout import verify_sglang_checkout


@triton.jit
def trace_kernel(X, W, S, IDS, TRACE, VAR, INV, SUM, OUT,
                 H: tl.constexpr, WEIGHT_FIRST: tl.constexpr):
    tok = tl.program_id(0)
    h = tl.arange(0, H)
    w = tl.load(W + h).to(tl.float32)
    acc = tl.zeros((H,), tl.float32)
    for k in tl.static_range(8):
        row = tok * 8 + k
        live = tl.load(IDS + row) >= 0
        # The production kernel skips invalid slots before loading poisoned data.
        if live:
            x = tl.load(X + row * H + h).to(tl.float32)
            y = x * w if WEIGHT_FIRST else x
            v = tl.sum(y * y, 0) / H
            inv = tl.rsqrt(v + 1.e-8)
            norm = y * inv
            if not WEIGHT_FIRST:
                norm = norm * w
            score = tl.load(S + row).to(tl.float32)
            weighted = norm * score
            acc += weighted
            tl.store(TRACE + (row * 4 + 0) * H + h, x)
            tl.store(TRACE + (row * 4 + 1) * H + h, y)
            tl.store(TRACE + (row * 4 + 2) * H + h, norm)
            tl.store(TRACE + (row * 4 + 3) * H + h, weighted)
            tl.store(VAR + row, v)
            tl.store(INV + row, inv)
    tl.store(SUM + tok * H + h, acc)
    tl.store(OUT + tok * H + h, acc.to(tl.bfloat16))


def comparison(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    diff = a.float() - b.float()
    return dict(bit_equal=torch.equal(a.view(integer), b.view(integer)),
                different_bits=int((a.view(integer) != b.view(integer)).sum()),
                max_abs=float(diff.abs().max()) if diff.numel() else 0.,
                rel_l2=float(diff.norm() / b.float().norm().clamp_min(1.e-30)))


def fixture(tokens, pattern, magnitude, seed):
    g = torch.Generator().manual_seed(seed)
    scales = dict(ordinary=1., tiny=1.e-8, near_epsilon=1.e-4, zero=0.)
    x = (torch.randn(tokens, 8, 4096, generator=g) * scales[magnitude]).bfloat16()
    gain = (0.75 + 0.5 * torch.rand(4096, generator=g)).bfloat16()
    scores = torch.rand(tokens, 8, generator=g).softmax(1)
    ids = torch.arange(8, dtype=torch.int32).expand(tokens, -1).clone()
    if pattern == "mixed":
        ids[torch.rand(tokens, 8, generator=g) < .75] = -1
    elif pattern == "remote":
        ids.fill_(-1)
    poison = torch.tensor([float("nan"), float("inf"), -float("inf"), 1.e30]).bfloat16().repeat(1024)
    x[ids < 0] = poison
    return dict(x=x, gain=gain, scores=scores, ids=ids)


def stock_trace(x, gain, scores, ids, weight_first):
    # Exact operations from the pinned helper, not a higher-precision oracle.
    f = torch.where((ids >= 0).unsqueeze(-1), x, 0).float()
    y = f * gain.float() if weight_first else f
    var = y.pow(2).mean(-1)
    inv = torch.rsqrt(var + 1.e-8)
    norm = y * inv.unsqueeze(-1)
    if not weight_first:
        norm = norm * gain.float()
    routed = norm * scores.float().unsqueeze(-1)
    summed = routed.sum(1)
    return [f, y, var, inv, norm, routed, summed, summed.bfloat16()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sglang-checkout", type=Path, required=True)
    p.add_argument("--s1-patch", type=Path, required=True)
    p.add_argument("--evidence-dir", type=Path, required=True)
    args = p.parse_args()
    verify_sglang_checkout(args.sglang_checkout, args.s1_patch)
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as helper
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(event="environment", torch=torch.__version__, triton=triton.__version__,
                         gpu=torch.cuda.get_device_name(), source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())), flush=True)
    names = ["masked", "pre_norm_gain", "variance", "inverse_rms", "normalized_gain", "route_weighted", "sum_fp32", "output_bf16"]
    count = 0
    with torch.inference_mode():
        for tokens in (0, 1, 8, 16, 31, 32, 33):
            for pattern in ("local", "mixed", "remote"):
                for magnitude in ("ordinary", "tiny", "near_epsilon", "zero"):
                    key = f"t{tokens}-{pattern}-{magnitude}"
                    path = args.evidence_dir / (key + ".pt")
                    torch.save(fixture(tokens, pattern, magnitude, 91027 + count), path)
                    loaded = torch.load(path, weights_only=True)
                    x, gain, scores, ids = (loaded[n].cuda() for n in ("x", "gain", "scores", "ids"))
                    for weight_first in (True, False):
                        kw = dict(norm_weight=gain, slot_expert_ids=ids, weight_first=weight_first)
                        stock = helper(x, scores, **kw)
                        repeat = helper(x, scores, **kw)
                        s1 = helper(x, scores, min_tokens_for_fused=1, **kw)
                        record = dict(event="case", tokens=tokens, pattern=pattern, magnitude=magnitude,
                                      weight_first=weight_first, input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      stock_repeat=comparison(repeat, stock), s1=comparison(s1, stock))
                        if tokens:
                            reference = stock_trace(x, gain, scores, ids, weight_first)
                            if tokens < 32:
                                assert comparison(reference[-1], stock)["bit_equal"]
                            trace = torch.zeros((tokens, 8, 4, 4096), device="cuda")
                            var, inv = (torch.zeros((tokens, 8), device="cuda") for _ in range(2))
                            summed = torch.empty((tokens, 4096), device="cuda")
                            out = torch.empty_like(stock)
                            diagnostics = {}
                            for fusion in (True, False):
                                trace_kernel[(tokens,)](x, gain, scores, ids, trace, var, inv, summed, out,
                                                       H=4096, WEIGHT_FIRST=weight_first, num_warps=8,
                                                       num_stages=2, enable_fp_fusion=fusion)
                                valid = ids >= 0
                                observed = [trace[:, :, 0], trace[:, :, 1], var, inv,
                                            trace[:, :, 2], trace[:, :, 3], summed, out]
                                stages = {name: comparison(a[valid] if i < 6 else a, b[valid] if i < 6 else b)
                                          for i, (name, a, b) in enumerate(zip(names, observed, reference))}
                                diagnostics[str(fusion)] = dict(stages=stages, trace_vs_production=comparison(out, s1),
                                    first_different=next((n for n in names if not stages[n]["bit_equal"]), None))
                            record["diagnostics"] = diagnostics
                            if pattern == "remote" or magnitude == "zero":
                                assert not stock.count_nonzero() and not s1.count_nonzero()
                        print(json.dumps(record), flush=True)
                        count += 1
    print(json.dumps(dict(event="complete", cases=count)), flush=True)


if __name__ == "__main__":
    main()
