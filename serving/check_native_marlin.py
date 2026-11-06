"""Watchdog-wrapped tile and first-GEMM differential probes; not scored checks."""
import argparse
import json
import time
from unittest.mock import patch

from bench_marlin_stock import make_stock_layer, SEED
from bench_precomp import routes, errors
from native_marlin import NativeState


def emit(**kw):
    print(json.dumps(kw), flush=True)


def tile_oracle(torch, weight, scales, phase, expert, unit, stage):
    # Independent vectorized address oracle; CUDA uses a per-thread register map.
    n = torch.arange(256, device="cuda")
    if phase == 0:
        n = unit * 128 + (n // 16) * 8 + n % 8 + ((n % 16) // 8) * 4096
    else:
        n = unit * 256 + n
    k = torch.arange(stage * 256, (stage + 1) * 256, device="cuda")
    N = 4096 if phase else 8192
    word = ((k[None, :] // 16 * (N // 64) + n[:, None] // 64) * 128
            + ((n[:, None] % 8) * 4 + (k[None, :] % 8) // 2) * 4
            + (n[:, None] % 64) // 16)
    shift = 4 * ((k[None, :] % 2) * 4 + ((n[:, None] % 16) // 8) * 2 + ((k[None, :] % 16) // 8))
    codes = (weight[expert].flatten()[word].long() >> shift) & 15
    transposed = (n % 8) * 8 + (n % 64) // 8
    swizzled = (n // 64) * 64 + (transposed & ~3) + ((transposed & 1) << 1) + ((transposed & 2) >> 1)
    enc = scales[expert].view(torch.uint8)[k[None, :] // 16, swizzled[:, None]].long()
    bits = (((enc & 128) << 7) | ((enc & 127) << 4)).short().contiguous()
    scale = bits.view(torch.bfloat16)
    # Match stock's BF16 dequant multiply BEFORE undoing its exponent rebasing.
    # Some encoded bytes from subnormal FP8 input scales underflow here. A real
    # arithmetic FP4*FP8 oracle would silently change this pinned behavior.
    qbits = (((codes & 7) << 6) | ((codes & 8) << 12)).short().contiguous()
    product = qbits.view(torch.bfloat16) * scale
    return product.float() * (2.0**119)


def stock_stages(torch, layer, x, scores, ids):
    from sglang.kernels.ops.moe import moe_wna16_marlin as module
    original = module.moe_wna16_marlin_gemm
    captured = {}

    def capture(*args, **kwargs):
        if kwargs["top_k"] == 1:
            captured["swi"] = args[0].clone()
        value = original(*args, **kwargs)
        captured["gemm2" if kwargs["top_k"] == 1 else "gemm1"] = value.clone()
        return value

    with patch.object(module, "moe_wna16_marlin_gemm", capture):
        z, local = layer._marlin_expert_outputs(x, scores, ids)
    captured["local"] = local
    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tiles", "gemm1", "full"], default="tiles")
    parser.add_argument("--tokens", type=int, nargs="+", default=[128])
    parser.add_argument("--patterns", nargs="+", default=["uniform"])
    parser.add_argument("--offset", type=int, default=32)
    parser.add_argument("--edge-scales", action="store_true")
    parser.add_argument("--activation-scale", type=float, default=1.0)
    args = parser.parse_args()
    import torch
    g = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        layer = make_stock_layer(torch, g, edge_scales=args.edge_scales)
        start = time.monotonic()
        state = NativeState(layer, offset=args.offset)
        emit(event="prepared", seconds=time.monotonic() - start,
             resources=state.resources, workspace_bytes=state.workspace_bytes)
        for phase in (0, 1):
            for expert, unit, stage in ((0, 0, 0), (7, 3, 8), (31, 15, 15)):
                actual = state.ext.tile(state.maps, phase, expert, unit, stage)
                expected = tile_oracle(torch, state.weights[phase * 3], state.weights[phase * 3 + 1],
                                       phase, expert, unit, stage)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                emit(event="tile_pass", phase=phase, expert=expert, unit=unit, stage=stage)
        if args.stage == "tiles":
            return
        layer._expert_map.fill_(-1)
        layer._expert_map[args.offset:args.offset + 32] = torch.arange(32, dtype=torch.int32, device="cuda")
        for tokens in args.tokens:
            for pattern in args.patterns:
                x = torch.randn((tokens, 4096), dtype=torch.bfloat16, device="cuda", generator=g)
                x.mul_(args.activation_scale)
                ids = routes(torch, g, tokens, pattern, args.offset)
                scores = torch.rand((tokens, 8), device="cuda", generator=g).softmax(dim=1)
                if pattern == "padding":
                    scores.zero_()
                a = stock_stages(torch, layer, x, scores, ids)
                b = stock_stages(torch, layer, x, scores, ids)
                out = torch.full((tokens, 8, 4096), float("nan"), dtype=x.dtype, device="cuda")
                debug = torch.full((tokens, 8, 8192), float("nan"), dtype=x.dtype, device="cuda")
                local = torch.empty_like(ids)
                state.run(x, ids, out, local=local, debug=debug, phases=1 if args.stage == "gemm1" else 2)
                torch.cuda.synchronize()
                valid = ((ids >= args.offset) & (ids < args.offset + 32)).flatten()
                observed = {"gemm1": debug.view(-1, 8192),
                            "swi": state.scratch[2][:tokens * 8 * 4096].view(-1, 4096),
                            "gemm2": out.view(-1, 4096)}
                for stage in ("gemm1", "swi") + (("gemm2",) if args.stage == "full" else ()):
                    emit(event="stage", tokens=tokens, pattern=pattern, stage=stage,
                         null=errors(torch, b[stage][valid], a[stage][valid]),
                         candidate=errors(torch, observed[stage][valid], a[stage][valid]))
                if args.stage == "full":
                    assert not out.view(-1, 4096)[~valid].count_nonzero().item()
                    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as final
                    gain = torch.ones(4096, device="cuda", dtype=x.dtype)
                    kw = dict(norm_weight=gain, eps=1e-8, weight_first=True, slot_expert_ids=a["local"])
                    z0 = final(a["gemm2"].view(tokens, 8, 4096), scores, **kw)
                    z1 = final(b["gemm2"].view(tokens, 8, 4096), scores, **kw)
                    z2 = final(out, scores, **kw)
                    emit(event="combined", tokens=tokens, pattern=pattern,
                         null=errors(torch, z1, z0), candidate=errors(torch, z2, z0))


if __name__ == "__main__":
    main()
