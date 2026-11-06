"""Retain graphs of different sizes and alternate them on one scratch owner."""
import torch

from bench_marlin_stock import make_stock_layer, SEED
from bench_precomp import emit, routes, errors
from native_marlin import NativeState
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as final


def main():
    g = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        layer = make_stock_layer(torch, g)
        state = NativeState(layer, offset=32)
        gain = torch.ones(4096, device="cuda", dtype=torch.bfloat16)
        empty = torch.empty(0, device="cuda", dtype=torch.bfloat16)
        cases = {}
        for tokens in (128, 256, 32, 1, 257):
            x = torch.randn((tokens, 4096), device="cuda", dtype=torch.bfloat16, generator=g)
            ids = routes(torch, g, tokens, "uniform", 32)
            scores = torch.ones((tokens, 8), device="cuda") / 8
            out = torch.empty((tokens, 8, 4096), device="cuda", dtype=torch.bfloat16)
            local = torch.empty_like(ids)
            state.run(x, ids, out, local=local, debug=empty)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                state.run(x, ids, out, local=local, debug=empty)
                result = final(out, scores, norm_weight=gain, slot_expert_ids=local)
            cases[tokens] = (x, ids, scores, out, local, graph, result)
        sequence = (128, 256, 128, 32, 256, 1, 257, 128, 1, 32, 257, 256)
        patterns = ("uniform", "remote", "hot", "padding")
        for repeat in range(3):
            for i, tokens in enumerate(sequence):
                pattern = patterns[(i + repeat) % len(patterns)]
                x, ids, scores, out, local, graph, result = cases[tokens]
                x.copy_(torch.randn(x.shape, device="cuda", dtype=x.dtype, generator=g))
                if repeat == 2:
                    x.zero_()
                ids.copy_(routes(torch, g, tokens, pattern, 32))
                scores.fill_(0 if pattern == "padding" else .125)
                graph.replay()
                torch.cuda.synchronize()
                z, li = layer._marlin_expert_outputs(x, scores, ids)
                expected = final(z, scores, norm_weight=gain, slot_expert_ids=li)
                comparison = errors(torch, result, expected)
                assert not out.view(-1, 4096)[local.flatten() < 0].count_nonzero().item()
                if pattern in ("remote", "padding") or repeat == 2:
                    assert not result.count_nonzero().item()
                emit(event="alternating_replay", repeat=repeat, tokens=tokens,
                     pattern=pattern, zero_input=repeat == 2, candidate=comparison)
        emit(event="complete", replays=len(sequence) * 3)


if __name__ == "__main__":
    main()
