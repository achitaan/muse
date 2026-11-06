"""Complete expert-operation comparison with three captures and null controls."""
import argparse
import itertools
import random
import statistics
import time

from bench_marlin_stock import make_stock_layer, SEED
from bench_precomp import emit, routes, errors
from native_marlin import NativeState, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[128, 256, 32])
    parser.add_argument("--patterns", nargs="+", default=["uniform", "mixed", "hot"])
    parser.add_argument("--offsets", type=int, nargs="+", default=[32])
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    assert args.reps >= 6 and args.reps % 6 == 0 and args.blocks > 0
    import torch
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import fused_post_expert_norm_weighted_sum as final
    g = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        start = time.monotonic()
        layer = make_stock_layer(torch, g)
        state = NativeState(layer, offset=32)
        flush = torch.ones(64 * 1024**2, dtype=torch.float32, device="cuda")
        empty_debug = torch.empty(0, dtype=torch.bfloat16, device="cuda")
        emit(event="environment", preparation_seconds=time.monotonic() - start,
             resources=state.resources, workspace_bytes=state.workspace_bytes,
             torch=torch.__version__, gpu=torch.cuda.get_device_name())
        for offset in args.offsets:
            state.offset = offset
            layer._expert_map.fill_(-1)
            layer._expert_map[offset:offset + 32] = torch.arange(32, dtype=torch.int32, device="cuda")
            for tokens in args.tokens:
                for pattern in args.patterns:
                    case = dict(tokens=tokens, pattern=pattern, offset=offset)
                    emit(event="case_start", **case)
                    x = torch.randn((tokens, 4096), dtype=torch.bfloat16, device="cuda", generator=g)
                    ids = routes(torch, g, tokens, pattern, offset)
                    scores = torch.rand((tokens, 8), device="cuda", generator=g).softmax(1)
                    gain = (0.75 + .5 * torch.rand(4096, device="cuda", generator=g)).bfloat16()
                    out = torch.empty((tokens, 8, 4096), dtype=x.dtype, device="cuda")
                    local = torch.empty_like(ids)
                    assert not eligible(x, allow_decode_kernel=True, state=state)
                    assert not eligible(x, state=state, accepted=True)

                    def forward(native=False):
                        if native:
                            state.run(x, ids, out, local=local, debug=empty_debug)
                            z, li = out, local
                        else:
                            z, li = layer._marlin_expert_outputs(x, scores, ids)
                        return final(z, scores, norm_weight=gain, eps=1e-8,
                                     weight_first=True, slot_expert_ids=li)

                    if tokens == 0:
                        assert forward(True).shape == (0, 4096)
                        emit(event="empty_pass", **case)
                        continue
                    if pattern == "padding":
                        scores.zero_()
                    forward(False)
                    forward(True)
                    torch.cuda.synchronize()
                    for block in range(args.blocks):
                        graphs, outputs = {}, {}
                        for arm in ("stock_a", "stock_b", "native"):
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                z = forward(arm == "native")
                            graph.replay()
                            torch.cuda.synchronize()
                            graphs[arm], outputs[arm] = graph, z
                        emit(event="capture", block=block, **case,
                             null=errors(torch, outputs["stock_b"], outputs["stock_a"]),
                             candidate=errors(torch, outputs["native"], outputs["stock_a"]))
                        if pattern in ("remote", "padding"):
                            assert not outputs["native"].count_nonzero().item()
                        if not args.correctness_only:
                            order_rng = random.Random(SEED + tokens + block + offset)
                            for cache in ("cold", "steady"):
                                samples = {arm: [] for arm in graphs}
                                events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                                orders = list(itertools.permutations(graphs)) * (args.reps // 6)
                                order_rng.shuffle(orders)
                                for order in orders:
                                    for arm in order:
                                        if cache == "cold":
                                            flush.sum()
                                        else:
                                            graphs[arm].replay()
                                        events[0].record()
                                        graphs[arm].replay()
                                        events[1].record()
                                        events[1].synchronize()
                                        samples[arm].append(events[0].elapsed_time(events[1]) * 1000)
                                med = {arm: statistics.median(v) for arm, v in samples.items()}
                                emit(event="timing", block=block, **case, cache=cache,
                                     samples_us=samples, medians_us=med,
                                     reduction=1 - med["native"] / med["stock_a"],
                                     control_reduction=1 - med["stock_b"] / med["stock_a"])
                        # Change values, route skew AND local-job count without recapture.
                        for replay_pattern in ("remote", "hot", "uniform", "padding", pattern):
                            x.copy_(torch.randn(x.shape, dtype=x.dtype, device="cuda", generator=g))
                            ids.copy_(routes(torch, g, tokens, replay_pattern, offset))
                            scores.copy_(torch.rand(scores.shape, device="cuda", generator=g).softmax(1))
                            if replay_pattern == "padding":
                                scores.zero_()
                            for graph in graphs.values():
                                graph.replay()
                            torch.cuda.synchronize()
                            emit(event="fresh_replay", block=block, **case, replay_pattern=replay_pattern,
                                 null=errors(torch, outputs["stock_b"], outputs["stock_a"]),
                                 candidate=errors(torch, outputs["native"], outputs["stock_a"]))
                            expected_local = torch.where((ids >= offset) & (ids < offset + 32), ids - offset, -1)
                            assert torch.equal(local, expected_local)
                            assert not out.view(-1, 4096)[local.flatten() < 0].count_nonzero().item()
                            if replay_pattern in ("remote", "padding"):
                                assert not outputs["native"].count_nonzero().item()
                        del graphs, outputs, graph, z
                    emit(event="case_done", **case)
        emit(event="complete", seconds=time.monotonic() - start,
             peak_allocated_bytes=torch.cuda.max_memory_allocated())


if __name__ == "__main__":
    main()
