"""Paired stock/S1 checks on a synthetic Muse shard using normal SGLang imports."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import time

from bench_marlin_stock import make_stock_layer, SEED, HIDDEN, LOCAL_EXPERTS, EXPERTS, TOP_K
from check_marlin_nvfp4_layout import verify_sglang_checkout


def emit(**record):
    print(json.dumps(record), flush=True)


def routes(torch, generator, tokens, pattern, offset):
    ranks = torch.rand((tokens, EXPERTS), device="cuda", generator=generator)
    local = ranks[:, offset:offset + LOCAL_EXPERTS].topk(TOP_K).indices + offset
    remote_ranks = ranks.clone()
    remote_ranks[:, offset:offset + LOCAL_EXPERTS] = -1
    remote = remote_ranks.topk(TOP_K).indices
    if pattern == "uniform":
        ids = ranks.topk(TOP_K).indices
    elif pattern == "local":
        ids = local
    elif pattern == "remote":
        ids = remote
    elif pattern == "padding":
        ids = torch.full_like(local, -1)
    elif pattern == "hot":
        ids = torch.arange(offset, offset + TOP_K, device="cuda").expand(tokens, -1)
    else:
        ids = torch.cat((local[:, :2], remote[:, :6]), dim=1)
    order = torch.rand((tokens, TOP_K), device="cuda", generator=generator).argsort(dim=1)
    return ids.gather(1, order).to(torch.int32).contiguous()


def errors(torch, actual, expected):
    a, b = actual.float(), expected.float()
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise RuntimeError("nonfinite stock or candidate output")
    difference = a - b
    return {
        "bit_equal": torch.equal(actual.view(torch.int16), expected.view(torch.int16)),
        "rel_l2": (difference.norm() / b.norm().clamp_min(1e-12)).item(),
        "max_abs": difference.abs().max().item() if difference.numel() else 0.0,
    }


def run_case(torch, layer, generator, flush, args, tokens, pattern, offset):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    layer._expert_map.fill_(-1)
    layer._expert_map[offset:offset + LOCAL_EXPERTS] = torch.arange(
        LOCAL_EXPERTS, dtype=torch.int32, device="cuda"
    )
    x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=generator)
    ids = routes(torch, generator, tokens, pattern, offset)
    scores = torch.rand((tokens, TOP_K), device="cuda", generator=generator).softmax(dim=1)
    if pattern == "padding":
        scores.zero_()
    gain = (0.75 + 0.5 * torch.rand((HIDDEN,), device="cuda", generator=generator)).bfloat16()

    def forward(cutoff):
        z, local_ids = layer._marlin_expert_outputs(x, scores, ids)
        return fused_post_expert_norm_weighted_sum(
            z, scores, norm_weight=gain, eps=1e-8, weight_first=True,
            slot_expert_ids=local_ids, min_tokens_for_fused=cutoff,
        )

    emit(event="case_start", tokens=tokens, pattern=pattern, offset=offset)
    if tokens == 0:
        empty = torch.empty((0, TOP_K, HIDDEN), device="cuda", dtype=torch.bfloat16)
        result = fused_post_expert_norm_weighted_sum(
            empty, scores, norm_weight=gain, min_tokens_for_fused=1,
        )
        assert result.shape == (0, HIDDEN)
        emit(event="empty_pass")
        return

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        stock_first = forward(32)
        stock_second = forward(32)
        candidate = forward(1)
    torch.cuda.synchronize()
    emit(event="eager", stock_repeat=errors(torch, stock_second, stock_first),
         s1=errors(torch, candidate, stock_first))
    if pattern in ("remote", "padding"):
        assert not torch.count_nonzero(stock_first).item()
        assert not torch.count_nonzero(candidate).item()

    for block in range(args.blocks):
        graphs, outputs = {}, {}
        # Two identical stock arms expose capture/order bias.
        for arm, cutoff in (("stock_a", 32), ("stock_b", 32), ("s1", 1)):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = forward(cutoff)
            graph.replay()
            torch.cuda.synchronize()
            graphs[arm], outputs[arm] = graph, output
        reference = outputs["stock_a"].clone()
        emit(event="capture", block=block,
             null=errors(torch, outputs["stock_b"], reference),
             s1=errors(torch, outputs["s1"], reference))
        if not args.correctness_only:
            order_rng = random.Random(SEED + block + tokens + offset)
            for cache in ("cold", "steady"):
                samples = {arm: [] for arm in graphs}
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                orders = list(itertools.permutations(graphs)) * (args.reps // 6)
                order_rng.shuffle(orders)
                for order in orders:
                    for arm in order:
                        if cache == "cold":
                            flush.sum()
                        else:
                            graphs[arm].replay()
                        start.record()
                        graphs[arm].replay()
                        end.record()
                        end.synchronize()
                        samples[arm].append(start.elapsed_time(end) * 1000)
                medians = {arm: statistics.median(times) for arm, times in samples.items()}
                emit(event="timing", block=block, tokens=tokens, pattern=pattern, offset=offset,
                     cache=cache, samples_us=samples, medians_us=medians,
                     null_ratio=medians["stock_a"] / medians["stock_b"],
                     s1_speedup=medians["stock_a"] / medians["s1"])

        # Fresh device inputs and routes must be consumed by the same captured graph.
        x.copy_(torch.randn(x.shape, dtype=x.dtype, device="cuda", generator=generator))
        ids.copy_(routes(torch, generator, tokens, pattern, offset))
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        fresh_stock = forward(32)
        torch.cuda.synchronize()
        emit(event="fresh_replay", block=block,
             stock=errors(torch, outputs["stock_a"], fresh_stock),
             s1=errors(torch, outputs["s1"], fresh_stock))
        # T>=32 takes identical production code in both arms. Marlin's parallel
        # reduction can still differ between invocations; record the null arm
        # instead of pretending unchanged-code equality is a serving contract.
        emit(event="fresh_null", block=block, tokens=tokens,
             null=errors(torch, outputs["stock_b"], outputs["stock_a"]),
             candidate=errors(torch, outputs["s1"], outputs["stock_a"]),
             same_dispatch=tokens >= 32)
        del graphs, outputs, graph, output
    emit(event="case_done", tokens=tokens, pattern=pattern, offset=offset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-checkout", type=Path, required=True)
    parser.add_argument("--s1-patch", type=Path, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 16, 32, 128, 256])
    parser.add_argument("--patterns", nargs="+", choices=["uniform", "mixed", "hot", "local", "remote", "padding"],
                        default=["uniform", "mixed", "hot"])
    parser.add_argument("--offsets", type=int, nargs="+", default=[32])
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--reps", type=int, default=12)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    if args.blocks < 1 or args.reps < 6 or args.reps % 6 or any(t < 0 for t in args.tokens):
        parser.error("invalid measurement counts")
    if any(o not in range(0, 256, 32) for o in args.offsets):
        parser.error("offsets must be EP8 shard starts")
    verify_sglang_checkout(args.sglang_checkout, args.s1_patch)
    import torch
    if "H200" not in torch.cuda.get_device_name():
        raise RuntimeError("H200 required")
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    start = time.monotonic()
    with torch.inference_mode():
        layer = make_stock_layer(torch, generator)
        flush = torch.ones(64 * 1024**2, dtype=torch.float32, device="cuda")
        emit(event="environment", gpu=torch.cuda.get_device_name(), torch=torch.__version__,
             cuda=torch.version.cuda, seed=SEED, preparation_seconds=time.monotonic() - start,
             allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
             patch_sha256=hashlib.sha256(args.s1_patch.read_bytes()).hexdigest())
        for offset in args.offsets:
            for pattern in args.patterns:
                for tokens in args.tokens:
                    run_case(torch, layer, generator, flush, args, tokens, pattern, offset)
        emit(event="complete", elapsed_seconds=time.monotonic() - start,
             peak_allocated_bytes=torch.cuda.max_memory_allocated())


if __name__ == "__main__":
    main()
