"""Measure the pinned Muse Marlin path on one synthetic EP8 expert shard."""

import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

from check_marlin_nvfp4_layout import verify_sglang_checkout
from marlin_nvfp4_layout import SGLANG_COMMIT

SEED = 4207
EXPERTS, LOCAL_EXPERTS, OFFSET = 256, 32, 32
HIDDEN, TOP_K = 4096, 8
TOKEN_COUNTS = (1, 8, 16, 32, 128, 256)


def make_stock_layer(torch, generator, *, edge_scales=False):
    from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    # Supply the real method's post-load fields without starting a model server.
    layer = MuseSpark1xFp4Experts.__new__(MuseSpark1xFp4Experts)
    torch.nn.Module.__init__(layer)
    layer.quant_config = SimpleNamespace(group_size=16)
    layer.moe_runner_config = SimpleNamespace(is_gated=True)
    layer.params_dtype = torch.bfloat16
    layer.intermediate_size_per_partition = HIDDEN
    layer.num_local_experts = LOCAL_EXPERTS
    layer.global_num_experts = EXPERTS
    layer._expert_map = torch.full((EXPERTS,), -1, device="cuda", dtype=torch.int32)
    layer._expert_map[OFFSET:OFFSET + LOCAL_EXPERTS] = torch.arange(
        LOCAL_EXPERTS, device="cuda", dtype=torch.int32,
    )

    def random_bytes(high, shape, low=0):
        return torch.randint(
            low, high, shape, dtype=torch.uint8, device="cuda", generator=generator
        )

    for name, rows in (("w13", 2 * HIDDEN), ("w2", HIDDEN)):
        packed = random_bytes(256, (LOCAL_EXPERTS, rows, HIDDEN // 2))
        scale_bytes = random_bytes(127 if edge_scales else 64,
                                   (LOCAL_EXPERTS, rows, HIDDEN // 16), low=0 if edge_scales else 32)
        # ModelOpt collapses w13 to one scale per expert before this preparation.
        global_scale = torch.full((LOCAL_EXPERTS,), 2.0**-9, device="cuda")
        if edge_scales:
            global_scale *= torch.linspace(0.5, 1.5, LOCAL_EXPERTS, device="cuda")
            # The model loader collapses unequal gate/up globals to column 0
            # BEFORE prepare_moe_nvfp4_layer_for_marlin. Preserve that contract.
            if name == "w13":
                raw_global = torch.stack((global_scale, global_scale * 3.0), dim=1)
                global_scale = raw_global[:, 0].contiguous()
                layer._fixture_uncollapsed_w13_global = raw_global
        for suffix, tensor in (
            ("weight", packed),
            ("weight_scale", scale_bytes.view(torch.float8_e4m3fn)),
            ("weight_scale_2", global_scale),
        ):
            setattr(layer, f"{name}_{suffix}", torch.nn.Parameter(tensor, requires_grad=False))
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def measure_case(torch, layer, generator, flush, tokens, reps):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )

    def rand(shape):
        return torch.rand(shape, generator=generator, device="cuda")

    x = torch.randn(
        (tokens, HIDDEN), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    # Fixed synthetic mix: two distinct local routes and six distinct remote
    # routes per row. Shuffle slots to avoid privileging one reduction order.
    local = rand((tokens, LOCAL_EXPERTS)).topk(2).indices + OFFSET
    remote = rand((tokens, EXPERTS - LOCAL_EXPERTS)).topk(6).indices
    remote += (remote >= OFFSET) * LOCAL_EXPERTS
    order = rand((tokens, TOP_K)).argsort(dim=1)
    ids = torch.cat((local, remote), dim=1).gather(1, order).to(torch.int32)
    scores = rand((tokens, TOP_K)).softmax(dim=1)
    effective_gain = (0.75 + 0.5 * rand((HIDDEN,))).to(torch.bfloat16)

    def forward():
        expert_outputs, local_ids = layer._marlin_expert_outputs(x, scores, ids)
        return fused_post_expert_norm_weighted_sum(
            expert_outputs, scores, norm_weight=effective_gain, eps=1e-8,
            weight_first=True, slot_expert_ids=local_ids,
        )

    # Complete kernel compilation and allocator warmup before capture.
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        forward()
        expected = forward()
    torch.cuda.synchronize()
    if not torch.isfinite(expected).all().item() or not expected.any().item():
        raise RuntimeError(f"stock output invalid at T={tokens}")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    graph.replay()
    torch.cuda.synchronize()
    reference = output.clone()
    if not torch.isfinite(reference).all().item():
        raise RuntimeError(f"stock graph produced nonfinite output at T={tokens}")
    delta = (expected.float() - reference.float()).norm() / expected.float().norm()
    times_us = []
    identical = 0
    worst_replay_rel_l2 = 0.0
    reference_fp32 = reference.float()
    reference_norm = reference_fp32.norm().clamp_min(1e-12)
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for _ in range(reps):
        flush.sum()  # Read beyond H200 L2 before the start event.
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times_us.append(start.elapsed_time(end) * 1000)
        if torch.equal(output.view(torch.int16), reference.view(torch.int16)):
            identical += 1
        else:
            relative_error = ((output.float() - reference_fp32).norm() / reference_norm).item()
            if not torch.isfinite(output).all().item():
                raise RuntimeError(f"stock replay produced nonfinite output at T={tokens}")
            worst_replay_rel_l2 = max(worst_replay_rel_l2, relative_error)
    return {
        "tokens": tokens,
        "median_us": statistics.median(times_us),
        "samples_us": times_us,
        "bit_identical_replays": identical,
        "worst_replay_rel_l2": worst_replay_rel_l2,
        "eager_vs_first_graph_rel_l2": delta.item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-checkout", required=True, type=Path)
    parser.add_argument("--reps", type=int, default=15)
    args = parser.parse_args()
    if args.reps < 2:
        parser.error("--reps must be at least 2")
    verify_sglang_checkout(args.sglang_checkout)
    import torch

    gpu = torch.cuda.get_device_name()
    if "H200" not in gpu or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError(f"expected an H200, got {gpu}")
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    with torch.inference_mode():
        layer = make_stock_layer(torch, generator)
        flush = torch.ones(64 * 1024**2, dtype=torch.float32, device="cuda")
        print(json.dumps({
            "gpu": gpu, "sglang": SGLANG_COMMIT, "seed": SEED,
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "local_experts": LOCAL_EXPERTS, "local_offset": OFFSET,
            "local_routes_per_token": 2, "warmup_calls": 2,
        }), flush=True)
        for tokens in TOKEN_COUNTS:
            print(json.dumps(measure_case(torch, layer, generator, flush, tokens, args.reps)),
                  flush=True)


if __name__ == "__main__":
    main()
