# Muse serving MoE: single-H200 baseline

The first port milestone is to read SGLang's existing packed weights and scales without allocating an additional persistent expert bank. The starting point is the stock Muse Marlin path after pre-expert norm. Pin SGLang to `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. See the [serving design](https://github.com/achitaan/muse/blob/19e09dd694d748d6ef77bd0e09308f56c3b210ca/serving/MOE_PORT_DESIGN.md).

## Run order

1. Run `python serving/check_marlin_nvfp4_layout.py` on CPU. It checks address anchors and full nibble and scale coverage across tile boundaries.
2. On an H200 with the pinned SGLang checkout and compatible dependencies, run the GPU byte comparison first. Save its output. It compares against stock's CUDA repacker, including all nibble positions and every nonnegative finite E4M3FN scale byte.
3. Run the stock benchmark and keep its JSONL. It uses the stock weight prep, both expert GEMMs, SwiGLU, and stock finalizer. The fixture has 32 local experts at offset 32 and two live local routes per row. It records raw cold-L2 replay samples and stock output repeatability for T=1, 8, 16, 32, 128 and 256. It requires no model checkpoint.

```bash
# From this repository's root on the GPU machine.
export PYTHONPATH="/path/to/pinned/sglang/python${PYTHONPATH:+:$PYTHONPATH}"
timeout 10m python -u serving/check_marlin_nvfp4_layout.py \
  --sglang-checkout /path/to/pinned/sglang
timeout 20m python -u serving/bench_marlin_stock.py \
  --sglang-checkout /path/to/pinned/sglang > stock-marlin-h200.jsonl
```

These are synthetic local expert measurements. A candidate still needs a stock differential on the same inputs, fresh routes on graph replay, and a separate all-remote and padding probe with a watchdog. The latter checks the custom path's zero-job guard. Stock output agreement here does not establish identical generated tokens or serving throughput.

CPU address checks, the exhaustive scale transform comparison, and the H200 CUDA
repack comparison passed. Clean-stock baseline samples, stock/S1 paired measurements,
and the bounded direct-layout prototype are now preserved in
[PRECOMP_HANDOFF.md](PRECOMP_HANDOFF.md) and `results/precomp-20261105/`.
`bench_precomp.py` expands routing to uniform, hot, all-local, mixed, all-remote
and padding patterns across shard offsets, with separate cold/steady modes and
stock-versus-stock controls. The completed performance comparison used uniform,
mixed and hot routing at offset 32; the larger routing matrix is correctness/replay
coverage, not a timing claim for every combination.
