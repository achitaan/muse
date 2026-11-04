# BF16 projection lane: token-owned FP32 SK2 epilogue

## Candidate

Submit **`candidates/dense_owner_epilogue.py`** as an OKC **check**, not the
generated `.cu` file. No check/submit has been run in this lane.

Base: `board_best/dense_kbv2_scored_1.5886.py`. The base's Python AST is
unchanged except for exporting its own SK2 source/configuration helpers. The
new wrapper keeps the base as the fallback, including all its existing
prepare-time selections and all nearby-shape routes.

## Why this is a new incremental attempt

The snapshot ledger's `qkv / o / unpooler projections` row for **FP16 transport
of SK2 dense K-warp partials**, job `eabdf1c9`, is closed. That experiment made
partial transfers narrower. This attempt retains **FP32** partials, avoids
transferring the owning warp's partial altogether, and removes the barrier
needed to repurpose the live input/weight ring. It changes neither codec,
escape handling, launch geometry, stage count, nor the MMA loop. It is not a
repeat of the closed route/Lt, rk2/routertail, or sub-12-bit codec searches.

The target is the final shared-memory transpose/reduction in the existing
`sk2_kernel`, not the bandwidth-bound weight stream. The incumbent spills all
K-warp accumulators into the retired ring, then reads both K partials in raster
output order. At WSK=2, each K-warp can instead own eight tokens of each m16
tile, keep that half in registers, and send only the other half to its peer.

The new disjoint scratch begins immediately after the existing 128-byte zero
row. All K-warp 0/1 additions and all escape-correction additions remain in
the incumbent's FP32 order; there is exactly one final BF16 rounding. One
consumer-only barrier publishes the exchanged partials. Escape helpers keep
their unchanged completion barrier and never access this extra scratch.

## Expected effect / stop rule

One candidate, four cells: **o@32, unpooler@16, unpooler@32, pooler@32**.
Partial-buffer bytes in **each direction** are halved. The separate scratch
adds 4/2/8/4 KiB respectively; the analytic shared-memory occupancy ceilings
are unchanged (actual register/occupancy attributes require CUDA).

The most plausible >=2% win is o@32, where the fixed epilogue is a larger
fraction of the short-K kernel. There is no measured speedup claim. Wider
weight-stream-bound cells may gain little or nothing. The native output
stores are less coalesced than the incumbent's raster stores and may erase
the saving; extra register liveness is another risk.

Prepare validates normal/outlier/one-hot inputs and partial token tiles,
requires exact equality to the original SK2 math, and checks FP64 accuracy.
Each cell must then win **both median and paired ratios <=0.975 on three
fresh-input/fresh-graph draws**, using the incumbent's 40-block ABBA and
512 MiB vector READ flush. A fresh post-timing replay check precedes adoption.
Otherwise the incumbent remains selected. The 270-second absolute deadline
includes the unchanged parent prepare; if it consumes the budget, the new
gate is skipped rather than exceeding the prepare limit.

If the caller's one H200 check adopts nothing, close this incremental line;
do not relax the gate or launch another geometry/compression sweep.

## Local evidence

Run:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/cpu-torch-python scripts/verify_dense_owner.py
```

The generated report is `results/dense-owner-offline.json`:

- Incumbent AST preserved, except explicit source/helper exports.
- Generated producer/escape/MMA source is identical apart from the scratch
  pointer; one native math barrier instead of two.
- 336 bit-exact FP32 epilogue emulations across every partial token count,
  NACC=1/2/4, cancellation-heavy values, and 4-row output permutations.
- Bounds/coverage checks: each output written once, no writes past M; every
  shared scratch element has one producer and an initialized peer read.
- 524,288 BF16 weights reconstructed bit-for-bit, including 68 escapes,
  positive/negative zero and a tiny forced escape; stream indexing unchanged.
- 15 full-product CPU models: normal/outlier relative L2 <=0.001811 versus
  FP64; every one-hot case exact after BF16 rounding.
- Eight fresh-output wrapper calls, metadata/fallback checks, six mocked
  gate acceptance/rejection cases (including NaN ratios and wrong outputs).

**Not verified:** CUDA host/device compilation, actual MMA execution, GPU
memory/race behavior, register/spill counts, graph capture/replay, prepare
duration, or H200 timing. No nvcc/ptxas/NVRTC toolchain or GPU is installed on
this macOS host. `results/dense-owner-offline.cu` is the exact readable
translation unit for the caller's optional compile gate, not a submission.

`TRIED.md` was updated only in this worktree, respecting the lane's write
scope. The caller must merge the result row into the shared main ledger.
