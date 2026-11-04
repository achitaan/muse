# Dense restart: raw-BF16 clustered full-token Stream-K GEMM

## Profile selection and build decision (written before the kernel)

Source: public check `8ba14018820847fd`, 30-row table in the supplied profile.
Equal-weight load totals are 33.63/46.75/50.63/37.38/**59.73/80.82 us** for
decode1/16/32 and verify1/16/32. All shapes have weight 62, which cancels.
The lowest speedups are unpooler256 **1.046x**, o256 **1.058x**, o128
**1.083x**, unpooler128 **1.131x**, qkv256 **1.138x**. Pooler256 and
unpooler256 consume **58.46%** of verify32 time; qkv/o256 another **32.70%**.
Unpooler/pooler128 consume **58.45%** of verify16 time. These dominate the
restart, with o16/32 included because its current 1.244x/1.201x is weak.

Raw small-M pooler/unpooler are NOT attractive: their single-pass BF16
memory floor is ~14.1 us already, versus 13.48–14.71 us observed. They stay
on the incumbent compressed kernels.

Single-pass roofline: `max((2NK+2MK+2MN)/4.8e12, 2MNK/989e12)` seconds.
Weights are original BF16, accumulated in FP32. Neither compression nor
approximate arithmetic is assumed.

| cell | incumbent us | ideal raw floor us | design budget us |
|---|---:|---:|---:|
| qkv128 | 12.27 | 8.424 | 11.110 |
| o128 | 6.57 | 3.987 | 5.430 |
| pooler128 | 17.26 | 14.636 | 18.013 (fallback) |
| unpooler128 | 17.65 | 14.636 | 17.263 (below gain gate) |
| qkv256 | 17.60 | 9.771 | 12.635 |
| o256 | 8.83 | 4.478 | 5.976 |
| pooler256 | 24.15 | 17.371 | 21.343 |
| unpooler256 | 23.10 | 17.371 | 20.904 |
| o16 / o32 | 5.43 / 5.67 | 3.557 / 3.618 | 4.952 / 5.020 |

The model includes grid utilization and Stream-K imbalance. The design
budget assumes **90% roofline efficiency +1 us fixed cost +0.75 us for
FP32 split reduction**, and only adopts modeled ratios <=0.975. The resulting
board headroom is **11.59% ideal**, **4.00% design budget**: the >=3% build
gate passes. At 80% efficiency it is only **1.47%**; with +1.5 us fixed and
+1 us reduction it is **2.66%**. Thus 4% is an optimistic engineering target,
not a prediction or measured speedup. Dense is NOT at its physical floor.
`scripts/dense_redesign_roofline.py` reproduces all numbers and writes JSON.

## Fundamentally different core

Each two-CTA cluster owns two neighboring **64-weight-row x ALL-token**
outputs. One native SS-WGMMA warpgroup/CTA consumes raw BF16, without a codec,
escape walker, decode warpgroup, register-A reconstruction, or cuBLAS plan.
The second dimension is the entire native token tile (16/32/128/256), so each
weight is read **exactly once**, including at 256 tokens.

For qkv/pooler, **66 clusters** partition a flattened `(128-output-row pair,
K64 chunk)` stream into equal contiguous ranges. This balances 132 CTAs
without the long tail of fixed 3/4-way K splits. Private FP32 partials and a
non-spinning last-arriver reduction handle boundary tiles. For o/unpooler,
64 clusters already fill 128 SMs: use whole-output ownership with no split
workspace or counter traffic. No global grid barrier or residency assumption.

Within a cluster, one producer per CTA copies its own raw weights; rank 0
multicasts each activation tile to both CTAs. **Consumers re-arm the next
full-barrier phase before releasing a slot.** Rank 0's empty barrier counts
both consumers, ensuring both destinations are armed before multicast.
This removes a separate peer-ready handshake. Three stages overlap TMA and
WGMMA; a disjoint output transpose buffer protects prefetched future tiles
in Stream-K mode. A final cluster barrier protects remote shared-memory
lifetime. Shared memory is <=193600 B, including alignment slack.

Why this could win: eliminate decode/escape overhead relative to compressed
routes and avoid duplicate token-tile weight streaming by construction;
halve activation source requests versus two independent 64-row CTAs;
balance qkv/pooler across the full GPU; keep FP32 reductions single-launch.
Why it can lose: 90% efficiency is ambitious; L2 activation/partial traffic,
cluster credit round trips, register pressure and last-arriver tails are
not represented by the ideal HBM roofline. The GPU gate is mandatory.

This is not the closed CUTLASS plan sweep (`df119bda`), simple per-warp
FMA/mma.sync GEMVs (`33c77700`), or c12 DW decoder pipeline (`027ee79b`).
In particular it has no DW decoder group and no producer-side per-stage
peer-ready handshake. It shares only architectural PTX instruction forms
with the incumbent; scheduling and data ownership are a new implementation.
The closed owner epilogue (`8ba14018820847fd`) is not included. The pending
rotated-sign candidate is not read or incorporated.

## Acceptance / local verification plan

Keep the unchanged board-best file as fallback. Require FP64 normal/outlier
and one-hot accuracy, adjacent-M correctness, changed-input graph replay and
zero counters, then three independent READ512 ABBA draws with both median
and paired ratios <=0.975. Resource attributes and full build errors must be
reported. Never retain validation inputs/outputs in production state.

Offline gates must prove exact weight-stage coverage, output/register map,
swizzle roundtrip, partial-slot and counter ownership, arbitrary arrival
order/replay, and randomized asynchronous two-rank barrier schedules.
CPU numerical models and static checks cannot establish CUDA compilation,
GPU synchronization correctness or speed. No CUDA tools/GPU are available
locally; no OKC calls or pushes are authorized in this lane.

## Implemented attempt / local evidence

Standalone check file: **`candidates/dense_redesign.py`**. The complete
incumbent prefix is AST-identical; the raw core is appended independently.
All ten target cells are gated against the actual inherited production path.
Six intended template modes cover native 16/32 output ownership, and native 128/256
with either output ownership or balanced cluster Stream-K. Resource logs
report registers, local bytes and requested dynamic shared memory per mode.
Changed-input validation poisons private scratch, tests zero then fresh
normal data, requires bit-repeatability on identical inputs, and verifies
that counters reset in the timed kernel. Same-cell concurrent calls on
different streams would need separate scratch, as for the incumbent.

`scripts/verify_dense_redesign.py` passed:

- Exact single-pass weight-stage coverage, all partial contributor/slot
  bounds and capacity checks on the ten actual target geometries plus 36
  boundary K/N geometries. Dynamic shared memory <=193600 bytes.
- 1460 asynchronous two-rank credit schedules / 523892 events, including
  every actual qkv/pooler/o/unpooler cluster range. Independently delayed
  TMA destinations, producer stalls, async MMA completion, remote arrivals,
  segment flushes and one/two-stage short ranges all complete safely in
  the model. This is not a proof of hardware memory-ordering behavior.
- 432 native-M output-tail mappings with each output written once, no
  writes past M, and exact raw-BF16 SW128 layout roundtrips.
- 36 numerical models across all six modes and partial token tiles, with
  arbitrary contribution arrival order and repeated scratch reuse:
  relative L2 <=0.001681 versus FP64, probes and zero cases exact. Missing
  contributions are detected by counter/poison checks.
- 12 fresh-output wrapper calls, 30 fallback checks and six mocked
  acceptance/rejection cases. Timed-wrapper AST excludes value-dependent
  host work or synchronization.
- C++20 template/host-wrapper **syntax-only** preflight passed with CUDA
  and ATen declarations stubbed and every PTX instruction removed.
  This does **not** establish Torch ABI, PTX or CUDA compilation.

Evidence: `results/dense-redesign-offline.json`, exact generated readable
translation unit `results/dense-redesign-offline.cu`, and the separately
marked syntax-only `.cpp` artifact. No nvcc, ptxas, NVRTC or GPU is available.
Actual resource use, CUDA compilation/capture/replay, hardware races,
prepare duration and performance remain unverified. One caller-run H200
check is needed; if it adopts nothing, close this structure rather than
spending a sweep on it. Shared `TRIED.md` updates are local to this worktree
and must be merged into the shared ledger by the caller.
