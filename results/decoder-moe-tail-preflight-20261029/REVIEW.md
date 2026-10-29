# Decode: persistent MoE, unpooler and FFN boundary

This is an unmeasured candidate. No best source has changed. Parent D9 is
`d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988`.
The single-file candidate is `1ed5ba7b1c87ae0142bf0d52fba08e7938c1be9dcc5d3200d388fabee15c951f`,
956,861 bytes. Native CUDA is `95abcfa8857ca584e03c86a5ba1ac583a26b6abde4ff13b5c4fcafe5a2052091`.

## Opportunity and scope

The D9 source-bound profiles put the small-token unpooler and FFN boundary
together at approximately 7–8 microseconds. This is a summed profile duration,
not a measured saving or a critical-path fraction. Removing the pre-expert
norm alone has an optimistic tiny-load screen of only 0.61% at track level;
that proposal is parked without a run.

The new proposal embeds the existing tiny MoE final output, a BF16 unpooler
and the final FFN residual/norm in one persistent launch. Only M1–8 can use the
official 1024-element unpooler prefix. Every larger shape keeps D9's full path.
Attention, routing, pooler, pre-expert norm/ID mapping, expert GEMMs, SwiGLU,
expert postnorm and their BF16 boundaries retain the parent source. Unsupported
device, PRMT scales, cache or layout uses the original root tail.

The separately reviewed preparation package `307dd791` is the base. Its
runtime kernels are D9's; its preparation cleanup has no isolated measured
speed claim. New work adds two tiny extensions, one 8 MiB dense BF16 layout,
and a private copy of the approximately 157 MiB fixed MoE scratch. Original
packed expert weights are shared, not duplicated or expanded. Extra graph
outputs, compilation/transient packing and the 512 MiB comparison flush also
contribute to peak memory. Actual peak and 300-second preparation remain GPU
validation requirements; the historical parent took 256.897 seconds.

## Ownership and publication

The original four CTAs per token store all 4096 BF16 MoE outputs. Every writer
fences its stores, the consumer threads join, and thread zero publishes one
acquire/release arrival to that token's counter. The 64 projection CTAs wait
for all four segments of each input row. No consumer loads uninitialized
nonlocal expert data: that masking remains in the original finalizer.

Each projection CTA computes 16 output rows. Its eight warps own disjoint
512-element K partitions; FP32 partials reduce before storing BF16. The new
layout is a permutation of original BF16 dense weights. Its fragment ownership
comes from the [NVIDIA PTX BF16 MMA specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float).

After projection, all grid CTAs publish and join a second, distinct generation
counter. The first M CTAs then process complete 8192-element residual rows.
The branch has an implicit zero tail, but both RMS denominators remain 8192.
The boundary preserves BF16 unpooler output, BF16 input-scale product, BF16
postnorm branch, separately rounded FP32 residual products, FP32 updated
stream/pre-norm input and BF16 output/gain. Two disjoint shared reduction banks
avoid reuse before all readers finish. Each row owner clears its readiness
counter only after every projection consumer has joined.

The new backend needs private sequential scratch. Interleaving the old backend
on the same scratch could leave the second barrier's alternating counters
stale. Separate scratch is explicit in the Python state and CPU host tests.
TINY and TINY2 share only this new backend's private scratch. No concurrent
same-state call contract is added.

The host uses a cooperative launch after a native occupancy/capability query.
Every CTA must fit resident simultaneously; otherwise the path is inadmissible.
No guessed occupancy, unresident spin waiter, timed host data reads, cached
routing, input-dependent allocations or lazy compilation is allowed.

## Completed CPU evidence

- Exact inverse edits recover the original MoE CUDA before the dedicated
  namespace and host query are added. Root inverse recovers all of `307dd791`.
- All 4,194,304 original BF16 weight values reconstruct bitwise, including
  signed zero, subnormals and all payload bits. MMA A/B and output ownership
  cover every K contribution and output exactly once.
- 2,048 replay schedules cover alternating shapes and generations. Premature
  publication, omitted second join and omitted readiness clear are detected.
- Thirty boundary arithmetic fixtures match the independent retained-Triton
  CPU model bitwise; eight analytic impulse GEMMs pass. These do not emulate
  hardware MMA or rsqrt.
- Both complete tiny proxy builds use 150 registers with zero stack/spills,
  versus the saved 146-register original proxies. Shared arrays are unchanged.
  This is NVRTC 13.0 device-only evidence, not the evaluator's NVCC ABI or
  actual H200 resource measurement.
- CPU host tests exercise all M1–8 argument lists, separate output allocations,
  family selection and larger/unsupported fallbacks. Twenty-five corrupted
  result fixtures are rejected. The positive fixture is synthetic.
- All six freshly downloaded official Decode files match the pinned copies.

## One-check experiment contract

Setup must first validate actual H200 resources/cooperative residency, lossless
packing, 24 complete-tail numerical fixtures including duplicate/all-nonlocal
routes, independent FP64 dense oracles and 16 changing-input/route graph
replays with mixed shapes and old-backend interleaving.

Ten cold-L2 paired graph samples, alternating arm order, compare the complete
MoE + unpooler + FFN tail at M1/8 and nearby M3/7 with two fixed fixtures each.
Both scored sizes must save at least 4.6 microseconds and beat old by 3%; nearby
ratios must be <=1.02. This prospective absolute saving screen motivates a
possible 1.5% full Decode gain. It is not a prediction or significance proof.
A losing setup terminates with diagnostics instead of benchmarking a retained
fallback and reporting its score as new-kernel evidence.

Retention additionally requires all official 13 tests/12 benchmarks/integrity,
the expected new profile markers only on M1/8, actual weighted six-load
reconstruction, >1.5% aggregate and control-normalized gain, >=2% normalized
gain in each affected load, and no affected-case regression beyond 0.5%.
Controls must remain within [0.95,1.05]. Raw pairs are preserved. These are
prospective decision rules, not a joint confidence interval.

No flag sweep or noise retry follows a loss. A retry needs a diagnosed bounded
defect. Current pacing, reserves and 8 p.m. Toronto authorization still apply.
