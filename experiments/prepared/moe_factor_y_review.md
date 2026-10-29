# MoE BF16 factor scratch and register cache

Prepared October 29, 2026. CPU evidence only; no CUDA compilation, resource
measurement, GPU correctness result or speedup is claimed at this checkpoint.
One fixed candidate, one paced official check, no parameter sweep.

## Source binding and scope

Baseline: `best/moe_verified.py`, SHA256
`331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.
Its source-bound primary check d600231c8a8244cb achieved 2.365437416088433x;
publication b6c70c9b83724d2e achieved 2.348299935218348x.

Frozen candidate: `results/moe-factor-y-preflight-20261029/submission.py`,
SHA256 `85a317b7bd44cad6f34bf0e3d6d83fc62569867acc7f46c223822d44c36fa741`,
336922 bytes. Alternate CUDA SHA256
`30d46586137332537599fd7db75df3c7e019bdfcbdceab7865e44841404947c1`.
The original submission remains an exact prefix. A reversible edit ledger
recovers every original CUDA byte. The original small-M and non-PRMT paths,
weight packing, scheduling, fixups and nonempty ownership algebra remain exact.
Actual new execution has device namespace `okc_factor_y::moe_fused`.

## Observed work and proposed mechanism

Source inspection shows the retained WGS phase B writes an FP32 product of a
BF16 scaled GEMM result and the fixed BF16 effective norm gain. Its finalizer
reads that FP32 row twice, for norm statistics and weighted reduction.

Store the same pre-gain BF16 result bits in a compact prefix of the existing
y-buffer region. Each finalizer thread loads eight columns for each of eight
route slots once, caches the bits, and reconstructs the FP32 gain product for
both passes. Logical y-buffer traffic falls from 49152 to 16384 bytes per
local route. This counts requested intermediate bytes, not measured HBM bytes.
It is not a prediction of a two-thirds kernel-time saving: cache hits, other
work, extra conversions, multiplication and register lifetimes all matter.

The changed family serves M>16 when scales are PRMT eligible. This is distinct
from the rejected scale compression, occupancy, cache policy and skinny16
experiments. It does not change expert weight bytes or token geometry.

## Numerical and synchronization boundaries

The existing BF16 GEMM accumulator, global-scale conversion and scaled-product
rounding stay intact. Bit extraction and promotion recover that exact BF16
product. The effective gain remains the retained BF16 `(1+w)`. Explicit
`__fmul_rn` reconstructs the original FP32 gain product before either reduction;
there is no extra BF16 gain-product rounding. Norm denominator is full H4096,
epsilon is 1e-8, and route order, final BF16 output and all three finalizer CTA
barriers remain unchanged. NVIDIA documents the noncontracting RN multiply:
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html

Invalid slots are masked before loading or arithmetic. CPU fixtures poison
invalid values with NaNs. An explicit `nj==0` range return precedes the job-table
read/division in the alternate CUDA. Inspection found the original empty range
reads an uninitialized job entry; this is a source finding, not an observed GPU
failure. All-remote analytic probes run the new path only and require exact zero.
The original implementation remains the fallback for unselected configurations.

All K-stage contributions, readiness publication, generation counters, final
arrivals, grid join and stream-K fixup offsets remain unchanged. CPU ownership
checks establish unique stores and prefix/fixup separation; GPU races and replay
correctness still require the official check and fresh-input probes.

## Resource and preparation budget

At chunk capacity, the separate candidate scratch bank is 164668960 bytes
(about 157.04 MiB): 16416, 71329280 and 93323264 bytes across its three tensors.
No extra expert layout is allocated. Original inputs, packed weights, incumbent
scratch, candidate scratch, compilation buffers, captured outputs and the
128 MiB cold-L2 flush buffer are all additional live allocations to budget.
Scratch capacity is not per-call traffic. The new compact prefix ends at byte
25165823 relative to ybuf, before the unchanged fixup region at byte50331648.

Each finalizer thread caches 32 uint32 payload words and eight FP32 gains,
alongside existing live state. These are source-level values, not allocated
register counts. Register pressure and local-memory allocation are principal
risks. The fixed gate requires both BIG variants to have 1..112 registers,
zero local bytes, zero static shared bytes, 231168 dynamic shared bytes,
at least one resident block, 576 threads and grid1..160. Query occupancy only
after setting the dynamic shared-memory attribute. A failed resource gate
skips new-kernel launches and selects the incumbent. Do not relax this gate
after seeing the result.

Only one extra MAIN extension is built. Original preparation over185 seconds
skips it; total compilation over270 seconds skips probes. The case loop stops
starting new cases after280 seconds. Every reachable extension is built during
prepare. Partial evidence cannot enable the candidate. No timed host reads,
data-dependent allocations, fixture reuse or lazy compilation are introduced.

## CPU evidence and fixed GPU design

The frozen packet records all65536 BF16 bit-pattern pack/promote checks,
1566720 finite gain-product comparisons to an independent FP64-to-FP32 oracle,
signed zero/subnormal/overflow cases, 262144 unique phase-B stores, six shape
fixtures, 100 host dispatch cases and25 chunk cases through M769. Selection
and full-result guard mutation tests reject11 bad cases each. These are CPU
filters; they do not establish GPU correctness or performance.

Fresh official TASK/reference/checker/baseline/nvfp4/AGENTS bytes matched all
six pinned files. The identity manifest records verification time separately
from an unavailable download timestamp; the earlier approximate-time manifest
is preserved as historical evidence.

Within one authorized check, normal paired probes are fixed at
M17,31,32,33,63,64,65,127,128,129,191,192,255,256,257,384,385, plus skewed M128.
Each uses ten cold-L2 graph pairs with balanced randomized AB/BA order.
CUDA-event milliseconds convert to nanoseconds with1e6. These replays are not
independent jobs. M32 also replays two fresh hidden/route/weight inputs.
All-remote M17,33,384,385 must be exactly zero; duplicate-route M33 compares
against the original. Errors must stay within0.5% relative and2% worst-row.

Enable the single new family only if every paired shape loses at most2%,
each normal M32/128/256 improves by more than3%, their geometric-mean ratio
is below1/1.03, and every resource, analytic and replay gate passes.
The full suite must then pass5tests/6benchmarks/integrity, with new namespace
in affected profiles2/4/5 and absent from small controls0/1/3.

Full-result retention requires affected-load GM gain>3%, control-normalized
gain>3%, each affected load at least99% of normalized baseline, control GM
within[0.95,1.05], and complete-track speedup>2.4009189773297592x (1.5% above
the checked baseline). These are conservative engineering gates, not a joint
statistical confidence statement. Empty choice, skip, inconclusive evidence
or a losing path retains the incumbent; no score or flag sweep follows.

## Overall contribution and next decision

The affected three of six MoE loads dilute an equal per-load improvement s to
sqrt(s) at track level. Exact whole-layer gain remains unmeasured; previously
saved MoE profiler sums do not establish its exposed critical-path fraction.
This candidate has a viable Decode transfer: use the same MAIN implementation
for matching eligible shapes, replacing its corresponding build rather than
building both alternatives, while retaining all small paths and layer rounding.
The CPU-only Decode package provides source/preparation headroom. Integrating
it requires a retained standalone gain and new whole-layer evidence first.

Freeze the packet before replacing `candidates/moe_active.py`. Use the existing
foreground experiment controller and cached baseline, wait locally until at
least03:22:01.371647UTC, launch once, persist the job ID and follow that handle.
Only one GPU job may be active. Publication is a separate decision after real
new-path correctness and material performance evidence; quota is a reserve.
