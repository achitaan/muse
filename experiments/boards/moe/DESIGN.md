# FS1: small-batch fused expert tasks

Scope: this worktree only. Parent is `moe_b13.py`; retain all original builds
and dispatch for unsupported inputs or a failed prepare-time correctness probe.
No local GPU execution or implementation commits.

## Task and numerical boundaries

One task owns 16 intermediate indices of one touched expert. Consume two
36-KiB A units, form BF16 SwiGLU intermediates, then consume one 36-KiB W2
K-slice and accumulate encoded FP32 partial outputs. A task has no dependency
on another CTA's A work. Each CTA handles a contiguous range of the 256 tasks
per expert and flushes once per expert segment using FP32 global reductions.

Use the existing A unit layout: each A stage holds 16 packed gate/up rows over
all 4096 K, producing eight intermediate indices. Two such stages produce the
task's 16 intermediates. This has exactly the proposed byte volume but avoids
changing A's eight-warp K partition and accumulator/reduction order. W2 is
repacked into natural m16n8k16 lane order, with each row's original K16 scale.

Preserve the existing Marlin boundaries: encoded A accumulator -> BF16 ->
BF16 global-scale multiply -> FP32 SiLU/product -> BF16 inter. Reduce encoded
W2 partials across tasks/CTAs BEFORE applying the B accumulator BF16 rounding
and BF16 global-scale multiply. Finalization then multiplies the parent's
already-effective BF16 norm weight, forms FP32 sum of squares, and combines
routes with `topk_w * rsqrt(ss/4096 + 1e-8)`, returning BF16. B accumulation
association changes; board tolerance, not bit equality, is the acceptance goal.

## Shared-memory adjustment required

The requested four 36-KiB slots plus an eight-token FP32 output accumulator
alone need 278,528 bytes, exceeding Hopper's per-CTA shared-memory limit.
The initial proposal was four slots for <=4 tokens and two slots for 5..8.
That compromises the requested pipeline on the recorded R8 five-token expert.
The implemented, user-approved choice is a hybrid accumulator: keep
tokens0..3 in shared memory and distribute tokens4..7 over all consumer lanes
using four shuffles per W2 MMA, adding64 FP32 accumulator values per thread.
This retains four slots but may exceed the practical register budget once x
and A accumulators are live. The user approved this hybrid option; no Oracle
consultation will be made. M<=4 gets a specialization without upper-token
register accumulators. Register pressure is measured below.

The chosen option needs212,992 bytes for ring+shared
accumulator, plus a4-KiB A-reduction buffer,256-byte intermediate tile and
routing/barrier metadata. There is no explicit per-task global output array or
extra weight pass; compiler local-memory spills remain on the hybrid path.
The original all-shared/eight-token/four-slot combination
cannot be implemented within the device limit.

## Routing, pipeline and replay

Eight consumer warps and one producer warp. CTA-local routing scans at most
64 IDs, builds touched experts and unique token lists (duplicate routes reuse
the same computed expert/token output and are combined separately at the end).
The producer starts after routing and streams three stages per task. Consumers
retain x registers per expert segment; partial output uses the hybrid layout.
Warp0 scans routing with one lane per expert and a ballot/prefix compaction.
The shared flush maps segment token positions back to actual input tokens.

Use double-buffered zero-initialized global y scratch. Each launch clears the
inactive copy, computes into the active copy, and uses one generation-counted
grid barrier before finalization. All clearing is inside the timed launch;
there is no caching of results across invocations. Empty routing still takes
the barrier and writes exact zero output. Scratch requires sequential replay,
as in the parent.

Honor PDL with `griddepcontrol.wait` before routing/x/topk-weight reads and
`griddepcontrol.launch_dependents` before the final grid barrier/finalizer.
Expose the same enable/disable launch contract as the parent.

## Validation and acceptance

- Offline sm_90a NVRTC/ptxas: resources, spills and device-source identity.
- CPU: complete task/output coverage; repack/inverse-repack equality for every
  original code and scale byte; scratch/generation and ring bounds.
- Prepare: supported shapes/scales/alignment checks; deterministic random and
  skewed routings versus the original kernel, with finite/max-error/relative-L2
  reports and fallback on failure. Compare representative inputs to an explicit
  dequantize/Marlin-boundary/reference-normalization formula as well.
- `fsdiag.py`: uninstrumented BASE versus FS, R1/R8 judge-like seeds, both
  orders, cold L2 and >=40 repetitions; additional M2/M4 and zero-route cases.

## Delivered implementation and numerical guard

`moe_fs1.py` starts with the exact bytes of `moe_b13.py`, followed by an FS
extension, lossless repacker, guarded prepare and dispatch. All parent packing,
builds and M>8 selector behavior remain available. The extra packed tensor is
905,969,664 bytes: `[32,256,3,36864]`. Parent layouts are retained. Scratch adds
8 MiB for two `[32,8,4096]` FP32 copies plus three counters. Each timed call
clears the inactive4-MiB copy and the inactive counter; no host scratch reset is
needed on replay. No cached outputs or intermediate results are reused.

One task consumes original A rowblocks2i/2i+1 followed by natural W2 K16i.
The producer warp issues four bulk-copy slots using full/empty mbarriers.
Eight consumer warps each arrive once on an empty barrier, after a warp sync
ensures every lane has finished reading. Header5760B + ring147456B + shared
accumulator65536B = **218,752 bytes** dynamic shared memory. Each expert
segment's first four tokens flush with vector FP32 reductions; upper-token
register owners flush scalar reductions. All output is reduced BEFORE the
Marlin B boundary. Finalizer y loads use L1-bypassing loads after the barrier.

Prepare validates Hopper, resident grid, EP8/widths/dtypes, positive-normal
PRMT-compatible E4M3 scales and packed alignment. The host checks actual CUDA
occupancy before launching the persistent grid. Unsupported scale sets/build
errors/probe failures print `FS_FALLBACK` and retain the parent dispatch.

Acceptance is copied from local `muse_moe_w4a16/reference.py:25,110-116` and
`checks.py:31-55`: global relative L2 <=0.015, each row <=0.075, row denominator
floored at0.1 times the mean reference row norm (and1e-12). Nonfinite values
fail. Max absolute/relative errors are reported, not used as invented limits.
The independent formula decodes original E2M1 bytes, uses original E4M3 scales,
FP32 matmul with TF32 disabled, and the reference norm `(1+post_norm_w.float())`.
A second formula mode applies the parent's BF16 accumulator/global-scale/inter/
effective-norm boundaries. Neither reference reads FS packed data.

Seed20261101 probes random and all-token expert routings at M1/2/4/8, plus
outlier x channels, zero routing interleaved with populated replay, and repeated
routes. M8 with nine routes/eight unique tokens compares to the formula only:
launching TINY2 on nine entries would exceed its capacity. Representative M1/M8
outputs compare to BOTH the raw board formula and Marlin-boundary formula.
These probes are implemented but **have not run on a GPU here**.

## Offline evidence and resource trade-off

Reproducible tooling and readable CUDA are under `.fs_offline/`:

```
python -B .fs_offline/build_source.py
python -B .fs_offline/build_diag.py
python -B .fs_offline/test_fs.py
python -B .fs_offline/test_diag.py
python -B .fs_offline/compile_fs.py
python -B .fs_offline/compile_diag.py
```

NVRTC/ptxas targets sm_90a, with CUDA cache disabled to retain resource reports.
Compilation removes the Torch host section only; diagnostic BASE uses the
same forward-declaration adapter used by prior preflights. This is **not** a
Torch host-extension compile or a GPU run.

The final hybrid build uses168 registers,80B stack,112B spill stores and128B
spill loads; low-token build151 registers, zero stack/spills. Some64 upper
accumulator values spill. A bounded allocation experiment found that requesting
224 registers eliminates spills but is NOT launchable: CUDA13
`cuda_occupancy.h:1523-1545` rounds9 warps to12 for the hardware allocation test,
so224 registers requires86,016 registers, over65,536.168 requires64,512 and fits.
Keep `__launch_bounds__(288,1)`; do not substitute the attractive nonlaunchable
variant. Unrolling the A half loop reduced spills modestly but was not adopted
without a runtime speed measurement. Final resource logs are authoritative.

Final standalone and diagnostic FS compilations both report151/168 registers
for low/high specializations and the same spills. Diagnostic BASE TINY/TINY2
both compile at146 registers and zero spills in this offline toolchain. The
recorded judge used144 for BASE; compiler/toolchain resources can differ.
Candidate211,309 bytes, SHA256 `e7a7bc0c01146ef2d586ecd6041d88775a43510e04e8bca45de1f4c076ccc7a3`.
Diagnostic84,081 bytes, SHA256 `4945291e433dd91cd6461cb7fc72e08f3c8f1189cd344b6bdeebb835c732ef02`.
Full manifest and compile options are in `.fs_offline/manifest.json` and
`.fs_offline/compile.json`.

CPU checks passed:
- Full one-expert A+B repack/inverse recovery of25,165,824 original code bytes
  and3,145,728 original scale bytes, exact, including all possible scale bytes.
  Independent dq4 algebra is used for inverse recovery, not `_POS` inversion.
- Actual W2 packed fragment coordinates match PTX row/K coordinates; hybrid
  ownership covers all8*4096 outputs once, with correct destination-lane parity.
-15 task/ring coverage models (0/1/2/6/32 experts; grids8/17/132),100 scratch
  generations with empty calls,96 routing models over all M1..8 and local
  starts0/32/224, including duplicate and sparse-token routes.
- Analytical one-coordinate E2M1=1 reference solution, both raw and Marlin
  boundaries; checker rejection of wrong/nonfinite outputs.
- Exact parent prefix, assembled CUDA identity and diagnostic payload hashes.
- Three-TU thin-host diagnostic linkage/source checks;40 cold-L2 samples with
  flush outside events;18 mocked timings in both orders and4 ratio summaries.
  Mock values establish control flow only, not accuracy or speed.

## Diagnostic and remaining risks

`fsdiag.py` embeds the exact candidate and pinned parent hash. It builds only
BASE TINY, BASE TINY2 and FS (three CUDA TUs, one binding), bypassing the parent
MAIN selector. R1/R8 use the recorded judge-like routing seeds; S2/S4 force an
expert across all tokens, and ZERO8 never calls the unsafe parent zero-job path.
Both case/variant orders,40 repetitions each,256-MiB cold-L2 flush outside CUDA
events. Resource/error/median/min/ratio lines use `FS_*`. It runs the same
prepare guard before timing; failure yields no misleading FS speed claim.
The prefill-shaped return is deliberately a diagnostic placeholder, not a
submission output. Weight distribution matches prior diagnostics but generating
only32 experts changes offsets versus a full256-expert weight draw.

No GPU execution, judge checks, direct git operations or implementation commits.
Runtime numerics, actual residency/PDL memory ordering, performance, and the
235-second diagnostic/full prepare host-build budgets remain unverified.
Hybrid spills, four-shuffle redistribution, shared accumulator traffic, local
routing, four-MiB scratch clear and low-parallelism finalization can erode the
anticipated gain. The ringdiag bandwidth floor is motivation, not an FS result.
Scratch is single-device and sequential-replay only, as in the parent; concurrent
calls sharing one state are unsupported. A hung/faulted device cannot be safely
rescued by a Python fallback; only completed probe failures can fall back.
