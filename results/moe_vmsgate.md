# VMS routing-lightness candidate — remote validation pending

## Artifact and provenance

- Branch: `loop/moe-vms-gate`, created from `63e253f`.
- Standalone check file: `candidates/moe_vmsgate.py`.
- Source: read-only `muse-engine/kb/exp/moe_vms5.py`, whole-file SHA256
  `18404b6c154c3cadcfa08fa10d55b6e96a39c81db7704d57af8a33a465130546`.
- Its incumbent is **ps1/st1f4**, including the bs8 WSI specialization and
  zero-local-job fix. This is not the older `best/moe_verified.py` in this
  checkout. Existing attribution notices are retained.
- No main commits, pushes, submissions, or judge checks were performed.

## Evidence and decision

Read the shared ledger before building. VMS4/5 checks `c5800fa6` and `adbd3ba4`
showed bs16 paired ratios around 0.986–0.993, with the light-draw ratio tail
around 0.917; bs32 was near a tie. The closed experiments changed pipeline
geometry. This experiment is materially different: **measure the per-draw
distribution and test whether a routing feature predicts the wins**.

The incumbent's route counts live in each CTA's private shared memory, not in a
global histogram a Python dispatcher can inspect. Ordinary graph capture records
both branches' launches; host branching on a device flag is not capturable.
Predicating the existing paths needs a feature-producing kernel plus four
skipped VMS launches on an incumbent draw (or a skipped incumbent launch on a
VMS draw). That is not a credible sub-1-us dispatch without measurement.
Putting both engines into one device entry would be a new persistent-kernel
architecture: their CTA geometry, shared-memory ownership, synchronization,
and finalizer grids differ. Native CUDA conditional graph nodes are possible
in principle, but need a new, toolkit-dependent integration with torch capture.
**No impossibility claim is made about that separate implementation.** We cannot
certify its overhead without GPU validation and do not ship an unvalidated one.

Accordingly this implements the requested **diagnostic fallback**, with optional
strict static per-batch adoption. The learned threshold is never applied in
`kernel_fn`. There is no runtime histogram launch, host routing read, dynamic
predicate, synchronization, or graph-conditional construction in this artifact.

## Prepare experiment and gates

For each of bs16 and bs32:

1. Compile only the established VMS5 control: 4-slot/512-K at bs16, 3-slot/512-K
   at bs32, both split-K=2 GEMM2 and the two-step deterministic finalizer. Two
   extension builds run concurrently with incumbent preparation. Every
   `load_inline`, including the incumbent, explicitly uses **C++20 for both
   host and NVCC**. No experimental or wrong-result diagnostic kernels are
   instantiated.
2. Repack existing raw unit-order NVFP4 bytes into the control's slot layout.
   Codes, block scales and global scales are unchanged; no half-slice copy or
   additional quantization is made.
3. On one reused graph, mutate x/ids/routing weights through all-remote, hot,
   all-local, skewed/outlier and normal cases. Check the public independent
   FP32 reference, bit equality against the incumbent, and VMS repeatability.
4. Generate **32 independent public-distribution draws**, with fixed recorded
   seed formula `710000 + 1009*bs + 37*draw`. Measure D, R_max, total local
   routes. This is the same heavy-tailed popularity/top-8/busiest-rank
   relabel distribution as the incumbent's `_gen_input`, not uniform routing.
5. Capture three separate graphs per draw: incumbent, independent incumbent
   null, and VMS. Replay after capture before checking captured outputs.
   Require public FP32 error bounds (0.015 global, 0.075 worst row), strict
   **BF16-bit equality with the incumbent**, and VMS repeatability.
6. Two passes per draw, each with six rounds. Each round runs ABBA and BAAB
   for VMS-vs-incumbent and for null-vs-incumbent. Alternate which comparison
   comes first; read a 512 MiB buffer **before each replay**, outside the
   event interval. Never compare unpaired timings from different draws.
   Times are raw CUDA-event graph replay microseconds, without assuming a
   fixed graph-overhead subtraction.
7. Static VMS adoption requires every draw's null in **[0.995, 1.005] in both
   passes**, all numerical/exact/repeat/stress checks, and paired geometric
   mean AND elapsed-time ratios <=0.99 in both passes on **each** 16-draw half.
   Any exception, deadline or sticky kernel error retains the incumbent.
8. Fit a one-sided lightness threshold on D, R_max or route count using the
   first 16 draws (minimum four draws on each side). Report the fixed policy's
   zero-cost counterfactual ratio on the other 16 draws. Do not refit on the
   held-out half or apply the policy at runtime.

At the end, `MOE_VMSGATE` prints every completed draw's D, R_max, routes,
inc_us, vms_us, paired ratio, null ratio (also both individual null passes),
bit-exactness and reference error. Per-bs summaries print p10/p50/p90 ratios,
static geometric mean, null validity, correctness/repeat/stress counts and the
held-out threshold result. The tested complete 64-draw log plus summaries is
10,286 bytes, within the judge's 16,000-character log tail.

### Budget

Build wait cutoff: 170 s from prepare entry. Diagnostic deadline: 265 s,
checked between stress/draw work and every ABBA quad. Only two control builds
and one numerical/timing arm are added, rather than all VMS5 ablations.
Given the supplied 38–42 s incumbent selector and 40–50 s builds, expected
preparation fits 300 s, but **elapsed time is not validated locally**. The
deadline is cooperative: a running compiler or CUDA operation cannot be
interrupted, and `ThreadPoolExecutor` timeout does not kill its compiler.

## Diff report

- Entire artifact is new relative to branch base `63e253f`.
- Incumbent Python/CUDA text is inherited byte-for-byte except identifying
  header comments and explicit C++20 build flags; its scheduling, arithmetic,
  selector behavior and source patches are not changed.
- VMS device numerical code is byte-for-byte identical to VMS5. Only the
  control variant table, host `pick` dispatch and identifying comment change;
  this removes expensive unused template instantiations.
- The former VMS5 broad ablation/gate block is replaced with the diagnostic
  and conservative static policy described above.
- `scripts/build_moe_vmsgate.py` regenerates the inherited prefix and CUDA
  source from a read-only VMS5 file, retaining the handwritten candidate gate.
  `scripts/test_moe_vmsgate.py --reference ...` checks exact regeneration and
  all numerical-device-source bytes after removing only the stated host ranges.

## Local verification

```
PYTHONDONTWRITEBYTECODE=1 python3 scripts/test_moe_vmsgate.py \
  --reference /Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse-engine/kb/exp/moe_vms5.py
```

**16 tests passed**: AST, host/CUDA/Python ABI alignment including negative
control, host/device C++20 flags, control-only template dispatch, runtime
shape dispatch/fresh outputs/no host reads, 4,624 actual repack byte-address
checks with a dependency-free lazy tensor adapter, histogram edge cases,
all static gate vetoes and null-bound endpoints, no null cancellation,
train-only threshold fitting/held-out failure, mocked event/read-flush ABBA
ordering, captured-output replay initialization, numerical threshold endpoints,
deadline, source provenance/diff, and complete log-tail retention.

No torch, CUDA GPU or nvcc is available locally. These are CPU/static and
mocked-policy tests, **not** GPU numerical, launch, graph, build, timing or
prepare-budget validation. Remote check and independent reviewer remain for
the calling session. The shared ledger was searched using `scripts/tried.py`;
only a branch-local ledger row is written because the shared helper would push
to main, explicitly forbidden for this task.

## Expected effect / risks

- Expected opportunity: **0–2%**, not a measured speedup. Diagnostic-only
  fallback has no GPU dispatch overhead and likely keeps the incumbent.
- VMS's different reduction order and incumbent nondeterminism may fail strict
  bit equality on every draw. That is intentional: do not loosen the exact
  gate to force adoption. Timings still answer the routing-lightness question.
- Feature correlation may be weak or fail held-out draws. The printed fitted
  threshold assumes zero dispatch cost, and is not evidence of a real speedup.
- Many strict per-draw null checks can veto adoption because of residual noise.
- Remote Torch/CUDA C++20 build, dynamic shared-memory limits, PDL graph behavior,
  kernel bounded-wait errors, actual reference tolerances and prepare<300 s
  must be confirmed by the judge. No local GPU/compiler assurance is claimed.
