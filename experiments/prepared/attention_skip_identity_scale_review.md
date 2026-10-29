# Attention identity-rescale short circuit

One fixed hypothesis, October29. Implemented and CPU-audited; GPU correctness,
generated instructions, branch frequency and speed remain unverified.

## Exact sources and scope

Baseline9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f,
`best/attention_checked.py`, scored8b767ea7b2b74de6 at2.5370005142373167x.
Reuse that source-bound full16tests/12benchmarks/integrity result. Do not spend
a fresh baseline check or assign an older omitted-boundary result to it.

Candidateed810c7abd8a2c852959609cbe9977ae562c61b85f57f5dce93bafc3b8925dcd,
173419bytes. CUDAd9637b769cc6af0e032da8373e18f792598dd611f3800ff6a308a2678e7c55bf.
Frozen packet: results/attention-skipscale-preflight-20261029/. The builder
reverses every CUDA edit to the exact original; all other Python AST is exact
except the extension's distinct module name. Preserve licenses and fallbacks.

Only L1 flat kernels change arithmetic control flow. Under the retained plan,
these are multiple-request full-context decode cases, including scored B16/B32.
L1 per-request and all L8 math stay original. The change retains QK/PV operands,
FP8 probability scale256, every key and mask, cache writes, BF16 norm/RoPE/
attention-output boundaries, split ownership, finalizers and all scratch layouts.
It does not repeat the rejected FP8-QK, operand-swap or occupancy trials.

## Mechanism and numerical argument

The original L1 loop computes a fresh maximum and multiplies16 running output
values by `ex2(old_max-new_max)` on every nontrivial key tile. If the maximum
is unchanged, that scale is exactly1. A warp vote before the per-row finite
branch tests whether any lane's maximum changed. If any changed, the complete
old update runs for every active lane. Otherwise it skips the exponent and
identity output multiplication and adds the unchanged probability sum directly.

NVIDIA specifies ex2.approx.f32(+0) and ex2.approx.f32(-0) as exactly+1:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2
For finite FP32 state, multiplying by+1 preserves its bits, including signed
zero and subnormals. FMA(l,1,ls) and RN(l+ls) have the same exact real sum and
one FP32 rounding. The maximum assignment, probabilities and PV contributions
remain unchanged. A genuine maximum change always executes the original
arithmetic; there is no approximate equality threshold or lazy maximum policy.

The vote is warp-uniform and occurs before the finite-state branch. All named
consumer lanes participate; producer roles occupy separate complete warps.
An entirely masked warp retains the old inactive update. No CTA barrier,
arrival, replay counter, memory ordering or finalization ownership changes.

The saved full-context kernels take about86.93us at B16 and166.29us at B32;
source inspection finds repeated identity work in their loop. It does not
establish how much of those times is exposed rescaling. Added vote/branch and
predicate/register lifetimes may outweigh the skipped work, especially when
maxima change often. This is the concrete uncertainty for one check.

## CPU and resource evidence

CPU checks cover261153 finite FP32 identity products and sum-rounding cases,
all256 subsets of eight query quads, and503 online state steps spanning stable,
growing, random, masked-edge and signed-zero traces. State bits match. Mutant
counterexamples show why a positive change threshold or an all-lanes vote is
wrong. These traces are synthetic CPU state screens, not GPU branch-frequency
or timing measurements. The independent state argument does not approximate
or replace the retained NVIDIA exponent/Tensor Core operations.

All shared-memory members and offsets match the parent. Dynamic shared bytes:
L1 nonflat58368, L1 flat67584, L8 nonflat94208, L8 flat114688. L1 flat retains
256threads, MINB2, base128, producer56 and consumer200register budgets. Two
blocks need65536registers by this source model; CUDA allocation/occupancy must
be measured. No additional cache, scratch bank, weight representation or
input-dependent allocation is introduced. Static code grows about1KiB.

At native L1 initialization, query the actual new flat function after setting
its existing dynamic shared attribute. Require1..128registers, zero local bytes,
at least two resident blocks and the exact67584-byte dynamic allocation. Failed
resources stop the check before launching kernels; do not relax the gate later.
Diagnostic marker ATTN_SKIP_RESOURCES binds this to the actual function.
The distinct device symbol is attn_kernel_skipscale<1,true,...>.

Fresh official TASK/reference/checker/baseline/AGENTS matched all five pinned
files. CPU result-fixture tests reject13 malformed admissions, including missing
tests, forged aggregate weights, wrong backends, resources, controls and weak
gains. Fixtures are not measurements. No GPU job has been started by preflight.

## Fixed evaluation and stopping policy

One full official H200 check, no sweep or selector, no fresh baseline run.
The source always uses the new short circuit in its declared regime. Preserve
the current evaluator's complete16tests/12benchmarks/integrity and BF16 guard.
Actual profiles3/5 must show the new L1 flat symbol; nonflat L1 controls0/1/2/4
and L8 controls6..11 must retain their declared template regimes. Reject missing
or ambiguous resource diagnostics and any failed numerical/integrity case.

Use each case's baseline and candidate times to reconstruct six weighted loads,
46sliding+16full. Never geometrically average the twelve shape rows. Compare
affected case speedup ratios to the exact cached baseline: affected GM>1.04,
control-normalized GM>1.04, each affected/control>=1.02, unchanged control GM
within[0.95,1.05], and complete track>1.015times baseline, i.e.
2.575055521950876x. These engineering gates are not a paired-job confidence
interval or proof of statistical significance. A failed/inconclusive result
retains9e922c35 with no score, transfer or unmotivated retry.

Use scripts/measure_attention_skipscale.py and its fixed guard through the
foreground reversible controller, scoped to candidates/attention_active.py.
The driver waits locally until at least04:22:28.724212UTC, queries quota once,
reserves four checks, persists upload intent and job ID, and follows that same
handle. A missing upload response requires reconciliation, never a blind
re-upload. One active GPU job, no new launch in the last20minutes of the
authorized October29 20:00Toronto window. Scored publication follows only a
verified material gain; it is not automatic check spending.

## Overall value and Decode transfer

Only two of six Attention loads change. Conditional arithmetic using retained
case weights shows why a4% speedup on both full kernels alone produces only
about1% track gain; the complete-track gate requires more. No rank or speedup
is promised. This is a small, exact-work elimination candidate with a bounded
single-source experiment, not evidence it reaches top50overall by itself.

The same three arithmetic edit anchors exist in retained DecodeD9. A validated
winner can be applied to those anchors while keeping D9's original FP16 PV
implementation; do not transfer the tied standalone FP8-PV backend. The whole
layer must retain routing/cache/residual checks and correct load weights.
At a hypothetical4% kernel gain, roughly86/165us full-attention spans would
save about3.3/6.3us on their full layers, or0.9/1.6us averaged over62layers.
These are conditional arithmetic, not a measured critical-path fraction,
prediction, or overall-board aggregation. An actual layer gain still needs
its own full source-bound evidence before publication.
