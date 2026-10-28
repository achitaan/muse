# One bounded MoE skinny16 comparison

Source `f20caadbdafb5761c782543ad7b30351f948571ab9fc8fd5e5e8bccbb97e3f34`,
329542 bytes. GPU-unmeasured. Parent33114871 checked2.365437416x as
d600231c8a8244cb; scored2.348299935x as b6c70c9b83724d2e. Current bests
remain unchanged. Main chat only; reuse the existing native top50 Goal.

Read `moe_skinny16_review.md` for the fixed alternate's mathematics, layout,
numerical contract and CPU evidence. The full checked parent and the complete
921dc83e prototype are exact prefixes. Default CUDA and every incumbent
algorithm remain exact. The new kernel reuses each hot expert's weight/decode
stream for two eight-token tiles, retains four FP32 accumulators per tile,
and restores a pruned second float4 store. This replaces two skinny jobs
only when an expert has9..16 routed tokens. No new weight format/copy or
precision reduction is introduced.

## Additional upstream cross-check

Public file-head metadata still points to f35a41ee for Dense, MoE, Decode and
Attention. There is no newer file revision to transplant at this observation.
The commit before dead-code pruning is
5f9f7c7b0fe83ca53f5f434e3bef2227549eaf5e. Its public MoE source is214505 bytes,
SHA2568bc17e47731b5dd4de9fefd3fd0db97e63e6104cacc53f01c8187c3553d03e5d:
https://github.com/vjeux/okc-2026/blob/5f9f7c7b0fe83ca53f5f434e3bef2227549eaf5e/muse_moe_w4a16.py

The pruning audit resolves the explicitly listed conditions inside static
skinny_ring for SM90/NW8/NACC4/EPI4off/SKT16, with both PRMT and LUT scale
decoding. After accounting for the incumbent's known BF16 Marlin/norm fixes,
the missing second store is the only remaining difference from the unpruned
static function. This checks for additional lost operations in that path;
it is not a CUDA compilation or a general preprocessor verification.
The original unpruned source, provenance and audit are archived with this packet.

## Prepare comparison and fresh-call behavior

Base preparation runs unchanged, then one alternate compiles if the device is
H200-class SM90 and elapsed preparation<=210s. No geometry alternatives are
tried. The old and new modules reuse the same sequential scratch contract.
The new Smem ABI estimate214656bytes fits the opt-in limit, but actual CUDA
resources and compilation remain unverified. Original preparation took163.434s.

GPU probes compare with the checked incumbent on self-generated M9/15/16,
each with diffuse, two-hot-expert and mixed-popularity routing. IDs are unique
within a token and route weights are positive/normalized. Mixed inputs include
a near-zero row. Require finite outputs, relative L2<=0.005 and worst row<=0.02,
using the0.1-average row denominator floor. All-remote probes at all three
sizes must yield exact zeros. Those twelve cases also alternate the scratch
between modules and sizes; they are not cached as actual-call outputs.

Only after numerical probes pass, time M16 on the same three routing patterns.
Each gets eight paired graph replays, alternating arm order. A128MiB L2 flush
runs outside every measured event interval. Retain every old/new raw time and
recompute the median paired ratio. Enable the whole M9..16 bucket only if the
three medians' geometric mean is below0.97 and none exceeds1.02. Preparation
budget gates at265/270s reject incomplete comparison, and synchronize at return.
Final compact diagnostics contain every probe and paired sample.

State adds only a boolean choice. The timed wrapper computes the complete
fresh call via the alternate for enabled M9..16, retaining original paths
for M<=8, M>=17 or rejected selection. Its dtype/layout handling matches the
incumbent. No input-value branching, host synchronization, cached activation
or output is added to the timed call.

## Fixed measurement and keep policy

Use `scripts/measure_moe_skinny16.py`, metric speedup, guard
`scripts/verify_moe_skinny16_evidence.py`, scope candidates/moe_active.py.
The exact cached primary parent result supplies baseline5/6/integrity with
no API call or new GPU run. The driver checks positive official MoE check
quota before launching one full check, freezes source bytes, persists the
handle and resumes after observation failure. Unknown upload outcomes require
official-job reconciliation, never blind retry. Daily reset is00UTC.

Require the named5tests/6benchmarks, skewed test, H200, preparation<=300s,
integrity, selection exactly(16,), all twelve probes and all24 paired samples.
The new symbol must appear in the actual16-token profile and no other profile.
Actual16-token speedup must improve>3%, also>3% after dividing by the five
unchanged controls' geometric change. Whole-MoE speedup must improve>1%
over the cached2.365437416x check. These practical thresholds are fixed before
launch and do not establish statistical significance from cross-job controls.
Only a passing material result can justify one scored confirmation. Never
score an empty selector or rerun for favorable noise. No nearby tuple sweep.

CPU selection audit:48routing cases,16 guaranteed hot second-tile cases,
770actual wrapper-dispatch cases and both CUDA precision contracts. Native
audit:524288 operand words,16ownership grids,4096K coverage, second-store
mutant detection, shared ABI and FP64 numerical screens. Evidence-tool audit
rejects17 incomplete, inconsistent or numerically invalid payloads.
The existing GPU suite remains authoritative; no GPU result is claimed here.

After a retained MoE result, evaluate transfer to Decode from affected cost
and full13/12 evidence, then Attention and Prefill in the user's order. If
MoE loses, save the negative result and keep33114871. Dense4d443b13 is another
fully prepared fixed trial for reset; do not spend unrelated remaining runs
just to exhaust a quota. The broader top50 Goal remains unmet and active.
