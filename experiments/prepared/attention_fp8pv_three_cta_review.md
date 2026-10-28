# Next inspection: three resident CTAs for the L1 flat loop

Unimplemented, unmeasured. First finish the exact-source scored confirmation
3e2a317e289a4cc8 of checked FP8PV a310b988. This packet does not justify
claiming a new gain or spending a run before the resource/dispatch audit.
Main chat only. Retain all BF16 boundaries and the same FP32/FP8 math.

Preflight update: source65527644 is prepared and CPU audited, not GPU
measured. Parent publication finished2.548282764x with full16/12/integrity.
The following inspection prediction is superseded where noted below.

The audit found an inherited64-segment truncation for skewed large batches.
L1 flat seg storage and its B>32 loop now cover MAXB512; segment count is at
most B, so capacity suffices for all supported B. All other Smem fields,
L8/nonflat sizes and arithmetic are unchanged. Actual modeled new L1 flat
sizeof66,560, dynamic67,584, three-CTA shared202,752bytes. This remains
below233,472. Register budget remains61,440. An additional public-shape
_fits bound protects total prefix+virtual-padding space from int32 overflow,
falling back when B*(max_pages*P+L+600)>=2^31. No host tensor values read.
This correctness fix is necessary even if the occupancy hypothesis loses;
preserve it separately rather than silently restoring a known truncation.

CPU audit:660 partitions,144,675 segments,434,025 paged-end checks and256
actual Python shape plans; old truncation reproduced30times, maximum366
segments in a CTA. New intervals exactly cover every request with no gaps or
duplicates, every new cache key is owned once, and counters total n_keys.
The forty actual Smem declarations are parsed for ABI offsets/alignments.
Parent sizeof61,440 is reproduced; unchanged layouts are exactly equal.
Warpgroup budgets use legal multiples of8; each producer releases3,072
registers, exactly its consumer's requested increment, within its CTA pool.
NVIDIA semantics: https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg

Predeclared guard requires full16/12/integrity, actual occupancy>=3, exact
resource layout, actual new full B16/B32 profiles, affected/control ratio>1.03,
neither changed ratio below0.99, and aggregate ratio>1.015 vs cacheda310.
Local-memory size is reported rather than assumed. These are empirical
admission thresholds, not proof of significance. One full H200 check next.

Closed at22:00UTC: checke6202683302640b4 passed16/12/integrity at
2.540543288x [2.502435139,2.586864626]. Actual resources matched the model:
numRegs80, local bytes0, occupancy3, dynamic67,584, sizeof66,560, consumer104,
132SMs,65,536regs/SM,233,472sharedbytes/SM. No spills, yet affected full
B16/B32 GM0.982897573 vs unchanged controls0.983429618; normalized0.999459.
The controller discarded the trial before the performance guard (metric
declined); do not claim that guard ran. Actual profiles87.5/166.1us.
Do not retry nearby register budgets or extra-CTA counts without new evidence.
Correctness repair is isolated as c05090e2, preserving the original2CTA
register/grid configuration. It needs an exact-source check before retention.

FP8Q was a closed negative result. FP8PV a310 passed16/12/integrity and its
fixed guard at2.580662730x. Changed six-case GM1.032137 vs unchanged-control
GM1.030139 means the cross-run points do not isolate a causal gain. Do not
hide this uncertainty or resubmit the same source for timing fluctuations.

Full-layer Decode transfer was screened using corrected1fb5's measured
Attention profile fractions. Even granting the largest observed L1 gain,
1.06301045, uniformly to every affected full/sliding case predicts only
about0.4984% whole-Decode gain. L8/M8/128/256 are unchanged. This is an
optimistic screening estimate, not a measured transfer; preserve the two
scarce Decode checks. Evidence: results/resumed-attention-fp8pv-20261028/
decode-transfer-screen.json. Standalone full uses coherent inputs whereas
Decode full uses soft inputs, adding uncertainty to extrapolation.

Potential larger Attention gain: FP8PV removes V expansion and halves PV
tensor instructions. The flat L1 loop may now fit a smaller consumer register
budget, allowing three CTAs/SM to hide page-load latency. Current Cfg hardcodes
MINB2 and BASEREGS=65536/(2*NTX). For L1,F=true NT128+NP128=256; producer
56registers, consumer200, average128, two CTAs. L1 full grids use2*SM splits.
Measured full B16/B32 remains about87/167us vs bandwidth floors63/126us.

Single coherent proposed change, only L1,F=true:

- MINB3 and use MINB in the BASEREGS formula; all other instantiations retain2.
- Average register budget80 (85 rounded down to8); producer56 unchanged;
  consumer104. Three blocks use3*(128*104+128*56)=61,440registers vs65,536.
- Source-derived Smem<1,true> layout model: sizeof61,440, launch dynamic
  sizeof+1,024=62,464; three blocks use187,392 vs H200233,472bytes/SM.
  These are ABI layout predictions, to confirm with actual CUDA attributes.
- Host full multi-request L1 quota3*SM; retain existing2*SM for L8 and all
  nonflat kernels. A larger grid without three-block resources is insufficient.
- Report real numRegs/localSizeBytes/active-CTA count/dynamic shared bytes
  during the existing host launch initialization. Require actual three-CTA
  occupancy. No tensors read for host diagnostics, no actual-call caching.

Risk is register pressure:104consumer registers may spill or serialize code.
Do not infer sufficiency merely from arithmetic. No geometry/knob sweep. If
this single candidate loses, do not retry nearby register budgets blindly.

Before one check: derive the complete Smem layout, prove all original math
and nonflat/L8 configuration constants unchanged, audit weighted flat
partition bounds and queue counters for the larger grid on nearby B/page
sizes, and verify no new grid barrier is introduced. L1 flat defaults coop0;
existing last-CTA protocol permits queued blocks. Keep launch-time residency
checks for any cooperative override. Maximum splits remain below1024 onH200.

Fix a stronger performance guard before testing: compare affected full B16/
B32 to their cached a310 parent and normalize that ratio by unchanged
controls. Require material gain beyond those controls (for example3%), full
16/12/integrity, actual new-path profiles/resources and aggregate improvement.
The prior guard passed a3% affected rise alongside a3% control rise, so its
threshold alone cannot attribute improvement. No new baseline GPU evaluation.
