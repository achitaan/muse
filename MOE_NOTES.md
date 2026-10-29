# October29: paired cluster activation multicast rejected

Job6627d54067774130 evaluated exact sourceb7043df7/nativeCUDA2701a25e.
The cooperative2x1x1 launch, full66-cluster residency query and graph captures
work on H200. Both BIG variants report96registers, zero actual local/static
bytes,231168dynamic bytes and576threads. Preparation took218.312seconds.
All19 synthetic comparisons, four changing-input/route graph replays, four
all-nonlocal zero cases and one duplicate-route probe passed the tighter
incumbent-comparison bounds. Changed FP32 partial ordering is not bitwise exact.

Normal128/256 median new/old times are1.014418810/1.006965887:1.44%/0.70%
slower. Chunked385 is2.29% slower and also fails the nearby guard. Therefore
selected_main=false. Official full5tests/6benchmarks/integrity at2.348327668x
times original paths; it is not full new-kernel correctness/performance evidence.
The saved packet is results/moe-cluster-x-check-20261029/, with exact source,
paired samples, result/profiles, review and controller logs.

Controller revertedb7601d1 and archived20261029-090013. Retain33114871; no
score, Decode transfer, parameter sweep or noise retry. Logical activation
request halving did not earn a whole-kernel gain in this fixed design. This
does not isolate HBM bandwidth, remote-barrier cost or scheduling as the cause.
Its resource/graph admission is useful evidence for a future distinct design,
but is not an invitation to sweep cluster/grid flags. DerivedMoE13checks and
10scores remain; next new check globally no earlier than09:53:13.031328UTC.

# October 29: cluster activation multicast feasibility

The current local investigation is a fixed two-CTA cluster design pairing output
units at equal K stages, preserving stream-K. The source/model halves logical
activation-copy requests, with unchanged destination bytes and median maximum
estimated CTA work. Some skewed cost maxima rise4.33%. Hopper primitives compile
locally; no kernel has executed and no full runtime candidate exists yet.
Review: `experiments/prepared/moe_cluster_x_review.md`; CPU evidence:
`results/moe-cluster-x-feasibility-20261029/`. Ring lifetime, full-source resources,
cooperative cluster progress and graphs remain open. All verified code is exact.

The separate disjoint-cut model passed53960 ownership/cost cases but removed
only0..2 split units per phase; it is parked without a GPU trial.

# October 29: paired epilogue store trial closed

Repaired check `5262c2c560c94197` / source `9fc2993a` compiled, used 96
registers with zero runtime local/static shared bytes, and passed all fixed
synthetic correctness/replay probes. It failed the timing gate: M32/M128
were 0.30%/0.39% slower, M256 0.77% faster. No new backend was selected in the
official suite. The controller reverted it; retain `33114871`. No score,
Decode transfer or nearby-shape selector follows. Full source-bound evidence:
`results/moe-epair-store-repaired-check-20261029/`. Later entries below are history.

> LATEST RESULT, October29: e8e3086d/check98e3538e85244c1d failed resource-query
> compilation (okc_epair_y vs actual okc_epair). No device correctness or timing
> evidence. Original33114871 restored, no score or Decode transfer. One check.
> Nine-reference repair9fc2993a preserves CUDA9d6c4354 and every scientific gate;
> four new query-wiring mutants and twelve full-guard mutants are rejected.
> Packet: results/moe-epair-store-repair-20261029/. No active job/controller;
> earliest next check06:27:37.228843UTC. Expected MoE15checks left.

> PREPARED CANDIDATE, October29 at05:08UTC: e8e3086d changes only WGS
> phase-A store representation, with original numerical expressions and
> scheduler. CPU audit covers8,519,680 writes/all counts1..64/all4096columns,
> 65,536 bit pairs,100 dispatch cases and12 full-guard mutants. No GPU
> result. Packet: results/moe-epair-store-preflight-20261029/. One paced
> check eligible after05:22:50.375730UTC; retain33114871 until full gates pass.
> Lane command was rejected by automatic review under main-chat-only rules;
> no retry/delegation/model switch occurred. Work continues directly here.

> LATEST OFFLINE SCREEN, October 29 at04:50UTC: adjacent-row register
> redistribution for WGS gather stores passed16,384 layouts and1,048,576
> exact uint4 ownership checks. Synthetic large-token fixtures reduce modeled
> store-sector requests; no hardware counters, speedup or GPU result implied.
> Preserve33114871. No runtime candidate, new check or score. Packet:
> results/moe-gather-pair-store-screen-20261029/. Next check no earlier than
> 05:22:50.375730UTC; earlier quota/waiter/candidate headers are historical.

> LATEST RESULT, October29: MoE factor trial92577a6426f74605/85a317b7
> compiled at96registers/0local bytes and passed its new-path synthetic probes,
> but paired M32/128/256 were0.36..0.84%slower. selected_main=false.
> Official full5/6/integrity at2.325355016x timed incumbents, not the new kernel.
> Controller discarded/reverted33114871; no score or Decode transfer. Evidence:
> results/moe-factor-y-20261029/. No GPU job/controller is active now.
> Derived remaining checks: Dense19,MoE16,Decode20,Attention20,Prefill20;
> ten scores per track. No new check before04:22:28.724212UTC.
> Continue distinct source/profile-led work, prioritizing whole-layer value.
> All earlier candidate/running/waiter headers below are historical. Goal active.

> CURRENT CANDIDATE, October29 03:07UTC: MoE BF16-factor scratch/cache
> source85a317b7 is implemented and CPU-audited; no GPU result or speedup yet.
> Frozen packet: results/moe-factor-y-preflight-20261029/. Review and fixed
> numerical/resource/performance gates: experiments/prepared/moe_factor_y_review.md.
> One paced check is eligible no earlier than03:22:01.371647UTC. No GPU job
> is active at this checkpoint. The previous diagnostic missed resource
> admission; its timestamps remain inadmissible as incumbent phase fractions.
> Preserve33114871 and all other incumbents; score only a verified material gain.
> Earlier running/waiter headers below are historical. Main chat only.

> Latest checkpoint: October28,22:53UTC. No pending job/controller. Goal active; top50unmet.
> Two checks this continuation, no scored runs. Prefill28d04382 passed5/8/integrity
> at1.680162079x but fixed normalized-profile guard failed1.004422(<1.02); reverted.
> MoEf5737777 passed5/6/integrity at2.348851666x; native65280gate audit passed,
> but aggregate declined vs2.365437416; controller discarded before guard. No Decode transfer.
> Retain MoE33114871(scored2.348299935x),Prefill93e57ab4(scored1.663469245x),
> Attention9e922c35(scored2.537000514x),DecodeD9checked1.990224880x/unscoredrepair.
> API limits: Dense0/0,MoE0/7,Decode1/4,Attention9/5,Prefill18/9(checks/scores).
> Total28checks/25scores; reset00UTC. Latest overall snapshot remains150/1447at22:34UTC.
> Next source inspection: dense_pinned_source_review.md; Dense cannot evaluate before reset.
> Conserve other tracks: no new candidate justified yet. Main chat only.

> Current checkpoint: October28,21:00UTC. MoE33114871 remains2.365437416x
> checked/2.348299935x published. Phase13's two-group design passed full5/6,
> actual two-CTA occupancy and zero spills, but slowed every changed load.
> Discarded at2.045556897x with no score; one MoE check remains. No pending
> job/controller. Next ordered focus: Decode. The top50Goal remains active.

# Scale-six rejected - October 28, phase6

Check daa5323ad4644c26 passed all 5 tests, 6 benchmark cases and integrity with
the compressed path enabled. Speedup fell to 2.198081320x, interval
[2.165110100,2.231052539], versus retained 2.365437416x. All three WGS loads
regressed; unchanged small paths stayed close. The controller discarded it.
One check used, no score. Both best files and candidate retain exact 33114871.
Full evidence and audits: results/resumed-moe-phase6-20261028/.

The lossless representation is correct, but its extra decode cost outweighed
the 2.7778% lower weight traffic. No encoding sweep or partial fallback-only
retention. Next read experiments/prepared/moe_wgs32_review.md for one bounded
17..32-token WGS specialization using original byte scales. GPU implementation
is not written or measured. All controllers archived; no job pending.

---

# Next MoE work - October 28, 17:58 UTC

MoE remains 53/994 at 2.348299935x; exact 33114871 is checked and published.
The decoder publication improved to 1.993956388x, overall 146/1436. No job is
pending. Next read experiments/prepared/moe_scale6_review.md and its CPU audit.
The lossless scale-six representation prototype passed 1,310,720 value round
trips, exhaustive adjacent field pairs and all 256 eligibility bytes. It could
reduce WGS weight bytes by 2.7778%, at extra decode cost. GPU code is not yet
implemented or measured. Use original fallback for wider scales, preserve all
precision boundaries and audit full layouts before one focused MoE check.

---

# Current MoE publication - October 28, 2026

Scored job **b6c70c9b83724d2e** passed all 5 tests, 6 benchmarks and integrity at
**2.348299935x**, CI [2.313075436,2.383524434]. Source is 33114871 in both
best/moe_verified.py and best/moe_published.py. It ranked **53/991** at 16:48:43 UTC;
50th place was 2.3528x, about 0.2% faster. Overall was 150/1432. Top 50 overall
remains unmet. No job or experiment controller is pending.

Publication used one scored run after two independent full checks of identical
timed native CUDA: d600231c8a8244cb at 2.365437416x and ca76bcd1b6ce43b4 at
2.350439250x. The second was the rejected backend selector with no choices.
Folding that empty metadata branch recovers the first kernel_fn AST exactly;
CUDA and packing are identical. The individual intervals overlap the old score;
this was a bounded scored confirmation of a repeated improvement, not a claim
of formal statistical significance. Replays were not counted as independent jobs.
The scored point is about 1.5% above the previous 2.312613084x publication.

Proof and full score records: results/resumed-moe-phase4-20261028/ and
results/resumed-moe-phase5-20261028/. The older withheld-publication statements
below are superseded. The published decoder still embeds the older dd6049cb
MoE; its newer unpublished 1.994319167x check includes this 33114871 implementation.

---

# Backend selection rejected - October 28, 2026, phase5

One full check ca76bcd1b6ce43b4 passed all 5 tests and 6 benchmarks at
2.350439250x, CI [2.315182661,2.385695838]. The two-algorithm comparison selected
neither alternative: the mixed skinny/fat fallback was about 6.1% slower on
average for the 16-token synthetic patterns and 7.1% slower for 32 tokens.
All six output comparisons were finite, relative L2 below 0.000814. Existing
incumbents were retained, so there was no change to timed native compute.
The controller discarded the trial; no score or follow-up sweep was spent.
Prepared-state selection returned metadata only; source is retained as negative
evidence in results/resumed-moe-phase5-20261028/. Prepare took 208.4 seconds.
The earlier 2.365437416x check and 2.312613084x publication remain unchanged.

Next decoder trial combines that already-checked exact MoE implementation with
an audited small-token FFN padding/boundary fusion. It also removes a dense
prepare-only sweep whose alternatives were unselected in the observed runs,
to protect the 300-second prepare budget. GPU behavior is not yet verified.

---

# Latest MoE check - October 28, 2026, phase4

Checked source best/moe_verified.py: SHA256
`331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.
Check d600231c8a8244cb passed all 5 tests, 6 benchmarks and integrity at
**2.365437416x**, CI [2.329955855,2.400918977]. This is not the published score.
The interval overlaps the prior source's check upper bound 2.334294331,
so no scored run was spent. best/moe_published.py retains dd6049cb and the
2.312613084x publication b4a10b76347e4cb0. The published decoder still embeds
that older MoE implementation; no uncertain standalone gain was transferred.

Two measured changes: paired BF16 epilogues (375685be8be14e1b) passed at
2.318616664x, CI [2.283837414,2.353395914]. Pairing retains the BF16 accumulator,
BF16 global-scale and RN-even BF16 product boundaries, followed by FP32 norm
arithmetic. All 1,566,720 products in the independent CPU audit matched analytical
FP64 RN-even rounding, including signed zeros, subnormals and overflow.

The second change uses Lee Penkman's exact eight-operation FP4 placement and a
matching prepare-time bijection of all 32 packed bits. It changes representation
only: same NVFP4 bit count, no requantization, same BF16 fragments, scales,
routing, reductions and norm math. CPU audits checked 1,048,576 codes and 65,536
scale bytes across MMA, natural-K, fat, unit and WGS layouts. All 256 possible
inputs to the masked multiply were enumerated, proving no carry interference
in its selected output bits. Source inversions recover all original native
CUDA except the explicitly changed helpers. GPU speed is established by the
full check, not by source-level operation counting.

Provenance: native Vjeux source remains pinned f35a41ee; exact placement derives
from lee101/openkernels c565e373d34575c5f7df862b05f8be7619078b0f, Apache-2.0,
copyright 2026 Lee Penkman. The pinned root had no NOTICE file. Full Apache
license and additional attribution are retained in the source and notices.

Evidence: results/resumed-moe-phase4-20261028/. Builders and CPU audits are in
experiments/prepared/ and read immutable baseline sources. The controller is
archived, no job is pending, and the overall Goal remains active. This phase
used two checks, no repeated baseline and no scored submission. Further work
needs a material hypothesis; do not spend a run merely to chase interval noise.

---

# Next MoE hypothesis - paired epilogue, GPU unmeasured

Prepared source 6862c3a54429fb4149e7fd78f63b24db71d70648a1947172826ea7f134082409
pairs eleven independent GEMM-output sites through BF16x2 RN-even operations.
BF16 accumulator/global-scale/product boundaries and FP32 norm intermediates
remain. CPU audit matched 1,566,720 products to independent FP64 rounding,
including signed zeros, subnormals and overflow. Reversing the transformation
recovers all original native CUDA bytes. This is not a measured speedup.
One full MoE check will test it against cached baseline dd6049cb at 2.299797371x;
no geometry sweep or automatic scored publication. See
experiments/prepared/moe_paired_epilogue_review.md and saved CPU audits.

---

# Latest native MoE checkpoint - October 28, 2026

Check `6792a5c832ee4aff`: **2.299797371x**, all 5 tests, 6 benchmarks and
integrity checks passed. Exact source `best/moe_verified.py`, SHA256
`dd6049cbf7a2a41666f2d9e46010f81e0dd22c0840881bc5f187ac5f3ba1d3a3`.
Scored publication `b4a10b76347e4cb0` passed at **2.312613084x**. Do not duplicate.
This supersedes the previous 1.268945620x checked Marlin candidate below.

Derived from Vjeux's Apache-2.0 source at revision
f35a41ee1c13e316dabbf772830d47b23b453393. One persistent native grid performs
routing, both NVFP4 matrix multiplies, SwiGLU and final norm/weighted sum.
All local routes remain present; routing is rebuilt from fresh inputs each
call. Weight layouts are bit/byte repacks, without requantization or persistent
BF16 expansion. CPU inversion recovered 1,048,576 weight codes and 65,536 scale
bytes across MMA, natural-K, fat, unit and WGS layouts, with seed 20261028.

Local changes remove diagnostics and restore pinned SGLang Marlin's BF16 GEMM
outputs: BF16 accumulator boundary, BF16 global scale and BF16 product before
SwiGLU or post norm. Effective (1+w) norm weights are prepared in BF16; scaled
norm values and reductions stay FP32 until final BF16 output. This preserves
Dot's precision correction while using the native scheduler. The structural
precision guard now recognizes both this source and the prior Triton finalizer.
CPU/static checks alone do not establish GPU correctness or speed.

Scratch is prepared once, fully overwritten/reset by the kernel, and shared
between sequential calls. Concurrent calls sharing this state are unsupported.
Full source/license and exact check evidence are in
`results/resumed-moe-phase3-20261028/`. The native source includes Apache license
and local change notices; preserve them. Main-chat-only work, no sweeps.

The next single hypothesis integrates the separately checked native attention
and MoE into the whole decoder. The FP32 router path, expert ID rotation and
residual boundaries are preserved. The draft is unmeasured. Prefill stays last.

---

# MoE optimization notes — October 28, 2026

Current order: MoE W4A16, Decode Layer, Attention Decode, Prefill Layer last.
Main chat only. Conserve checks and scored runs; every trial needs a concrete hypothesis.

## Completed phase 1

- Baseline 03449dfa701d4a95: 0.999760595x, all 5 tests and 6 benchmark cases passed.
- Small-M fused finalizer 74ba53564cfd42e4: 1.149649851x, full official validation passed.
- Smaller EP8 M tiles bd37e0d81b0542fa: 1.271471562x, CI [1.252399489,1.290543636], full validation passed.
- Direct single-token CUDA NVFP4 GEMV 67f2f03d27b44ede: correct but slower, 1.070874526x aggregate; single-token path about 148us versus 54us retained. Reverted. No follow-up parameter sweep.
- Evidence, frozen source hashes and controller history: results/resumed-moe-phase1-20261028/.
- Four MoE checks used, no MoE scored submissions.

## Precision review and new baseline

Dot supplied a review (no performance measurements) identifying the same small-M
fallback and highlighting the difference between the public starter and pinned
SGLang. Independently verified pinned helper:
https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py

SGLang's effective norm weight is BF16(1+w), but y * effective_weight stays FP32
through RMSNorm and route-weight reduction, with only final output cast to BF16.
The public starter adds an intermediate BF16 rounding. Phase 1 followed that
starter and passed all official checks; the new baseline removes the extra
rounding to match pinned SGLang's FP32 computation more closely. Do not roll back
to the older rounded variant just for a small timing point change. Establish a
fresh measured baseline for the stricter precision contract first.

Mask remote expert routes before loads/arithmetic, output zero for all-remote
rows, and overwrite outputs on every graph replay. Preserve FP32 accumulation,
original NVFP4 codes, FP8 block scales and FP32 global scales. In-kernel
dequantization is permitted; persistent BF16 predecoded weights are not.

## Prepared hypotheses, unmeasured

The original-layout single-token CUDA attempt is retained as negative evidence.
A grouped Triton W4A16 GEMM addon is being prepared under .tools. Its first CPU
check caught multiplication-order differences from the task dequantizer; fix
value * (scale * global_scale) before any H200 evaluation. It is not a speedup.
Marlin launch/scheduling remains another hypothesis; no library/harness patches.

## Other tracks

See TRACK_NOTES.md for earlier attention/decoder work. Attention scored job
f6b899f674914ef2 completed at 1.116457695x and passed. Decoder baseline
65ab5fffad0c4995 completed at 1.002013444x. The next focus after MoE is decoder
layer, then attention and prefill. No Dot advice authorizes external spending.

## Phase 2 checkpoint and move to Decode Layer

FP32-intermediate baseline 0535f51d98d548ee passed all 5 tests, all 6 benchmarks
and integrity checks at **1.268945620x**, CI [1.249911436,1.287979804].
Exact source: best/moe_verified.py, SHA256 eab0e08d75f29658ec33a8f370d947ca023a9c10e3d09666684e6df1ef05f92d.

Grouped Triton check c8a6c6e4c46142e1 failed compilation because a masked FP8
load used an integer zero. One diagnosed retry 8f5c231df84b4147 compiled and
passed all correctness checks but was much slower: 0.448401955x aggregate, with
M128/M256 around 7.7/10.6ms. Reverted. No further geometry sweep.
Evidence and immutable controller history: results/resumed-moe-phase2-20261028/.

Seven MoE checks were used. One scored publication of the stricter FP32
candidate is now pending: 650aa66fd3334d8c. Observe that job rather than
resubmitting it. Remaining quota at launch: 13 checks and 9 scored runs.

Next focus is whole Decode Layer, carrying over the small-M FP32 finalizer and
M-tile choices. Whole-layer baseline 65ab5fffad0c4995 is already measured and
cached at 1.002013444x. Reuse it without another GPU baseline run.


## Phase 7: four-tile WGS rejected (18:41 UTC)

Full check 47f44d0ff8704a44, exact source d0ae4a9a775b28d614fee1768d37fad609024c44ad2189058565263b03c9f9b4,
passed all 5 tests, 6 benchmarks and integrity at 2.353666801x, interval
[2.318361799,2.388971803]. Specialization was enabled. Baseline remains
33114871, checked at 2.365437416x and published at 2.348299935x. No score used.

The 32-token speedup was 1.638191005x versus baseline 1.642387556x.
Actual resource queries: original 96 registers/thread, zero local bytes,
231168 dynamic shared bytes; specialization 95 registers, 8 local bytes,
183552 dynamic shared bytes (both BIG instantiations agree). Shared memory
fell, but register capacity barely moved and a spill appeared. Separate-job
timings do not isolate the spill's cost. Do not retry tile sizes as a sweep.

The discarded wrapper also lacks a final prepare synchronization after its
new scratch allocation. It was never published. Any future reuse must fix
that readiness issue and check the resulting exact source. There is no reason
to spend another check repairing this losing experiment alone.

Complete frozen source, raw results/profiles, CPU/source audits and immutable
controller history: results/resumed-moe-phase7-20261028/. Controller reverted
the candidate to the verified baseline and was archived at 18:43 UTC.
One MoE check consumed after the 18:24 quota snapshot (expected 7 checks/7
scores remaining, not a fresh API quota observation). Goal remains active.


## Phase 8: scheduler estimate rejected (18:52 UTC)

Full check35f15c0fcf9247d6, source df1871fad29b04f70aed36c7e3e1a5085066346c966aee5d07fe96e1a928a843,
passed5tests/6benchmarks/integrity at2.341433747x, CI[2.306312241,2.376555253].
WGS32/128/256 speedups1.624659818/2.557936989/2.856700906 did not improve
on the primary baseline1.642387556/2.590667418/2.874509152. Separate-job
intervals overlap; no isolated code-layout or instruction-cost claim. Controller
reverted to33114871 and was archived. One check, no score. Full source, raw
profiles/results, exact-owner CPU audit and history: results/resumed-moe-phase8-20261028/.
Do not transplant this quotient change to Decode or Attention.

Next reviewed hypothesis: const __grid_constant__ Params in the native MoE
kernel declaration. All device helpers already take const references and the
parameter object has no member mutations; pointed-to scratch/output stays
mutable. This may avoid per-thread parameter copies, but no generated-code
or timing gain is established. See experiments/prepared/moe_gridparams_review.md.
No pending job at this phase boundary. Expected MoE quota6checks/7scores after
two checks since18:24; not a fresh quota API observation. Goal remains active.


## Phase 9: grid-parameter annotation rejected (18:57 UTC)

Full checke5ed457feaff4119, source1613f98e749538dde707677bd268fb6d94fe39b45341147d2eaadae243067c71,
passed5tests/6benchmarks/integrity at2.354389959x, CI[2.319074110,2.389705809].
No aggregate gain over retained33114871; reverted without a score. The source
audit recovers the exact original kernel body and module AST. This experiment
does not establish whether parameter copies were present or eliminated in SASS.
Raw sources/profiles/results, source audit and controller history are saved in
results/resumed-moe-phase9-20261028/. No trial from phases7..9 is published or
transferred to Decode. Baseline remains2.365437416x checked/2.348299935x scored.

Next investigation is design-only: experiments/prepared/moe_register_split_review.md.
A complete producer warpgroup may allow explicit register redistribution, but
compile-time resource admission, all thread roles and barrier counts need proof
before implementation/evaluation. Keep GS2 and every original numerical path.
Do not apply setmaxnreg to a partial producer warpgroup or mix in losing trials.

All controllers are archived. No official job is pending. Three checks and
zero scores used since18:24. Expected MoE quota5checks/7scores remaining, an
accounting estimate rather than a fresh API observation. Overall rank remains
the last observed146/1438 at18:24; no new placement claim. Top50Goal is active.
Public Vjeux and LeePenkman repository HEADs still match their already-pinned
f35a41ee andc565e373 revisions; no newer public patch was found in those repos.


## Phase10: full producer warpgroup, no retained speedup (19:08 UTC)

Checkfc4ff22b0b184caf/source17adb3e6 passed5tests/6benchmarks/integrity at
2.340478636x, CI[2.305371456,2.375585815]. Resource preflight was
(640,96,96,0,0,231168,1,1,65536,132): both initial allocations96, no static
local bytes, one resident block/SM. Split enabledTrue. No changed WGS load
improved over primary33114871. Reverted, archived, no score. All math helpers
and GS2 were unchanged. Static attributes do not directly measure dynamic
ownership or establish a compiler-stall cause. Full evidence and audits:
results/resumed-moe-phase10-20261028/. No official job pending at this boundary.

An inactive Decode transfer draft21bf949f is991255bytes with all non-MoE AST
unchanged, but it was never evaluated or promoted because standalone timing
did not improve. Retain09badc9b in Decode and33114871 in MoE.

One source-backed follow-up: four-step WGMMA groups on this admitted register
split. Original source specifically says GS4 serialized16..32-token variants
at96registers; GS4 needs16more live A-fragment registers, matching the reviewed
96to112compute budget change. Test that paired mechanism once, retaining all
geometry, original layouts and GSFCI3; capture ordinary compiler diagnostics.
Do not retry tile/encoding/grid parameters or score a flat result. Expected
MoE quota4checks/7scores after four checks since18:24 (not a fresh API query).


## Phase11: GS4 serialized, rejected (19:16 UTC)

Checkb47f9fd54cbb4d25/source844c0165 passed5tests/6benchmarks/integrity at
2.282485686x, CI[2.248248401,2.316722971]. Compiler diagnostics explicitly
report C7512 for both BIG instantiations: WGMMA serialized due to insufficient
register resources. Initial96registers, zero reported local bytes, one resident
640-thread block/SM. Full result/audit/source/profiles/controller preserved in
results/resumed-moe-phase11-20261028/. Reverted, no score, no Decode transfer.
Close the GS/allocation tuning path; do not sweep group sizes or register counts.

A two-compute-warpgroup design was assessed, not implemented/evaluated. It
would need128-row WGS units and a repack, double phase-A/B units/staged X reads,
change readiness indexing, and preserve512virtual gather/finalizer threads to
cover4096columns and the old FP32 norm tree. Extra activation traffic makes
this a poor next check without stronger bottleneck evidence.

Next: one isolated WGS-weight cache policy from retained33114871. Fat jobs
split at64tokens, so multiple jobs on a hot expert reread identical weights
within a launch; the existing evict-first comment assumes weights stream once.
Own NumPy public-formula routing draws (seed20261028,64draws/shape) found repeats
in50/64draws atM256, mean logical load/unique-weight ratio1.07969, and17/64
atM128, ratio1.01287. These are synthetic opportunity counts, not official
inputs, cache-hit measurements or a speedup prediction. CPU model is saved
in phase11. Preserve intermediate evict-last stores and all numerical work.

No job/controller pending at this boundary. Expected MoE3checks/7scores left
after five checks since18:24, not a new quota API observation. Goal active.


## Phase12: WGS normal weight priority rejected; move to Decode (19:28 UTC)

Check7b9052f66cab49a2/source273c1f1a passed5tests/6benchmarks/integrity at
2.312809011x, CI[2.278116875,2.347501146]. WGS-normal-enabledTrue. All changed
WGS loads lost about3% relative speedup against primary33114871. Reverted, no
score. Do not sweep eviction constants or transfer this change. Full frozen
source/results/profiles, CPU audits and history: results/resumed-moe-phase12-20261028/.
The GS2cache variant's captured compiler log reports96registers/16barriers,
zero stack/spills and no C7512. This is not SASS or cache-hit attribution.

Inactive Decode cache draft4c910181 is988840bytes and has unchanged non-MoE
AST; it was never evaluated/promoted because standalone timing lost. Both
Decode09badc9b and MoE33114871 stay published/retained.

Six MoE checks and zero scores used since18:24. Expected MoE2checks/7scores
left (accounting estimate, not fresh API observation); reserve those for new
evidence rather than more register/cache tuning. Other track quotas unchanged
by this work. No job/controller pending. Overall remains last observed146
at18:24, no new rank claim. Native top50Goal stays active.

Next in the requested order: Decode. Correct profile attribution at one token:
router_fp32~11.6us, pooler v18s_gemv~13.4us, unpooler smtrace_gemv~4.0us.
At8tokens router~14.9..17.9us. Do not attribute pooler time to unpooling or
spend a check on pruning a tiny zero unpooler path. The full-FP32 router input
currently reloads the same1024-column K-split for all8warps/CTA. Assess one
shared-input cache per CTA:4KB atM1,32KB atM2..8, fresh each launch. Preserve
FP32weights/input, every __fmaf_rn and shuffle/reduction order, sigmoid/top8/ties,
128block counter protocol, output format and side-stream ordering. No BF16
router approximation. No new Decode source or check yet.


## Two-consumer-group architecture prepared

Draft8f6b8c76 has exact inverse/source wiring, full-size weight permutation,
CPU ABI/shared-memory, wide-integer scheduler and fair protocol audits. See
experiments/prepared/moe_two_warpgroup_review.md. It replaces the fast main
module with320threads, FT_T32, NX2, half weight units and an occupancy-checked
two-CTA grid. Tiny/MID/ineligible fallback paths remain original. GPU resources,
correctness and speed are unverified. One primary check is authorized next; no
new baseline, no parameter sweep, no score until the full guard passes.


## MoE phase13: two consumer groups rejected, October28 20:59UTC

Check0fb1da41657d4346, source8f6b8c76, passed5tests/6benchmarks/integrity
and analytic all-remote probes. Actual resources match the CPU layout:
320threads, Smem22744bytes, dynamic110848bytes,96registers,0local bytes,
2active CTAs/SM for both variants,132SMs and264CTA capacity. The new path
ran in all M32/128/256 profiles. This proves the intended resource change,
not a speedup: whole point2.045556897x, CI[2.014873544,2.076240250].
Affected relative speedups were0.80046,0.74681,0.72405. Controller discarded
and reverted; guard was not run because whole metric already lost. No score.
Retain33114871 (checked2.365437416x/published2.348299935x).

All changed loads lost, including M32 where smaller job capacity need not
increase jobs. Do not blame the result solely on larger-load repeated weights
or try neighboring occupancy geometry without new evidence. Preserve the last
MoE check for a stronger idea. Full source, result, profiles, edits, CPU audits
and controller history: results/resumed-moe-phase13-20261028/.
No job/controller pending. Next ordered focus is Decode; corrected1fb5d05d
remains checked1.988085030x/published1.986659677x.


Padding inspection: retained331 WGS gather already stores only r<cnt rows
(d[k] nonnull); padded rows are counted in xready but not globally written.
Producer bulk copies ceil(cnt/8)*8 rows, not FT_T rows. There is no easy
full-padding zero-fill removal to spend the last MoE check on. No new MoE
evaluation used. Native local expert count is32 of256 global experts;
MMAX384 is a token-capacity constant, not the expert count.
