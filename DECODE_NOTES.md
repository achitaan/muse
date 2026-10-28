# Next Decode investigation - October 28, 2026, 19:28 UTC

MoE phases7..12 produced no retained improvement; none is transferred here.
Decoder09badc9b remains checked/published at1.994319167x/1.993956388x.
No job or controller is pending. Read
experiments/prepared/decoder_router_shared_review.md before implementation.
The proposed change caches each FP32 router K-split once per CTA, reducing
eightfold duplicate logical input loads while retaining the exact original
FP32 FMA/reduction/top8 computation. No new Decode candidate/check exists yet.
The earlier SK2 projection trials and profile misattribution stay rejected.

# Latest publication - October 28, 2026, 17:58 UTC

Retained 09badc9b is now also the published source. Scored job
441eb580f3074669 passed all 13 tests, 12 benchmarks and integrity at
**1.993956388x**, interval [1.964047042,2.023865734]. Overall is **146/1436**,
Decode **146/869**, at 17:58:08 UTC. Previous 1.965324622x publication is
superseded; its exact source/evidence remain in phase5. No job/controller pending.

Phase9's output comparison 008ff9489dec46d2 passed at 2.007000100x but selected
no alternate: its fixed kernel was 4.9% slower. The guard discarded it. Correct
profile mapping: m16trace_ is O16 (~5.6 us), dd_cap2_b_cpssm is unpooling (~14.6 us).
Do not attribute the latter to O16 or count the empty-choice aggregate as a gain.
Full frozen records, profiles, audits, history, publication and placement are in
results/resumed-decoder-phase9-20261028/.

One scored confirmation followed the primary 09badc9b check and three full
incumbent-only observations. Intervals overlap, so no formal significance claim.
MoE's later scale-six trial passed correctness but was slower and discarded.
Next focus is the WGS32 specialization review described in HANDOFF.md.

---

# Phase8 checkpoint - October 28, 2026

Shared-build trial 50eae88b3dd744ae passed 13/12/integrity at 1.998784201x,
CI [1.968802438,2.028765964]. All six bounded projection comparisons ran.
None selected a new path: pooler and sliced unpooler were about 3..5% slower,
QKV tied/slightly lost. Maximum synthetic relative L2 was 0.0001871. The guard
discarded the trial; no score was used. The full official calls still used
incumbents, so this is not full official numerical coverage of new paths.

Prepare took 254.61 s. Existing CUDA bytes and router precision remain exact;
three unused router setups were disabled. Retained 09badc9b and published
a61735ee are unchanged. Full source, profiles, CPU audits and controller:
results/resumed-decoder-phase8-20261028/. No job is pending at this checkpoint.

Next prepared trial replaces the three rejected instantiations with one fixed
output projection for 9..16 tokens, targeting its profiled ~15 us at 16 tokens.
See experiments/prepared/decoder_output_projection_review.md. GPU unmeasured.

---

# Phase7 checkpoint - October 28, 2026

The small-projection trial afd35730c7d3476f passed all 13 tests, 12 benchmark
cases and integrity at **1.997916935x**, CI [1.967948181,2.027885689]. The
230-second safety gate skipped the new paths: old preparation took 257.03 s.
No new packing, extension build or projection comparison ran on GPU. Their GPU
correctness and performance are unverified despite the whole-decoder pass.

The guard correctly discarded the trial because it lacked an explicit nonempty
SMALL_PROJ_CHOICES tuple. Retained 09badc9b and published a61735ee remain unchanged.
The trial is an independent timing observation of the retained computation with
an empty-choice forwarding wrapper, not a new speedup. No score was submitted.
Source, result, profiles, CPU audits and controller: results/resumed-decoder-phase7-20261028/.
One check used; no job/controller is pending. Overall was 150/1432 at 17:04:44 UTC.

The prepared builder adds three fixed public SK2 configurations in a separate
extension. CPU tests recover 1,441,792 BF16 values and 233 correction entries
exactly, prove stream bijections and test fallback, dispatch and store geometry.
Trial source 92fe19e2 is 995430 bytes. All previous source bytes are unchanged.
Next, reduce preparation overhead or share the existing extension build before
spending another check; do not simply increase the safety cutoffs. The baseline
prepare observations of 215 s and 257 s show that headroom varies materially.

---

# Current retained decoder check - October 28, 2026, phase6

Checked source best/decoder_verified.py, SHA256
`09badc9b956d00795ff2eeb61c63244b25347736ff5819e20a8f692a863d5a67`.
Job 09b34f4e921e49a6 passed all 13 tests, 12 benchmarks and integrity at
**1.994319167x**, CI [1.964404380,2.024233955]. This has not been scored: its
interval overlaps the published source's check. best/decoder_published.py
retains a61735ee and the 1.965324622x publication 02fd620987114785. No job or
controller is pending. Overall was 150 at 16:48:43 UTC; top 50 remains unmet.

## Measured phase6

The combined exact-MoE and implicit-padding FFN trial 8589b0fde4344ff4 passed at
1.972242299x, CI [1.942658665,2.001825934]. It transferred literal native MoE
CUDA from the fully checked 33114871 source. Other attention/dense/router CUDA
is unchanged. Global/nonlocal declarations are namespaced and all global
bindings resolved. This MoE is now separately published at 2.348299935x.

The small-token unpooler retains only its 1024 local output columns. A new
Triton boundary supplies its known zero tail implicitly, preserving the full
8192-column RMSNorm domain, BF16 product/normalized-branch round trips, FP32
residual update and FP32 pre-norm input. The original boundary is retained
for larger shapes and unsupported layouts. CPU models agreed on 18 cases;
an analytic constant case catches the erroneous 1024 denominator.

The first profile exposed a latency regression: the new FFN kernel took about
5.5..5.6 us for one token, versus the old boundary plus padding around 4.4 us.
One follow-up moved independent stream/gate/pre-weight loads before the first
reduction. Its statement AST multiset, arithmetic and launch geometry are
identical. The profile then showed 3.8 us for one token and 3.4..3.8 us for eight.
The aggregate check rose to 1.994319167x; no geometry sweep or intermediate
score was used. Individual stage timings across jobs are diagnostic evidence,
not controlled independent speedup estimates.

The unselected dense WG preparation sweep was removed. Published check/score
logs both retained all three incumbents; those timed dense paths are unchanged.
Latest observed preparation was 214.96 s versus 279.08 s in the earlier check;
these setup observations are not the optimized runtime metric or a controlled
setup benchmark. The 300-second prepare budget remains important.

Full sources/profiles/CPU audits/controller history:
results/resumed-decoder-phase6-20261028/. Builders in experiments/prepared/
reproduce exact 427cebf1 and 09badc9b sources from immutable files. Source size
988314 bytes. Next: review curated small-batch SK2 projection components in
experiments/prepared/decoder_small_projection_review.md. No new compressed
projection candidate has been implemented or measured yet.

---

# Latest decoder publication - October 28, 2026

Scored job **02fd620987114785** passed all 13 tests, 12 benchmarks and integrity
at **1.965324622x**, CI [1.935844752,1.994804491]. Exact a61735ee source is in
both best/decoder_verified.py and best/decoder_published.py. Overall placement
was **149/1428**, and Decode **149/869**, at 15:47:59 UTC. The current Decode
50th-place score was 2.139x; top 50 remains unmet. No decoder job is pending.
Full evidence and placement snapshot are in results/resumed-decoder-phase5-20261028/.

The pending-publication statements in the historical checkpoint below are now
superseded by this passing result. MoE's later standalone 2.365437416x check has
not been scored or transferred; the decoder's embedded MoE remains dd6049cb.

---

# Latest decoder check - October 28, 2026, phase5

Checked source: best/decoder_verified.py, SHA256
`a61735ee0cb7774ae8e187526d2e104f3df4e2e95453f66ee3b46fdc33f9186d`.
Job 34131378611b46ab: **1.968554229x**, interval
[1.939025916, 1.998082543], all 13 tests, 12 benchmarks and integrity passed.
One scored publication is pending: **02fd620987114785**. Preserve that handle;
do not submit again. Until it passes, best/decoder_published.py retains the
previous 660f4ac5 source and 1.874889991x publication. Overall rank 193 is the
last observed snapshot for that earlier publication, not a rank for this check.

## Two checks, one publication

First, pre-expert RMSNorm and expert-ID rotation were fused into one Triton row
kernel. It reads the existing prepared BF16 effective norm gain, uses FP32
intermediates/reduction and writes BF16 normalized input plus int32 local IDs.
The positive modulus 256 is implemented exactly as a bit mask. No input-derived
values persist between calls. CPU audit: 50 manufactured numerical cases,
maximum relative L2 0.001993 to analytical FP64 after BF16 rounding, and all
2,048 ID/start combinations exact. H200 check 90e0e5d105eb4773 passed fully at
1.939554165x, interval [1.910460852, 1.968647477]. Its 1.6..2.6 us kernel replaces
normalization at 4.4..6.1 us and a separate rotation launch. The interval still
overlapped the earlier source's check, so this intermediate was not scored.

Second, the whole decoder now actually uses the wide-router kernels already
prepared by its checked dense component. Above 32 tokens it chooses the
existing 128-row arm up to 128 rows and 256-row arm thereafter. Both use dynamic
M, TMA boundary handling and masked output stores; complete whole-layer tests
include nearby 33/40-row cases. Unsupported device, layout or prepared state
retains the original two-GEMM path. No new CUDA payload, weight layout or prepare
compilation was introduced. BF16 hi/lo router weights and input have FP32
accumulation/output. Small-token full-FP32 routing and the 9..32-token fusion
are unchanged. CPU inverse recovered all 4,194,304 hi/lo words; the permitted
matrix approximation's maximum relative L2 to FP64 was 2.611e-6 in seven row
sizes, including 257. This model does not emulate wgmma reduction order.
Profiles show the transferred wide router at 6.3..8.6 us, replacing the
separate GEMMs and their reductions on large verify shapes.

The combined lower interval 1.939025916 exceeds the published source's checked
upper interval 1.913738826. That is the publication gate; the intermediate
check is not an independent baseline or replication. Both phases use the
service's paired SGLang timings and cold L2, with complete official coverage.
No repeated baseline, blind sweep or scored intermediate was used.

Full sources, profiles, CPU audits and controller history:
results/resumed-decoder-phase5-20261028/. Promoted builders and CPU audits are
in experiments/prepared/ and read immutable measured sources. The phase is
archived; no controller is active. The overall Goal remains active, top 50 unmet.
MoE is still the largest component of several whole-layer shapes. A possible
next investigation is pairing scalar BF16 GEMM-output epilogues while preserving
every rounding boundary; no such change has been implemented or measured.

Earlier checkpoints below are historical.

---

# Current decoder checkpoint - October 28, 2026

Current checked and published source: `best/decoder_verified.py` and
`best/decoder_published.py`, SHA256
`660f4ac563357a66f3a45632abf3d38113277734025cb0ad10a51a728f0ba1ff`.
Check 7f5007bdf0784a65: 1.885456971x, interval [1.857175117, 1.913738826].
Score e5b95129043d442d: **1.874889991x**, interval [1.846766641, 1.903013341].
Both passed all 13 tests, 12 benchmarks and integrity checks. Latest placement
is 193/866 for Decode and 193/1427 overall at 15:05 UTC. Rankings can change.

## Changes and evidence

For 9..32 tokens, the pooler and router share one native kernel. BF16 pooler
weights use the existing lossless representation with exact escape corrections;
router weights retain BF16 hi/lo components with FP32 accumulation. Unsupported
weights or workspace bounds use the previous implementation.

For 1..8 tokens, a new SIMT CUDA kernel performs the full FP32 router GEMM and
top-8 selection. Inputs, weights, products and accumulation remain FP32; no
TF32 or BF16 approximation is introduced. Partial sums are published with a
memory fence before the completion counter. Every consumed partial is rewritten
on each call. A low-seven-bit counter cycles after 128 producer blocks. Selection
adds router bias, breaks ties by the lowest expert ID and normalizes the unbiased
sigmoid scores. The original FP32 F.linear path remains as fallback.

CPU audits cover all 2,097,152 row/K pairs, counter wrap, manufactured router
cases including ties and outliers, lossless pooler/router packing and fallback
AST identity. These audits are separate from the passing H200 results. Shared
scratch requires sequential calls. All original residual, MoE precision, cache
write and attention semantics remain covered by the whole-layer checks.

The pooler/router-only candidate bd3fad49 checked at 1.836827648x with overlapping
uncertainty, so it received no scored run. The combined candidate's checked lower
bound exceeds the previously published source's checked upper bound. Only then
was one scored publication launched. The initial oversized source was rejected
by HTTP validation before a job existed. Packaging removed Python docstrings and
comments, compacted indentation and deduplicated complete license copies; CUDA
literals and executable AST were preserved. Current source is 984,265 bytes.

Profiles show the fused FP32 router at 11.7us for one token and 14.3us for eight;
the one-token shape is not a uniform win. The aggregate improvement is verified.
No repeated baseline or parameter sweep was used. Full sources, profiles, CPU
audits and immutable controller history are in results/resumed-decoder-phase4-20261028/.
Previous published source 53424d85 and its 1.804641371x score remain in phase3.

## Reproduction

The promoted builders in experiments/prepared/ read immutable phase3/phase4
sources. build_decoder_fp32_router.py reads the tracked fp32_router_addon.py and
reproduces SHA256 660f4ac5 exactly. The preceding build_decoder_fused_pr.py also
requires the pinned Vjeux upstream decoder source at .tools/vjeux-decoder-upstream.py
(commit f35a41ee1c13e316dabbf772830d47b23b453393); its output is already frozen
as bd3fad49 in phase4. Frozen measured sources can be used directly without the
upstream file. All files must be read and written as UTF-8 or exact bytes.

The current phase is archived. No active controller and no pending scored job.
The historical entries below record earlier work and are superseded here.

---

# Latest decoder checkpoint - October 28, 2026

Check `2f41872843fe4490`: **1.815511785x**, all 13 tests, 12 benchmarks and
integrity checks passed. Interval [1.788279108, 1.842744462]. Exact source
`best/decoder_verified.py`, SHA256
`53424d85d65aaa61d518c044d2a78e10453a9e4f6d13c885c1c633f1f8240024`.
Scored publication `95bbfd2c81224715` passed at **1.804641371x**. Do not duplicate.

This integrates the checked native attention and corrected native MoE while
retaining the checked lossless dense implementation, original SGLang residual
boundaries, expert-ID rotation and FP32 router for small token counts. Native
MoE preserves BF16 GEMM/global-scale boundaries and FP32 norm intermediates.
The original Marlin weight bank is not built because native MoE owns its exact
packed NVFP4 copies. Shared scratch requires sequential calls.

First integration check `42673a30ea944f2b` failed before numerical execution:
the AST renamer omitted a Global declaration. The controller reverted it. The
fixed transformer now renames declarations, and a local audit verifies 94
global bindings and calls the attention factory twice, confirming one build.
Only this specific fix was retried; no parameter sweep was run.

The first scored upload hit a TLS error. The authoritative jobs list confirmed
no new run before one retry, which returned job 95bbfd2c81224715. Evidence and
reconciliation are in results/resumed-decoder-phase3-20261028/. Preserve Apache
native-source notices and the dense MIT notice. The older checkpoints below
are historical, not current source identities.

---

# Whole decoder-layer notes â€” October 28, 2026

The user requested MoE first, then Decode Layer, Attention Decode, Prefill last.
All work was in the main chat; checks were used before scored publication.

## Verified progression

- Existing SGLang baseline 65ab5fffad0c4995: 1.002013444x. Reused from cache for this phase; no repeated baseline GPU charge.
- Transfer of the checked FP32 MoE finalizer and EP8 M tiling, job 38c6e73267b8443f: 1.106991211x.
- Namespaced integration of the lossless dense projection kernels, job 5976b8f072e94fc0: 1.150084629x.
- Transfer of checked Q/K norm, RoPE, FP8 conversion and cache writes, job 03de660946744adb: **1.168638714x**, CI [1.1511091334927925, 1.186168294918969].

Every retained step passed all 13 correctness tests, all 12 benchmark cases and
integrity checks. Small point changes remain subject to timing uncertainty.
Total decoder budget used: four checks, one scored publication now pending.
Observe scored job **88f8b06932fa43b7**; do not launch a duplicate.

Exact source: best/decoder_verified.py, SHA256 5ed23e1eba17e7abfc5145fbdf5b607f501163517b9869af433363bb48ac10c2.
Metadata: best/decoder_metadata.json. Frozen sources, raw judge records, profiles
and controller history: results/resumed-decoder-phase2-20261028/.

## Fidelity and provenance

The FP32 router-input path up to eight tokens remains unchanged. Above that,
the original BF16 hi/lo router GEMMs and FP32 output remain. Dense kernels cover
QKV, output, pooler and unpooler only. Original BF16 weights are losslessly packed;
NVFP4 expert weights remain in their exact source format. FP32 norm intermediates
are retained; remote expert routes are masked before arithmetic.

The dense integration derives from Nathen Fernandes' MIT-licensed v18. Its
upstream notice and full license are embedded in the submission and retained in
THIRD_PARTY_NOTICES.md. AST namespacing preserves all literal CUDA payload bytes.
FA3 still attends to every required key, with unchanged causal/window semantics.
New cache entries, routing, residual streams and next attention input were all
checked by the official whole-layer judge.

## Next track

Attention Decode is next. Existing checked source best/attention_checked.py is
1.120441329x; scored job f6b899f674914ef2 is 1.116457695x. A partitioned attention
prototype is prepared with CPU interval/softmax audits and exact raw-byte FP8
views. Its GPU compilation, correctness and speed remain unverified. Prefill
remains last. No paid GPU or external model provider is authorized.


## Decode phase10: shared router input rejected after causal review (19:46 UTC)

Checka2960d1516fc4243/source41d475ae passed13tests/12benchmarks/integrity at
2.003057078x, CI[1.973011222,2.033102934]. New xshared kernel appears in all
four small-token profiles. Its affected load speedup ratios against09badc9b
were0.98336/0.99071/0.97434/0.97027: every changed load lost. Router spans
were12.7/11.0/17.0/18.3us versus11.6/11.6/14.9/17.9us. The whole-layer point
gain came from unchanged controls, notably32-token loads. Do not claim a gain.

The controller correctly recorded its numeric keep by the configured whole-layer
metric; its immutable log remains intact. After archiving, review explicitly
restored09badc9b in commit2324c43. No score and no best-source promotion.
Full source/profile/audits/history: results/resumed-decoder-phase10-20261028/.
This demonstrates why future router guards must require affected-load and
component improvement, in addition to full correctness and selected-path proof.

Next: replace top8's repeated float/ID/raw shuffle tree with exact unsigned
ordered-FP32-key max plus integer ID min and one raw-score broadcast. This
changes selection implementation, not score precision. Signed zeros need one
canonical key because float comparisons equate them; NaNs never win the
original local comparisons. A preliminary CPU proof matched old shuffle and
scalar global selection on404vectors/3232picks, including ties/extremes and
NaNs, with raw-score bits unchanged. No CUDA candidate yet. Preserve all
FP32FMA, split reduction, sigmoid, normalization, counters/fences and shape
dispatch. Main chat only; one check after complete review, no repeated baseline.

No job/controller pending at this boundary. Decode quota expected4checks/5scores
after one new check; not a fresh API observation. Published09badc9b remains
1.993956388x. Native top50Goal active; no updated rank claim.
