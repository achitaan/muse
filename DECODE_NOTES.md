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
