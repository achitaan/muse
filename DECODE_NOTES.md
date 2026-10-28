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
