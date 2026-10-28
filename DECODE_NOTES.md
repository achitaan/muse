# Whole decoder-layer notes — October 28, 2026

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
