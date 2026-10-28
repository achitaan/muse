# Decoder exact MoE transfer and implicit-padding FFN boundary

Baseline a61735ee: full passing check 34131378611b46ab at 1.968554229x;
publication 02fd620987114785 at 1.965324622x, rank 149 at the last snapshot.
No repeated baseline is needed.

Transfer the separately checked MoE 33114871 (2.365437416x, full 5/6/integrity),
including paired BF16 epilogues and the exact DQ8 bit layout. Its standalone
improvement did not clear the conservative publication gate, so it was not
scored. The source builder preserves its literal native CUDA byte for byte;
all other attention/dense/router CUDA payloads remain identical. Namespacing
renames global/nonlocal declarations and a symbol-table audit finds no
unresolved global references.

For at most eight tokens, the official unpooler returns this GPU's first
1024 columns and pads the remaining 7168 with zeros. The new Triton boundary
reads that compact input directly, uses implicit zeros, and fuses the existing
post-norm, gated FP32 residual update and next pre-attention norm. Critically,
the norm denominator is still 8192. BF16 input-scale products and normalized
branch boundaries are retained; residual stream and pre-norm input stay FP32.
The original module handles unsupported layouts, empty input and larger batches.
Only static prepared norm weights and residual coefficients are cached.

Primary source verification:
https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py
https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh
Both the BF16 round-trips and FP32 pre-norm domain were verified against the
actual single-CTA CUDA path. The new FFN boundary does not change the separate
attention boundary or small-token FP32 router input/weights.

CPU proof: 18 manufactured cases, including zero branch, zero stream, outliers,
small values and constants. Cropped-domain and explicitly padded calculations
agreed with zero measured relative L2 for both outputs. An analytic unit-prefix
case requires residual prefix 2.828125, catching an incorrect 1024 denominator.
This is a CPU arithmetic proof, not GPU compilation or performance evidence.

The previously unselected dense WG preparation sweep is removed to protect the
300-second prepare limit. Both the baseline check and scored logs say WG_KEEP
for all three shapes, with candidates slower by substantial margins. Timed
dense dispatch retains those measured incumbents; no new dense algorithm is
introduced. This changes preparation cost, which is not the speed metric.

One combined whole-layer check must pass all 13 tests, 12 benchmarks and
integrity. Inspect per-stage profiles, not just the aggregate. Publish only if
its interval clears the actually published source's prior check interval.
No small-token FFN warp sweep or standalone intermediate score is planned.
Source is 988314 bytes, below the service's 1,000,000-byte limit. Preserve all
Apache/MIT notices and additional Lee Penkman/SGLang attribution.
