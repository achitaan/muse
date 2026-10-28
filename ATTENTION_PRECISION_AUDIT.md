# Attention BF16 boundary audit

October 28, 2026. Numerical correction takes priority over further publication.
This audit also applies to the native attention embedded in Decode.

The pinned SGLang revision is a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f.
Its [FusedQKNorm source](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_qk_norm.py)
rounds the normalized row to the input BF16 dtype before RoPE (lines81..87).
Q/K output allocations use BF16 when the operation is called without a fused
KV store (lines299..308). Consequently the RoPE result is also BF16 before the
baseline converts Q and new cache entries to FP8. Full attention has no RoPE,
but still materializes the normalized Q/K as BF16. Our pinned local
muse_attention_decode/baseline.py additionally casts attention output to BF16
before rms_norm_gate. TASK.md requires BF16 wherever SGLang keeps BF16.

The inherited b56e07b1 attention source omitted these round trips: it used
FP32 norm/RoPE values for conversion and FP32 attention output for RMSNorm.
Its pr8 helper was an identity. Earlier full correctness passes only establish
tolerance to the public reference; they do not establish these operation
boundaries. The same CUDA literal is embedded in Decode09badc9b and earlier
native variants. Earlier published speedups describe those historical sources;
do not use them as evidence for the corrected source or publish them again.

The correction rounds the normalized Q/K before RoPE, rounds the RoPE output,
and rounds attention output in every one of the five finalizers. Reductions
stay FP32, caches stay FP8, and scheduling, split reduction, all keys, cache
writes, weight preparation and scratch lifetime stay unchanged. It also
corrects the PyTorch fallback: BF16 norm/RoPE, and the query scale folded into
the attention score after those boundaries. Power-of-two cache scales1/64
and1 preserve the relevant BF16 values exactly.

Prepared scripts and audits live in experiments/prepared/:
build_attention_bf16_boundaries.py, audit_attention_bf16_boundaries_cpu.py,
build_decoder_attention_bf16.py. The CUDA edits reverse byte-for-byte to the
parent. All Python AST outside the corrected fallback is unchanged. The Decode
transfer uses exactly the corrected CUDA and renamed fallback AST; every other
Decode AST node is unchanged, including the original FP32 small router.

CPU evidence covers1,579,010 rounding values, RNE ties and neighbors, signed
zero, normal/impulse/zero rows and both cache scales. Missing-norm/RoPE mutants
change thousands of FP8 cache bytes; missing attention-output rounding changes
136,403 final BF16 elements. Twelve nearby fallback cases cover q_len1/3/8,
page1/8 and full/sliding, comparing cache writes and output to an independent
FP64 mathematical oracle with the BF16 boundaries. Maximum relative L2 is
4.89e-9. Torch2.3 CPU lacks FP8 advanced indexing, so that CPU test indexes an
equivalent uint8 view of the bytes; submitted GPU arithmetic is unchanged.
This model does not establish CUDA FMA equivalence or GPU correctness.

Attention draft24782695 (170315bytes) is checking as1a16e2d658a8400d.
Decode draft1fb5d05d (988827bytes) is prepared, not GPU measured. Wait for the
Attention full16test/12benchmark/integrity result before a Decode check.
No scored correction has been launched yet. No active autoresearch controller;
phase11 was archived after its guard rejected the router experiment. These
are required numerical repairs, retained based on conformance rather than
discarded solely because an older nonconforming source might time faster.
