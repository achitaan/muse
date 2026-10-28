# Decoder expert-input fusion

Baseline: exact published source 660f4ac5, check 7f5007bdf0784a65 at
1.885456971x, interval [1.857175117, 1.913738826]; scored publication
e5b95129043d442d at 1.874889991x. Every official case passed.

The profile shows pre-expert RMSNorm at 4.4..6.1 us, plus a separate route-ID
rotation launch. One Triton row kernel computes both. Normalization uses BF16
input, effective gain and output with FP32 multiplication/reduction. The gain is
the existing prepared module's BF16 effective weight. Positive modulus 256 is
exactly a low-eight-bit mask, including negative ID differences. Every output
element is overwritten; no input data is retained between calls.

The original dense projections, routing scores/selection, native expert CUDA,
residual boundaries and attention are byte-identical. The full FP32 router for
up to eight tokens is unchanged. No extra compilation variants are introduced.

Primary contract sources:

- SGLang pinned RMSNorm implementation:
  https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/srt/layers/muse_spark_layers.py
- PyTorch RMSNorm dispatch:
  https://github.com/pytorch/pytorch/blob/v2.9.0/aten/src/ATen/native/layer_norm.cpp
- Local official muse_decode_layer/TASK.md and baseline.py.

CPU audit: deterministic seed 20261028, 50 manufactured cases (zero, constant,
normal, outlier and small-magnitude inputs), maximum relative L2 to analytical
FP64 RMSNorm 0.001993 after required BF16 rounding. All 2,048 expert-ID/start
combinations match the old rotation exactly. This is not GPU correctness proof.

Experiment design: one source change, one official full-case check against the
service's paired SGLang timings with alternating order and cold L2. Reuse the
cached baseline. Assess per-shape profiles as well as aggregate speedup; repeated
replays are measurements within a job, not independent job replications. Keep
only passing candidates. Publish only if the checked gain over the currently
published source exceeds the reported uncertainty. No tile/warp sweep is planned.

Risks: Triton reduction order can change final BF16 rounding; official complete
coverage is required. Expected launch savings are several microseconds per
shape, so the aggregate gain may still overlap measurement uncertainty.
