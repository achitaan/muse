# Native MoE paired epilogue, not yet measured

Baseline source dd6049cb, full passing check 6792a5c832ee4aff at 2.299797371x;
published b4a10b76347e4cb0 at 2.312613084x. Whole-layer profiles still show
native MoE taking roughly 40..64% on several shapes.

The existing correction converts each scalar accumulator to BF16, rounds the
global scale to BF16, multiplies and rounds the product to BF16, then converts
back to FP32 for SwiGLU or post-normalization arithmetic. The prepared trial
pairs independent values through __floats2bfloat162_rn and __hmul2_rn, followed
by __bfloat1622float2. All BF16 boundaries remain. Norm-scaled values, sums of
squares and route combination stay FP32. No clipping, saturation, activation
quantization or weight changes are added.

The potential saving is fewer conversions, unpack operations and scalar
multiplications in the epilogues. Compiler scheduling and register pressure
could cancel the benefit; no instruction count or speedup is claimed yet.

Primary CUDA semantics:
https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT162__ARITHMETIC.html
The RN variant preserves multiply rounding and prevents contraction with
additions. This uses the same permitted BF16 arithmetic as the scalar baseline.

The CPU audit enumerates all 65,280 finite BF16 accumulator values against 24
BF16 scales spanning both signs, signed zero, subnormals, normal exponents and
overflow. All 1,566,720 products match independent analytical FP64 RN-even
rounding bit for bit. CPU BF16 and the scalar float32-then-BF16 model both match.
This tests algebra/rounding and does not emulate CUDA compilation.

The builder replaces eleven paired output sites, changes the extension cache
tag, and proves that reversing those replacements recovers every original
CUDA byte. Weight packing, routing, reduction order, SwiGLU and norm operations
are otherwise unchanged. The precision guard recognizes scalar and paired forms,
checking that every old scalar call was replaced in the trial.

Next experiment: one full MoE check, cached baseline, unchanged paired official
timings, complete five-test/six-benchmark coverage. No geometry sweep or scored
publication is justified before seeing the result. Keep only a passing gain,
and transfer into the decoder only if the benefit is worth another check.
Numerical equivalence is evidence for correctness, not speed.
