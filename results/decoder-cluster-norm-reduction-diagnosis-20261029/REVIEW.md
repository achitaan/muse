# Reduction-order diagnosis after setup rejection

Check63acbd4d99324fea/sourcef382e1af reached native resources successfully
(32regs/0local/72static/248active clusters/132SMs, all four modes). A fixed
experimental incumbent-comparison gate then rejected FP32 residual differences
of5.487757880e-5global and2.108618646e-4worst row, versus2e-5limits. There were
no official correctness tests or benchmarks after this setup failure. It is
not an established speed regression or an official-task accuracy verdict.

The original diagnostic omitted the failing M/kind and actual input. This
prevents proving its exact cause from the saved result. Future bounded repairs
must record that context before numerical comparison; do not infer completed
probe/replay counts from partial progress.

The pinned scalar SGLang implementation uses256logical threads, four contiguous
vec8packs per thread, ordered FMA chains and an eight-warp XOR reduction tree.
The first candidate used1024logical threads with strided elements and linear
combination of four CTA sums. It preserved mathematical precision boundaries,
but its FP32 addition order differs. Finite CPU fixtures reproduce changed
row sums; the small-value fixture also changes90BF16 postnorm values. These
are toy reproductions, not the unavailable failed GPU input.

A bounded repair can preserve the original256logical threads and exact FMA/
warp-partial tree while distributing64threads to each of four CTAs. Each CTA
owns two original warps; all eight warp partials must be collected before the
original final tree. The CPU model matches all768reference row sums bit-for-bit.
This is source-order/ownership evidence, not native correctness or performance.
No native repair or retry is present yet. Keep all thresholds unchanged.

Do not copy the pinned upstream cluster variant's four-moment prenorm shortcut.
Its moment expression precedes the required BF16 normalized-branch materialization
used by the selected scalar baseline. The replacement must form that BF16 branch
and the actual updated FP32 residual before computing its second norm statistic.

Primary pinned sources are preserved with original notices in pinned-upstream/:
[scalar and cluster kernel](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh),
[CTA reducer](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/include/sgl_kernel/cta.cuh),
[warp reducer](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/include/sgl_kernel/warp.cuh).
