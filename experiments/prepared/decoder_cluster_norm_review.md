# Decode: cluster4 residual/norm boundaries

Terminal result supersedes the preflight discussion below. Check63acbd4d99324fea
passed native resource admission in all four modes (32regs,0local,72static,
248active clusters,132SMs), then failed the fixed setup numerical comparison:
FP32 residual global5.4878e-5/worst-row2.1086e-4 exceed2e-5. No official full
tests or benchmarks followed; no speedup or official correctness claim exists.
Reverted96fd1b7, controller archived20261029-161455, all44protected hashes restored.
Full failure packet: results/decoder-cluster-norm-setup-failure-20261029/.

Pinned SGLang source shows a different logical-thread/FMA/reduction order.
A768-row CPU model supports an exact-order distributed repair, but the actual
failed GPU row and M/kind are unavailable, so cause remains a hypothesis.
No native repair/retry yet; gates stay unchanged. Diagnosis and original
upstream sources: results/decoder-cluster-norm-reduction-diagnosis-20261029/.
Original preparation notes below are historical.

Parent D9: `d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988`.
Prepared candidate: `f382e1afd11d53edc1f203df0e58502dc704d7de15da66c4780fb0e9fd1eecaf`.
Native CUDA plus launcher: `c84debd7016155722eda211709914721b254f8383d091e9154a72f97807e87c6`.
No GPU result or measured speedup yet. Full packet:
`results/decoder-cluster-norm-preflight-20261029/`.

## Mechanism and expected opportunity

The two residual/norm boundaries precede the router/pooler branch and the layer
return. Small-token saved profiles show roughly8..9microseconds combined.
Those are separate profiler-run durations; they are not benchmark critical-path
fractions. This candidate leaves every projection and its geometry intact and
uses four CTAs in a cluster per complete8192-value row. Each CTA handles2048
values, with two distributed FP32 row reductions and a final DSM lifetime join.
It targets underfilled small-token rows, not higher occupancy in the MoE mainloop.

This differs from the rejected cooperative O1/O8 projection fusion: no dense
weight ownership changes, no grid-wide spin or join, and no projection code
inside the boundary. All cluster participants are scheduled together by the
cluster launch; actual native capacity still needs admission.

The CPU weighted-load screen charges a hypothetical net saving in each complete
layer, across both boundary changes and all overheads. A1us net saving for
M1/8/16/32 corresponds to about1.00514track ratio;2us to1.01034;3us to1.01562.
These are conditional scenarios, not predictions or an overall-board formula.

## Arithmetic and scope

The attention mode retains BF16 branch normalization, separately rounded FP32
alpha/residual and beta/branch products/addition, FP32 updated residual, FP32
pre-normalization and BF16 output. Small-token router input remains FP32.
FFN modes also retain the BF16 effective post gain and its BF16 product
materialization before RMSNorm. Only T<=8 uses the task's1024-column implicit
zero tail; both row norms always use the full8192 denominator. Above8tokens
the unpooler output is complete and cannot be discarded as zeros.

The timed root function and all original module nodes are an exact prefix of
the existing source-preserving307dd791pack. Added wrappers select boundaries
only for M1..32 after one shared prepare-time decision. M>32 and unsupported
layout/dtype/cache modes use original fallbacks. No actual input or route is
host-read or cached for selection. No persistent extra device scratch is added.

## Current evidence and fixed gates

All four semantic modes compile offline at32registers,72static shared bytes,
zero dynamic shared memory and zero stack/spills. Native H200 attributes,
cluster capacity, graph behavior and elapsed time remain unverified.
CPU60numerical cases have maximum modeled relativeL2about4.805e-5 against a
higher-precision row oracle.256DSM schedules detect shared-stat overwrite and
early-exit mutations; full-row ownership and the wrong1024 denominator are checked.
All six freshly downloaded official evaluator files match pinned copies.
Independent CPU wrapper/signature checks and27fault-injected result fixtures pass.

One future check runs20component correctness cases and8changed-input graph
replays. Paired cold-L2 graph comparisons alternate arm order and retain raw
samples for each boundary at M1/8/16/32. Every regime must save at least1us
combined, improve combined time by at least3%, and neither individual boundary
may regress more than2%. Selection is all-or-none across the family.

Retention also requires full13tests/12benchmarks/integrity, both new boundary
kinds actually executing in all eight affected rows, original controls, correct
46sliding+16full load reconstruction, affected and normalized-affected load
gain>=1.005, raw and control-normalized track gain>=1.003, no affected load
below0.995normalized or individual changed case below0.98normalized. Controls
must stay within0.95..1.05. None of these gates establishes statistical significance.
If the design loses, preserve D9 without a noise retry or post-hoc shape selector.

The cluster mapping and DSM lifetime design follow
[NVIDIA's cluster programming model](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters).
Mechanism support is not evidence of profitability.
