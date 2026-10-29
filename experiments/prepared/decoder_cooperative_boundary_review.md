# Decode cooperative output-projection boundary: CPU admission work

Later terminal GPU evidence supersedes the readiness discussion below:
check7f85bd70a3484dc7 validated resources/80synthetic probes but rejected both
fused families as slower. D9 restored; no publication or neighbor sweep.
See decoder_cooperative_boundary_admission_review.md and the source-bound
results/decoder-cooperative-boundary-check-20261029/review.json. CPU evidence
below remains historical and is not the final performance conclusion.

Baseline is retained D9, `best/decoder_verified.py`, SHA256
`d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988`.
No active source changed, no host launcher exists, and no GPU job was used.
The broad top50 objective remains active. This is a different completion
mechanism from the previously rejected serial last-block finalizer.

## Exact contract and mechanism

The pinned SGLang commit a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f is saved in
`results/decoder-cooperative-boundary-screen-20261029/pinned/`. Source URLs
and hashes are in `screen.json`; existing Apache attribution and license in
THIRD_PARTY_NOTICES.md and licenses/ apply. The retained H200 automatic norm
policy uses the single-CTA JIT backend, not the newer cluster policy enabled
only for capability major >=10.

For this attention boundary, post-norm has no weight/input scale; residual
gates are channel-wise FP32; pre-FFN effective gain is BF16; pre-norm input
stays FP32. Preserve the BF16 GEMM result, FP32 postnorm with eps1e-8, BF16
normalized branch, two-product FP32 gated residual and FP32 new stream,
FP32 pre-norm over all8192columns with eps1e-5, BF16 gain and BF16 FFN output.
The selected CUDA JIT source writes unrounded FP32 router values when
pre_norm_bf16=false. Its native/Triton alternatives round that router output.
Do not substitute a different backend's formula into the retained CUDA path.
Reduction grouping and compiler contraction still require GPU validation.

Prototype selected O1/O8 GEMV device prefixes are unchanged under an exact
inverse edit; only the additional parameter, helpers and final call are added.
All CTAs first finish their original BF16 output stores, then participate in
one cooperative grid barrier. The first M CTAs finalize one whole token each;
the others exit after the barrier. Every row therefore retains its own CTA.
The selected RED0/WK8/non-diagnostic paths have no earlier reachable return.
Other modes have not been adapted and must never select this prototype.

Three streaming passes avoid retaining a full FP32 row and all gate vectors
in registers: branch statistics; rounded branch/gated residual and pre-norm
statistics; new-stream reread and output/router stores. Two distinct128B
reduction scratch areas avoid a WAR race between returned sums. At M8 the
extra logical reads versus a register-held boundary are128KiB BF16 branch
and256KiB FP32 new stream. These are requested bytes, not measured HBM traffic.
Original output/scratch and residual/output tensors require distinct ownership.

## Completed local tests

`preflight_decoder_cooperative_boundary.py` builds exactly two families with
both packed and original-BF16 formats. NVRTC13.0 sm90a/C++20 proxy compilation:

| Path | Threads / CTAs | Dynamic shared | Packed / BF16 regs | Stack / spills |
|---|---:|---:|---:|---:|
| O1 |1024 /128|16384B|48 /38|0 /0 /0 both|
| O8 |256 /512|4096B|50 /32|0 /0 /0 both|

Baseline proxy registers are50/32 for both families. No profitability follows
from register changes. Arithmetic residency budgets on132SMs are one block
per SM for O1 and four for O8, enough in the simplified model. This is not an
actual occupancy query or guaranteed cooperative launch. The eventual host
must query capability, actual SM count, function resources and active blocks
using the exact dynamic shared size before enabling either family. Reject
unsupported/oversized grids before capture; retain the original path.

`model_decoder_cooperative_boundary.py` tests40manufactured numerical cases,
36row ownership cases across M1..8, and three intentionally broken contracts.
The corrected v2 packet uses source-bound epsilons: BF16 output and FP32
residual match the CPU higher-precision row-sum comparison in these fixtures;
maximum router relative L2 is1.787e-7. Dropped postnorm BF16 rounding, rounded
router values and a1024-column denominator are each detected. This finite CPU
model does not execute the GPU kernel or establish exact GPU numerical parity.
The original model.json used a generic1e-8 pre-epsilon and is superseded by
model-v2.json; the draft CUDA always accepted both correct runtime epsilons.

## Opportunity and next admission gates

Weighted D9 loads use46sliding+16full and six token bands. Hypothetically saving
1/2/4.5us from every affected M1/M8 case yields track gains1.003234/1.006512/
1.014904. These are optimistic screens, not forecasts;4.5us assumes removal
of nearly the whole separately profiled boundary despite replacement work.
The official overall aggregation remains unknown. Potential gain is modest.

Continue only with a source-bound host integration that reuses retained packed
plans, preserves all unaffected paths, fits preparation/source-size limits,
and proves exact cooperative capture/launch admission. NVIDIA documents
cooperative grid synchronization and graph capture support:
https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
https://developer.nvidia.com/blog/cuda-11-features-revealed/
Capability and capture behavior on the evaluator still need actual validation.

Before one future paced check: freeze the complete source; independently
compare fused results with the actual retained JIT boundary on fresh/skewed
data and changing inputs; test graph replay, nearby shapes and both formats;
require real new-path execution, full Decode tests/benchmarks/integrity and
native resource markers. Use a fixed material affected/control performance
gate decided before launch. Retain the incumbent if the evidence is tied or
inconclusive. No parameter sweep, score or transfer is authorized by this CPU
packet alone; existing user authorization and evaluation pacing govern runs.
