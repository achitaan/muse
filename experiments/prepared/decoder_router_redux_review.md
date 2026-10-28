# Exact warp reductions for full-FP32 router top-eight

Status: implemented and CPU-audited; GPU unmeasured. The failed shared-input
cache is excluded. Retained decoder09badc9b remains the baseline, checked
1.994319167x/published1.993956388x. Main chat only, one scoped check.

The router's final warp chooses eight biased-score winners. The old selector
uses five shuffle levels for best score, raw score and ID in each round. Keep
its local per-lane comparisons and replace that global tree with:

1. Map the complete FP32 winner bits to an unsigned monotone order key.
2. Warp unsigned max chooses the winning score key; integer min chooses the
   lowest expert ID among equal keys.
3. Broadcast the winning owner's unchanged raw score once.

For nonnegative floats, XOR of the sign bit preserves numerical order. For
negative floats, complement reverses the negative encoding's order. Canonical
zero bits make +0/-0 keys equal, as the old floating comparisons do. The
original local comparisons cannot select a NaN, so NaN encodings never enter
the key max. Tie IDs are reduced separately; no score bits are truncated to
make room for an ID. Raw scores, sigmoid, renormalization and all dot/split
FP32FMA/shuffle operations remain original. This is a selection implementation,
not a lower-precision score representation.

All32lanes of each remaining last-block warp participate. The original
warp>=M return is warp-uniform; no new memory-order assumption is made from
the reduction intrinsics. Every existing volatile partial, threadfence,
CTA barrier,128block counter and output instruction stays unchanged.

Primary documentation confirms signed/unsigned reductions on SM80and newer
and the participating-lane requirements (H200isSM90):
https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html

## Source and CPU evidence

The minimal core0d98e250 is988656bytes. Its only changed literal is_RF_CUDA:
the global top8tree and kernel names. The inverse recovers the entire original
CUDA literal and module AST. The core is not an independently GPU-checked file.

audit_decoder_router_redux_cpu.py compares the original warp shuffle tree,
the new key reduction and a scalar global selection oracle on404vectors and
3232eight-round picks. It covers ties, adjacent/extreme/subnormal values,
signed zeros, infinities and ignored NaNs, with raw-score bits unchanged.
A mutant without signed-zero canonicalization chooses ID1instead of ID0 and
is detected. FullFP32dot and split-reduction source bytes are unchanged.

## Controlled diagnostics in the submitted draft

Draftc378ea501dcb2adde88d6d607ecb9e75951dbb677e763609532be1bd03758962
is998174bytes, below the1MBlimit. build_decoder_router_redux_paired.py adds
an exact-source incumbent kernel/export in the same router extension and a
prepare-only diagnostic wrapper. No extra module or weight copy. The wrapper
reverses to the minimal core AST; its incumbent kernel reverses byte-for-byte
to the old selector. Other layer payloads remain exact.

Own syntheticFP32inputs use a private torch.Generator, never global seeding
or actual call inputs. GPU probes compare bitwise outputs/IDs at M1..8on
normal inputs and analytic zero-input/zero-bias ties. The latter must choose
IDs0..7with weights1/8. Then M1/8, normal/impulse patterns, each get eight
alternating cold CUDA-graph pairs using the same actual static weights and
bias. An owned128MiBflush exceeds L2; event timings exclude the flush. Preserve
all samples and paired ratios. This is diagnostic setup, not call work moved
outside the timed path.

Every actual timed router call always uses Redux. No selector, synthetic
output, graph or timing choice is returned as model state. Probe calls use
the existing parts/counter contract: every call adds128, so the counter low
seven bits remain zero, and every consumed partial is overwritten next call.
All graph work completes before diagnostic tensors/graphs are released.

Guard requirements are fixed before evaluation: full13tests/12benchmarks and
integrity, Redux visible in profiles0/1/6/7, all GPU bitwise/tie probes pass,
all four controlled paired medians exceed1.03, and the four actual affected
loads improve in geometric mean against cached09badc9b. The controller still
uses official whole-layer speedup as its primary metric, target2.15for this
phase only. Native top50Goal cannot be completed from this local target.

The last shared-input trial's aggregate keep was rejected after review:
unchanged controls caused the whole-layer point gain while every changed
load lost. Do not repeat that mistake or score a merely favorable aggregate.
One official check now establishes compilation, prepare budget, controlled
GPU behavior, full-layer correctness and timing. No baseline GPU repeat.
