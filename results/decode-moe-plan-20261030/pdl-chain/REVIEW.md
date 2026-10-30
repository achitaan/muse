# O / fused pooler-router / unpooler PDL at 16 and 32 tokens

CPU implementation packet. **No GPU check, score, correctness result or speedup
belongs to this draft.** The scored incumbent remains `c15df9d3`, 2.075466170x.
The final Decode check is preserved. Top 50 remains unmet.

Frozen draft: `experiments/decode_moe/decoder_pdl_chain_draft.py`, SHA256
`985abda3f1a91e6daca07a38e9f91a397b4942ef892285704ddbd1792bff9a40`,
984,569 bytes, 15,431 bytes below the limit. `binding.json` binds its CUDA,
C++ declarations, runtime, layer clone and exact parent prefix. The older
O-only `f377b6ce` draft and its evidence remain intact.

## Exact scope

The new layer mask is 1=O, 2=fused pooler/router, 4=unpooler. Only B16/B32,
q_len=1 can select it. Sliding and full context are gated separately. Every
other regime retains the scored parent. No TR8 change is transferred.

O16 uses the retained M16TRACE C12 instance; O32 uses the active **r3
direct-store** O32 source, rather than the unused generic OLAYER template.
Their provenance, 72/42-register screens and wait ordering are in `../pdl16/`.

U16 uses the later CPSSM override: WR2/WK4/KU16/CP2, MT2/ST2/RED0. Initial
asynchronous weight copies precede the dependency wait; coherent activation
loads follow. U32 uses the later WK2/PF2/Q2/CAP2, split-2 cluster source.
Its two escape loads and six vector weight loads precede the wait. Initial
activation TMA moves after it. The original FP32 partial-reduction order,
cluster barriers, peer lifetime and BF16 output store remain unchanged.
Both derivations have exact inverse proofs, saved under `../pdlu/`.

The fused pooler/router reuses its original compiled extension and public
launch-setting API. Fixed depths are 5 at M16 and 3 at M32. The host setting
is restored to zero in `finally`, including launch errors. Kernel arguments
and attributes are captured by value; no graph-replay host flag is consulted.
This relies on sequential preparation/eager launch configuration and the
existing sequential scratch contract; broader host-thread safety is unproved.
Its arithmetic, scratch and BF16 hi+lo router calculation are unchanged.
The exact FP32 router at <=8 tokens remains parent.

## CPU evidence

`check_pdl_chain_cpu.py` compiles all four private kernels together. The
combined source passes with 72/42/109/88 registers (O16/O32/U16/U32), zero
stack and zero spills. U16/U32 floating-point instruction counts match their
parents. Buffer models cover the U16 24,576-byte raw ring plus 8,192-byte
reduction region; U32's four K partitions cover all 64 units, and its 512
output items have one writer each. U32 reduction storage fits in its unchanged
18,432-byte allocation. These integer models do not simulate async execution.

`check_fpr_pdl_cpu.py` binds the existing FPR source to the scored parent and
reproduces its two compiled instances: 85/126 registers, 1,024 static shared
bytes, zero stack/spills. Weight-copy instructions precede the producer wait;
activation TMA follows it, with a later all-thread wait. Dynamic branches mean
static instruction positions are not an executed-stage count. Source bounds
P by both the ring depth and each CTA's available stages.

The pinned stock norm source supports architecture-selected PDL and an
outgoing trigger. Copies and origin hashes are in `stock-norm-pinned/`.
Its trigger is near the final stores, so its presence alone does not establish
a useful FPR overlap window. These are historical pinned sources, not a fresh
verification of the currently deployed evaluator.

NVIDIA documents completion/visibility semantics in the [PTX grid dependency
control specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-griddepcontrol).
Its [PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
describes stream-captured programmatic edges and opportunistic overlap.
CUTLASS's [SM90 TMA kernel](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized.hpp)
waits before dependent TMA loads. These support the mechanism; they do not
validate this implementation or its performance.

## GPU gates and stopping conditions

All compiler threads join before new graph captures. New unit probes start
only while the preparation clock is below 274 seconds. Each eligible component
must match parent bytes on three changing graph inputs. The FPR probe includes
the real stock norm predecessor, residual output, logits and routing results.

There is one registered union of passing components per shape. No performance
mask sweep is performed. The whole layer must then match all output and cache
bytes and beat its exact parent by >3% in **both** interleaved cold-L2 arm
orders. No new diagnostic starts after 278 seconds. Failure, missing plans,
unsupported formats, nonzero local memory, no residency or budget expiry keeps
the parent. Preparation <300 seconds remains unverified on H200.

Actual-execution evidence must include P16_RESOURCE, P16_COMPONENTS,
PDLI_BITWISE/PDLI_TIME and P16_SELECT, plus private kernel markers in the
targeted profiles. An empty selection is fallback evidence. Full-context
3% gates still require approximately 6.4/9.4 microseconds at B16/B32; adding
three handoffs is not proof that they can save this much.

GPU admission remains false: no 16/32 measured overlap ceiling or whole-layer
gain has been established, and the last Decode check is reserved. Do not
publish this draft. Any eventual scored upload also requires 13/13 official
tests, integrity, targeted latency improvement and a clear gain over the
current 2.075466170x incumbent; the original prompt's 2.0510x is superseded.
