# L1 flat Attention: fixed128-key/two-stage draft

This is CPU-screened development work, with no GPU job, installed candidate,
measured speedup or rank claim. Retain Attention9e922c35 and DecodeD9 unchanged.

## Sources and scope

Baseline Attention SHA256:
`9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f`.
Draft SHA256:
`7ef8ab72409ef250979fcd14802af38ec3b7e966f9c662d3ce535f650eced553`.
CUDA SHA256:
`20aeed18f70ef0f55e75e412f1d1af69142a94f618de166601e36cc58a1fd70f`.

Full draft, baseline/new CUDA literals, eight exact inverse edits, CPU report,
compiler inputs/options/logs/disassembly and resource review are in
`results/attention-l1-wide-tile-preflight-20261029/`.
The builder also proves the remaining Python AST unchanged except its extension
module tag. This draft still needs host resource admission and a final binding
before an official evaluation; its historical baseline result does not belong
to this source.

Only L1 flat kernels change: the existing planner selects them for full context
with2..MAX_BATCH requests. L8, all non-flat kernels, fallbacks, planner, virtual
key partitioning and finalizers retain their original behavior. This includes
nearby unscored shapes; there is no case-specific performance selector.

## Mechanism and limits

The retained L1 flat path stages64keys and gives each key-half warp32keys.
The draft stages128keys and gives each half64keys. Each half performs two
FP8 K32 PV operations, preserving every key and original scale256. Per-key QK,
exponentiation, value contribution and cache representation remain required.
The maximum/rescale/denominator update and producer stage handoff occur half
as often for long runs. This is an algorithmic blocking change, separate from
the failed identity-rescale branch and already-retained sequence-length
balancing.

The fixed pipeline depth is two, chosen to keep the memory model within two
blocks. Nominal data-stage capacity rises from192 to256keys. It is not a
Cartesian tile/depth sweep. Producer and consumer thread counts stay128each;
register allocation limits stay128base/56producer/200consumer. The page-index
window grows from80 to144entries per slot, and its TMA transaction uses the
local tile width. All local tile/stage offsets and the cross-segment tile
counter use the same width.

Logical cache bytes per useful key are unchanged. Do not claim fewer HBM bytes
or a measured synchronization bottleneck. Extra registers, shared-memory
traffic, larger code and different tail utilization can erase a benefit.

The saved D9 profile reports substantial full-attention kernel durations at
M16/M32, but those are separate profiling-pass durations, not critical-path
fractions. This draft affects two of six standalone weighted loads through
their full-context rows. Any gain is diluted on the whole layer and cannot be
translated into an overall rank without the unavailable aggregation model.

## Precision and CPU evidence

BF16 Q/K norms, BF16 RoPE materializations, FP8 caches, BF16 attention-output
materialization before norm/gate, final BF16 store and large-L8 fallback remain.
The existing source precision guard passes. Increasing the running-maximum
group changes internal probability rounding and summation order: bitwise
equivalence is not asserted. All keys are still processed, including tails.

The450finite CPU ownership/bounds cases cover producer chunks, QK/PV key
ownership, aligned page-index windows and sequential stage reuse. They do not
prove CUDA barrier memory ordering or deadlock freedom. The42NumPy numerical
proxies preserve the existing256-scaled positive E4M3 PV probability boundary.
They remain finite; worst new/oracle relativeL2 is3.545%, also reached by the
baseline on that fixture. Worst new/baseline difference is3.037%. These are
arbitrary-score mathematical screens, not official test results or an exact
emulation of CUDA MMA/exp2/norm/gate rounding. No tolerance pass is inferred.

Offline NVRTC13.0.88/sm90a compiles baseline, wide and disabled-wide forms.
All instantiated kernels have zero stack/spill fields. Flat L1 keeps128base
registers. Compiler sizeof probes report84992dynamic shared bytes including
the retained1024-byte slack, versus67584baseline. Actual occupancy and dynamic
register redistribution still require H200 admission.

After normalizing only compiler-generated local symbol names and labels, the
disabled draft's three instantiated instruction bodies equal the retained
baseline. L8-flat and L1-nonflat instruction bodies also equal the baseline in
the enabled draft. Full disassembly is retained; this is not runtime evidence.

## Before the single focused GPU check

Add a source-bound native resource marker and require actual H200 zero local
memory, at most128base registers, at least two active blocks, exactly84992bytes
dynamic shared memory and the intended128-key/two-stage specialization. Freeze
the final source hash, expected profile markers, fresh official task/reference/
checker/baseline identity, capacities and result guards before uploading.

Keep the existing cached9e result as baseline; no redundant baseline job.
Require full16tests/12benchmarks/integrity, all numerical/cache/gate boundaries,
and real new-path execution on the affected full-context L1 cases. An invalid
or ambiguous path/resource admission cannot establish performance.

Conservative proposed retention gates, to freeze before launch: affected
geometric ratio>1.04, affected/control-normalized ratio>1.04, each affected case
normalized>=1.02, unchanged controls within[0.95,1.05], and reconstructed six-load
track ratio>1.015. Use actual46/16load weights. These engineering gates are not
a paired-job confidence interval. A failure retains9e without a noise rerun,
selector, score or Decode transfer. Preserve raw evidence and failed history.

Transfer, if justified later, must apply the tile/pipeline geometry to D9's
original FP16 PV body and preserve its numerical boundaries. Do not import
standalone FP8 PV as a claimed improvement; its earlier publication was tied.
Then require complete whole-layer validation before any Decode publication.

All work remains in the main chat. Current pacing/reserve/window policy still
applies; an available evaluation slot is not a reason to launch an incomplete
candidate.
