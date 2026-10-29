# MoE disjoint partial fixups: CPU draft parked

No GPU check, scored submission or active-candidate change was made for this
study. Keep retained MoE33114871, checked2.365437416x and scored2.348299935x.
The current native CUDA draft is
`26575142c54068507cdd961c31ea391ca9ae80b7fd9c45c361cd8ae78f1b7db7`.

Baseline source SHA256:
`331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.
Baseline native CUDA:
`1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c`.

Full inverse edits and CUDA drafts are under
`results/moe-private-fixup-preflight-20261029/`. Compiler inputs, flags,
adaptations, logs and complete compressed disassembly are in its `compiler/`
directory. The latest ownership model is
`results/moe-private-fixup-model-20261029/v4/model.json`.

## Evidence that changes the opportunity screen

The official generator adds heavy-tailed expert popularity and rotates the
busiest shard into the local expert range. Uniform independent routing is not
an appropriate estimate of active experts or traffic for the challenge. The
packed weight codes themselves are uniform random nibbles; do not assume
entropy compression from quantized-normal sparsity.

The retained implementation already lets phase B acquire individual phase-A
slices through `doneAs`; there is no global A-to-B join to remove. Its fixed
whole-unit alignment alternative also has negative saved CPU tail-cost evidence.
Neither is a new optimization proposal.

The separate source of possible overhead screened here is split-unit fixup:
contributors atomically add FP32 vectors, and a completing warpgroup reads and
zeroes the buffer before applying the retained epilogue. The public-formula-
inspired saved draws generally have around120split units per phase on the
normal32/128/256-token cases. Their logical atomic payload is only a few MiB
per phase. These are synthetic operation counts, not measured HBM traffic,
atomic cache transactions or evidence of exposed latency.

## Fixed ownership proposal

Give each CTA two disjoint endpoint slots per phase. A leading partial uses
endpoint0, a trailing partial endpoint1. A range lying inside one unit uses
endpoint0. Every contiguous range has at most two partial units, so these
slots cannot alias within a phase. Reordering the trailing A unit does not
change that ownership.

Keep original counter and `doneAs` addresses. Endpoint0 reuses the original
value area. Endpoint1 occupies an additional20MiB after the original fixup
and metadata region. The first contributor to a split unit stores endpoint1;
later contributors store endpoint0. Once the existing completion protocol
admits a finisher, it reads contributors in increasing CTA order, adds in FP32,
and invokes the unchanged BF16/global-scale epilogue. All 16 stages along K remain.
No output atomics, new input host reads or persistent expanded experts are added.

The original atomic implementation expects its value area zeroed. This draft
leaves disjoint values there. Therefore an incumbent and candidate must use
separate scratch allocations in any future paired comparison. Sharing weights
is fine; alternating these algorithms on one scratch state is unsafe. An
additional20MiB is the new state's increase, not the total comparison allocation.
Keeping both states live would require roughly157+177=334MiB of scratch, plus
all weights, inputs, layouts and transient preparation allocations.

For a future production wrapper, scope the draft to EP8 PRMT MAIN and retain
all original TINY/TINY2/MID, non-PRMT, other-shard and nearby-shape fallbacks.
That wrapper and its prepare-time admission have not been implemented.

## Publication, replay and progress obligations

Each contributor drains its WGMMA accumulators, writes its unique FP32 slot,
then reaches the retained128-thread named barrier before its leader publishes
the stage count. A finisher must observe every contributing stage through the
original GPU-scope acquire/counter protocol, pass the warpgroup barrier, and
read only freshly written contributor slots. All invalid/unused slots may
remain poisoned; none may be read or multiplied by zero as a substitute for
masking. Phase banks are separate.

The logical dependency graph gains no new spin wait: completion waits and
phase readiness remain the retained ones. Still, the native draft has no GPU
memory-order, liveness or numerical validation. Require all original stages,
arrivals, generation resets and final grid completion. Never weaken those
conditions to make a prototype run. Sequential same-state calls are the
retained contract; concurrent same-state calls are not established.

NVIDIA's [PTX memory model](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#memory-consistency-model)
defines barrier synchronization, release/acquire patterns and observation
through atomic RMW chains. It supports the ordering argument above; a passing
finite CPU model would not establish that a CUDA implementation satisfies it.

The1638CPU ownership cases check endpoint uniqueness, exact stage coverage,
contributor ranges, fresh writes across generations, chunk384+1 handling and
the precomputed lookup used by the finisher. Revision1's mislabeled balanced
remainder and unsplit385 stress setup are preserved and superseded. Revision4
also checks a division-free last-owner walk against the retained inverse.
It relies on the unchanged bound `total_cost * grid < 2^32` and covers grids
through160. These are mathematical/layout screens, not numerical GPU checks.

## Bounded compiler revisions and decision

Both BIG variants report the same following offline NVRTC13.0.88 fields:

| Form | Base regs | Stack bytes | Spill-store bytes | Spill-load bytes |
| --- | ---: | ---: | ---: | ---: |
| Retained | 96 | 16 | 56 | 124 |
| Disjoint partials, inverse owner in finisher | 96 | 184 | 768 | 1216 |
| Division-free last-owner walk | 96 | 88 | 612 | 992 |
| One scalar accumulator tile | 96 | 32 | 136 | 304 |
| Precomputed endpoint owners | 96 | 24 | 100 | 204 |

Each correction addresses observed register liveness, not a geometry sweep.
The final form computes endpoints during range setup, before matrix
accumulators are live, and appends16bytes of shared metadata without changing
original shared-field offsets. Compiler sizeof probes give22248bytes for
Smem and231168bytes for the full dynamic allocation, unchanged after alignment.
No C751x serialization warning appears. All original weight packing, matrix
geometry and numerical epilogues remain source-identical; the FP32 reduction
order intentionally changes.

The final local-memory burden is still above the retained proxy. Offline
stack/spill fields are not actual H200 LocalSizeBytes or measured traffic;
this does not prove a runtime regression. However there is no admissible
critical-path evidence that removing these atomics would overcome the added
owner setup, contributor reads and local-memory risk. Park without a GPU
ticket. Do not retune registers, commit-group flags or geometry merely to
force admission. Reopen only with a distinct compiler result or credible
exposed-time evidence. All verified sources and current quota remain unchanged.
