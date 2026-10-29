# CPU screens during the factor-trial pacing wait

No additional GPU jobs, API calls or speed measurements. These screens do not
enable a kernel or alter an incumbent. Preserve with the factor trial's final
review after its foreground controller releases the repository.

## Tiny route construction

`screen_moe_tiny_route.py` binds retained MoE33114871. Its one-warp model uses
entry peer masks, one leader per expert, a warp prefix and stable entry ranks.
It matches an independent grouping reference across85620 cases: exhaustive
65536 four-symbol patterns at M1,20000 randomized patterns at M1..4, and all
local counts0..32. Expert counts/offsets, stable entries/positions, fat and
skinny jobs and fat queue order agree. This does not establish CUDA execution,
synchronization safety, resource use or performance. Duplicate patterns verify
metadata equivalence; they do not establish baseline GPU support for duplicates.

The original route has four CTA barriers. Its per-warp count workspace is not
read outside route. Under retained FAST1, early_issue is a no-op; the independent
weight prefetch at kernel start must remain unchanged. A future implementation
could keep route work in one warp, then publish all metadata at one final CTA
barrier. Warp synchronization around shared count/offset publication remains
necessary. This is a mechanism to review, not an implemented CUDA candidate.

Only one of six scored MoE loads is M1. A1.5% whole-track improvement from just
that load requires a1.093443x load improvement, removing about8.55% of its time.
The retained M1 benchmark is about24.85us, so that means roughly2.12us recovered
after charging all new work. No admitted trace establishes that route work has
this much recoverable latency. Park the idea; do not spend a GPU check solely
because the CPU mapping passes.

## Decode transfer binding

`audit_factor_decode_transfer.py` proves that retained MoE33114871 and DecodeD9
contain exactly the same139873 CUDA bytes, SHA256
1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c.
Their CPP interface and16 listed dispatch/layout constants also agree. Thirteen
packing, eligibility, geometry and scratch helpers have identical ASTs after
namespace normalization and docstring removal. Neither prepare nor the complete
runtime wrapper is claimed AST-identical by this audit.

This supports a precise transfer route if the standalone factor trial wins:
replace the eligible MAIN build only, retaining old small and ineligible-scale
paths. The alternate CUDA can use the existing lossless source loader. Avoid
compiling both fast MAIN versions inside Decode's preparation budget.
Before reusing one scratch bank across small and large shapes, prove every new
valid read is preceded by a completed write, invalid routes remain masked,
fixup/counter offsets remain exact, and alternating-shape replay is safe.
The standalone trial's two separate banks do not prove shared-bank integration.
Whole-layer correctness, cache updates and weighted latency still need their
own authorized full check after a genuine retained standalone gain.

## Specific FP4 shortcut rejected

A finite exact-cover search found no four pairwise-disjoint cyclic rotations
of the8-bit placement mask0x81C081C0 that partition all32 source bits. There
are48 disjoint triples among16 distinct masks. This rejects only a four-mask rotation-only shortcut;
it does not prove a global instruction lower bound or rule out other bit
permutations or instruction families. Do not reopen a GPU flag experiment
from this finite screen.
