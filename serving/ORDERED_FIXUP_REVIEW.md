# Ordered partial-buffer follow-up: pre-GPU review

Baseline: commit 6aa03df, frozen CUDA SHA256
`3eb38b8d658e3becd0d314c2ce4ad717786b9859c7570cee4392458411340736`.
The original source, patch, results, and empty acceptance set remain unchanged.

## Hypothesis and ownership

Replace CAS additions only in the WGS split-unit handoff. Keep routes, job order,
weight loads, GEMMs, BF16 boundaries, whole-unit epilogues, counters and finishers.
Each CTA owns a contiguous half-open stage range [a,b). Its first and last units
are the only possible partial units; both can be the same unit. Endpoint 0 stores
the partial whose unit is a>>4, and endpoint 1 stores the other partial. Reordering
execution of those two units does not alter their assigned addresses.

For each phase/CTA/endpoint/warpgroup/element there is one writer. Each split unit
has 16 stages, so at most 16 nonempty contributors. The finisher enumerates CTAs
from the exact owner of stage 0 to the owner of stage 15, ignores empty/nonoverlapping
ranges, loads only the job's valid n-tiles, and adds in increasing owner/stage order
using explicit FP32 RN addition (including subnormals). No numerical equivalence
with stock's differently partitioned GEMM is inferred from determinism.

## Publication and replay

Vector partial stores precede the existing 128-thread barrier and release counter
publication. Phase A retains the acq_rel last-arrival election; phase B retains the
designated finisher and acquire wait. The finisher's own stores are included. The
existing dependency graph and resident-grid requirement are unchanged. All arrivals
are published before reading contributor storage. The counter is reset after the
epilogue. Existing phase/final joins must complete before the next serialized call.

Partial values need no clearing: every read addresses a nonempty contribution from
the current call, whose publication was acquired, and no endpoint is reused within
that phase. Phase A/B banks are disjoint. Zero-local-job calls read no partials.
Retained graphs share scratch only under the existing serialized owner contract.
These arguments require GPU replay and sanitizer probes; CPU tests are not a race proof.

## Memory and cost

Allocate 2 phases * 160 CTA slots * 2 endpoints * 4 warpgroups * 4096 FP32 values:
40 MiB, replacing the prior 20 MiB CAS bank. Other counters/layouts are unchanged.
One bound layer: 196,135,200 bytes. Sixty layers: 196,180,512 bytes (~187.1 MiB),
below 256 MiB/rank. No persistent expert-weight bank is added.

Each contributor writes and the finisher reads 65,536 bytes for an eight-n-tile
partial unit across its four warpgroups; smaller jobs scale with n-tiles. Counter
traffic remains. This replaces atomic read/modify/write plus final reads/clears,
but introduces contributor enumeration and serial FP32 additions. Added launches: 0.
Complete-operation timings must include all routing, finalization and stock combine.

The CPU model checks exact 16-stage coverage, unique endpoint writes, contributor
ordering and the source's corrected inverse owner mapping. It covers 1,284 cases
including empty, singleton, hot and skewed jobs, both phases and grids 1..159
(selected sizes). Maximum observed contributors: 11; mathematical upper bound: 16.
An initial audit omitted the source's integer inverse correction loops; that audit
was corrected before GPU execution. No kernel change was made to those loops.

## Bound and decision

Only T128 is initially authorized for candidate GPU probes. Advance to T256 only
after correctness passes. T32 stays stock; no routing-dependent eligibility.
Unresolved differences are serving-equivalence failures, not toleranced passes.
At most two focused corrections, $7 for this hypothesis, no replacement hypothesis.
Retain only >=10% complete-operation gain in every one of three captured blocks
for uniform/mixed/hot routes, beyond controls, with all correctness gates passed.
