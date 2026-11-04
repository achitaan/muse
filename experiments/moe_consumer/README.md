# JOINFULL: WGS stage-completion coalescing

**Status: offline candidate, not a measured speedup. No OKC check/submit run.**

Submission candidate: `experiments/moe_consumer/moe_joinfull.py` (standalone,
readable CUDA embedded as a raw multiline string). Base:
`board_best/moe_m3_scored_2.3797.py`.

## Evidence and hypothesis

The supplied ledger has already closed ring-depth changes, early slot release,
producer reordering, epilogue variants, KPERM3, SC6, and multiple fresh engines.
In particular:

- `TRIED_main.md:82`, job `c967ffa3`: MAIN phase B's weight producer is
  back-pressured 83% of its time; consumers spend 66% in dequant/MMA and 15.5%
  in weight/x waits. Instrumentation perturbed that build, so these are only
  directional measurements.
- `:52`, `7044c658`/`0414170f`: earlier weight-slot release did not help.
- `:63`, `79036669`/`22bb61fd`: re-splitting consumer warpgroups did not help.
- `:24`, `63c17c9c`: removing the final grid barrier at small M did not help.

**Distinct change:** remove one of the two CTA-local *full-slot* wait loops per
WGS stage, not any global completion flag, slot-release wait, grid barrier,
copy, computation, or route. No matching ledger row was found by searching
`barrier`, `completion`, `joined`, `combined`, `mbarrier`, and `ring` in both
the supplied ledger and the checkout's ledger. This does not overlap the
five-bit-scale, global-readiness/PDL, duplicate-job, or epilogue lanes.

The existing three-slot weight and x rings use identical stage ordering and
slot cycles. Each full barrier now starts with **two arrivals**. The weight
producer arrives with 36 KiB expected and copies to it; the x producer arrives
with its own `npad * 512` expected and copies to the same barrier. A consumer's
single wait cannot complete until both arrivals and all bytes complete. The
independent weight-empty and x-empty barriers, and their release points, stay
unchanged. No extra storage, registers, global accesses, or arithmetic state
is introduced at source level. Actual compiler resources are unverified.

The candidate also ports the already-kept zero-local-job guard (`:73`, job
`2cba38a5`); this is a correctness fix, not a new optimization claim. The
original prepare-time build selector is retained unchanged to isolate host
behavior; it may choose a non-WGS build, where JOINFULL has no effect. Its
limitations are documented in ledger rows 40/42/44/252. Do not mistake a
different build selection for evidence supporting JOINFULL.

## Correctness invariants

1. Weight producer, x producer, and consumer traverse the same phase-A reorder
   and phase-B order, with one continuous stage sequence across both phases.
2. Weight and x slot counts are equal (compile-time `static_assert`).
3. Each producer arrives exactly once, with its own expected byte count,
   **before** its own TMA copy. This remains valid when the first transfer
   completes before the other producer arrives: the second arrival is still
   pending, so the phase cannot complete.
4. Full-slot phase completion protects both TMA destinations. Weight reuse
   still requires all consumer warps to finish reading weights; x reuse still
   requires all consumer warps' outstanding WGMMA reads to retire.
5. Every dequantization, MMA, accumulation order, BF16 rounding boundary,
   epilogue, normalization and routing expression is unchanged. NVFP4 packing
   and BF16 activation/weight storage are unchanged. Atomic timing can still
   change the incumbent's non-bit-repeatable stream-K summation.
6. Zero local jobs generate zero stage ranges for both producers and consumers.

## Local gates

Run from the worktree root:

```sh
python3 experiments/moe_consumer/test_joinfull.py
git diff --check
```

The test suite checks:

- 874 exhaustive single-phase arrival/partial-completion interleavings against
  the independent predicate "both copies issued and both transfers complete";
- 600 control and 600 joined-ring traces, with four independently progressing
  consumer warpgroups, two producers, partial transfers, 1–65 stages, slot
  wraparound, random segment drains and continuous phase parity;
- schedule-independent direct dot products of integer-valued payloads, plus
  payload-version checks before use and before releasing each slot;
- deliberately broken initial count and x destination are rejected by the
  emulator (premature readiness/deadlock respectively);
- 2,014 producer/consumer order cases, including empty ranges, short split
  segments, phase-A reorder and phase B;
- exact host-AST equality except the CUDA string/extension cache name;
- an exact whitelist of changed CUDA lines (no arithmetic changes);
- byte-identical **preprocessed** TINY/TINY2/MID/fallback sources, each with
  PRMT scales enabled and disabled. SDK stubs only support preprocessing;
  this is explicitly not CUDA compilation.

Machine: ARM64 macOS, Apple Clang 21. No Torch, NumPy, nvcc, ptxas, NVRTC,
NVPTX Clang target, Linux container runtime, or local GPU was available. No
full-size MoE numerical reference, GPU tolerance, CUDA graph replay,
deadlock/sanitizer, compiled-resource, prepare-time, or runtime-speed gate has
been performed. CPU emulation is not proof of GPU correctness or performance.

## H200 check handoff (parent only)

Exact candidate to check: **`experiments/moe_consumer/moe_joinfull.py`**.

The compile-time control is `-DOKC_JOINFULL=0`. This restores the two original
full barriers/waits while retaining the same zero-local-job guard. For a
resource-matched paired diagnostic, build the ordinary MAIN define tuple plus
this flag using `_ext`, and call `okc_run` directly with the same state/inputs
as the ordinary MAIN module. Do not compare different prepare-selector
decisions. Keep ordinary M=1/8/16 paths as negative controls.

Required GPU gates: original correctness suite and all-remote/mixed graph
replay; compare MAIN `numRegs`, local/spill bytes and shared memory; cold-L2
**read-flush** ABBA on the same many routing draws at M=32/128/256; inspect
nearby M=33/257 too. Do not accept an aggregate score movement alone. Stop
after one check if per-draw paired medians do not support a repeatable >=2%
gain on at least one target shape.

Expected effect is deliberately uncertain: fewer barrier-test/branch/address
instructions on MAIN, potentially low-single-digit improvement if their
consumer issue cost is exposed. It does not remove the real DMA/MMA waits.
A sub-2% or zero result is plausible; there is **no evidence yet of a board
win**, and this is not a reason to force another tuning series.

## Scope / ledger

The prescribed `.tools/tried.py` helper is absent. `scripts/tried.py` exists
but fetches/pushes main; the lane expressly forbids main commits and all pushes.
Consequently this branch's `TRIED.md` records the local claim/result, and the
parent must merge the row into the shared main ledger. No other worktree was
modified. Official `okc init` was used only to read the MoE TASK/AGENTS locally;
the generated problem/harness files are not part of this change.
