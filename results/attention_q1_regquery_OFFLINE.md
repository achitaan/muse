# RQ-Q1: keep per-request decode queries in registers

**Status: offline-ready experiment, not a measured improvement.** One substantial
attempt; no CUDA execution, no okc check/submission, no push. Branch
`loop/attn-q1-register-query` starts directly at **63e253f**, not the PVT branch.

## Exact check payload

`candidates/attention_q1_regquery.py` is standalone. It retains the board-best
attribution and all original CUDA source strings and base production dispatcher.
The new translation unit is constructed by five fail-closed, readable prologue
edits plus one helper. Submit that file for a **check** only if the caller opts
to do so; this worker did not spend a GPU check.

Payload SHA256:
`d335fed2f52d1e5f693ac0585ec62f43178066ad1caa2694a0b356b691b8c2fb`.

## Bottleneck evidence and idea

The ledger's decode-1 timeline identifies several 0.4–1.1 us phases (row 171),
and its cold-instruction diagnostic rejects instruction-cache warming (row 182).
Source inspection identifies an untried avoidable query dependency in the Q1
per-request path of `attention_kb1h2_scored_2.6536.py`:

1. Raw queries: cp.async to shared memory (2 KiB per CTA).
2. Eight threads/head load and normalize them, rotate with the RoPE table,
   round to pre-scaled FP16, and store the fragments to shared memory.
3. A CTA barrier publishes those fragments.
4. Four threads/head in each of two key-half warps reload the fragments.

RQ-Q1 instead gives each consumer lane its eventual 16 query dimensions and
keeps the raw and prepared fragments in registers. It removes **10 KiB of shared
query traffic per CTA** (2 KiB cp.async writes, 2 KiB norm reads, 2 KiB FP16
writes, 4 KiB consumer reads) and the query-publication CTA barrier. The barrier
is retained when the CTA has new K/V to publish. All RoPE-table, initial
page-index/constant, and main-loop K/V acquire barriers remain.

This is a **latency / redundant shared staging** optimization, not a different
attention approximation. The two key-half warps duplicate Q loading/norm/RoPE;
the additional arithmetic and global load issue are real costs, not omitted
from the hypothesis. Both loads are 16-byte aligned and coalesced, with the same
2 KiB of unique Q data; the existing KV loads can hide the extra query work.

The original eight FP32 8-element partial sums are preserved. Adjacent partials
are combined in each lane, then two quad shuffles reproduce the original
three-level eight-lane binary tree. The epsilon, FP32 norm/RoPE/scaling order,
and FP16 rounding are unchanged. The original half-word permutation is produced
directly in the QK register fragments.

**Expected effect, not a result:** a plausible 0.1–0.25 us saving (roughly 2–4%)
on some short Q1 sliding shapes if extra query arithmetic stays under the KV
load latency. Flat decode-full B16/B32 is intentionally untouched. Q1 B1 full
may be neutral. A regression from duplicated work or register pressure is quite
possible; the candidate is not automatically selected.

## Why this is independent / not a closed repeat

The caller's `TRIED_main.md` Attention and Open ideas sections and the worktree's
`RESULTS.md` / `EXPERIMENTS.md` were read before implementation. Read-only
`tried.py search` searches used terms Attention/query, Attention/prologue and
Attention/register against that exact snapshot.

- **Row 191 / 0acc8428** kept *page indices* in registers. This leaves page
  staging and every cache address untouched and changes *query* staging only.
- **Row 174** closed one-query *FP8 QK*. This keeps FP16 queries, FP16 QK/PV,
  the same MMA instructions, scores and softmax.
- **Rows 189–190** changed softmax updates or removed split/merge. Neither is
  changed here; every launch plan and the entire merge protocol are retained.
- **Rows 197, 199–200** concern verify-full / PAGE=1 / physical-range access.
  This changes no KV traversal, cache layout, byte count, physical order or
  verification path.
- **01323f0 / PVT-Q1** changes the PV MMA axes. This changes no MMA axis,
  instruction or fragment interpretation in QK or PV. It branches independently
  from 63e253f; the PVT branch was not edited.

The configured `.tools/tried.py` is missing locally. `scripts/tried.py` was run
with its fetch/ledger functions overridden **in memory** for read-only snapshot
searches; its normal `add` pushes to main, forbidden for this lane. This branch's
`TRIED.md` contains the claim/result. **Caller must copy that row into the shared
ledger before spending a check.** No main checkout, ref or worktree was changed.

## Gate and local evidence

Command (the Python environment is read-only; temporary C++ files are created
and removed inside this worktree):

```sh
PYTHONDONTWRITEBYTECODE=1 \
  /Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse/.venv/bin/python \
  scripts/validate_attention_q1_regquery.py
```

**9/9 tests passed**, including:

- Actual new C++ helper compiled and executed with Apple clang++ C++20 under
  CPU BF16/FP16/shuffle shims: **258,048 packed words match bit-for-bit** with an
  independent model of the incumbent's norm, shared permutation and reload.
  Cases include distinct heads, zero/constant/random BF16, amplitude 2^-12 to
  2^12, positions 0/1/2048/80000/1000000, RoPE on/off, unit and 1/64 scaling.
- Dense FP64 RMSNorm + RoPE oracle: normalized/rotated half-fragment relative
  error < 0.0003. Zero and constant-query analytical solutions pass.
- Exhaustive pointer coverage: every dimension of every query is loaded exactly
  twice; every load is 16-byte aligned and within the Q slice.
- Original CUDA/host strings and production metadata/fallback guards unchanged.
  The entire QK/PV loop, cache update block, tile traversal and post-loop merge
  code are byte-identical. Source edits reject absent/duplicate anchors.
- No split sweep: exactly four RQ targets (d1S, d16S, d32S, d1F), plus re-gating
  the incumbent's reproduced v32S one-head arm; all other cases are controls.
- Mocked admission executes the actual selector. Both median and paired ratios
  must be <= **0.98** in the search **and** independent confirmation. Failure of
  either metric in either pass rejects RQ. The inherited whole-cache exactness
  and no-worse global/row error checks remain unchanged.

Python AST and CPU-only Torch import also pass. On CPU, import reports
`skip no-cuda-or-capture` with an empty selection table; no GPU timings are
fabricated. The original 512 MiB **read** flush and ABBA graph timing protocol
are retained. Actual calls add only shape dispatch, not tuning, host sync,
printing, output caching, or retained inputs/results.

The candidate-versus-incumbent whitespace delta check passes. A whole-new-file
staged check needs `core.whitespace=-blank-at-eol`, because byte-preserving the
original CUDA strings retains their pre-existing trailing spaces.

**Not verified:** CUDA translation-unit/host-extension compilation, NVRTC/PTX,
ptxas, registers/spills, Hopper rsqrt/FMA/instruction scheduling, device memory
ordering, actual graph replay, judge tolerances and performance. No nvcc,
NVRTC or ptxas is installed/discoverable on this machine. Host helper tests use
FP contraction off and model the reduction tree, not Hopper's instructions.

## Remaining risks

1. Twice the query arithmetic and extra LDG issue can erase the avoided staging.
2. Compiler register allocation could regress occupancy or spill; unmeasured.
3. Compiler FMA selection may differ despite identical expression/tree order;
   the strict no-worse GPU error gate may reject it.
4. The inherited 85-second import budget includes another extension build, so
   late builds or comparisons can time out safely to the incumbent.
5. Passing a CPU shim is not a substitute for the caller's H200 check. No
   speedup has been established; stop this idea if all four RQ arms fail.
