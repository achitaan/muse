# PVT-Q1: transpose the PV MMA axes, not the cache

**Status:** offline-ready experimental candidate; no GPU execution, no okc check
or submission, no speedup established. Branch: `loop/attn-pv-transpose`.

## Payload

Submit exactly **`candidates/attention_q1_pvt.py`** for an okc **check** if the
caller elects to spend one. This is standalone and retains the board-best file's
attribution, original base dispatcher, safe fallback and import-time admission.
The author of this branch has not submitted any check and has not pushed.

## Evidence and change

In `board_best/attention_kb1h2_scored_2.6536.py:1674-1680`, the Q1 PV loop feeds
`mma_f16` eight times per 16 keys with A fragments `(P0, 0, P1, 0)`. Only eight
of the 16 MMA query rows are useful. Its epilogue consequently reads only
`o[...][0:2]`, discarding the other two accumulators.

The candidate leaves QK and softmax alone. It computes the *same* PV product as
`V^T @ P`, using 16 dimensions as MMA M and eight heads as N:

- four useful m16n8k16 MMAs, instead of eight half-empty ones, per 16 keys;
- two existing `ldmatrix.x4.trans.b16` instructions, instead of four LDS.64
  instructions, to get the same FP8 V bytes into register fragments;
- byte permutation and exact FP8-to-FP16 conversion; **P stays FP16, V stays
  exact FP16, and accumulation stays FP32**;
- per-head alpha broadcasts for the transposed accumulator lanes;
- the same 18-float/ lane warp-half exchange, with four alpha broadcasts;
- eight lane shuffles plus one scale broadcast at the split epilogue, repacking
  four float4 values per lane into the **unchanged** workspace row layout;
- identical logical key traversal, split plan, page indexing, cache updates,
  row-sum path, atomics/TMA merge, counters, fences and finalizer.

The expected tradeoff is fewer tensor/load issue instructions versus extra
once-per-split packing and per-tile alpha shuffles. The long Q1 sliding splits
(especially decode32 sliding) are the most plausible target. A **2–5% shape
gain is a hypothesis, not a measurement**. Decode-full is memory-bound, so it
may gain nothing. One-tile latency can regress. Rejected arms retain base.

## Not a closed repeat

The ledger snapshot used was the caller-provided `TRIED_main.md`:

- Attention row 177 / `0898ae00` moved the **Q8 FP8 V transpose to producers**;
  this is Q1, uses no producer work/new staging/extra barrier, and keeps FP16 P.
- Row 174 closed **one-query FP8 QK**; QK and query precision are unchanged here.
- Rows 189–191 closed lazy softmax, direct-finalize and private partials;
  none of those algorithms/protocols is changed here.
- Rows 199–200 concern physical-range traversal/staging; physical cache access
  order and bytes are untouched here.

The real helper's `search` entrypoint was run in read-only mode against the
provided snapshot (terms: Attention/PV, Attention/transpos, Attention/q1).
The `.tools/tried.py` path is absent on this machine. Its `scripts/tried.py`
replacement normally fetches/pushes main; those operations were disabled to
honor this lane's no-main/no-push scope. The PVT-Q1 row in this branch's
`TRIED.md` is the local claim/result; **the caller must merge it into the shared
ledger** before spending a check.

## Local validation

Command (Python with NumPy; main-worktree environment is read-only):

```text
PYTHONDONTWRITEBYTECODE=1 \
  /Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse/.venv/bin/python \
  scripts/validate_attention_q1_pvt.py
```

Ten tests cover:

1. all 254 finite E4M3FN encodings convert exactly to FP16;
2. every modeled ldmatrix/A-fragment coordinate, both heads, both key halves,
   both K steps, every dimension tile; all row addresses 16-byte aligned;
3. old/new PV products versus an independent FP64 dense P@V oracle, with
   uniform, one-hot and random probabilities; bit-identical old/new CPU models;
4. distinct per-head alpha rescaling, warp-half merge and final v4 packing;
   every workspace coordinate written exactly once;
5. stable online softmax over lengths 1/17/31/32/33/63/64/65/129/2048/2055,
   including empty second halves, partial tiles, multiple splits, unit and
   1/64 V scales; relative error below 0.001 versus dense stable softmax;
6. fail-closed source anchors, unchanged QK/Q8/protocol regions and exact
   production dispatch/metadata guard;
7. host Clang syntax check of the **new helper C++ only**;
8. mocked admission: both median and paired ratios must pass **0.98** in the
   search **and** an independent confirmation. Either metric failing either
   comparison rejects the new layout.

Also: candidate/test Python AST parsing and real CPU-only Torch import (safe
`skip no-cuda-or-capture`, no selection) pass. The candidate delta passes
`git diff --no-index --check` against the incumbent. A plain staged whitespace
check reports the incumbent's existing trailing whitespace because this is a
new standalone copy; it is intentionally retained for byte-identical original
CUDA strings. The staged check passes with only `blank-at-eol` disabled.

Payload SHA256: `ead89b63affb5e2293fda0a0a1a56dbb1d2c0333b6da416a04fbbecd3a2b3222`.

The CPU model uses the documented PTX fragment mapping, sequential FP32
accumulation and a dense oracle. It does **not** verify Hopper's actual
instruction accumulation tree, assembler constraints, resource counts,
instruction scheduling, GPU memory ordering, runtime correctness or timing.
`nvcc`, NVRTC and `ptxas` are not installed/discoverable locally. Full CUDA
compilation and actual graph replay are **unverified**, not passed.

## Admission and remaining risks

Only one novel route is tested, for the six Q1 scored layouts, without any
split/knob sweep. The incumbent's reproduced Q8 v32S `1h-s4` arm is re-gated;
the other closed one-head extensions are removed from the search. The base
production kernel is unchanged. Each PVT arm must synchronize safely, preserve
whole-cache bytes, pass the judge-like error bound, and have global/row errors
no worse than base before timing and after repeated graph execution.

All timing uses the inherited 512 MiB **read** flush and ABBA graph protocol.
Both metrics must be <=0.98 in both search and independent confirmation.
Kernel calls contain no new synchronization, input-dependent host work or
output caching. Only device scratch/constants and shape choices survive.

Main risks: actual ldmatrix/MMA codegen and resource use have not been compiled;
added shuffles can erase savings; identical algebra need not be bit-identical
on tensor cores, so the strict error admission can reject it; the inherited
85-second import budget could expire before late full-context arms. No broader
kernel redesign was attempted after this single substantial implementation.
