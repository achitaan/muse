# One bounded attempt: M16/M32 SK2 owner-warp decode

## Handoff

- Branch: `loop/dense-m1632-sk2rot`, from `63e253f`.
- **Exact check file:**
  `/Users/achita.anantachina/.local/share/opencode/worktree/e72615/loop-dense-m1632/candidates/dense_sk2_rotcodes.py`
- No OKC calls, submissions, pushes, or writes to any other worktree.
- This is an independent copy of `board_best/dense_kbv2_scored_1.5886.py`.
  It does **not** include the other lane's M1/M8 changes. The incumbent AST is
  identical apart from additional exports in `_make_vjeux`'s final return.
- Scope is only the eight `(qkv_gate/o/pooler/unpooler, M16/M32)` SK2
  specializations, using their existing owner warps, geometry, barriers,
  escape tables, FP32 MMA/reduction and BF16 output stores. No scalar GEMV,
  router, neighboring-shape or other subsystem changes.

## First-pass bandwidth and launch/tail analysis

Measured times below are the exact values in the supplied
`dense_cg_rotcodes.ec0ea8d1555443ce.txt`, not the slightly different rounded
times in the task brief. The fixed graph replay subtraction is already in
the reported `yours` times; **do not subtract another 4 us**.

| Cell | Exact 12-bit W stream | Time, us | W-only TB/s | % of 4.8 TB/s | W-only floor, us |
|---|---:|---:|---:|---:|---:|
| qkv_gate:16 | 28,311,552 B | 9.70 | 2.919 | 60.8% | 5.898 |
| o:16 | 12,582,912 B | 5.22 | 2.411 | 50.2% | 2.621 |
| pooler:16 | 50,331,648 B | 13.98 | 3.600 | 75.0% | 10.486 |
| unpooler:16 | 50,331,648 B | 13.79 | 3.650 | 76.0% | 10.486 |
| qkv_gate:32 | 28,311,552 B | 9.80 | 2.889 | 60.2% | 5.898 |
| o:32 | 12,582,912 B | 5.76 | 2.185 | 45.5% | 2.621 |
| pooler:32 | 50,331,648 B | 14.97 | 3.362 | 70.0% | 10.486 |
| unpooler:32 | 50,331,648 B | 14.65 | 3.436 | 71.6% | 10.486 |

These are achieved **useful encoded-weight bytes/time**, not a claim of
measured HBM transaction bandwidth. Escape-table bytes, input/output,
transaction overhead and repeated activation reads are not included. The
weights are streamed once at 1.5 B/value. The first eight cases in the
analysis JSON contain the reproducible arithmetic.

The profile has **one kernel per target cell**: qkv16/32 use `rq`, o16 and
pooler16 use `gv`/`gv_predxor`, and o32/pooler32/unpooler16/32 use `sk2`.
There is no extra finalizer node or inter-kernel gap to eliminate. The
standalone SK2 source also has no `griddepcontrol` operation; decode-layer
PDL findings cannot simply be transplanted here. o32's profiler duration
is 4.9 us vs its timing-table 5.76 us, but the aggregate profiler does not
provide a GPU timeline, so that difference cannot be labeled an actual
launch gap or internal stall. Startup, drain and epilogue remain unmeasured.

The o stream is 12 MiB, not an empty near-launch-floor kernel: its 2.62 us
stream floor still leaves ~2.6–3.1 us above that ideal. However, this is a
small latency window and a different warp/epilogue topology is too risky
without an internal timeline. Ranked options:

1. **Selected:** remove integer reconstruction work in the existing owner
   warps, with no communication changes. M1 already supplies GPU evidence
   that this specific decode reduction can help another kernel family.
2. Remove a launch/finalizer or tune PDL: no such extra node is present.
3. Change geometry, split strategy or epilogue: closed prior attempts or
   unsupported by the available profile, so not repeated.

## Representation change, and why it is not a closed repeat

Every existing SK2 B unit contains 512 unmodified low bytes plus 256 code
bytes (two uint32 words per lane, eight weights per word). Rotate only the
sign fields of each word in untimed preparation:

```text
old low-nibble signs:  3+8i -> 7+8i
old high-nibble signs: 7+8i -> (11+8i) mod 32
all 24 exponent-offset bits: unchanged
```

The two high-byte reconstructions become:

```c++
ha = (nw & 0x87878787u) + base4;
hb = (__funnelshift_r(nw, nw, 4) & 0x87878787u) + base4;
```

The window base is at most 120; each byte's maximum is 255, so no carry
crosses byte boundaries. The actual Torch transformer checks the inverse
on **every word**, checks that every low byte stayed fixed, and refuses
malformed metadata. Existing original-BF16 escape payloads, corrections and
row permutations are reused unchanged. No smaller exponent format or
lower-precision computation is introduced. Storage stays 12 bits/value.

This is the incumbent chenwainuo register-A family's `dec1` representation,
not a novel codec claim. Unlike the other lane's scalar GEMV work, it is
applied to the **SK2 TMA/shared-memory B-fragment owner-warp decode path**.
There is no scalar cgemv code or scalar row-major packing change.

Relevant ledger rows, searched in the shared copy before implementation:

- `ee7d7b81` / `cbd91613`: sub-12-bit coding increased decode/escape work.
  Here the byte count and escape work are unchanged and decoding shrinks.
- `83c46449`: CURP/register-A ST/KS geometry sweeps gave no new cell wins.
  There is exactly one existing SK2 geometry per cell here, not another sweep.
- `33c77700`: new simple GEMV family was 1.7–10x slower. No new math family,
  stream scheduling or tiling is introduced here.
- `8ba14018820847fd` (caller-provided result): token-owned epilogue was
  slower. That communication/epilogue change is absent; source equality
  verifies the complete original epilogue and barriers.
- `ec0ea8d1555443ce`: M1 scalar rotcodes evidence is supporting prior art;
  these eight SK2 specializations were unchanged in that experiment.

## Gate, expected effect and stopping rule

Preparation overlaps one optional eight-specialization extension build
with the **unchanged** parent prepare/tuner. o32 is evaluated first, then
o16, unpooler, pooler, qkv. The comparison is always against the complete
incumbent selected by the parent, including any admitted `rq` arm—not just
against the old SK2. This is important for qkv, o16 and pooler16.

Each cell needs **three independent fresh-input/fresh-graph draws**. Every
draw checks exact equality with the original SK2 twin, independent FP64
accuracy and agreement with the full incumbent on normal, outlier, two
one-hot batches and zero inputs. It checks replay stability, runs the
existing 40-block ABBA with a 512 MiB vector READ flush, then validates
again with fresh inputs. Both median and paired ratios must be **<=0.975
in every draw**. Bad builds/packs, errors, budget exhaustion or insufficient
gain leave that cell on its unchanged incumbent. Only winning weight
packs/module metadata remain in returned state; no inputs, references,
graphs, outputs or losing packs are retained. Overall budget is 285 s from
prepare entry; actual GPU build/prepare duration is unverified.

At source level this removes about **three integer operations per eight
weights**, without changing PRMT, tensor-core work or memory traffic. o's
full matrix removes ~3.15 million thread-instructions; pooler/unpooler
~12.58 million each. Compiler fusion/overlap can hide all of that. A
**1–4% per-cell improvement on active SK2 cells is a plausible hypothesis,
not a measured claim**; the strict gate deliberately rejects small/noisy
gains. Already-fast `rq` and `gv` routes may win and remain selected.

One H200 check is appropriate. If nothing is adopted, close this exact
attempt rather than weakening admission or sweeping geometries. This is
the one substantial attempt for this lane; no OKC check was spent locally.

## Local verification

```sh
PYTHONDONTWRITEBYTECODE=1 \
  /Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse/.venv/bin/python \
  scripts/verify_dense_sk2_rotcodes.py
```

Evidence: `results/dense-sk2-rotcodes-offline.json`.

- Incumbent AST preserved except factory helper exports.
- All eight generated specialization lines match the parent's native
  specialization lines; the entire CUDA SK2 body is literally identical
  after reversing helper/kernel renames and the generated dispatch.
- **63,438,848** exhaustive BF16 payload/position/window cases and
  **1,000,034** random/basis code-word permutations passed.
- Actual Torch repacker matched an independent NumPy permutation on
  **64,064 words**; five malformed pack/base cases were rejected.
- Actual old/new decoder expressions compiled under clang++ C++20 and
  passed **7,929,856 word/window cases** with CPU PRMT/funnel semantics.
- Actual B-fragment packing, stage-major stream and escape tables were
  reconstructed via independent index formulas across all eight native
  geometries (two CTAs each, native K). Every payload reconstructed exactly,
  with **2,752 escapes**, and qkv's balanced row maps were checked.
- **32 numerical models**, using real SK2 K-warp partitions, NACC selection
  and escape-warp order: worst relative L2 **0.0017792** vs FP64. Tested
  one-hot BF16 outputs and zero outputs exact. NumPy dot16 rounding is
  **not** an emulator of actual Hopper `mma.sync` rounding.
- Eight fresh-output wrapper pairs and **48 fallback cases** passed.
- Actual three-draw gate control was exercised with CPU graph mocks:
  successful admission, exact-twin failure, post-timing corruption and
  marginal-win rejection. Actual prepare control was exercised for build
  failure, selective adoption and deadline after an earlier valid winner.
- Python AST/compile and `git diff --check` passed.

**Not verified:** CUDA host/device compilation, NVRTC/ptxas, registers/spills,
SASS instruction counts, barriers or actual GPU graph execution, H200
numerics, total GPU prepare time, or performance. `nvcc` and `ptxas` are
not on PATH; `ctypes.util.find_library` finds no NVRTC/CUDA/runtime library.
CPU C++ compilation is not CUDA compilation. No speedup is claimed.

`TRIED.md` is updated only in this scoped worktree. The shared helper was
absent on this machine; the caller must merge its parked/result row into
the shared main ledger without changing another lane's worktree.
