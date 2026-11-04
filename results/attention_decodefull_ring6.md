# Decode-full: one bounded six-stage gather-ring attempt

## Artifact and scope

- Branch: `loop/attn-decfull-gatherpipe`, based on `63e253f`.
- Check file: `candidates/attention_decodefull_ring6.py` (self-contained).
- New route: **only d16F/d32F**, q_len=1, full, PAGE=1, 16 Q / 2 KV heads,
  head_dim=64, scored coefficients/layout/device.
- No OKC check/run was submitted; no push or main commit was made.
- The shared helper was absent at `.tools/tried.py`. The available
  `scripts/tried.py` pushes to main by design, so it was not invoked to write.
  The claim and offline result are recorded in this branch's `TRIED.md` only;
  the caller must merge the ledger row using its authorized workflow.

## Cheap bandwidth calculation

Read-only inputs: `.tools/okc show muse_attention_decode`, scored source
`board_best/attention_kb1h2_scored_2.6536.py`, and the baseline JSON read with
`git show loop/attn-redesign:results/attention_redesign_baseline.json`.

**KV read bytes = sum(seq_lens) x 2 caches x 2 KV heads x 64 dims x 1 FP8 byte.**
GQA shares these reads across eight Q heads: do NOT multiply bytes by 8.
The new token is produced locally in shared memory and written to the cache,
not read as another historical token. Full context is not the 2048-token window.

At the scored nominal context **80,000 cached tokens/request**:

| case | useful historical KV bytes | incumbent us | useful TB/s | /4.8 TB/s | ideal byte us |
|---|---:|---:|---:|---:|---:|
| d16F | 327,680,000 | 86.4889 | 3.78869 | 78.93% | 68.2667 |
| d32F | 655,360,000 | 165.7343 | 3.95428 | 82.38% | 136.5333 |

Neither nominal case crosses the requested 85% stop threshold. These rates use
the reported **overhead-subtracted** judge times; gross replay times give lower
rates. The public layout has 80K +/-7.5% per-request lengths, and the baseline
does not expose a particular run's exact seq_lens. Exact per-draw historical
bytes are therefore **not recoverable from the baseline JSON**. The nominal
numbers are not represented as a measured exact HBM transaction count.
At all-74K/all-86K bounds, the useful rates are d16F 3.505/4.073 TB/s and d32F
3.658/4.251 TB/s. Page-table reads add ~1.56% before accounting for cache reuse,
and Q/new-KV/gate/workspace traffic is small relative to KV.

## Evidence and selected hypothesis

The incumbent already has:

- **264 CTAs = 2 x 132 SMs**, flat, equalized work over requests, not naive
  per-request ceil-divided splits.
- Virtual 400-token cost at request boundaries to compensate extra prologues.
- Three 64-key K/V stages, four specialized producer warps, four consumer warps.
- **TMA bulk page-index prefetch** 3 tiles ahead in four slots. KV itself is
  random 128-byte rows, gathered by `cp.async.cg` 16-byte chunks, not contiguous
  runs suitable for bulk/TMA tensor boxes.
- FP32 merge, last-arriver finalization in the same launch. There is no separate
  merge kernel/launch gap to remove; standalone full decode uses no PDL.

CPU partition calculations for three independently seeded public-distribution
draws show d16F tile means 75.66–77.65 with maxima 76–78; d32F means 150.39–152.34
with maxima 151–153. Tail imbalance is small (max <1.4% above mean), not a 10%
explanation. These are analytical tile counts, not measured CTA runtimes.

The remaining candidate lever is the **three-deep KV ring's gather-latency
coverage**. Double it to six stages, preserving partition, logical key order,
query representation, all arithmetic, new-KV writes and FP32 merge exactly.
This is a hypothesis to test, not an asserted measured root cause.

Prior real-PAGE1 streaming diagnostic `be84a6c4` measured ~3.96/4.15 TB/s raw
random-row reads; relative to current useful rates this suggests only a small
ceiling. **Expected effect is unmeasured, plausibly 0–5% on the targets** if
gather stalls are exposed, about 0–1% on the attention board. Regressions are
plausible and are rejected by setup admission. A 10% target gain is not promised.

## Why this is not a closed repeat

References are to the supplied `TRIED_main.md` snapshot:

- Attention row 179 explicitly suggests deeper gather pipelining after a
  copy-only ceiling diagnostic. Process row 251 (`b7a92e20`) and Attention row
  199 (`be84a6c4`) supply random-gather ceilings, including real PAGE1 patterns.
- Rows 169/186 (`a7c4e6e6`, `129ffe91`) change runtime launch plans, not this
  compile-time ring depth. **No split-count sweep here.**
- Row 181 (`a4bd8523`) changes q8 page-index prefetch. This leaves page-index
  TMA unchanged and specializes **q1 KV** storage/in-flight depth only.
- Rows 177/180/183/188/194 concern verify transpose/exponent/index/head paths;
  there is no verify-full candidate here.
- Row 200 (`dbb907cf`) stages physical-order traversals; this does not build an
  inverse map, stage a reordered cache, or change any logical traversal.
- Row 197's broad floor diagnosis remains a serious negative prior. This small
  controlled depth test does not claim that diagnosis was disproved.

The incumbent's scored v32S one-head route is independently re-gated as before;
its already-closed full-context one-head arms are **not** retried during setup.
Other untaken tuning arms were removed, not replaced by new sliding/verify work.

## Admission and fallbacks

- JIT builds run during module import, never on actual call inputs.
- Three independent synthetic PAGE1 scattered/coherent input draws per batch.
- On **each** draw: synchronized uncaptured probe, graph capture, poisoned
  new-KV slots, whole-cache byte equality, public-reference whole/row error,
  proximity to incumbent and essentially non-degraded reference error.
- Two independent 16-block READ512 ABBA passes/draw. **Both** gross median and
  paired ratios must be finite, positive, and <=0.988 on every draw/pass.
- 33 further graph replays and poisoned-new-KV recheck.
- Only fully confirmed batches publish. Build failure, safe resource rejection,
  numerical rejection, slow/noisy timing, unsupported layout, or budget exhaustion
  retains the incumbent; unsafe device faults are surfaced, not hidden by fallback.
- Setup has a 220-second total-import budget. No inputs/outputs/graphs survive
  setup; only modules, device identity, selected batch numbers and numeric
  diagnostic summaries survive. Scratch is initialized normally for real calls.

## Local verification

Run `python3 scripts/check_attention_decodefull_ring6.py`.
Machine-readable output: `results/attention_decodefull_ring6_cpu.json`.

- AST and Python compile: passed.
- Original CUDA_SRC byte-for-byte equal to scored incumbent: passed.
- Generated device code delta: **only q1 flat STAGES 3->6**; host launcher also
  narrows instantiation and checks occupancy/errors.
- Native C++20 layout: 111,616-byte Smem +1024 alignment slack =112,640 bytes/CTA;
  2 CTAs use225,280 of227,328 available shared bytes. Non-flat/q8 layouts unchanged.
- 2,750 randomized ring schedules: exact logical row sequence, tail zeroing,
  consumer new-token patch, stage reuse/parity, multiple wraparounds: passed.
- Independent dense-vs-tiled full GQA attention/gating model: max abs error
  8.89e-16 (double precision model, **not** a CUDA tolerance claim).
- 41 metadata rejection cases, 7 mocked publication/budget/device-fault policies,
  8 timing gate edge cases, 5 equivalence gate cases: passed.
- Incumbent-to-candidate delta whitespace check: no new whitespace warnings;
  all other staged files pass `git diff --cached --check`. A whole-file staged
  check reports the incumbent CUDA strings' inherited trailing spaces, retained
  deliberately for byte-for-byte source identity (not new experiment code).
- CUDA/NVRTC/ptxas/H200 conformance, compiled register count/spills, occupancy,
  CUDA graph correctness and timings: **not run** (no torch/CUDA/toolchain here).

## Remaining risks / caller action

Submit exactly the check file above if this bounded test is worth one check.
Inspect `DF summary` and build diagnostics; do not infer a win from board-level
noise. The only production override is a fully admitted d16F/d32F route.

The six-stage ring has only 2 KiB shared-memory slack at two CTAs/SM; actual
device occupancy must confirm register/static-smem constraints. Deeper queues
may not help because the producer or memory system already saturates, and the
larger carveout can regress performance. CPU models cannot prove CUDA memory
ordering or instruction scheduling. The existing global scratch/cached launcher
state inherits the incumbent's single-stream, single-device invocation assumption.
