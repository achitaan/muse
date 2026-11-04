# Attention check-log profiler (diagnostic only)

Candidate: `candidates/attention_sliding_diag.py`

The first 411,544 bytes are exactly `board_best/attention_kb1h2_scored_2.6536.py`
(SHA-256 `6b65dec7c1aaeb4675815555ddbb0ee8a0de17ab93e56523c3d1d386899998ca`).
No incumbent Python/CUDA code, tuning, dispatch or production entry point was
changed. There are no diagnostic JIT builds. Do not submit this file to the board.

## Execution

Stateless attention has `kernel_fn(data)`, not a weights/prepare interface.
An appended module-tail adapter calls an idempotent `prepare()`; all additional
CUDA work runs inside that function, in try/except. A later explicit prepare call
is a no-op. Deadline: **280 s from incumbent import start**, including the original
85 s tuner; 20 s of the stated 300 s limit is reserved for cleanup. Late incumbent
builds cause a diagnostic skip, not an unbounded compilation/join.

Synthetic inputs reuse the incumbent's public-layout/distribution generator:
80K context, agentic length spread for B16/B32, page size 1, scattered physical
pages, soft sliding keys and coherent full values. These are the twelve official
shape classes, not the judge's private inputs/seeds. Each captured incumbent
graph gets eight event-only controls and eight CUPTI-profiled replays. Every
replay follows a **512 MiB read-reduction flush**, outside its timing interval.
Flush-only calibration identifies and removes all flush kernels. Missing graph
activities are errors; there is deliberately **no misleading eager fallback**.

Scratch/constants/property-cache dictionaries are isolated and restored by
identity; RNG state is forked/restored. No inputs, graphs or outputs survive.
Kernel choices remain the original import-time tuner's choices for this device.

## Reading the log

- `ATTN_PROF <case> <kernel> us=... n=... pct=...`: mean summed device duration
  for this named kernel per replay, mean launches per replay, share of summed
  device durations. Names retain template arguments; none are truncated.
- `ATTN_PROF_TOTAL`: `total_us` sums GPU durations; `span_us` is first GPU start
  to last GPU end; **`gap_us = span_us - total_us`**, signed. PDL overlap can make
  it negative. `idle_us` is actual uncovered span; `overlap_us` is sum minus
  interval union. Event/control times include the graph launch envelope;
  `profiler_delta_us` diagnoses CUPTI perturbation. `outside_span_us` is NOT
  counted as possible savings (it includes the common replay envelope).
- `ATTN_PROF_STAGES`: gather/main attention, separate split merge, separate
  QK-prep/norm/cache-write, and other. **Zero separate prep/merge time does not
  mean zero fused cost.** Most incumbent routes do those operations internally.
- `ATTN_PROF_FLOOR`: unique causal KV-union bytes / 4.8e12 bytes/s, in us.
  Sliding verify reads the union of 2047 old + 8 new positions, not 8 copies of
  the window. Full uses actual synthetic lengths + L. Both caches, two KV heads,
  head dim 64, FP8 one byte are counted. `cached_kv_bytes` excludes new tokens
  that can be forwarded in shared memory. This is a semantic read-once floor,
  not measured HBM traffic; it excludes QKV/output/page metadata, writes,
  transaction amplification and duplicated reads. `pct_of_floor = 100*span/floor`;
  `hbm_peak_pct = 100*floor/span` is the inverse efficiency measure.
- `ATTN_PROF_NOMERGE`: for a recognized two-kernel main->merge graph only,
  optional raw CUDA graph APIs disable its terminal merge node, replay on the
  same **synthetic scratch** buffers eight times, verify main-only activities,
  then re-enable it in finally. No CUDA source is changed. Recent PyTorch raw
  graph handles and `cudaGraphNodeSetEnabled` are required. Fused routes, old
  PyTorch, or unavailable APIs report `status=unavailable`, never a fake timing.
  Output is intentionally wrong and never returned to a production call.
- `ATTN_PROF_SUMMARY`: per-case potential = max(0, gap + separate merge +
  separate prep); this nets out overlap. Weighted sums use 46S + 16F per load.
  Each load's hypothetical time subtracts half its weighted potential;
  `half_combined_board_gain_x` is the geometric mean of all six load ratios.
  The gap/merge/prep-only halving ratios are separate hypothetical bounds, not
  additive predictions. Incomplete six-load results print `NA-incomplete`.

## Headroom interpretation (pending GPU evidence)

The source already has one-launch fused paths for sliding decode and verify-1,
and optionally selected one-head verify-16/32. Only the unselected VFG route has
an externally removable merge. Thus a zero launch-gap/separate-merge report
does **not** rule out substantial sliding prologue, duplicated QK prep, atomic/
cooperative merge or last-CTA tail cost inside the main kernel. That would need
a follow-up matched-code phase diagnostic, not a launch-fusion claim. Conversely,
if VFG survives the tuner and shows a large exposed merge or dependency bubble,
the no-merge control localizes that overhead without a new kernel build.

Full-context high `hbm_peak_pct` would corroborate the limited bandwidth-only
upside. The tiny sliding KV-only floor shows why latency/parallelism and fused
tail costs can dominate there. No new performance conclusion is claimed before
the check log exists.

## Offline verification

`python3 scripts/validate_attention_sliding_diag.py`

Tests parse the AST without importing torch/submission, enforce exact prefix and
addition-only diff, exercise flush exclusion and signed-overlap accounting,
KV union byte formulas, actual kernel-name stage grouping, 46/16 weighting and
six-load geometric gains, native API call shape/topology/reversible disable with
64-bit fake handles, all-case line formatting/log-tail size, and cache restoration
under injected profiler/stream-join failure.

The copied incumbent contains pre-existing trailing whitespace, intentionally
preserved byte-for-byte; new appended lines are checked separately as clean.

**No local NVIDIA GPU, nvcc, or nvidia-smi. No check, submit, push, or GPU run
performed. CUPTI collection, native API availability and elapsed budget remain
unverified until the caller runs a check.**

Ledger helper was absent. The authoritative `origin/main:TRIED.md` was searched;
this diagnostic differs from the closed cold-instruction-cache and launch-fusion
ideas by measuring CUPTI launch spans/stages of the current exact incumbent.
Only this branch's ledger is updated, because the task explicitly forbids writes
to main or another worktree. Caller must reconcile its row to main before a check.
