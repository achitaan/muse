# Rotcodes re-audit: controlled addresses and mandatory flanking nulls

**Check file:** `candidates/dense_rotcodes_null.py`.
Branch `loop/bf16-rotcodes-null` starts at `404a116` (`loop/bf16-rotcodes2`).
This changes measurement/selection, not CUDA kernels or arithmetic.

## Finding: an identifiable protocol confound, not yet a GPU root attribution

The supplied sibling check `bf7cb2fb7aad4812` measured complete-incumbent vs
complete-incumbent nulls of 0.9497/0.9632 at o32, 0.9672/0.9771 at o16,
0.9824/0.9878 at un32, and similarly non-unit controls elsewhere. This is
large enough to explain prior apparent component wins.

The old `_rc_capture` / `_dt_abba` protocol couples all of these factors:

- incumbent always captured first, candidate second;
- final outputs allocated inside different graph-private pools;
- each graph replayed immediately after its own capture, so recency differs;
- incumbent always at ABBA's outer positions, candidate at the inner ones;
- only three eager warmups and one non-cold replay per graph;
- event objects first initialized while measurements are already starting.

There is additional concrete setup behavior: local `torch/cuda/graphs.py`,
`graph.__enter__`, synchronizes CUDA and calls both `torch.cuda.empty_cache()`
and `_host_emptyCache()` on every capture. The incumbent's own lower-level
`_capture` helper already documents this behavior and its measured 28 ms vs
1.2 ms capture overhead. The newer gates inadvertently used the high-level
context again. Repeated allocator resets/idle periods are not a neutral
way to assign the first and second algorithms.

These source facts plus the nulls prove that the OLD PROTOCOL is not an
identified algorithm comparison. They do **not** prove whether physical
buffer coloring, launch slots, cache writebacks, graph-instance effects or
clock transitions dominate on H200. No GPU is available locally, so this
attempt does not claim to have empirically isolated that hardware cause.

## Repair

1. **Exact common addresses.** Every graph in a draw reads one common,
   64-KiB-aligned input and writes one common, separately allocated aligned
   output. Twenty-four explicit runtime output-allocation sites in our own
   source delegate to `_ng_empty`. During our private preparation benchmark,
   a thread-local lease supplies that output instead of allocating another.
   Every call must use exactly one matching output allocation and return
   that exact pointer, shape, dtype and stride, or the cell is skipped.
   Unsupported internally allocating paths are never measured anyway.

2. **Production behavior unchanged.** There is no Torch monkeypatch. Without
   the explicit temporary lease, `_ng_empty` delegates to ordinary
   `torch.empty`. Every borrowed call still runs its FULL computation.
   Leases are restored in `finally`, isolated per thread, and cleared on
   preparation exit. No validation input, output or graph enters returned
   production state. Native CUDA strings and all numerical logic are intact.
   Non-native allocators and deterministic empty-buffer initialization are
   rejected, because borrowing must not remove real allocation/fill GPU work.

3. **Independent counterbalanced captures.** Four independent graph pairs
   per phase; two captured A->B and two B->A. Orders are seed-varied without
   consulting timings. Each graph is a fresh object: nulls never reuse a
   pair with reversed labels to manufacture reciprocal ratios.

4. **Direct capture and equal warmup.** One explicit side stream and direct
   `capture_begin/end`, avoiding the high-level allocator-cache resets.
   All graphs are captured before replaying any. All events are primed;
   every graph receives eight cold READ512 warmups before collection.

5. **Balanced event positions.** Panels rotate across batch positions;
   ABBA and BAAB alternate in complete eight-round cycles. With 32 rounds,
   every panel/role occupies each of the 16 event positions four times.
   Each phase has 128 quads, versus the old single-pair 40. Flush behavior
   remains the same allocating vector READ over 512 MiB before EVERY replay.

6. **Actual flanking nulls.** Every candidate draw performs independent
   base/base, base/candidate, then base/base phases: 24 fresh captures.
   Before AND after nulls must have BOTH median and paired ratios in
   **[0.995, 1.005]**, for **every raw panel**, each capture-order stratum,
   and their aggregate. Opposite 0.95/1.05 errors cannot average to a pass.
   Candidate ratios must still be <=0.975 in every panel on three fresh
   input/graph draws. No null rescaling, timing retries, outlier removal,
   fixed-overhead subtraction or threshold relaxation is used.

7. **No aliased correctness comparison.** Because benchmark outputs alias,
   correctness is checked against separate ordinary-call outputs, not by
   comparing two aliases. All captured graphs are rechecked against those
   independent values. The prior lossless, escape, FP64, probe and exact
   arithmetic/replay checks remain in the existing validation functions.

M=1 `_rc_measure` and ALL rotcodes2 `_r2_measure` calls use this protocol.
The optional old ILP-only timing diagnostic is disabled rather than left
unguarded. No historical rotcodes selection is carried forward: each new
prepare re-evaluates M=1, so invalid nulls can retract all previous wins.
Other legacy dispatcher/weight-layout bakes are unchanged; this is not an
audit of every historical autotuner in the base file.

## Reading the check log

The final `NG protocol` header and one compact line per measured arm are
retained at the end of preparation. `NG RC` identifies the M=1 re-audit;
`NG R2` identifies pooler/tail/SK2/router arms.

- `pre[min,max]`, `post[min,max]`: extrema across ALL raw, stratum and
  aggregate null ratios, both estimators, over all completed draws.
- `C{median/paired,...}`: aggregate candidate ratios for each draw.
- `cmax`: worst candidate ratio over all panels/strata/draws; this also
  must be <=0.975, not just the aggregate displayed in `C`.
- `ordP`, `ordQ`: first pre-null and last post-null capture-order medians,
  AB then BA. They expose surviving capture-order effects.
- `PASS`, `NO_GAIN`, `NULL_REJECT` or an explicit binding/error rejection.

If the pre-null is bad, candidate/post values are diagnostic only and cannot
be adopted. Residual H200 bias simply causes rejection; the implementation
does not force or calibrate the null into the band.

## Local evidence and remaining measurement gap

Run `scripts/verify_rotcodes_null.py` with CPU Torch. Passed:

- AST normalization proves only 24 explicit output allocation sites changed
  in the parent; native CUDA and numerical/production dispatch logic intact.
- Ordinary calls allocate fresh outputs; nested/exception/thread-local
  lease cleanup and six aligned-view shapes/dtypes pass.
- Exact role/event-slot balance across 16 independent scheduling seeds.
- Ten protocol tests: output-address and ABBA artifacts that falsely admit
  a 0.9524 old null become unit nulls; a real simulated gain survives;
  residual capture bias is rejected EVEN WITH UNIT AGGREGATE; post-null,
  incorrect results, unsupported binding, capture failures and incomplete
  deadlines all veto adoption.
- Mock graphs are all distinct, retain the COMPLETE incumbent (two fake
  GPU operations, not a shortcut), use identical input/output pointers,
  receive symmetric cold warmups and are checked without alias tautologies.
- M=1/R2 policy wiring rejects invalid nulls and returns `None`, not a
  false numeric winning score, to the rotcodes2 selector.
- Unchanged kernel regressions pass again: 132 SK2, 42 scalar and 10 router
  numerical models, lossless payload tests, and 100,000 compiled CPU
  two-row FMA-order cases.

Evidence: `results/rotcodes-null-offline.json`. These are **CPU protocol
models**, NOT H200 measurements. Native CUDA compilation, stream/capture
behavior, prepare duration, actual null band and surviving speedups remain
unverified. There is no new speedup claim. The stricter controls may reject
everything, and unsupported allocating paths are intentionally skipped.
No OKC operations, pushes or other-worktree edits were performed. The shared
ledger update is worktree-local and must be merged by the caller.
