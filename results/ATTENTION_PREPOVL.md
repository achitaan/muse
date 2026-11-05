# Pair q1 prep-overlap candidate

Check file: `candidates/attention_prepovl.py`, branch `loop/attn-prepovl`.
Base: `43a0f06`; incumbent byte prefix unchanged (411544 bytes).

## Evidence and scope

Phase check `6a1bc00e6e494c2a` measured 1.36/2.26/2.72 us prep at d1S/d16S/d32S.
Inspection found both proposed broad changes already present: cached KV is issued
before Q/K math, `has_new` skips new-KV prep for nonowning splits, global cache
writes are range-owned, and q1 normalization uses all 128 threads.

The new change targets the remaining prep serialization, not those existing
optimizations. Only q1 non-flat instances change:

1. Stage inverse frequencies/scales/cache locations with the page-index group.
2. After that group's existing wait and CTA join, compute identical RoPE angles
   independently in each warp while raw Q/KV and cached-KV copies are outstanding.
3. Full-warp shuffles distribute each thread's four angles BEFORE divergent K/V
   branches. Reuse those registers in Q and new-K math, with unchanged arithmetic.
4. Remove the Q-copy CTA join (each q1 raw chunk is read by its issuing thread)
   and shared-RoPE-table CTA join. Preserve per-thread copy waits, all commits,
   final prep barrier, new-row patch, mainloop, merge, and cache-store ownership.

The candidate compiles with `-DATTN_PREPOVL=1`, C++20 and SM90a in a separate
in-process `load_inline` extension. Macro default is zero. An independent phase
extension records one cold replay, excluded from admission timings.
Targets: d1S, d16S, d32S, d1F. VP2/one-head routes remain incumbent.

## Admission

Independent synthetic data only, import-time idempotent prepare; no call-time
tuning. Each shape gets two passes of 16 ABBA blocks with a 512 MiB READ before
every replay. Both median and block-paired ratios must be <=0.985 in BOTH passes.
Each pass also runs actual-incumbent vs independent actual-incumbent graph ABBA.
Both null log-ratio centers must lie within the measured paired-log IQR around
zero, and that IQR must be narrower than the 1.5% margin. Bad nulls fail closed.
Reference/judge tolerance and poisoned-new-slot whole-cache byte checks run
before and after timing. Only fully confirmed keys enter the production table.

Logs: `ATTN_PREPOVL ... old_us new_us ratio paired null null_paired null_spread
null_ok exact valid admitted`. `exact` means whole K and V cache bytes.
`ATTN_PHASE ... variant=prepovl` reports the instrumented sampled owner's phases.

## Validation / limits

- `python3 scripts/validate_attention_prepovl.py`: passed.
- `python3 scripts/validate_attention_phase_diag.py`: passed (incumbent diagnostic regression).
- AST parse and process-start grep: passed; no forbidden APIs in check source.
- No local GPU or nvcc: CUDA compilation, numerical/cache equivalence and speed
  are **unverified** until the caller check. No speedup claimed.
- Main ledger untouched per caller restriction; branch-local `TRIED.md` needs
  reconciliation after the check. Nothing submitted or pushed.

Risks: replicated trig and eight live angle registers may outweigh barrier
savings; the conservative null gate can reject small genuine improvements.
One instrumented replay is directional evidence, not a stable timing estimate.
