# Same-route MoE headroom diagnostic — production unchanged

Standalone check file: **`experiments/moe_headroom/moe_headroom.py`**.
Its first 185,756 bytes are exactly `board_best/moe_m3_scored_2.3797.py` from
`63e253f`. Only new lines are appended. No JOINFULL, EXACTDIV, SS64/32, source
patch, flag change, kernel_fn override, build-selection change or new production
backend is present. A new prepare wrapper returns the original state object,
without new keys or selections. Scratch undergoes ordinary incumbent launches
and resets, just as it already does in the original prepare-time selector.

## What runs in prepare

After the original prepare (including its original selector), a separate tiny
extension builds a once-read TMA stream. The existing `_moe_routes` helper and
the same x-before-routes draw order generate judge-like, busiest-rank routes.
Seed=20261104. Measure **8 draws each at M16/M32** and **4 each at M8/M128/M256**.
Counts are the actual D/R/histograms of **these diagnostic draws**, not estimates
from SOL, and not observations of the judge's later timed inputs.

Each draw uses identical routes/weights for incumbent and stream. The incumbent
is the unmodified `kernel_fn`, including its actual selected build. The stream
reads the same physical allocations it uses:

- WGS: `state['w'][12:14]`;
- TINY/TINY2/MID: unit copies `[10:12]`;
- non-WGS MAIN fallback: fat `[8:10]` for experts with >8 routes, unit copies
  `[10:12]` otherwise.

The stream uses the incumbent's L2 weight hint, grid size and a 3 x 36 KiB TMA
ring. Per expert it reads **18 MiB gate/up + 9 MiB down**, codes and full E4M3
block scales together, **once**. Ordinary loads additionally read 8 B of global
scales per touched expert and the 8 KiB effective BF16 norm weights once. The
full TMA byte count must complete before reusing a slot; one word per 128-byte
line is sampled into a checksum, with lane-varying word offsets to avoid shared
bank conflicts. No full dequantization/MMA/activation work is
performed. The checksum is diagnostic-only and never a MoE output.

Eight **ABBA quads** compare each draw. Every arm READ-flushes 512 MiB using
`amax()` outside its CUDA-event interval. The buffer is written only once at
setup, then read/synchronized once before measurements. Each quad's A and B
times are the means of its two symmetric repetitions; reported draw values
are medians over quads. There is no timing subtraction, L2 write-flush, input
reuse across draws, output replay shortcut or incumbent/stream selection.

## Reading the output

`MOE_HEADROOM` prints one line per completed draw:

- `case`, `M`, `draw`, `D`, `R`, `hist`: case, draw index, unique local experts,
  local route pairs and the full 32-expert route-count histogram.
- `bytes`: once-read **logical weight bytes**, `D*27MiB + D*8 + 8192` (D>0).
- `inc_us`, `stream_us`: raw single-node CUDA-graph event times, including the
  common replay/dispatch cost. Not directly identical to judge-adjusted times.
- `unique_TBps = bytes/(inc_us*1e6)`: effective unique-weight rate of the full
  MoE; **not measured DRAM traffic**, especially if jobs duplicate experts.
- `stream_TBps = bytes/(stream_us*1e6)`: the matching stream's logical rate.
- `inc_pct_stream`: median paired `100*inc/stream`; 110 means the incumbent
  takes 110% of stream time. Weak summaries also give its reciprocal
  `floor_eff_pct = 100*stream/inc`.
- `paired_gap_us`: median paired `inc-stream`; common dispatch cost cancels.
- `dup_jobs` (draw) and `job_bulk_bytes_med` (summary): estimated excess
  <=8/64-token jobs and their raw code+block-scale bytes. These distinguish unique-byte streaming from
  expert re-streaming at the large verify shapes. `layout`/`hint` identify the
  exact physical weight representation and cache policy measured.
- `flag=above_HBM_peak` (>5 TB/s) or `stream_slower` marks suspect floor data.

`MOE_HEADROOM_SUMMARY` aggregates the **paired per-draw** gaps/ratios, not a
comparison of unrelated routing medians. `MOE_HEADROOM_WEAK` repeats decode16
and decode32 gap, reducible-time percentage and D/R ranges at the end so they
survive log-tail limits. `MOE_HEADROOM_BOARD` shows the counterfactual gain if
stream times replaced incumbent times, using the **1/6** board exponent and
leaving unmeasured loads unchanged. It is **not an achievable MoE speedup or
a universal hardware upper bound**: this stream has its own launch/ring costs,
and omits activation traffic and all math/dependencies/finalization.

## Time budget / failure containment

Diagnostics skip before compilation if original prepare used >195 s. They stop
starting work at 270 s total, leaving ~30 s margin; only >=4 complete balanced
quads are reported. Weak cases run first. The extra extension is small, but
native `load_inline` compilation is not interruptible; an unusually slow build
remains a timing risk. All diagnostic Python exceptions, including logging
errors in the error handler, are swallowed without changing returned state.
A device fault/hang cannot be made recoverable by Python try/except: CUDA code
is locally uncompiled/unrun, so that remains the explicit GPU validation risk.

## Local gates and interpretation

```sh
python3 -B experiments/moe_headroom/test_headroom.py
```

Tests parse the whole candidate, prove the exact base prefix and addition-only
diff, validate physical source selection/byte counts, enumerate chunk ownership
and reciprocal splits (including rounding perturbations), model ring parity,
mock ABBA/deadlines/failure handling, and check read-flush event placement.
No Torch or GPU is needed for them. No CUDA/NVRTC/ptxas compiler or GPU exists
locally; **actual stream bandwidth, correctness and prepare duration are not
verified**. The supplied JOINFULL profiles remain untouched/untracked.

My prior: row 97 (`7f733a2b`) already measured 6.1–6.5 us per extra touched
expert (~4.4–4.6 TB/s), so no real >=3% board gain is currently established.
For just the two weak loads, a 3% board gain needs the product of their speedup
ratios >=`1.03^6=1.19405` (about 1.093 each). If their paired stream gaps fall
below that even before adding math, stop redesign work. Verify gaps may include
duplicate jobs; closed wide-job/L2 attempts are not reopened by this diagnostic.
Only a repeatable same-D/R residual sufficiently above that threshold would
justify another implementation. This file gathers that missing evidence; it
does not itself improve the production path.

The local TRIED row requires parent merge into the shared ledger. No main
commit/push, no other worktree edits, and no OKC check/submit was made.
