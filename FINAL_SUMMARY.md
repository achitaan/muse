# Final summary (Muse Open Kernel Challenge, branch argon/decode-next-20261030)

Stopped by user decision on Nov 2, 2026 at $195.49 of the $300 budget.

## Board bests (published on `main`, `board_best/`)

| Board | Best scored | Job | Rank at stop | #50 at stop |
|---|---:|---|---:|---:|
| Decode layer | **2.1056x** (was 2.0970) | `9dabcd9a` | 362 / 1175 | 2.2980 |
| MoE | 2.3797x | `ac80911e` | 202 / 1302 | 2.4652 |
| Attention | 2.5690x | `c8f75600` | 306 / 1246 | 2.7208 |
| Dense | 1.5278x | `6131bc96` | 333 / 1619 | 1.5883 |
| Overall | - | - | 362 / 1834 | - |

Decode source: `board_best/decoder_p36_scored_2.1056.py` (= `experiments/top50/decoder_p36_trig.py`).

## What helped
- Decode PDL/early-start work at 16/32 tokens (sk2 PDL, early-trigger norm) and small bit-identical MoE MAIN tweaks
  (sibling-job L2 hint, earlier weight-slot release, finalizer pass 2 from registers): each 0.5-2% on some shapes.
- Read-flush in-prepare A/B gates (judge-like) for the Decode per-shape selections; the 2.1056 scored run.

## Main lesson
Our diagnostic harnesses and the Decode in-prepare gate flushed L2 by *writing* 256 MB; the judge flushes by
*reading*. Write-flush inflated small-batch MoE times (bs1 31.1 vs 26.0 us) and produced false gains
(e.g. the TINY L2 hint). Always reproduce the judge's measurement method first.

## Closed (all in TRIED.md with numbers)
- MoE: ~20 kernel changes (EARLYW, BTAIL, wide jobs g=2/N80/N96, L2 hints, sibling interleave, skinny refill,
  strided units, x-direct, register re-split, ...) and two fresh designs (small-batch FS kernel, large-batch
  re-split/wide jobs): all flat or slower.
- Attention: q8 fpart path, FA4 exp2 emulation, index prefetch, launch-plan autotune (write and read flush);
  cold instruction fetch measured as not a cost.
- Decode: PDL twins at 16/32 tokens, glue PDL at 128/256, trigger-after-wait variants: no gain.
- Dense: exhausted earlier (cuBLASLt sweeps, custom WGMMA at 1.09x of cuBLAS).

## Why we stalled
Incremental tuning of a shared public baseline reached a local optimum; real gains (0.3-1%) were below the
judge's +-1.5% noise; no local H200/profiler, so every experiment was a judge check; large redesigns came late and
lost to the tuned incumbent; the MoE kernel (35-50% of Decode) stayed ~16% behind the board leaders.
