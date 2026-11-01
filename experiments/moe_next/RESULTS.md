# MoE next steps (branch argon/decode-next-20261030, from main 2e50482)

## Small-batch phase diagnostic (`63c17c9caa984361`, MoE check, timing only)

- Instrumented copies of the TINY (1 token), TINY2 (8 tokens) and MID (16 tokens) builds of the MoE board-best kernel (`board_best/moe_m3_scored_2.3797.py`). They use the existing `OKC_DEBUG` per-CTA globaltimer stamps.
- Inputs are skewed synthetic routings from the incumbent's own `_moe_routes`, two seeds per size.
- The timed path is unchanged: 5/5 passed, 2.3669x (incumbent code).
- The instrumented launches match the originals within noise: graph time 25.62 vs 26.19 us, 78.82 vs 80.27, 53.31 vs 53.30, 62.24 vs 62.38 and 107.42 vs 107.52.

Medians over 132 CTAs, in us:

| M (build) | experts / routes | span | route | phase A | phase B | barrier wait | final | tail after latest B |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 (TINY) | 2 / 2 | 22.5 | 2.0 | 11.4 | 7.0 | 1.3 | 0.5 | 1.6 |
| 1 (TINY) | 2 / 2 | 21.0 | 2.1 | 10.0 | 6.8 | 1.1 | 0.3 | 1.6 |
| 8 (TINY2) | 10 / 14 | 76.3 | 2.2 | 44.0 | 25.4 | 3.8 | 0.4 | 2.5 |
| 8 (TINY2) | 6 / 12 | 48.1 | 2.2 | 27.2 | 15.2 | 1.8 | 0.4 | 1.8 |
| 16 (MID) | 7 / 22 | 57.9 | 2.5 | 31.6 | 18.2 | 4.7 | 0.2 | 1.8 |
| 16 (MID) | 14 / 25 | 101.9 | 2.5 | 60.7 | 34.5 | 3.3 | 0.2 | 2.0 |

- **Streaming rate** (28.3 MB per expert: 18.9 MB A + 9.4 MB B):
  - at M = 8 / 16, phase A runs at about 4.3 TB/s and phase B at about 3.7-3.8 TB/s (B includes readiness waits);
  - at M = 1, phase A runs at 3.3 TB/s and phase B at 2.7 TB/s, because the phases are short.
- **Tail after the latest phase-B CTA:** only 1.6-2.5 us at small M (5.8-8.8 us at M = 32-256 in the earlier diagnostic `e66873fa`).
- **Decision:** removing the grid barrier with per-token finalization is worth at most about 2 us at the shapes where it matters most. **Not pursued.**
- **Remaining overheads, each small:**
  - routing prologue: 2.0-2.5 us on every shape;
  - launch: graph time is about 3.5 us above the span;
  - phase-A end spread: phase B waits for a job's slowest A CTA, 1.5-3 us;
  - phase B streams about 12% slower than phase A.
