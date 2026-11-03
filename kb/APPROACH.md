# How we approach the Decode problem (achitaan, Nov 3)

Goal: the judge's Decode score, not microseconds. Small local wins only matter if they move log(score) on enough loads,
and isolated wins often vanish after integration. We pick experiments by **expected gain in log score per check**,
adjusted for correctness risk, and we prefer a few large structural bets over many 0.5% tweaks.

## 1. The objective, exactly
For load j (d1, d16, d32, v1, v16, v32): R_j = (46 B_jS + 16 B_jF) / (46 T_jS + 16 T_jF), Score = (prod R_j)^(1/6).
B = SGLang time, T = ours (judge-adjusted). `python kb/muse_score.py` replays this exactly (79/79 saved jobs).

Required uniform latency reduction on all six loads (from 2.2347):
| Target | Reduction |
|---|---|
| 2.30 | 2.84% |
| 2.3569 (Joey) | 5.19% |
| 2.38 (expected #30 bar) | 6.11% |
| 2.40 | 6.89% |
| 2.50 | 10.61% |

Value of 1 us saved on one shape (basis points of log score, from 1ca0668a):
| Shape | bp per 1 us | layer us | bp per 1% of that layer |
|---|---:|---:|---:|
| d1S | **18.4** | 65.7 | 12.1 |
| v1S | **11.5** | 106.0 | 12.2 |
| d16S | 8.5 | 127.5 | 10.8 |
| d32S | 6.7 | 146.7 | 9.8 |
| d1F | 6.4 | 73.3 | 4.7 |
| v16S | 4.8 | 233.0 | 11.3 |
| v1F | 4.0 | 112.2 | 4.5 |
| v32S | 3.5 | 305.1 | 10.8 |
| d16F | 2.9 | 201.2 | 5.9 |
| d32F | 2.3 | 300.3 | 6.9 |
| v16F | 1.7 | 321.7 | 5.4 |
| v32F | 1.2 | 477.1 | 5.9 |
Joey's lead is ~530 bp. One us on decode-1 sliding is worth 5x one us on verify-32 sliding. A % of a sliding layer is
worth ~2x a % of a full layer. Decode-16 alone cannot close the gap: we need independent wins on d1, v1 and d16.

## 2. Where the big pools are (bytes floor vs measured)
Small-T sliding layers are far from their byte floor. Example d1S (65.7 us measured): weights read once are roughly
qkv ~28 MB + o ~13 MB + pooler ~50 MB + unpooler slice ~6 MB (12-bit lossless) + FP32 router ~8 MB + touched experts
(~2.6 distinct x 28.3 MB ~ 73 MB) ~ 180 MB -> ~43 us at 4.2 TB/s, and KV/activations are small. So ~20 us of d1S is not
data movement: per-kernel ramp/drain, dependent launches, and phases that cannot prefetch across kernel boundaries
(sk2 only pre-issues 2 weight stages before griddepcontrol.wait). The MoE at T=1 alone is 29.7 us vs ~17.5 us of bytes.
These numbers are estimates to be confirmed by the timeline diagnostic below; they are the reason the next bet is
structural.

## 3. The research loop
1. **Generate hypotheses**: independent architectural ideas and parameter variations, each with a stated mechanism.
2. **Feasibility filter** (cheap, numeric, before building): smem/register/launch limits, byte floor, maximum possible
   score contribution (bp per us x exposed us), precision rules, and the ledger of past failures (kb/ref/TRIED.md).
3. **Rank** by expected Delta-log-score per check: gain x P(success) x P(transfer to the full layer), minus build cost.
   Prediction comes from kb/predict (zero-transfer and full-transfer baselines, held-out backtest), not from hope.
4. **Official-equivalent experiment**: fresh inputs, read-flush (cold L2) ABBA timing in prepare(), both orders, two
   passes, correctness and reproducibility checks, negative controls; one narrowly scoped change per arm.
5. **Integrate, verify or reject**: measure the whole Decode layer, update the ledger and the predictor, keep the best
   known-good source, publish only scored improvements.
Numerical feasibility first; no formal proofs. One human-reviewed decision process with narrowly scoped agent tasks, so
every failure is informative (each check prints its own attribution: phases, bytes, exposed waits).

## 4. Current bets, ranked by potential (update after every result)
| # | Bet | Shapes | Pool (us) | Max bp | Status |
|---|---|---|---|---|---|
| 0 | ~~Boundary glue~~: ceiling inflated by a routing artifact; faster kernel bit-exact but no gain (closed) | | | | |
| 0b | **MoE at T=1/8** (TINY/TINY2 run ~2.5 TB/s; MoE board bs1 2.2x SOL) | d1S/F, v1S/F (+ MoE board) | ~8-12 at d1 (est.) | ~150-250 | screen 62adc58a |
| 1 | ~~Layer stream at T=1/8~~ (timeline + PCX: no exposed gap, closed): one persistent pipeline for the small-T layer (pooler+router -> route -> MoE -> unpooler -> boundary, then qkv/o), where weight streaming for the next stage starts while the current stage computes; no per-kernel ramp | d1S/F, v1S/F | ~15-25 per layer (est.) | ~300-450 | timeline diagnostic first |
| 2 | VMS/VML small-batch MoE engine | d16, d32 (+ MoE board) | 10-20 | ~100-150 | iterating (iter 4 diagnosis) |
| 3 | Dense-board GEMVs in Decode (FP32 reduction) | d1, v1, d16 | 2.4-4.4 | ~80-110 | check running |
| 4 | Lossless scale coding for MoE weights (0.232 vs 0.5 b/value) | all MoE-bound | ~6% of MoE bytes | ~50-80 | idea |
Closed bets stay closed unless the new version differs materially (say how in the ledger row).

## 5. Rules of thumb from others and from us
- Standalone wins often tie or lose after integration (Anshul: 6 of 7). Always gate in the full layer.
- Use per-SM timelines and same-codegen controls (Keell); a big decode win can coexist with a verify collapse (Anvith).
- Break a counter reset on purpose to make sure a pass is real (Sapalidis).
- Our own: synthetic-route selection did not transfer (MoE build selection); a 5-7 us predicted gain regressed (7bae45c5);
  a tail-split "balance" fix doubled time (VMS iter 2). Measure, then believe.
