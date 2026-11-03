# Decode CHASE report (offline; data through jobs 8ffd6871 and c5800fa6, Nov 3 2026)

Tool: `python kb/predict/predict.py all` (stdlib only; reuses `kb/muse_score.py`). Output: `BACKTEST.txt`, `rows.csv` (job x shape x path), `family_rows.csv` (job x shape x kernel family). Replay: 63 Decode jobs (23 new-format + 40 old-archive format), max |calc - judge| 4e-16; 2 non-timing jobs rejected (b0caa4c9 prepare failure, d588efd5 duplicate).

**Start point.** Anchor 1ca0668a = 2.2347 (scored). Noise band = the user's CONTROLS (`muse_score_ref.py`, 7 unchanged-path checks): median 2.2433, range 2.2318-2.2607 (width 0.0289, half-width 0.0145); my 19-job replay of the same path is wider (2.2257-2.2722, sd 0.0120), per-shape sd 1-3%. Tool replays all 7 CONTROLS (max diff 4e-5) and 39 of the 43 `HISTORY` jobs exactly (the other 4 are not in my archive set); HISTORY/CONTROLS are reused, not re-derived. 2.37 needs +6.1% = **7.9 us on every one of the 12 shapes** (3/5/8 us -> 2.284/2.318/2.372). Matching Joey's ratio on every shape he leads gives only 2.356.

**Two results arrived after the brief (not yet in TRIED):** lane B check **8ffd6871: adopted 0**, every dense GEMV arm 1.00-1.025 slower (best qkv cgemv+pdl 0.995-1.000, ~+0.3 us; plan claimed 4.4/2.4/3.5 us). VMS iteration 4 **c5800fa6**: bs16 best arm 94-96 vs 98.3 us (median 0.955-0.981 but paired 0.991-1.006, so ~0 to -4 us); every bs32 arm 1.000-1.016 slower; stream-only A+B alone is 94.6 us. Five current lane rows: 2.3103 at design numbers, **2.2409 at measured numbers**; extra uniform saving still needed 3.3 us (design) vs 7.5 us (measured).

## Predictor (held out by revision group; positive = faster; MAE in us, errors capped at 20 us)
| stage | n rows / groups | zero | full | LOGO-shrunk | sign beyond noise |
|---|---|---|---|---|---|
| written-down claim -> full layer (incl. 7bae45c5: +5-7 claimed) | 27 / 9 | 6.5 | 8.4 | 6.5 (slope clipped 0) | 0/13; **0/27 claims realised** |
| claim -> board A/B (incl. 74e3e792: -7.5 claimed, +99.5 us) | 25 / 14 | 5.8 | 10.3 | 5.8 | 6/21; 0/25 realised |
| isolated A/B -> full layer | 24 / 10 | 3.5 | 1.7 | 1.7 (b=1.04 [0.43,1.38]) | 6/6 but 3 are slowdowns; **positive isolated wins realised 4/16** |
| adopted in-prepare gate -> judged parent/child delta (old archive) | 15 / 7 | 2.8 | 3.0 | 3.0 (b=0.45 [0.13,0.85]) | 8/10; score MAE 0.006 |
Unchanged-parent score MAE 0.0127x (n=19), 0.0147x (old, n=40). Old->keell rebase: shrunk slope no better than zero (iso 1.60 vs 1.23; claim 5.75 vs 5.75). **Negative result:** no source-only point forecast is supported; a claim carries "zero, range 0..claim". Slowdowns seen in isolation transfer; small isolated wins mostly vanish; a gate win reaches a judged job at ~0.45x and is below one job's noise. Caveat: written-down claims exist mostly for ideas that were later closed, so 0/27 is a floor, not a rate. 46 arms / ~20 groups cannot support a genetic search or calibrated intervals.

## CHASE (ranked by hit x score gain / checks; full list with per-shape numbers in BACKTEST.txt)
| # | experiment (parent -> change) | loads | gross -> net us (net = min(gross, floor cap) x haircut) | cond. score (x band width) / ceiling | evidence | kill | checks |
|---|---|---|---|---|---|---|---|
| 1 | **N2** kb v1 sk2 qkv -> rq/CURP with FP32 split reduce (fixes f2ab6d4c id flips) | d16, d32 | d16 S/F 0.4/0.2, d32 S/F 1.7/2.0 -> 0.2/0.1/0.8/1.0 (0.5 haircut assumed) | 2.2369 (+0.0022, 0.08x) / 2.3307 | gate (in-layer paired measured, numerics rejected) | ids flip, or ratio >0.99 at d32 | 2 |
| 2 | **VMS-16** kb v1 MID -> VMS r4+fin2, T=16 only (decode_vms.py) | d16 | design 10.4/7.6 (d16 S/F); measured-so-far 1.9/1.4 | 2.2392 (+0.0045, 0.16x) / 2.2610 | iso (board median -4%, paired -0.7%) | decode gate >0.988 median or paired | 3 |
| 3 | **N3** kb v1 boundary glue (8.6-13.7 us/shape) -> fold / PDL early trigger at T>=16 | d16,d32,v16,v32 (8 shapes) | 24% capture prior: 2.1-3.3 per shape -> same | 2.2516 (+0.0169, 0.58x) / 2.3067 | pre (hit 0.03 emp.; 0.0021/check at flat 0.25) | skip-launch ablation saves <1.5 us | 2 |
| - | N1 MoE T=128/256 consumer throughput (headroom 31-57 us/shape, no lever; P2-RS64, L1, L2 closed) | v16, v32 | 6% prior: 1.9-3.4 | 2.2410 / **2.3516** | pre | consumer A/B not <=0.98 | 3 |
| - | DG-T1/T8/T16 dense GEMV, VML-32 | | killed by 8ffd6871 / c5800fa6 | 2.2347 | | | |
Verify-32 sliding one-head attention in Decode: -0.5 us on v32S only -> +0.0004 (no priority).

## Recommendation
Signal is thin: **no candidate has a conditional gain above the 0.0289 CONTROLS band (best N3 0.58x, N2 0.08x, VMS-16 0.16x)**, so a judged score cannot confirm any of them; only the in-prepare gate can, and the tool's own rank (#1 N2) is worth +0.002 even if it works. Spend the next check on **N3's timing-only ablation** (skip each boundary launch / both, wrong numerics allowed, read-flush ABBA, all 12 shapes): one check, no correctness risk, and it converts the largest un-owned family (+0.017 conditional, 2.307 ceiling) from a claim into a measured ceiling. Build N3 only if the ablation saves >=3 us on d16/d32/v32; otherwise park it. Do not resubmit lane B or VML; hold VMS-16 until a board arm reaches paired <=0.97. Judge N3 over N2/VMS by the flat-hit column (the 'claim' class hit rate is biased low); that is a judgement, not the tool's top rank.
