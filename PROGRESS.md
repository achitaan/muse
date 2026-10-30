# Dense (`muse_bf16_proj`) — Optimizer loop progress

Mode: Optimizer, loop. Started 2026-10-30 22:04 UTC. Limits: 6 passes or 60 min (deadline 23:05 UTC).
Quota at start: 4 checks left today (16/20 used), 10 scored left; reset 00:00 UTC.
Anti-loop rule: nothing in `DENSE_LEDGER.md` section 1/1b is retried.

## Phase 1 — findings

- **Metric:** geometric mean over 6 loads (decode 1/16/32, verify 1/16/32) of SGLang load time / ours; load time = sum of 5 shapes (qkv_gate, o, pooler, unpooler, router), each a CUDA-graph replay, cold L2, paired ABBA median ratio, ~4 µs replay cost subtracted. 95% CI at least ±1.5%.
- **Correctness:** relL2 vs FP64 ≤ 1% (router ≤ 0.01%), single-entry power-of-two probe rows ≤ 0.1%, any row ≤ 5× bound, no NaN/Inf; 9 pre-tests incl. outlier distributions. Weights at own precision (lossless compression allowed; router hi+lo allowed). No caching, no host syncs.
- **Board (22:05 UTC):** ours 1.5278x, rank 229/1497 (board moved; was ~160). Leader 1.6498. **Top-50 cutoff now 1.5698** → need **+2.75%** (≈ 8 µs summed over loads at 0.2–0.48 %/µs).
- **Baseline run:** reused the latest incumbent-runtime check `0812a06eac9b42d8` (all S6 candidates fell back): 1.5156x (CI 1.493–1.538); board run 1.5278x. No separate baseline check spent (quota).
- **Profile (per-shape, µs after overhead; `results/shape_table_s6.txt`):** large-M shapes are cuBLAS-wall (closed). Small-M kernels run at 2.5–3.7 TB/s effective on c12 bytes. Outlier: **qkv_gate M1/M8 at ~3.0 TB/s** vs pooler M1 at 3.7 TB/s (9.3 / 9.1 µs kernel vs ≈ 7.6 µs at pooler efficiency).
- **Hypothesis for the outlier:** wave quantization. qkv has 2304 rows = 144 16-row tiles; incumbent `(1,4,8,4)` = 576 CTAs → 4.36 per SM on 132 SMs (max 5 → ~87% balance). S=8 gives 1152 CTAs (8.73 → 97%).
- **Research:** DFloat11 (NeurIPS 2025) / bf16-huffman-infer: entropy-coded exponents reach ~11 bits vs our 12, but Huffman decode in a fused GEMV runs at 80–90% of BF16 speed in public results; our S4 already showed decode cost dominates. Not a 60-min lever.

### Ranked hypotheses (expected score gain, all unmeasured)
1. **H1 qkv M1/M8/M16 wave-balanced split (S=8 / more warps):** up to ~1.2% if qkv reaches pooler efficiency; likely 0–0.6%.
2. H2 o M8 geometry (4.5 µs kernel vs 3.9 at M1): ≤ 0.3%.
3. H3 c12 decode cheapening (oracle #4): < 1%, diagnostic first.
4. M≥128 new kernel family: closed/too large for this window.

## Phase 2 — passes

| Pass | Hypothesis | Change | Validation | Benchmark | Verdict |
|---|---|---|---|---|---|
| 1 | H1 qkv M1/M8/M16 wave balance | `experiments/dense_p1.py`: 7 v18d geometries, per-shape gated selector (commit `677e0c2`) | NVRTC 14 instances, 0 spill | driver first refused (reserve); user allowed reserved checks; check `e26e498c24e8487b` passed (1.4901x, runtime unchanged = noise) | **reverted (all KEEP): every layout slower, 3–65%. H1 refuted**; the incumbent geometry is the local optimum |
| 2 | H2 o M8 wider CTAs + router M8/M16 alternative splits | `experiments/dense_p2_o8.py` (commit `fbdf87d`) | NVRTC o 50 regs, 0 spill; router one-hot exact, relL2 2.5e-6 | check `dc821d6f1ad94fbf` passed 1.5177x (runtime unchanged) | **reverted (all KEEP)**: o M8 (4,8,2,1) 1.025 (14/16), router M16 S16/WK4 1.028 (16/16) — real but below the 3% gate, ≈ 0.1% score each; router M16 S8/WK8 3% slower |

Loop stopped after pass 2 (22:22 UTC, ~18 of 60 min, 2 of 6 passes, 2 of the 4 freed checks).

**Continuous mode (user request after the self-review):** the loop resumes with one hypothesis per pass, the same gates and the ledger. Pass/time caps are lifted; checks are limited by daily quota.

| Pass | Hypothesis | Change | Validation | Benchmark | Verdict |
|---|---|---|---|---|---|
| 3 | H3: find what makes qkv M1 slow (DIAG ablation) + ST 3/4 loads in flight | `experiments/dense_p3.py` (commit `751a559`) | NVRTC 7 instances, 0 spill | check `a61fe75d5a5f427b` passed (1.5098x, runtime unchanged) | **reverted**: ST3/ST4 9–25% slower. Diagnostic: qkv M1 body ≈ 8.4 µs = 7.2 weight streaming + 0.4 cluster reduction + ≈ 0.8 decode/MMA/x (hidden). Oracle #4 dead | Reason: no remaining hypothesis that fits the window can close a +2.75% gap. Both measured levers were local layout neighbourhoods, and they're exhausted (P1 slower everywhere, P2 under +0.2% total).

## Self-review

- **Baseline vs final:** baseline is the board best of 1.5278x (latest incumbent-runtime check 1.5156x). Final is unchanged: no change was kept, and `best/verified.py` is untouched. Both pass checks reran the incumbent runtime (1.4901x, 1.5177x), consistent with ±1.5% judge noise.
- **Gap to cutoff:** top-50 is 1.5698x and we're at 1.5278x, so **+2.75% still needed** (rank 229).
- **Kept:** nothing. The ledger records every result (34 "SLOWED DOWN" mentions), and the new driver copy `scripts/measure_dense_lane.py` allows an explicit reserve override (`LANE_CHECK_RESERVE`).
- **Failed:** H1 (qkv wave balance) was refuted, 3–65% slower. H2 gave only sub-gate gains.
- **Measurement validity:** every candidate was timed against the live incumbent in the same process (ABBA, cold L2, CUDA graphs, raw samples logged), and the official checker passed on every run. No benchmark, input or validation code was touched.
- **Unverified:** whether the two P2 near-misses (≈ 0.2% combined) survive a frozen-route check. That gain is far below the judge's CI and not worth adopting under the ledger's 3% rule without a confirming run.
- **Next best idea:** the qkv M1/M8 kernel still runs at ~3.0 TB/s vs the pooler's 3.7 TB/s, and P1 shows it isn't layout. Next is a diagnostic of the v18d qkv kernel itself (oracle #4: c12 reconstruction cost, via a dummy-decode ablation at M1, one check). Only a confirmed decode-bound result would justify a cheaper-decode kernel (~1% upside). The +2.75% gap realistically needs a new M128/M256 kernel family (multi-day).
