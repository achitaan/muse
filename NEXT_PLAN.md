# Next optimization plan (written 2026-10-29 23:50 UTC, after the omos lanes' check window)

## Where we stand (okc leaderboard, 23:48 UTC)

| Board | Best | Rank | Note |
|---|---:|---:|---|
| Overall | - | 221 / 1628 | follows the Decode rank |
| Decode | 2.0071x | 221 / 1000 | top-50 cutoff ~2.18x -> need about +9% |
| MoE | 2.3507x | 90 / 1131 | same MoE kernel family runs inside Decode |
| Attention | 2.5624x | 149 / 1087 | split plans at a local optimum (A5 in-process autotune) |
| Dense | 1.5278x | 158 / 1411 | config levers exhausted (D1-D7) |
| Prefill | 1.6635x | 179 / 652 | not in the overall score |

The board keeps each team's best run, so scored runs cannot lower a published number.

## What tonight proved

1. Configuration selection is exhausted on every track: builds, splits, warps, cuBLASLt
   algorithms and neighborhoods, olayer configs, FA3 options, MoE ring depth and schedules were
   all measured on the judge GPU and the incumbents are at local optima.
2. Single official runs cannot resolve changes below ~3% per shape (run-to-run spread 2-5% on
   long shapes, 5-11% on tiny Dense shapes). A3 looked like +9% on one shape for four runs and
   vanished against same-night controls.
3. The judge GPU works as a profiler: a prepare-time harness can compile several variants, check
   them against the incumbent output, time them as interleaved cold-L2 graph replays and print the
   table, while the scored path stays the incumbent. One check = one multi-variant GPU experiment
   (300 s prepare cap; one nvcc build ~40 s, Triton builds are cheap).
4. Only structural (kernel-level) changes can move Decode by the ~9% top 50 needs.

## Decode time breakdown (D4 run 5cac5a65eb204c78, per-kernel GPU time)

| Shape | Median vs floor | Largest kernels |
|---|---|---|
| decode 1 sliding | 85.4 vs 40.0 us | moe_fused 29.1, v18s_gemv 13.3, **route_fp32 11.1**, native_gemv 9.5, attn 5.3, norm_residual 5.3, smallm2_gemv 5.1, _sfn 3.8, smtrace_gemv 3.7, _pn 2.0 |
| decode 32 full | 307.3 vs 208.1 us | attn 162.5 (about 90% of KV bandwidth), **moe_fused 87.8**, sk2 19.8, m32w_gemm 16.7 |
| verify 32 sliding | 322.5 vs 143.8 us | **moe_fused<true> 215.9 (63%)**, three nvjet GEMMs 74.2, attn 15.1 |

## Plan, in priority order

### Step 0 - one Decode base, shared harness (half a day, offline + 1 check)
- Merge the lanes: the board's Decode best (2.0071x) came from the other lane
  (`argon/top50-20261029`, scored run `646774bad23e4c7f`). Use that source as the single Decode
  parent and stop running two Decode lanes against one quota.
- Build `harness_prepare.py`: a reusable prepare-time A/B module (variant list -> compile ->
  correctness vs incumbent -> interleaved cold-L2 graph timing -> printed table), with the
  incumbent always serving `kernel_fn`. Gates frozen per experiment as tonight.

### Step 1 - Decode small-token latency (1-2 days; expected +2-4% Decode score)
- Router: `route_fp32<1>` takes 11.1 us at 1 token. Its weight is 8.4 MB (floor about 1.8 us) and
  the Dense track's router runs 3.9 us. Target <= 5 us (more CTAs / split-K, top-k fused in a tiny
  tail). Saves about 6 us on decode 1 and verify 1 loads (about 7% of those loads).
- Fewer boundary kernels at T <= 8: fold `_sfn`, `_pn` and one `norm_residual` into the producer or
  consumer GEMV epilogues (each separate kernel costs 2-4 us at T = 1). Target -5 to -8 us per load.
- Evidence path: prepare harness first (in-process A/B), then 2 paired official checks
  (candidate vs same-night parent control) before a scored run.

### Step 2 - MoE kernel at 32-256 tokens (multi-day; the largest lever, counts twice)
- `moe_fused` is 28-63% of the Decode shapes and is the whole MoE track. At 256 tokens it runs
  216 us against a ~109 us floor; at 32 tokens about 88 us inside Decode and 127 us vs 71 us on
  the MoE board. Tonight ruled out build choice, fallback build and ring depth (M1-M5).
- Candidates to test with the harness: expert-tile load balancing for skewed routing (the busiest
  rank gets far more rows), 2-CTA clusters with TMA multicast of the activations, overlapping the
  gate/up and down GEMM phases (the "phase schedule" gap noted earlier), larger N tiles for fat
  experts at T >= 128.
- Target -15 to -20% on moe_fused at T >= 32: about +4-6% Decode score and +8-10% MoE board.

### Step 3 - cheaper items in parallel (low risk, small gains)
- Prefill P7 (Triton KV gather + contiguous FA3): FA3 -3.1 to -5.2% in-process on four full shapes,
  gather 15 us. Needs 3 paired confirm runs (P7 vs parent control); then one scored run.
- Dense: TMA-bulk GEMV for <= 8 tokens (router 3.9 vs 1.75 us floor, sliced unpooler 3.9-4.1 vs
  1.75, O 5.0 vs 3.5). Develop in the harness; Dense-board only.
- Attention: only kernel-level work remains (split plans are optimal); lowest priority.

## Budget and process per day (20 checks + 10 scored per problem)
- Decode: about 6 harness checks + 4 paired confirmation checks + 1-2 scored runs.
- MoE: about 6 harness checks (shared kernel work with Decode) + 2 confirmations + 1 scored.
- Never upload two candidates at the same instant (two uploads failed together at 23:38:38).
- Rule for keeping a change: in-process gain >= 3% on the targeted kernel AND paired same-night
  official checks >= 1.02 on the affected shapes; then one scored run.

## Honest expectation
Step 1 + Step 2 at their targets give roughly +6 to +10% on Decode, which is about what the top 50
needs (+9%). Step 2 carries most of it and is genuinely hard kernel work; a GPU dev box (H100/H200)
would cut iteration time by an order of magnitude versus judge-only experiments.
