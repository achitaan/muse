# Optimization plan

Goal: get as close as possible to top50 overall. Main chat only, no deadline,
preserve the selected model, and conserve official checks and scores.
The ordered MoE -> Decode -> Attention -> Prefill pass is complete. The goal
remains unmet; completed experiments do not establish that optimization is done.

## What the evidence says

The latest saved public overall row is155/1451. The official API reference
explicitly describes the overall board as ranks only. It supplies no combined
speedup. A coincidence between overall and Decode rank does not establish a
formula. Public aggregate-score models do not reproduce the complete board;
per-load composition, screening and ties may matter. Do not invent an overall
multiplier by averaging or summing standalone speedups.

Reference: https://openkernelchallenge.dev/llms.txt
Public snapshots and model limitations: results/overall-ranking-screen-20261029/.

The current saved50th-place speedups are Dense1.554205, MoE2.361119,
Attention2.613234 and Decode2.148002. These are separate track thresholds,
not a proven threshold for overall. The corrected retained Decode check is
1.990225, about8% below that Decode target in speedup terms. Its API row and
Attention's API row still refer to older publications; keep source identities
separate from board positions.

## Ranked next work

| Priority | Work | Measured reason | Decision before another run |
| --- | --- | --- | --- |
| 1 | One prepared Dense cluster check | M128 QKV/pooler profiles are13.7/19.8us. Two fixed original-BF16 WGMMA cluster kernels target these spans. | Full9/30, nearby FP64 probes, nonempty choices, actual new profile symbols, paired wins, >3% affected/control and >1.5% aggregate gain. |
| 2 | Diagnose the larger MoE WGS path | MoE is about46% of corrected Decode's M16 sliding profile and about46–64% of M128 profiles. This offers greater whole-layer potential than tiny boundary changes. | Require a specific measured bottleneck and a distinct algorithmic hypothesis. Deeper GPU counters or focused external code/profile evidence would help. No new macro sweep. |
| 3 | Integrate compatible verified gains into Decode | Improving the actual whole layer directly tests the requested serving computation. Source989018bytes and last prepare256.897s leave little room. | Quantify expected whole-layer gain, preserve BF16/FP32 contracts, prove size/preparation feasibility, then one full13/12 check. Do not transfer a rejected standalone path. |
| 4 | Attention, then Prefill | Large full attention approaches its bandwidth floor; recent arithmetic rearrangements did not show causal wins. Prefill is outside overall. | Need new evidence beyond a nearby parameter change. Retain the user's order and leave Prefill last. |

Dense is a bounded component experiment, not a claim that its score alone
improves overall. Its optimistic listed-floor screen suggests roughly2.8%
standalone improvement, with much smaller whole-Decode impact. If it wins,
evaluate integration alongside other compatible verified gains rather than
spending a full-layer check on a negligible transfer.

For skinny16 MoE, a profile-based transfer screen estimated only0.66% whole
Decode gain from10% faster affected MoE, or1.75% from30% faster affected MoE.
This was an approximate cost screen, not a timing prediction. The actual
candidate lost badly, so no transfer is justified.

## Closed hypotheses to avoid repeating

| Experiment | Actual evidence | Outcome |
| --- | --- | --- |
| Skinny16 MoE weight reuse | Repaired kernel passed12GPU probes. New/old paired ratios1.3461/1.4563/1.4319; choice empty. Full5/6 timed incumbents. | Rejected; two checks including one compile failure, zero scores, no Decode transfer. |
| Two-warpgroup WGS | Full5/6 passed at2.04556x versus retained check2.36544x; measured two-CTA occupancy with no spills. | Rejected. More occupancy did not imply a faster kernel. |
| Attention QK operand swap | Full16/12 at2.50066x versus retained2.53700x; changed/control ratio0.9940. | Rejected despite fewer MMA instructions. |
| Prefill paired BF16/DQ8 transfer | Full5/8 at1.68016x; normalized profile gain1.00442 failed the fixed1.02threshold. | Rejected; no score. |
| Router shared-input cache | Full13/12 passed, but every changed load lost; aggregate gain came from controls. | Rejected. Protect affected paths as well as the aggregate. |

Evidence is in results/ and the matching experiments/prepared/*review.md files.
Handoff history is not a reason to retry a closed idea under a new label.

## Current Dense packet

Source169dbce77be377eea3c49b05428d67d3141c0a6f506f8dd23e57ad4cd7e269e3,
687291bytes. Original EDC source is an exact prefix. Only the event-time
conversion changed from the earlier unsubmitted4d443b13 packet: CUDA
milliseconds are now multiplied by1e6 for fields named nanoseconds. The
inverse restores the entire earlier source; CUDA and acceptance gates match.
The full CPU source/dispatch/ownership/numerical audit was rerun for this hash.

Review: experiments/prepared/dense_cluster_review.md.
Evidence: results/dense-cluster-preflight-20261029/.
Metric: scripts/measure_dense_cluster.py; guard: scripts/verify_dense_cluster_evidence.py.
Scope: candidates/active.py. Baseline: exact scored EDC1.527771327x, cached
locally or in the tracked sprint evidence. No baseline API call or GPU rerun.
The driver must confirm positive official quota before launching one check.
At most one scored confirmation after material gain and all evidence gates.

## Planning inputs that would help

- GPU performance counters for a specific hot kernel: bandwidth, memory stalls,
  register pressure, tensor-core utilization and synchronization cost. The
  current summaries give kernel duration; do not assume they provide counters.
- A focused review from Dot with code, numerical assumptions, expected whole
  Decode impact, nearby-shape behavior, and compilation/preparation cost.
- An authoritative description of the overall formula, if the organizers
  publish one. Aggregate leaderboard correlations are insufficient.

Keep a small ranked hypothesis list and an explicit keep criterion. Preserve
all failures, raw samples, exact sources and timestamps. Resume existing job
handles after observation failures; never infer termination from a timeout or
re-upload blindly. Quota is capacity, not a reason to launch weak experiments.
