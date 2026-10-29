# Optimization plan

Goal: get as close as possible to top50 overall. Main chat only, no deadline,
preserve the selected model, and conserve official checks and scores.
The ordered MoE -> Decode -> Attention -> Prefill pass is complete. The goal
remains unmet; completed experiments do not establish that optimization is done.

Current authorization, October 29 UTC: local implementation and CPU verification
only. GPU checks, profiling jobs and submissions remain unauthorized. The
user's evidence-led plan has been reviewed; the corrections below supersede
older automatic launch directions. Preserve every verified and active source.

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
| 1 | Review D1's value and integration feasibility before any run | Two fixed original-BF16 cluster families target M128 QKV/pooler. A 3% win in both gives only about 0.25% Dense track gain under unchanged controls. | Explain a credible 5.62us load saving for the existing 1.5% track gate, or a worthwhile Decode transfer. Resolve preparation and nearby-band performance first. Any GPU use requires new authorization. |
| 2 | Diagnose exposed MoE WGS readiness/fixup waits | Historical summed profiles motivate investigation but do not establish a critical-path fraction. Per-job/per-stage phase overlap already exists. | Identify exposed waits or fixups beyond existing deferred work and overlap. No global-join replacement premise, occupancy rerun or macro sweep. |
| 3 | Prepare compatible verified gains for Decode | CPU-only package bc528949 reduces source to 893369 bytes; setup remains the historical 256.897s. | Preserve exact numerical contracts and whole-layer state. Solve the 300s whole-layer preparation budget before an authorized 13/12 check. Do not transfer a rejected standalone path. |
| 4 | Attention, then Prefill | Large full attention approaches its bandwidth floor; recent arithmetic rearrangements did not show causal wins. Prefill is outside overall. | Need new evidence beyond a nearby parameter change. Retain the user's order and leave Prefill last. |

Dense is a bounded component experiment, not a claim that its score alone
improves overall. Its optimistic listed-floor screen suggests roughly2.8%
standalone improvement, with much smaller whole-Decode impact. If it wins,
evaluate integration alongside other compatible verified gains rather than
spending a full-layer check on a negligible transfer.

The recorded 65.731us M128 Dense load needs about 5.617us saved to yield 1.5%
track gain when it is the only changed load: 1.015^6 =1.09344 load speedup.
The listed QKV-only and pooler-only floors yield about 1.33% and 1.37% track
gain; both together about 2.84%. These are historical arithmetic screens,
not forecasts. Preserve the existing aggregate retain guard.

MoE already releases phase-A completion through `doneAs[job,stage]`; phase B
acquires each required 256-column slice. Its global completion barrier follows
phase B. Future overlap work must target waits inside that existing protocol.
Tiny routing changes must preserve the initial weight prefetch that overlaps
routing. Decode's router/pooler stream overlap also covers M128/M256; account
for the join when estimating D1 transfer value.

## CPU-only Decode package

Source `bc528949d73aa09ae9a330d030cadd5654f9ff7d20d48e8ff26bbeed96ae1c43`
losslessly encodes one static CUDA literal using the existing loader. It saves
95649 bytes and leaves 106631 bytes under the decimal 1MB cap. Independent static
verification recovers D9's CUDA, complete module AST and every original byte;
three mutations are rejected. The package is unmeasured and unpromoted.

Review: experiments/prepared/decoder_source_package_review.md.
Frozen evidence: results/decoder-source-package-20261029/.
The static preparation map records 21 dense jobs with four workers. Repeated
factory names have different inputs/options and do not prove duplicate builds.
The WG sweep is already disabled. No compilation/preparation work was removed,
and no setup speedup is claimed. Historical remaining setup time is 43.103s.

The raw D1 increment would fit this package at 943068 bytes before adaptation.
Its complete preparation feasibility, state integration and performance across
enabled bands remain unresolved. No GPU job is authorized by this packet.

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
After explicit GPU authorization and the readiness gates above, the driver
must confirm positive official quota before one check. Any scored confirmation
also requires authorization, material gain and every evidence gate.

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
