> LATEST TERMINAL RESULT, October29: Attention787962a762c24249/ed810c7a
> passed16tests/12benchmarks/integrity at2.640210548x with128registers,
> zero local bytes and two resident blocks. Affected GM1.043242463 versus
> controls1.037861777 gives only1.005184396 normalized gain: fixed gate failed.
> Retain9e922c35/scored2.537000514x. No score or Decode transfer;5af9 CPU
> draft parked. Original parser-format rejection is preserved; recognizing
> fals... locally changes no scientific threshold and uses no GPU rerun.
> Evidence: results/attention-skipscale-20261029/. Controller archived.
> No GPU job/waiter is active. Derived94checks/50scores remain; next new
> check no earlier than05:22:50.375730UTC. All older live headers are historical.

> CURRENT CANDIDATE, October29: Attention identity-rescale short circuit
> ed810c7a is frozen and CPU-audited; no GPU result or speedup yet. It changes
> only L1 flat update control, preserving all keys, precision and geometry.
> Packet: results/attention-skipscale-preflight-20261029/. Fixed review/gates:
> experiments/prepared/attention_skip_identity_scale_review.md. Next new check
> no earlier than04:22:28.724212UTC, one job, no fresh baseline or sweep.
> No GPU job/controller is active at this checkpoint. Retain every best;
> the rejected MoE factor trial remains closed. Earlier running headers
> below are historical. Goal remains active; main chat only.

> LATEST RESULT, October29: MoE factor trial92577a6426f74605/85a317b7
> compiled at96registers/0local bytes and passed its new-path synthetic probes,
> but paired M32/128/256 were0.36..0.84%slower. selected_main=false.
> Official full5/6/integrity at2.325355016x timed incumbents, not the new kernel.
> Controller discarded/reverted33114871; no score or Decode transfer. Evidence:
> results/moe-factor-y-20261029/. No GPU job/controller is active now.
> Derived remaining checks: Dense19,MoE16,Decode20,Attention20,Prefill20;
> ten scores per track. No new check before04:22:28.724212UTC.
> Continue distinct source/profile-led work, prioritizing whole-layer value.
> All earlier candidate/running/waiter headers below are historical. Goal active.

> CURRENT CANDIDATE, October29 03:07UTC: MoE BF16-factor scratch/cache
> source85a317b7 is implemented and CPU-audited; no GPU result or speedup yet.
> Frozen packet: results/moe-factor-y-preflight-20261029/. Review and fixed
> numerical/resource/performance gates: experiments/prepared/moe_factor_y_review.md.
> One paced check is eligible no earlier than03:22:01.371647UTC. No GPU job
> is active at this checkpoint. The previous diagnostic missed resource
> admission; its timestamps remain inadmissible as incumbent phase fractions.
> Preserve33114871 and all other incumbents; score only a verified material gain.
> Earlier running/waiter headers below are historical. Main chat only.

> CURRENT GPU AUTHORIZATION: checks enabled sparingly through October29,2026
> at20:00Toronto (2026-10-30T00:00Z). EVALUATION_POLICY.json governs pacing:
> one active GPU job, >=60minutes between new checks, four checks/track reserved.
> Score only after material source-bound gains; quota is not a spending target.
> This supersedes earlier no-GPU and review-only holds below. Main chat only.
> D1 is closed after a losing check. MoE db0f6481/checke66873fa2adc4030 is
> complete: full5/6/integrity and low overhead, but local8bytes missed the
> historical0-byte resource gate. No timing admission or diagnostic score.
> Next new check is no earlier than03:22:01UTC and needs a concrete candidate.

# Optimization plan

Goal: get as close as possible to top50 overall. Main chat only,
preserve the selected model, and conserve official checks and scores.
The current GPU evaluation window ends October 29 at 20:00 Toronto.
The ordered MoE -> Decode -> Attention -> Prefill pass is complete. The goal
remains unmet; completed experiments do not establish that optimization is done.

Current authorization includes local implementation, CPU verification and
sparing GPU checks under EVALUATION_POLICY.json. Public scores require a
material source-bound gain. The user's evidence-led plan has been reviewed;
these directions supersede earlier review-only/no-GPU holds. Preserve every
verified and active source.

## What the evidence says

The latest saved public overall row is155/1451. The official API reference
explicitly describes the overall board as ranks only. It supplies no combined
speedup. A coincidence between overall and Decode rank does not establish a
formula. Public aggregate-score models do not reproduce the complete board;
per-load composition, screening and ties may matter. Do not invent an overall
multiplier by averaging or summing standalone speedups.

Reference: https://openkernelchallenge.dev/llms.txt
Public snapshots and model limitations: results/overall-ranking-screen-20261029/.
Exact retained-source load/profile table and CPU reproducer:
results/retained-load-opportunities-20261029/review.md and
experiments/prepared/summarize_retained_loads.py. It reconstructs both retained
track speedups exactly, while keeping profiler sums separate from benchmarks.
The D9 M256 sliding profile is173.5us MoE/301.9us summed kernels; the older
200.3/331.3us pair is bound to a different source.

The current saved50th-place speedups are Dense1.554205, MoE2.361119,
Attention2.613234 and Decode2.148002. These are separate track thresholds,
not a proven threshold for overall. The corrected retained Decode check is
1.990225, about8% below that Decode target in speedup terms. Its API row and
Attention's API row still refer to older publications; keep source identities
separate from board positions.

## Ranked next work

| Priority | Work | Measured reason | Decision before another run |
| --- | --- | --- | --- |
| 1 | Audit source-backed MoE traffic/instruction opportunities | Diagnostic e66873fa completed but missed its resource gate; its traces cannot establish incumbent phase fractions. Full-unit and one-stage CPU ownership screens show tail risks. | Preserve the raw diagnostic, do not relax its gate or retry for nicer traces. Seek a mechanism beyond closed geometry/cache/scale flags and quantify required weighted-load saving before another check. |
| 2 | Select one distinct MoE runtime hypothesis | Current profiles are summed spans, not critical-path fractions. Existing occupancy, skinny16, scale compression and cache-policy trials are closed. | No runtime candidate is currently admitted. Budget added traffic/resources, numerical/replay invariants, execution markers and meaningful load coverage. A private diagnostic point alone cannot justify promotion. |
| 3 | Prepare compatible verified gains for Decode | CPU package bc528949 is 893369 bytes; df68deb6 bypasses two unreachable setup factories. Draft307dd791 also removes an unread512MiB RoPE table; all runtime functions stay exact. These drafts have no fresh GPU preparation evidence. | Preserve exact numerical contracts and whole-layer state. Address the 300s budget before a justified 13/12 integration check; no check merely to retime cleanup. Do not transfer D1. |
| 4 | Attention, then Prefill | Large full attention approaches its bandwidth floor; recent arithmetic rearrangements did not show causal wins. Prefill is outside overall. | Need new evidence beyond a nearby parameter change. Retain the user's order and leave Prefill last. |

Dense D1 is closed. Check62f428608bbf41aa passed9tests/30benchmarks/integrity,
but both M128 families lost about4..7% in their paired comparisons and choices
were empty. Official1.517702802x timed incumbents; it does not measure a new
algorithm gain. EDC remains retained. No score, Decode transfer or nearby
cluster-flag retry is justified. Preserve the historical opportunity screen
below as a bound, not a forecast or an open launch direction.

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
