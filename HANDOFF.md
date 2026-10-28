# Current handoff - October 28, 2026

Goal: approach top 50 overall, main chat only, no deadline. Conserve checks and
scores; no blind sweeps, repeated baselines, cosmetic resubmissions, subagents,
paid GPUs, external model providers, global changes or credential output.
The requested ordered pass MoE -> Decode -> Attention -> Prefill is complete
with full passing coverage. Top 50 overall is still unmet; Goal remains active.

| Track | Published speedup | Last observed place | Job |
| --- | ---: | ---: | --- |
| Dense | 1.527771327x | 97/1242, 15:05 UTC | 6131bc96427e4486 |
| MoE | 2.348299935x | 53/991, 16:48 UTC | b6c70c9b83724d2e |
| Decode | 1.965324622x | 149/869, 15:47 UTC | 02fd620987114785 |
| Attention | 2.562402309x | 80/945, 15:05 UTC | 20524d06e3894595 |
| Prefill | 1.663469245x | 122/573, 15:05 UTC | 23e7c8b4c3e54869 |

Overall: **150/1432 at 16:48:43 UTC**. MoE's 50th-place score was 2.3528x,
about 0.2% ahead. Decode's earlier 50th-place score was 2.139x. Placements can
change. Prefill does not count toward overall. No job or controller is pending.

The newer whole-decoder check is **1.994319167x**, job 09b34f4e921e49a6, exact
09badc9b source in best/decoder_verified.py, full 13/12/integrity pass. It adds
checked exact MoE, implicit small-token FFN zero padding, and early FFN input
loads. Its interval overlaps the published check; no scored run was spent.
best/decoder_published.py remains a61735ee. Source distinctions matter.

MoE checked and published sources are both 33114871: paired BF16 epilogues and
exact DQ8 bit placement, with all original norm precision retained. One scored
confirmation followed two independent passing checks of identical timed CUDA.
The rejected selector chose no alternate; folding its metadata branch recovers
the simpler kernel AST. Individual intervals overlap, so report measured points
and ranks without claiming statistical significance. No repeated source checks
or scored intermediates were used. See MOE_NOTES.md and full replication proof.

Next: review experiments/prepared/decoder_small_projection_review.md for
curated public SK2 small-batch pooler, QKV and sliced-unpooler configurations.
Read the pinned helpers/actual dispatch and prove exact packing/geometry before
implementation. Reuse existing generic helpers and preserve the full FP32
small-token router; this proposal concerns separate BF16 projections. No new
projection candidate is implemented or measured. Source is currently 988314
bytes, leaving 11.7 kB under the upload limit. Keep prepare below 300 seconds;
last observed setup was 215 s after removing an unselected dense WG sweep.

Read STATE.json, per-track NOTES, official TASK.md/AGENTS.md and notices. best/
holds exact sources and metadata; results/ holds frozen sources, raw official
records, profiles and immutable controller history. Decoder phase6 and MoE
phase5 are archived. Initialize one scoped phase with cached baseline before
trials, preserve handles across observation failures, never edit journals, and
push complete evidence at boundaries. Dense's last observed quotas were empty;
query other quotas only when needed.

Preserve BF16 GEMM/global-scale and effective-weight boundaries, FP32 norm
intermediates, every route/key, fresh inputs, cache writes and all residual
outputs. NVFP4 weights stay exact, no persistent BF16 expansion or output/data
caching. Shared native scratch requires sequential calls. Keep Apache/MIT
licenses and Vjeux, Nathen Fernandes, Lee Penkman and SGLang attribution.
