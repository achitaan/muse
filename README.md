# Muse Open Kernel Challenge

Current goal: approach top 50 overall, with all work in the main chat. The user
removed the original deadline and requested this order: MoE W4A16, whole Decode
Layer, Attention Decode, Prefill last. Conserve evaluations and publish only
fully checked candidates.

| Track | Latest verified result | State |
|---|---:|---|
| Dense projections | 1.527771x scored | Both daily quotas exhausted |
| MoE W4A16 | 2.348300x scored | Saved |
| Whole decoder layer | 1.965325x scored | Saved |
| Attention decode | 2.562402x scored | Saved |
| Prefill layer | 1.663469x scored | Saved |

Top 50 remains unmet. Placements are provisional; refresh the official overall
board rather than treating a historical rank as current.

## Retrieval and reproduction

- [STATE.json](STATE.json): exact sources, hashes, job IDs and latest snapshot.
- [TRACK_NOTES.md](TRACK_NOTES.md): cross-track handoff.
- [MOE_NOTES.md](MOE_NOTES.md) and [DECODE_NOTES.md](DECODE_NOTES.md): hypotheses,
  successful changes, failed attempts, precision review and evidence locations.
- [HANDOFF.md](HANDOFF.md) and [CLOUD_PROMPT.md](CLOUD_PROMPT.md): continuation instructions.
- `best/`: exact measured source bytes and metadata.
- `results/`: raw official records, profiles, frozen sources and immutable histories.
- `experiments/prepared/`: builders, CPU audits and hypotheses; measured status is recorded in the per-track notes.

GPU measurements use the free official H200 service. No paid infrastructure or
external model provider is provisioned. Credentials and local caches are ignored.
The dense implementation derives from Nathen Fernandes' MIT-licensed v18;
preserve [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and embedded notices.

Native attention attribution and full Apache license are preserved. See
[ATTENTION_NOTES.md](ATTENTION_NOTES.md) for cache lifetime, sequential replay
assumptions and exact evidence. User requested fewer checks and submissions;
remaining quota is capacity, not a reason to launch weak experiments.

Latest overall placement snapshot: **150/1432 at 16:48:43 UTC October 28**.
MoE is **53rd** at its new **2.348299935x** publication. Top 50 overall is unmet.
All five requested tracks have complete passing coverage; Prefill is excluded
from overall placement.

Latest decoder check: **1.994319167x**, full 13/12/integrity. It remains unpublished;
the completed published decoder score is **1.965324622x**. See DECODE_NOTES.md
for implicit-padding FFN fusion, early-load correction and exact MoE transfer.
MoE's new publication followed two independent checks of identical timed CUDA;
see MOE_NOTES.md for uncertainty and the rejected backend-selector evidence.

Next unmeasured hypothesis: curated small-batch compressed dense projections,
described in experiments/prepared/decoder_small_projection_review.md. No job or
experiment controller is pending. Preserve checked/published source distinctions.
