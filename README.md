# Muse Open Kernel Challenge

Current goal: approach top 50 overall, with all work in the main chat. The user
removed the original deadline and requested this order: MoE W4A16, whole Decode
Layer, Attention Decode, Prefill last. Conserve evaluations and publish only
fully checked candidates.

| Track | Latest verified result | State |
|---|---:|---|
| Dense projections | 1.527771x scored | Retained; new cluster candidate CPU-prepared |
| MoE W4A16 | 2.348300x scored | Saved |
| Whole decoder layer | 1.990225x checked | Corrected retained repair; unscored |
| Attention decode | 2.537001x scored | Corrected retained source |
| Prefill layer | 1.663469x scored | Saved |

Top 50 remains unmet. Placements are provisional; refresh the official overall
board rather than treating a historical rank as current.

## Retrieval and reproduction

- [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md): ranked next work, measured bottlenecks, closed hypotheses and keep criteria.
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

Latest saved public overall row: **155/1451**, October29UTC / October28Toronto.
The overall API supplies ranks only; exact composition is not established.
Prefill is excluded. Decode/Attention API best rows still reference older
publications, not the corrected retained sources shown above.

MoE skinny16 passed its GPU probes but lost all paired timings and was reverted.
Two checks including a compiler repair, zero scores. Next is the prepared Dense
cluster packet, with corrected timing units, cached exact baseline and full9/30
guard. No job or controller is pending at this pre-launch checkpoint. See
OPTIMIZATION_PLAN.md for the whole-layer impact criteria before any transfer.
