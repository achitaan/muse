# Muse Open Kernel Challenge

Current goal: approach top 50 overall, with all work in the main chat. The user
removed the original deadline and requested this order: MoE W4A16, whole Decode
Layer, Attention Decode, Prefill last. Conserve evaluations and publish only
fully checked candidates.

| Track | Latest verified result | State |
|---|---:|---|
| Dense projections | 1.527771x scored | Both daily quotas exhausted |
| MoE W4A16 | 2.312613x scored | Saved |
| Whole decoder layer | 1.815512x checked | One scored publication pending |
| Attention decode | 2.562402x scored | Saved |
| Prefill layer | Unmeasured grouped-MoE candidate | Current focus; last track |

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
- `experiments/prepared/`: explicitly unmeasured hypotheses and CPU audits.

GPU measurements use the free official H200 service. No paid infrastructure or
external model provider is provisioned. Credentials and local caches are ignored.
The dense implementation derives from Nathen Fernandes' MIT-licensed v18;
preserve [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and embedded notices.

Native attention attribution and full Apache license are preserved. See
[ATTENTION_NOTES.md](ATTENTION_NOTES.md) for cache lifetime, sequential replay
assumptions and exact evidence. User requested fewer checks and submissions;
remaining quota is capacity, not a reason to launch weak experiments.
