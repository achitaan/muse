# Muse Open Kernel Challenge

Current goal: approach top 50 overall, with all work in the main chat. The user
authorized paced GPU checks through October 29 at 8 p.m. Toronto and requested this order: MoE W4A16, whole Decode
Layer, Attention Decode, Prefill last. Conserve evaluations and publish only
fully checked candidates.

| Track | Latest verified result | State |
|---|---:|---|
| Dense projections | 1.527771x scored | Retained; cluster candidate rejected |
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

Latest saved public overall row: **162/1467**, October 29 at about 05:17 UTC.
The overall API supplies ranks only; exact composition is not established.
Prefill is excluded. Decode/Attention API best rows still reference older
publications, not the corrected retained sources shown above.

Latest completed check: MoE paired-store repair `5262c2c560c94197`. It compiled
and passed its synthetic probes but failed the performance gate, so the controller
reverted it. No score or Decode transfer follows. Raw source, paired samples,
profiles and review are in `results/moe-epair-store-repaired-check-20261029/`.
Two simple Attention L8 overlap edits were rejected by CPU compiler inspection;
both serialize matrix instructions. No GPU evaluation was used for that screen.

No job or controller is active. Derived balances after the latest accepted check:
Dense 19, MoE 14, Attention 19, Decode 20, Prefill 20 checks; 10 scored submissions
per track (92 checks and 50 scores total). These are the fresh prelaunch limits
minus that check, not a later live query. No new check before 07:28 UTC / 3:28 a.m.
Toronto. Preserve hourly spacing and reserves; no next runtime candidate is
qualified yet. See OPTIMIZATION_PLAN.md and the latest HANDOFF.md entries.
