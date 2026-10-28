Continue toward top 50 overall in this main chat only. No deadline. Requested
track order: MoE W4A16, Decode Layer, Attention Decode, Prefill last. Be sparing
with checks and scored runs: no blind sweeps, repeated baselines or frequent
quota queries. Review source and existing profiles before spending a run; aim
for material gains. Unused quota is not a reason to test weak hypotheses.

Read HANDOFF.md, STATE.json, TRACK_NOTES.md, MOE_NOTES.md, DECODE_NOTES.md,
ATTENTION_NOTES.md, THIRD_PARTY_NOTICES.md and the problem TASK.md/AGENTS.md.
Attention check ea693135be544501 passed 16 tests and 12 benchmarks at
2.589760797x. Exact source best/attention_checked.py, SHA256
b56e07b1dbc930ef8fd4adaeba108e6d17b62a0f390d68c024b98d3cd608aee4.
Scored attention job 20524d06e3894595 passed at 2.562402309x; never duplicate.
Decoder scored 88f8b06932fa43b7 passed at 1.172780190x. MoE scored
650aa66fd3334d8c passed at 1.268508633x. Dense 6131bc96427e4486 passed at
1.527771327x and both daily quotas are exhausted. Top 50 overall remains unmet.
Prefill has no run yet and does not count toward the overall board per TASK.

Preserve stricter FP32 MoE norm intermediates and the decoder FP32 router path.
Attention derives from Vjeux's Apache source. Cache changes remove redundant
initialization from graph replay and retain buffers per launch signature.
Same-signature calls require sequential replay. Preserve Apache/MIT licenses,
precision boundaries, all routes/keys, fresh inputs, cache writes and outputs.
No output caching, harness/timing changes or persistent BF16 expert expansion.
Two custom split-attention attempts were correct but slower; do not repeat them.

The attention experiment controller is archived; native overall Goal remains
active. Validate controller state and existing job handles before starting a
new scoped phase, reusing cached evidence. Never edit controller events by hand.
Publish evidence at phase boundaries. Use only the existing account and free
official H200 service; no paid GPUs, external providers, global configuration
changes or subagents. Never read/print/commit credentials. Report unmet targets
and provisional placement honestly.
