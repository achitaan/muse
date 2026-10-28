# Current handoff - October 28, 2026

Goal: approach top 50 overall in this main chat only. No deadline. Conserve
checks and scored runs: no blind sweeps, repeat baselines or frequent quota
queries. The requested ordered pass (MoE, Decode, Attention, Prefill last) now
has complete passing coverage on every track. Overall Goal is still active.

| Track | Latest verified result |
|---|---|
| Dense | 1.527771327x scored, 6131bc96427e4486; both daily quotas exhausted |
| MoE | 2.312613084x scored, b4a10b76347e4cb0 |
| Decode | 1.804641371x scored, 95bbfd2c81224715 |
| Attention | 2.562402309x scored, 20524d06e3894595 |
| Prefill | 1.639290758x checked, 9ca79c7f676f4b63; scored 23e7c8b4c3e54869 passed, 1.663469245x |

Latest overall snapshot: rank 221 after the new MoE/decoder scores. Top 50 is
unmet. Prefill is excluded from overall placement by its official TASK.
Read STATE.json, all per-track NOTES.md, THIRD_PARTY_NOTICES.md and relevant
TASK.md/AGENTS.md. best/ holds exact source bytes and metadata; results/ holds
full official evidence and immutable controller histories.

The first Prefill phase reached its local 1.5x threshold and is archived;
this does not complete the overall Goal. No experiment controller is active.
Observe existing Prefill scored job 23e7c8b4c3e54869 without duplication. Next
source-review hypothesis is experiments/prepared/decoder_fused_pr_review.md:
fused lossless pooler/router for 8<T<=32, supported by current decoder profile.
Keep the entire original FP32 router-input path for T<=8. Do not adopt a public
whole-layer pipeline wholesale or infer performance from another author's score.

Current sources preserve BF16 GEMM/global-scale boundaries, BF16 effective norm
weights, FP32 norm intermediates, every route/key and all residual outputs.
Native decoder shared scratch requires sequential calls. Preserve dense MIT and
native Apache notices. No output/input caching, harness/timing changes, dropped
routes/keys or persistent BF16 expert expansion. Dot's precision review is in
MOE_NOTES.md and retained in current sources.

Use the signed-in account and free official H200 service only. No subagents,
paid GPUs, external providers, global configuration changes or token output.
Initialize a scoped phase before further trials, reuse cached baselines, retain
job handles across observation failures, and publish evidence at phase boundaries.
Never hand-edit controller journals. One decoder upload failed TLS; official jobs
were reconciled before a successful retry. Complete raw evidence is preserved.

Latest: Prefill scored job 23e7c8b4c3e54869 passed at 1.663469245x.
The next decoder fusion draft has passed its bitwise codec/layout and Python
binding audit, but GPU correctness and speed remain unmeasured. One check
will test 9..32-token fusion. Preserve the complete FP32 small-token router.
Publish only if the checked gain exceeds measurement uncertainty.
