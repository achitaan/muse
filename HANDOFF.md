# Current handoff - October 28, 2026

Goal: get as close as possible to top 50 overall, main chat only. No deadline.
User requested fewer checks and scored runs: no blind sweeps, repeated baselines
or frequent quota queries. Require a concrete question and a material expected
gain. Original track order is MoE, Decode Layer, Attention Decode, Prefill last.
The final track is now prepared for its first full check.

| Track | Current verified evidence |
|---|---|
| Dense | 1.527771327x scored, 6131bc96427e4486; both daily quotas exhausted |
| MoE | 2.312613084x scored, b4a10b76347e4cb0; best/moe_verified.py |
| Decode Layer | 1.815511785x checked, 2f41872843fe4490; scored 95bbfd2c81224715 pending |
| Attention | 2.562402309x scored, 20524d06e3894595; best/attention_checked.py |
| Prefill | First grouped-MoE candidate prepared; unmeasured |

Read STATE.json, MOE_NOTES.md, DECODE_NOTES.md, ATTENTION_NOTES.md,
PREFILL_NOTES.md and official TASK.md/AGENTS.md for the active problem. Prefill
does not count toward overall placement. Last overall snapshot was 619 after
attention, before the new MoE and decoder scores. Top 50 remains unmet.

The current decoder keeps the FP32 router for small token counts and all
SGLang residual boundaries. It combines checked native attention/MoE with the
checked lossless dense path. The native MoE restores BF16 GEMM/global-scale
boundaries and FP32 post-norm intermediates, preserving Dot's precision review.
Native attention/MoE shared scratch requires sequential calls. Preserve Apache
native-source attribution and the dense MIT notice. Never drop routes/keys,
cache inputs/outputs, change timing/harness or persist BF16 expert expansions.

The decoder controller phase is archived; native overall Goal remains active.
Initialize one Prefill scope from the prepared candidate and use its full
5-test/8-benchmark verifier and cached-evidence guard. Do not interpret a local
controller target as completion of the overall Goal. Existing job handles must
be observed after timeouts, never duplicated. One initial decoder upload hit a
TLS error; official jobs confirmed no accepted run before the successful retry.

Use the existing account and free official H200 service only. No subagents,
paid GPUs, external providers or global orchestration changes. Never read,
print or commit tokens. Commit/push evidence at controller phase boundaries;
never edit immutable controller journals by hand.
