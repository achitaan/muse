# Current handoff - October 28, 2026

Pursue top 50 overall in the main chat only. Original 04:00 deadline was removed.
Requested order: MoE W4A16, Decode Layer, Attention Decode, Prefill last.
Native Goal remains active. Top 50 remains unmet. The latest user instruction
is to be more sparing with checks and runs. Require a concrete unresolved
question and a material expected improvement; avoid blind sweeps, repeated
baseline measurements and repeated quota/board queries.

Read STATE.json, TRACK_NOTES.md, MOE_NOTES.md, DECODE_NOTES.md,
ATTENTION_NOTES.md and the relevant official TASK.md/AGENTS.md before changes.
Reuse exact frozen sources and existing official job handles after timeouts.

| Track | Current evidence |
|---|---|
| Dense | best/verified.py; scored 6131bc96427e4486, 1.527771327x; both daily quotas exhausted |
| MoE | best/moe_verified.py; scored 650aa66fd3334d8c, 1.268508633x |
| Whole decoder | best/decoder_verified.py; scored 88f8b06932fa43b7, 1.172780190x |
| Attention | best/attention_checked.py; check ea693135be544501, 2.589760797x; scored 20524d06e3894595 pending |
| Prefill | No GPU evaluation yet; official TASK says it does not count toward overall |

MoE preserves pinned SGLang's FP32 post-norm intermediates. Do not restore the
older extra BF16 rounding for timing noise. Decoder retains the FP32 router
path, incorporates checked dense/MoE improvements and fused Q/K preprocessing.
Attention derives from Vjeux's Apache source; native CUDA algorithms are
unchanged. Cache changes eliminate initialization from graph replay. Scratch
requires sequential replay for calls sharing a device/launch signature.
All current sources pass the complete official correctness/benchmark coverage.

Preserve every attention key, weight codes/scales, activation boundaries,
cache updates, routing and residual outputs. No output caching, harness access,
dropped routes, timing changes or persistent BF16 expert-weight expansion.
Preserve dense MIT and attention Apache licenses and local change notices.

The attention controller phase is archived under
results/resumed-attention-phase2-20261028/controller/. No experiment controller
is currently active. Do not hand-edit immutable controller events. Initialize
the next scoped phase from cached evidence before any new trial. Publish
checkpoint documentation outside active candidate-only scopes.

Use the existing authenticated free official service; never print tokens.
No paid GPUs, external providers, subagents or global orchestration mutations.
Quotas are per track and reset at 00:00 UTC; unused quota need not be consumed
on weak hypotheses. Commit/push measured evidence at phase boundaries.
