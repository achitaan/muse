# Current handoff — October 28, 2026

Pursue top 50 overall in the main chat only. Original 04:00 deadline was removed.
The requested order is MoE, Decode Layer, Attention Decode, Prefill last. Current
focus is Attention Decode; conserve checks and scored submissions. Native Goal
remains active and is the sole foreground continuation controller.

Read STATE.json, TRACK_NOTES.md, MOE_NOTES.md, DECODE_NOTES.md and the current
problem's TASK.md and AGENTS.md before changing code. Validate the active
experiment controller and existing official job handles before launching work.
Never resubmit a job because observation timed out.

Dense projections: exact best source best/verified.py, scored job
6131bc96427e4486, 1.527771327x. Both daily quotas are exhausted. Historical rank
85 is not a live placement claim.

MoE: best/moe_verified.py and best/moe_metadata.json identify the stricter FP32
norm candidate. Scored job 650aa66fd3334d8c passed at 1.268508633x. The older
1.271471562x check followed the public starter's extra BF16 norm rounding; the
current candidate preserves pinned SGLang's FP32 norm intermediates instead.
Do not revert that precision contract for timing noise. See MOE_NOTES.md for
Dot's review and negative CUDA/grouped-kernel evidence.

Whole decoder: best/decoder_verified.py and best/decoder_metadata.json identify
the 1.168638714x checked source, passing all 13 tests, 12 benchmarks and integrity
checks. Scored job 88f8b06932fa43b7 is pending; observe it rather than launching
another. The candidate integrates lossless dense kernels, the checked MoE changes,
and fused Q/K preprocessing while preserving the original FP32 router path.

Attention: best/attention_checked.py passed 16 tests and 12 benchmark cases at
1.120441329x; scored job f6b899f674914ef2 passed at 1.116457695x. Next hypothesis
is partitioned online-softmax attention with every visible key included, exact
raw-byte FP8 reads and fused output norm/gate. CPU interval coverage, stable
merge and FP8-value preservation were checked; GPU correctness/speed are untested.
Read the prepared addon and audits under experiments/prepared/.

Preserve original BF16/NVFP4 weight formats, permitted FP32-router hi/lo splitting,
FP32 accumulation, all activation boundaries, every attention key, fresh inputs,
cache updates, routing and residual outputs. No output caching, harness access,
timing changes, dropped routes or persistent predecoded BF16 expert weights.
Preserve the dense upstream MIT notices. Full official tests are authoritative;
CPU proofs and static guards do not establish H200 correctness or performance.

Use the existing signed-in free official service; never print or commit tokens.
No paid GPUs, external providers or global orchestration changes are authorized.
Commit/push measured evidence at phase boundaries. Do not edit run.json/events by
hand or commit outside a controller's approved path scope during an active run.
