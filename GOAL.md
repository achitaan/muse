> CURRENT GPU AUTHORIZATION: checks enabled sparingly through October29,2026
> at20:00Toronto (2026-10-30T00:00Z). EVALUATION_POLICY.json governs pacing:
> one active GPU job, >=60minutes between new checks, four checks/track reserved.
> Score only after material source-bound gains; quota is not a spending target.
> This supersedes earlier no-GPU and review-only holds below. Main chat only.
> Next: one MoE db0f6481 prepare-only phase diagnostic, no earlier than
> 2026-10-29T02:21:46UTC. D1 was checked and rejected; retain all incumbents.

# Active objective

Get as close as possible to top 50 overall in the Open Kernel Challenge, using
the current account and free official H200 service. Work in this main chat only
and preserve the selected model. The user removed the original 04:00 Toronto
deadline. The objective remains unmet; the current authorized GPU window ends
at 20:00 Toronto on October 29, 2026 (2026-10-30T00:00:00Z).

Current authorization: local implementation, CPU verification and sparing GPU
checks. EVALUATION_POLICY.json governs launch spacing and quota reserves.
Public scored submissions require material source-bound gains. The supplied
plan has been reviewed. No subagents, model changes, paid services or global
changes.

The requested ordered pass MoE -> Decode -> Attention -> Prefill is complete
with full passing coverage. Continue justified optimization, conserving checks
and scores. Do not run blind sweeps, repeated baselines or cosmetic submissions.
No paid infrastructure, external model routing, subagents or credential output.

Use OPTIMIZATION_PLAN.md for current priorities and closed hypotheses, and
STATE.json/HANDOFF.md for exact source identities, job handles and snapshots.
The latest saved overall row is155/1451; the API gives ranks only and its exact
composition is not established. Do not treat an average of track speedups as
an official combined speedup. Corrected retained sources differ from older
Decode/Attention API best rows; preserve that distinction.

MoE skinny16 is closed: one compiler repair, passing GPU probes, all three
paired timing patterns slower, no selected bucket or score. Retain33114871.
Decode packaging is now CPU-verified at bc528949: 893369 bytes, with all D9
CUDA and the expanded executable AST exact. It remains an unevaluated draft;
best/decoder_verified.py and candidates/decode_active.py remain D9. See
experiments/prepared/decoder_source_package_review.md for frozen evidence.
The CPU-only df68deb6 follow-up bypasses two unreachable dense-router setup
factories while preserving all runtime functions. Its preparation savings are
unmeasured; historical D9 setup was 256.897s/300s. D1 is closed: both fixed
families passed GPU probes but lost at M128 and were not selected. Do not
transfer or rescore it. Next use one bounded MoE phase diagnostic to identify
exposed work before a new runtime hypothesis; no diagnostic promotion.
Preserve numerical precision, exact weights, all routes/keys and fresh outputs.
