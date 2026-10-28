# Current handoff - October 28, 2026

Goal: approach top 50 overall in this main chat only. No deadline. Conserve checks
and scored runs: no blind sweeps, repeated baselines or frequent quota queries.
The requested pass (MoE, Decode, Attention, Prefill last) is complete, with full
passing coverage on every track. The overall Goal remains active and top 50 unmet.

| Track | Published speedup | Place at 15:05 UTC | Job |
| --- | ---: | ---: | --- |
| Dense | 1.527771327x | 97/1242 | 6131bc96427e4486 |
| MoE | 2.312613084x | 108/991 | b4a10b76347e4cb0 |
| Decode | 1.874889991x | 193/866 | e5b95129043d442d |
| Attention | 2.562402309x | 80/945 | 20524d06e3894595 |
| Prefill | 1.663469245x | 122/573 | 23e7c8b4c3e54869 |

Overall: 193/1427. Prefill is excluded from overall by its official TASK.
All scored jobs above are terminal and passing. No pending publication.
No experiment controller is active; phase4 Decode is archived and exported.

Read STATE.json, per-track NOTES.md, THIRD_PARTY_NOTICES.md and applicable
TASK.md/AGENTS.md. best/ holds exact measured source bytes and metadata; results/
holds full official records, profiles, frozen sources and controller histories.
The latest decoder combines fused pooler/router for 9..32 tokens and full-FP32
router/top-8 for 1..8. Original F.linear is retained as fallback. See DECODE_NOTES.md
for numerical boundaries, exact hashes, CPU audits and clone reproduction.

Next: inspect profiles for a material MoE improvement. Do not spend a run on a
weak tile sweep. Current expert weights remain exact NVFP4; BF16 GEMM/global-scale
boundaries and effective norm weights are preserved, with FP32 norm arithmetic.
Keep every route/key, fresh input, cache write and residual output. Shared native
scratch requires sequential calls. No output caching or persistent BF16 expansion.

Use only the existing account and free official H200 service. No subagents, paid
GPUs, external providers, global configuration changes or credential output.
Dense's last observed daily quotas were exhausted. Check other quotas only when
needed. Initialize one scoped experiment phase before trials, reuse cached
baselines, and preserve job handles after observation failures. Never hand-edit
controller journals. Commit and push complete evidence at phase boundaries.

Pinned public sources: Vjeux/okc-2026 f35a41ee1c13e316dabbf772830d47b23b453393
(Apache-2.0), NathenFernandes/muse-bf16-proj
7385e5d8f70980cfd5322ea254edc0b2b8ca90f7 (MIT). Preserve all embedded notices.
