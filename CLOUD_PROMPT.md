> Current focus: whole Decode Layer, transferring verified MoE improvements. Read MOE_NOTES.md, TRACK_NOTES.md and STATE.json. MoE scored publication is pending; observe its existing job rather than duplicating it.

> Latest focus: MoE experts (muse_moe_w4a16), at the user's request. Preserve quota. Read TRACK_NOTES.md for retrieval of attention/decoder results and pending jobs. Dot's additional ideas are pending.

Continue toward top 50 overall, using the main chat only. The user removed the original deadline and requested whole decoder layer as the current focus, with conservative use of evaluations. Read the latest STATE.json, the resume note at the top of HANDOFF.md, THIRD_PARTY_NOTICES.md, and muse_decode_layer/TASK.md and AGENTS.md. Use checks for focused hypotheses before spending scored runs. Preserve current best sources and immutable official evidence. Inspect the active controller and existing job handles before launching anything.

Attention preprocessing passed 16 tests and 12 benchmark cases at 1.120441329x in check job 89f9f7d227424d20. Exact source is best/attention_checked.py; scored job f6b899f674914ef2 was launched once and must be observed, not duplicated. Whole-layer work should first establish the supplied SGLang baseline, then assess local expert ID mapping and 32-expert sorting. The split-attention addon under experiments/prepared is unmeasured.

The following describes the historical dense-projection sprint, whose original deadline is no longer the current goal deadline:

The October 28 sprint used all 20 checks and 10 scored submissions. Its 04:00 Toronto / 08:00 UTC deadline must not be silently extended. Best scored job 6131bc96427e4486 passed all 9 tests, 30 benchmark cases and integrity checks at 1.527771327x, provisional rank 85; rank 50 was 1.5448x at the snapshot. Refresh live state rather than assuming those numbers remain current. Top 50 was not achieved.

Exact best source: best/verified.py, SHA256 edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b. A simpler independently validated alternative is best/lean.py, scored at 1.525309711x. Their small score difference overlaps the judge's uncertainty interval. The source derives from Nathen Fernandes' MIT-licensed v18 at commit 7385e5d8f70980cfd5322ea254edc0b2b8ca90f7. Preserve all attribution and license notices.

Use the main chat only, the existing Codex subscription and the free official H200 service. No paid GPUs, external model providers, credits or global orchestration changes. Establish the current baseline and read existing failures before new hypotheses. Preserve exact BF16 weights/activations, the allowed router hi/lo decomposition, FP32 accumulation, fresh inputs and complete products. Do not cache outputs, inspect or patch the harness, exploit probes, or alter timing.

Authenticate through an existing OKC_TOKEN environment secret or the human login flow. Never print or commit credentials. Verify source hashes and official full-case results; CPU codec checks and compilation are not H200 runtime measurements. Keep one controller, immutable measured source copies, per-shape records, confidence intervals and honest keep/revert history. Finish by updating evidence and pushing the handoff, reporting the actual placement and any unmet target.
