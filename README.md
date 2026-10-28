> Resumed October 28: the user removed the deadline and requested top 50 overall, main chat only. Current focus is whole decoder layer, with conservative check-first use of quotas. Dense-projection quota remains exhausted. Attention preprocessing passed all 16 tests and 12 benchmarks at 1.120441x checked; scored job f6b899f674914ef2 is pending. The earlier sprint snapshots below are historical. See STATE.json.

# Muse Open Kernel Challenge

Optimization of `muse_bf16_proj` on the official remote H200 evaluator.

## Final October 28 checkpoint

- Best scored result: **1.527771x SGLang**, provisional rank **85**.
- Rank 50 cutoff at the snapshot: **1.5448x**. Top 50 was not achieved.
- Job `6131bc96427e4486` passed all 9 tests, 30 benchmark cases and integrity checks.
- 95% interval: [1.504855, 1.550688]. Small point differences are not conclusive wins.
- Exact scored source: `best/verified.py`, SHA256 `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b`.
- Lean alternative: `best/lean.py`, independently scored at 1.525310x.

All **20 checks and 10 scored submissions** are used. Further H200 optimization
is blocked by the official daily quotas; reset is after the user's 04:00 Toronto
/ 08:00 UTC deadline. A new deadline and renewed quota are required to continue.

The implementation derives from Nathen Fernandes' MIT-licensed v18. Local work
extended the exact additive decoder and evaluated alternative layouts. Preserve
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and embedded license notices.

## Reproduction and handoff

- [HANDOFF.md](HANDOFF.md) and [STATE.json](STATE.json): exact current state.
- [CLOUD_PROMPT.md](CLOUD_PROMPT.md): continuation prompt.
- `results/sprint-20261028-evidence/manifest.json`: full official evidence and frozen source hashes.
- `results/sprint-20261028-phase1/`, `phase2/`, `phase3/`: controller histories.
- `results/sprint-20261028-phase3/report.html`: standalone experiment report.
- `results/sprint-20261028-aggregate-audit.json`: independent recomputation of the score from all 30 cases.
- `scripts/verify_codec.py`: CPU bit-pattern checks; these do not establish GPU speed.

Editing ran locally in the main chat; GPU measurements used the free official
H200 service. No paid infrastructure is provisioned. Credentials and local caches
are excluded from Git. Historical October 23 documents remain under `results/`.
