# Muse Open Kernel Challenge

Optimization of `muse_bf16_proj` on the official remote H200 evaluator.
Read [HANDOFF.md](HANDOFF.md) and [STATE.json](STATE.json) before continuing.

## Latest verified result

- Scored job `1183fb1c393c4422`: **1.521764x SGLang**, all 9 tests and 30 benchmarks passed.
- Leaderboard snapshot at 05:13 UTC October 28: **106th**; rank 50 was 1.541x.
- Exact source: `best/verified.py`; SHA256 `41a3c15f1f94560314f227adf718e21d4b17a03c5d58bc864e17827048c21bdf`.
- Check job `60a349bee6944644`: 1.522875x, 95% interval [1.500031, 1.545718].

The candidate derives from Nathen Fernandes' MIT-licensed v18 implementation.
Local changes extend its exact additive decoder. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Small point-score gains inside the judge's uncertainty interval are not conclusive performance wins.

The current user deadline is **October 28, 04:00 Toronto / 08:00 UTC**.
Work stays in the main chat. Code editing runs locally; GPU evaluations run on the
challenge service. No paid GPU infrastructure is provisioned.

## Evidence and continuation

- `results/sprint-20261028-evidence/manifest.json`: source hashes and raw official results.
- `results/sprint-20261028-phase1/` and `results/sprint-20261028-phase2/`: immutable experiment controller histories.
- `results/sprint-20261028-cpu/`: exact codec and layout checks; these are not GPU timing evidence.
- `muse_bf16_proj/`: unmodified official task and reference material.
- `CLOUD_PROMPT.md`: continuation prompt with the current deadline and source attribution.

Credentials, CLI binaries, compiler caches and local measurement caches are excluded from Git.
