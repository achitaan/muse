# Muse Open Kernel Challenge

GPU-kernel optimization sprint for [`muse_bf16_proj`](https://openkernelchallenge.dev/problems/muse_bf16_proj).

Start with [HANDOFF.md](HANDOFF.md). Paste [CLOUD_PROMPT.md](CLOUD_PROMPT.md) into a Codex Cloud task attached to this repository.

The coding session ran locally in Codex Desktop. Official evaluations run on the challenge's remote H200s. This repository does not provision paid GPU infrastructure.

## Verified so far

- Official PyTorch starter: 9/9 correctness tests and 30/30 scored shapes passed on H200; **0.908671× SGLang**, 95% interval **[0.895041, 0.922301]**. This is slower than SGLang.
- Initial Triton candidates: 17 SM90 specializations compiled through cubin; five follow-up specializations also compiled.
- Independent numerical review and reproducible CPU router models passed, including negative controls. These do not establish GPU correctness or performance.
- Six optimization checks were pending at the handoff snapshot. Inspect their current status before drawing conclusions or launching replacements.

The small-batch candidate also passed official correctness at **0.912009x SGLang**, 95% interval **[0.898329, 0.925689]**. Neither completed candidate beats SGLang overall. No scored submission has been made.

## Files

- `muse_bf16_proj/`: unmodified official task, baseline, reference, checks and starter.
- `candidates/`: standalone optimization candidates.
- `best/starter.py`: currently verified fallback; replace only after authoritative evidence.
- `results/`: complete official result, pending job metadata, frozen payloads, CPU checks and compiler evidence.
- `scripts/official.py`: Linux/Windows CLI driver with immutable payloads and a hard deadline for new checks/submissions.
- `scripts/analyze_results.py`: per-shape official result summaries.
- `scripts/offline_compile.py`: optional SM90 compile-only verification.
- `scripts/verify_arithmetic.py`: deterministic CPU router arithmetic models.

Credentials, downloaded tool binaries, compiler wheels and caches are excluded from Git.
