# Open Kernel Challenge handoff

Snapshot: October 23, 2026, approximately 12:40 PM America/Toronto (16:40 UTC). Refresh cloud job status immediately; this document is a checkpoint, not a live status feed.

## Objective and execution ownership

The original user objective is preserved verbatim in `GOAL.md`: optimize one kernel legitimately, use only the existing Codex subscription, no paid infrastructure/providers/credits, and stop at **2026-10-23 13:00 America/Toronto / 17:00 UTC**. The user subsequently requested a GitHub handoff and deletion of the local workspace. Local work stops for that transfer. No Ralph background runner was launched; one native goal owned the local sprint. All native implementation/review agents finished.

Continue in one Codex Cloud task only. A cloud environment has not been created or started by this repository. Check the clock first: if the original deadline has passed, do not restart optimization or launch new evaluations unless the user supplies a new deadline. Read-only retrieval of already queued results can still document what happened.

## Competition and selected kernel

Official site: https://openkernelchallenge.dev/

Selected `muse_bf16_proj`, because it is a bounded BF16/FP32 projection task with plausible small-M and router gains. Compute `x @ W[:n].T`; BF16 outputs except FP32 router. Read the official files in `muse_bf16_proj/`, particularly TASK.md and AGENTS.md.

BF16 weights and inputs must retain their precision. Router FP32 weights may use BF16 hi+lo with FP32 accumulation. Its relative L2 tolerance is1e-4; ordinary projections1e-2; power-of-two probe rows1e-3. Preserve graph compatibility, full computations, fresh inputs and all odd/outlier test cases. Do not cache outputs, modify the harness or exploit evaluation details.

## Current evidence

Starter job `05a52f7e67084663` passed9/9 tests and30/30 scored shapes on H200. Speedup **0.9086713229984033× SGLang**, 95% interval **[0.8950412531534272,0.9223013928433792]**. This is not a performance improvement. Raw result: `results/05a52f7e67084663-result.json`. Summary: `results/official-results.md`.

Non-router starter projections roughly match SGLang. The FP32 router is expensive: larger-M times19.7–46.0µs versus SGLang14.8–17.7µs. At M8 both take about26µs. This supports router fusion as the highest-priority measured bottleneck.

Independent Astra review found no blocking declared-shape defect in initial small, large and router candidates. CPU high/low models passed; BF16-only negative controls failed. See `results/independent-review-01.md` and `results/cpu-arithmetic.json`. All17 initial and5 follow-up SM90 compiler specializations passed. This evidence is **not GPU runtime validation**.

## Existing cloud jobs — poll these, do not recreate them

| Job | Source | Status at snapshot |
|---|---|---|
| `05a52f7e67084663` | `best/starter.py` | Passed;0.908671× |
| `9b29fd36a8d646dd` | `candidates/small_batch.py` | Passed; 0.912009x |
| `3714c74106a44f6e` | `candidates/large_batch.py` | Queued |
| `6e71899c989747d9` | `candidates/router_only.py` | Queued |
| `2f290a08be49498d` | `candidates/hybrid_v1.py` | Queued |
| `179a7589fc6d4bf0` | `candidates/small_adaptive.py` | Queued |
| `020a3c8891a1489c` | `candidates/router_gemv.py` | Queued |
| `4f2f2262cdc14b0e` | `candidates/router_packed.py` | Queued |

Quota at last query: **8/20 checks used;12 checks left;0/10 scored submissions used**. Limits reset00:00 UTC; refresh with `okc problems --json`. The queue was the main constraint: the starter took about11 minutes to return; the next jobs remained queued longer. Do not treat a queued or observation-timeout result as terminal.

Small-batch job `9b29fd36a8d646dd` has now passed 9/9 tests and 30/30 shapes: **0.912009x SGLang**, 95% interval **[0.898329, 0.925689]**. Six optimization jobs remain queued. No overall speedup over SGLang is established. Raw evidence is saved alongside the starter result.

## Candidate map

- `small_batch.py`: contiguous-K FP32-accumulating GEMV for M1; BF16 Tensor Core split-K for M2–32 with FP32 scratch/reduction. Split8 generally;16 for sliced unpooler. Larger projections and router use PyTorch fallback.
- `large_batch.py`: conventional grouped tiled BF16 GEMM for M>=32. BM32/64,BN64,BK64,4 warps,3 stages; modest split-K for small CTA grids. FP32 accumulation and masked tails.
- `router_only.py`: fused BF16 hi/lo dot products, separate FP32 accumulators, split-K reduction. BF16 projections use cuBLAS. Assumes declared router N256; generalization would need a separate weight stride.
- `hybrid_v1.py`: initial small-batch kernels plus fused router, retaining cuBLAS for M>32. This is already queued.
- `small_adaptive.py`: reduces splitting based on output-tile count; wide-N o/unpooler write directly, eliminating scratch/reduction. May lose bandwidth occupancy; measure before selecting.
- `router_gemv.py`: direct original-FP32 weight reduction for router M1, fused high/low path otherwise.
- `router_packed.py`: concatenate router hi/lo weights along output width in prepare; one cuBLAS BF16 GEMM with FP32 output, then sum the two halves. No Triton dependency.
- `integrated_unverified.py`: all three initial custom kernel families; **not submitted**. Do not mistake it for the best candidate.

Initial local file SHA256s are in `results/initial-hashes.json`; the files were frozen before upload. Later submissions use immutable `results/payloads/<sha256>.py` copies and launch records. Server `content_hash` is a separate identifier; it differs from local file SHA256 and is not assumed equivalent.

## Cloud setup

Use an existing signed-in Codex subscription and this GitHub repository. Standard Codex Cloud CPU environments are sufficient to edit code and use the official remote evaluator; do not assume they contain a GPU. Enable permitted network access to the official challenge endpoint if needed.

Install the official CLI in the checkout:

```sh
OKC_INSTALL_DIR="$PWD/.tools" OKC_NO_MODIFY_PATH=1 sh install-okc.sh
.tools/okc version
.tools/okc whoami
```

Local login does not transfer to the cloud. The user accepted the challenge terms and signed in as `achitaan`; optional Meta recruiting-email disclosure was declined. No credential is committed. Use a securely configured `OKC_TOKEN` secret in the cloud environment or complete the official human login. Never print or commit tokens. The official AGENTS.md explains the authenticated token handoff; the user must provide it through a secret mechanism, not source control.

The included CLI wrapper now runs directly on Linux and through Ubuntu WSL on Windows:

```sh
python scripts/official.py quotas
python scripts/official.py jobs
python scripts/official.py status 6e71899c989747d9
python scripts/analyze_results.py
```

Check/submit freezes source bytes and records its SHA256. These commands refuse new work after the original deadline:

```sh
python scripts/official.py check candidates/hybrid_v1.py
python scripts/official.py submit best/submission.py
```

Do **not** repeat that example check blindly: hybrid_v1 is already queued.

## Verification and next actions

1. Retrieve every existing job. Save full JSON; inspect correctness, errors, paired per-shape ratios and95% intervals.
2. Preserve the best complete correctness-passing artifact. Both `best/starter.py` and `candidates/small_batch.py` pass official correctness. Small batch is 0.912009x SGLang (95% interval [0.898329, 0.925689]); neither beats SGLang overall. Keep the starter recoverable; the small difference is within the reported uncertainty.
3. Compare per-shape improvements, especially router and small-M projections. Avoid retaining gains within noise or guessing that fewer splits is faster.
4. Combine measured winners into one self-contained submission; no local imports. Reuse `scripts/build_candidate.py` as appropriate, then independently review the actual final file.
5. Run a meaningful official check of the integrated artifact and submit a verified improvement through `okc submit` if time permits. Preserve the exact checked bytes.
6. At the hard deadline stop new work, save all results/pending jobs/next actions, and stop agents. Do not change the deadline without user instruction.

For CPU precision checks, install/use PyTorch and run `python scripts/verify_arithmetic.py`. It was tested with Python3.10 and PyTorch2.3.1+cpu. This is a numerical model, not Triton execution.

Optional offline SM90 compilation used Triton3.8.0, Python3.14.4, and no GPU. Recreate a matching project-local Triton installation under `.tools/offline-triton/site-packages` (not committed), or adapt the script to an existing installation; see `results/offline-README.md`. Run `python scripts/offline_compile.py --suite all` and `--suite followup`. Compilation validates lowering only; the judge may have another Triton version.

## What was excluded from Git

Downloaded okc binary, Triton wheel and compiler/cache files, Python caches, and all authentication material. Code, original task, candidate payloads, raw evaluation evidence, scripts and this handoff are committed. Do not restore credentials from repository history; none should be present.
