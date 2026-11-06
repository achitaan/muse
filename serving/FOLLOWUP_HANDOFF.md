# Serving follow-up: S1 isolated; ordered MoE rejected

Completed November 5, 2026 Toronto time (raw timestamps use November 6 UTC).
Branch: `codex/serving-precomp-20261105`; starting commit `6aa03df` is preserved.
**S1 remains the unchanged first official-check candidate. Both native MoE
experiments remain disabled. This follow-up establishes no new speedup.**

## S1 result

The actual imported helper consumed identical saved BF16 expert-output tensors in
both arms, without invoking GEMMs. Coverage: T=0/1/8/16/31/32/33; local/mixed/remote
slots; ordinary/tiny/near-epsilon/zero inputs; weight-first and weight-after-norm;
poisoned invalid slots. There were 168 comparisons and 84 saved fixtures.

- Stock repeats were bit-identical in all 168 comparisons.
- S1 differed in 37 cases, all below T32. All 37 first differ at FP32 variance
  reduction; masking and pre-norm gain application matched.
- The instrumented fused kernel's final output matched the actual S1 helper in all
  144 nonempty cases. Thus its intermediate evidence is bound to observed helper output.
- Maximum observed S1 relative L2 was 0.00010304358 (~0.0103%). This is an observation,
  **not a serving tolerance or permission to claim answer agreement**.
- Disabling FP fusion in the diagnostic still left 31 small-token cases different.
  It does not fix the differing variance reduction. No production flag was changed.
- All-remote and zero cases remained exactly zero. T32/33 select identical fused
  code in both actual helper arms and matched exactly.

The observed differences are consistent with FP32 reduction ordering, rather than
a missing BF16 materialization. No unintended precision boundary was identified;
the approved correction condition was not met. Zero numerical correction cycles
were spent. Retain the original S1 patch and its previously measured 23–40 us savings;
no complete-operation timings were repeated because the patch is unchanged.
Actual-import/kernel tests: **115 passed**.

## Ordered partial-buffer MoE result

The sole follow-up replaces contended CAS partial accumulation with one uniquely
owned buffer per CTA endpoint, then FP32 RN reduction in increasing contributor
order. It preserves the original scheduling, finishers and publication counters.
See [ORDERED_FIXUP_REVIEW.md](ORDERED_FIXUP_REVIEW.md) for ownership, replay,
publication, traffic and memory arguments. The CPU model passed 1,284 cases.

The exact candidate compiled in 65.31 seconds. Measured resources were 132 CTAs,
576 threads/CTA, 231,168 bytes dynamic shared memory and one resident CTA/SM.
Workspace was 196,135,200 bytes for one bound layer; the modeled 60-layer total is
196,180,512 bytes (~187.1 MiB), below 256 MiB/rank. No extra persistent weight bank.
All six tile-dequantization comparisons passed.

T128 compared GEMM1, SwiGLU, GEMM2 and stock norm/combine for uniform, mixed, hot,
all-local, all-remote and padding routes. Uniform, mixed and all-local stock repeats
were identical while the candidate differed starting at GEMM1. Combined relative L2:

| Routing | Candidate versus stock | Stock repeat |
|---|---:|---:|
| Uniform | 0.0530% | 0 |
| Mixed | 0.0537% | 0 |
| Hot | 0.0637% | 0.0137% |
| All-local | 0.0650% | 0 |
| All-remote / padding | 0 | 0 |

These are diagnostic differences, not acceptance thresholds. Invalid outputs were
initialized and all-remote/padding results were exactly zero. The changed summation
does not resolve the existing GEMM equivalence problem. **Rejected at T128**; no
T256, performance measurements, edge-scale expansion, broad replay sweep or sanitizer
run followed this failure. No new kernel correction or replacement hypothesis was
attempted. The separate experimental patch has an empty accepted-shape set.

The disabled experimental patch applies cleanly to pinned stock and composes with S1.
All **44 integration tests passed**, and combined `git diff --check` passed. These
tests cover imports/dispatch and do not confer numerical acceptance on the GPU kernel.

## Artifacts and reproduction

| Artifact | SHA256 |
|---|---|
| `s1_postnorm_v2.patch.diff` (unchanged, 7,375 bytes) | `ea7f03abf143c671694d4caa421530f8d76bcc531a7c4b6063a358a1aabf9a7f` |
| `experiments/ordered_fixup/native_marlin_wgs.cu` | `9e3a734f5d1245459654d03fde6c03398ba38a32108f58f05003583bf629698c` |
| `experiments/ordered_fixup/moe_ordered.experimental.patch.diff` (168,808 bytes) | `0a9ccac0622f4fc675d94c7f4523629c3714478cb6a5b12251469ed4d0d658d5` |

Both patches satisfy the freshly fetched allowed-path/suffix rules, 409,600-byte cap,
400-character added-line limit and forbidden-word checks. The earlier v3 patch,
sources, hashes and raw results remain unchanged. [FOLLOWUP_RESULTS.json](FOLLOWUP_RESULTS.json)
contains the stage comparisons; [FOLLOWUP_SOURCE_HASHES.json](FOLLOWUP_SOURCE_HASHES.json)
binds the scripts and sources. Readable logs are under `results/followup-20261105/`.
`raw-evidence.tar.gz` contains all 97 exported files, including the 84 saved input
tensors, and the remote checksum manifest. Every file matched before pod deletion.

Use the image and runtime in [PRECOMP_REPRODUCE.md](PRECOMP_REPRODUCE.md).
`followup_prepare.sh` creates fresh pinned stock/S1 checkouts and runs isolation.
It assumes `/workspace/muse` contains these sources and no checkout exists at its
target paths; it must not be rerun over an existing setup. GPU commands below cost
rental time and are recorded for reproduction, not a request to repeat failed work.

```bash
# Offline; leaves frozen native v3 untouched.
python serving/audit_ordered_fixup.py
python serving/build_ordered_fixup.py

# H200, after fresh setup with followup_prepare.sh:
export PATH=/opt/sglang/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=/workspace/sglang-s1/python
export MAX_JOBS=4 TORCH_CUDA_ARCH_LIST=9.0a
timeout 8m python -u serving/run_ordered_probe.py check_native_marlin.py \
  --stage full --tokens 128 --patterns uniform mixed hot local remote padding
python serving/build_native_patch.py --sglang-checkout /workspace/sglang-stock \
  --source-dir serving/experiments/ordered_fixup \
  --output serving/experiments/ordered_fixup/moe_ordered.experimental.patch.diff

# Local saved-evidence summary; requires no GPU.
python serving/summarize_followup.py
```

Cold stock Marlin JIT compilation accounted for several minutes before the full
probe produced comparisons. First SSH setup also required a restart: WSL cleared
the initial `/tmp` key between invocations. The replacement key used a task-owned
persistent directory and was deleted after cleanup. No account SSH keys were added.

## Budget, cleanup and opening

Pod `0q6kvxe9ribr00`, one H200 in US-CO-1 at $5.29/hour, existed from
04:11:35.188 to at most 04:25:58 UTC. Conservative compute estimate **$1.26785**;
80-GB ephemeral disk estimate **$0.00266**; follow-up total **about $1.27**.
Billing had no posted records for this new pod at cleanup, so these are estimates.
Using the earlier conservative $5.16 estimate gives about **$6.43 total** and
**$43.57 remaining** under the original $50 allowance. The additional $15 cap
and its S1/MoE/reserve sublimits were all respected; none is a spending target.

Deletion returned 204 and a fresh list showed no pods. No network volume was created.
All exported evidence is local; no task-created compute/storage remains billing.
See [FOLLOWUP_SESSION.json](FOLLOWUP_SESSION.json) for timestamps and cost allocation.

The official manifest was fetched again in this follow-up: same pinned SGLang
`a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f` and image digest; serving opens November 10
at noon Toronto, three C16 checks/day and one scored run/day. It was closed during
this work. Preparation cutoff remains November 9 at 8 p.m. Toronto.

At opening, re-fetch the manifest and use one C16 check for **S1 alone**. Reserve
additional checks for diagnosed repairs. Only a passing official answer/TTFT gate
permits the scored C1/4/16/32 run. **Zero official checks or submissions used here.**
No future job or background watcher was installed; the opening-day action is pending.

## Self-review

Reviewed source binding, original-source preservation, isolated-input authority,
diagnostic-versus-production output equality, subnormal-preserving additions,
endpoint ownership, counter publication, workspace offsets, disabled dispatch,
patch composition, numerical stopping decision, cost allocation and cleanup.
No independent review or broad GPU safety certification is claimed. Differences
remain explicit, and no new performance gain is inferred from source changes.
