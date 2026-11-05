# bs32-only VMS static tolerance admission

Check file: `candidates/moe_vms32.py` on `loop/moe-vms32`, from `4ccd503`.

Prior caller-provided measurement: `e115610794554d17` had bs32 static
geomean 0.987 with clean base/base nulls, but bit-exact agreement failed for
VMS AND repeated incumbent runs. This revision removes that unjustified
bit-equality veto; it does not change either CUDA engine or assume adoption.

## Admission

- Only bs32 is considered. bs16 always calls the unchanged incumbent.
- Independent FP32 public reference with TF32 disabled, restored on exit.
- Each draw's VMS global relative L2 <= 0.015, row error <= 0.075,
  and global error <= 1.10 times the incumbent's own same-draw error.
  Both first and last measured outputs are checked; finite incumbent
  errors must also meet the public bounds. Sticky kernel errors veto.
- Retain five graph-replay/input-change stress cases under the same bounds.
- Main phase: 24 freshly generated judge-like draws, seeds
  `1132288 + 37*i`, disjoint from the earlier diagnostic's seeds.
- READ512 eviction, captured graphs, six rounds of ABBA and BAAB, interleaved
  VMS/incumbent and independent captured incumbent/incumbent null arms.
- Every draw's null median in [0.995, 1.005]; geometric mean of paired
  per-draw ratios <= 0.990 and paired median <= 0.995.
- Only after passing main and stress, confirm on eight independent seeds
  `1732288 + 37*i`. Geomean and median must both be <= 0.995, with the same
  per-draw numerics/null vetoes. No retries, threshold fitting, or cherry-picking.
- Runtime dispatch depends only on shape and compatible local expert range.
  No routing readbacks, output caching, host synchronization, or per-call tuning.
- `MOE_VMS32` prints each draw's phase, seed, ratio, null, errors and timings.
  `MOE_VMS32_PICK` prints adoption, both phase summaries and total prepare time.

## Local validation

`python3 scripts/test_moe_vms32.py`: 18 tests passed, including numerical
boundary/veto cases, main and confirmation admission, actual prepare
orchestration with fake CUDA, fresh-seed flow, failed-arm cleanup, TF32
restoration, graph/event ordering, read-flush exclusion, signatures, static
bs32-only runtime dispatch, fresh outputs, and 4,624 repack-byte addresses.

AST parsing passes. All four embedded CUDA/C++ strings are byte-identical
to the parent. Every load_inline has host/NVCC C++20 and explicit
`-gencode=arch=compute_90a,code=sm_90a`. Forbidden process-start grep is clean.

No local GPU or nvcc; CUDA 13 compilation, real error ratios, timing admission
and <285-second prepare walltime remain unverified. Gate deadline is cooperative
at 265 seconds; the inherited incumbent build and any in-flight threaded
load_inline cannot be forcibly interrupted. One additional VMS build (not two)
and one timing pass per draw reduce work substantially from the prior diagnostic.

No check was submitted and nothing was pushed or committed to main.
The shared TRIED helper is absent; the branch-local ledger row needs caller
propagation under the task's no-main-write restriction.
