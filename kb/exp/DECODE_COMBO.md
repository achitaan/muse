# Decode KV + router-stream combination

Branch `loop/dec-combo`, parent `51fe86a` (`loop/decode-kvpre`).
Standalone check file: **`kb/exp/decode_combo.py`**, 988,742 bytes.

## Changes

- KV: the exact `_kp_source` transform from `51fe86a`; tile-zero cached
  K/V loads before the attention dependency wait. No arithmetic, group-count,
  new-row patching, length-miss retry or incumbent source changes.
- POOL: the exact source transform and lossless layout from `0714254`;
  router stages use one 16 KiB transaction instead of eight 2 KiB transactions.
- d1S/d1F consider KV; v1S/v1F consider POOL and KV+POOL. B1 L8's existing
  non-WS attention plan supports KV. All other shapes remain incumbent.
- Removed disabled/non-selecting MFAB diagnostics and isolated-attention timing.
  Omitted redundant Python comment-only lines after the attribution banner.
  Every incumbent AST node before MFAB and every embedded CUDA source string
  remains unchanged from the KV parent.
- All fourteen extension build sites use the inherited C++20/SM90a wrapper,
  including host C++20. Candidate builds run in threads alongside base prepare;
  no process creation, external task imports, or versioned graph APIs.

## Admission

Three independently seeded whole-layer graph passes per shape, sixteen measured
quads per arm/pass: eight ABBA and eight BAAB. A separate captured base/base null
participates in each pass. Arm/null order rotates between rounds and passes.
Every replay has an actual 512 MiB float32 READ reduction outside CUDA events,
and actual cache destination rows are reset before replay. No isolated timings
or no-op arms influence selection.

Admit only if all three passes complete, every paired candidate/base ratio is
strictly below 1, their median is <=0.995, and both raw/paired null ratios are
inside [0.995,1.005]. Best median paired ratio wins among admitted arms.
`DEC_COMBO` reports every arm/pass; `DEC_COMBO_PICK` reports production choice.
`admitted=1` means that arm independently passed; PICK resolves multiple passes.

Each pass includes 21 incumbent correctness snapshots and four snapshots per
candidate. Ordered expert IDs must equal the incumbent, including the initial
incumbent replay whose IDs parameterized the mathematical reference. All
untouched cache bytes must remain unchanged. Final BF16 hidden, FP32 stream,
carry-subtracted residual update and new FP8 rows must:

1. Meet the public task's relative L2/worst-last-dimension-row checks. Hidden and
   update tolerances are calibrated with BF16-vs-FP8-query mathematical reference
   outputs using exactly min(0.15,max(0.03,2.5*error)); cache tolerance is 0.08.
   Full-stream checking is an additional conservative check.
2. Have global and worst-row error versus reference no greater than the maximum
   of those same metrics across incumbent repeats on the same input. There is
   no epsilon, multiplier or fixed floor added to this incumbent error envelope.

Numeric outputs are **not** required to be bit-equal. A more accurate candidate
is allowed even if it differs from all incumbent repeats. The router weight
pack/unpack remains byte-exact. Incumbent routing is also checked against the
public routing constraints; candidates have identical IDs by the stronger test.

Prepare-only reference math is embedded from the public task's documented
reference/check/NVFP4 semantics, with no runtime imports of evaluator/task files.
It never writes the input caches and is never used by production kernel_fn.
Only synthetic prepare inputs use the reference. Input streams use the public
fixed seed-5354 basis (including the router's mean alignment) and fixed 80k B1
context, rather than the KV parent's independently redrawn mean directions.

Build joins stop at prepare+230 s; the optional gate's soft deadline is +275 s.
Reference expert loops check it too. Partial/incomplete arms never publish.
Cleanup restores launch globals and checks guards before publishing choices.
These deadlines are not proof of <285 s total preparation: inherited base
preparation, CUDA compilation and asynchronous device faults are not preemptible.

## Local validation

- `python3 kb/exp/build_decode_combo.py`: reproducible standalone assembly,
  AST parse and strict file limit pass.
- `python3 kb/exp/test_decode_combo_cpu.py -v`: ten stdlib tests pass; four
  torch-dependent tests skip on system Python (torch absent).
- With a worktree-local CPU torch venv and `DEC_COMBO_TASK_REF` pointing at the
  locally cached public task: **all fourteen tests pass**. Tests include full
  8192-wide front, full 4096-wide one-local-expert back, L1/full and L8/sliding,
  both BF16 and FP8 queries against public ref_front/ref_back; exhaustive FP4
  code pairs/scales; exact lossless router layout; every gate boundary, NaNs,
  null failures, ID changes, improved outputs, TLS restoration; and all three
  timing-pass protocols with CPU graph stubs (21 base/four arm snapshots,
  sixteen quads per arm/null, balanced orders).
- `python3 kb/exp/test_decode_kvpre_cpu.py -v`: nine parent tests pass.
- Forbidden grep and `git diff --check`: pass.

No local GPU/nvcc. No CUDA compilation, GPU correctness/performance or prepare
duration claims. No check, submission, push or main-branch commit performed.
The configured `.tools/tried.py` is absent; the alternate helper pushes to main,
so the ledger claim/result is recorded only in this branch as instructed.
Caller must publish the claim before spending a GPU check and record its result.
