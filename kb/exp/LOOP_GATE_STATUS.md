# D1: gated output-projection overlap, `loop/decode-gate`

Check file: `kb/exp/decode_gate.py` (976494 bytes, readable standalone source).
Rebuild with `python3 scripts/build_decode_gate.py`; the generator now reads
editable diagnostic sources, not a possibly stale generated diagnostic file.
No GPU checks, submissions, pushes, or main commits were made by this lane.

## Changes and audit

- Based on `argon/gate-20261104` / `7a8589c`. Every inline build in the generated
  candidate and diagnostic explicitly uses host/NVCC `-std=c++20` and NVCC
  `-gencode=arch=compute_90a,code=sm_90a`. All other incumbent compiler flags,
  especially fast-math/FMA and numerical definitions, are preserved. No reliance
  on racing writes to `TORCH_CUDA_ARCH_LIST`.
- Prefix is 54 CTAs / 1296 rows, including the mixed tile's 16 gate rows;
  suffix is 42 CTAs / 1008 rows. Stage-major streams and per-CTA escape tables
  are repacked consistently. Original arithmetic and rounding are unchanged.
- Every gate CTA triggers dependents **at entry** and has no wait on the prefix:
  its input is the original `x`, not prefix output. PDL therefore cannot release
  attention until every gate CTA has started. Prefix CTAs have already started
  before gate launch; neither producer branch waits for attention. Resident
  attention cannot starve an unstarted gate CTA, and a cooperative attention
  barrier can only delay until the independent gate work retires.
- Prefix consumers rendezvous after all output stores, then each CTA publishes
  one acq_rel RMW. Attention acquires count=54 and synchronizes its CTA before
  reading Q/K/V. The four actual global gate loads retain a preceding true PDL
  wait. Supported shapes are B=1, L=T=1/8, page=1, 16 Q heads / 2 KV heads:
  these use native non-flat, non-cluster attention only. The output projection's
  ordinary PDL wait joins completed attention and therefore the gate stores.
- Fixed an inherited late-wait race: with a speculative length miss, a split
  with no **real** keys could reach the prefix-ready wait after active splits
  had reset the count. Rejecting a miss at segment entry, before async or QKV
  work, makes each real active split consume readiness exactly once; real empty
  splits return without waiting. Existing last-merge leaders then safely reset
  slot 32767. Both last-CTA and cooperative resets are covered. Prepare clears
  that slot after every shape, including rejected/aborted arms.
- Fixed the new fast path's bypass of the incumbent fresh-host-thread CUDA
  context binding.
- Synthetic B=1 page-table width is now 80000+L, matching its fixed context and
  the scored shapes. The old width=86000+L forced a speculative miss in every
  timing replay and did not model production's fast prologue.

## Admission and diagnostic output

For each d1S/d1F/v1S/v1F shape, two fresh-input passes validate exact byte
identity of all QKV rows (including g), attention output, both final outputs,
expert ids, and newly written K/V cache rows. Baseline itself must repeat
exactly. This is intentionally conservative about upstream atomic nondeterminism.

Each pass measures CUDA graphs after a **512 MiB read-only flush**, outside
the event interval. Twelve blocks alternate ABBA/BAAB. Admission requires both
the ratio of medians and median paired ratio <=0.992 in **both** passes; each
order's paired median must pass too. An equal-sample base-vs-base null arm must
land in [0.995,1.005] for median, paired, ABBA and BAAB ratios in both passes.
Null placement alternates before/after the real arm. `GATE_ARM` prints all
requested fields; `GATE_PICK` reports adoption. Any failure uses the incumbent.

Before admission, `GATE_CEIL` prints two-pass isolated full-QKV vs prefix-only
projection timings and retained-row exactness. It runs even if the attention
extension build fails or the overlap arm loses. Prefix-only never publishes
readiness, so its isolated graph needs no attention consumer or counter reset.
These diagnostics are prepare-only, never production dispatch.

Optional builds run alongside incumbent prepare and are waited for only until
240 s. Timing loops check a 280 s overall deadline, preserving 20 s headroom.
The actual compilation/prepare duration still needs judge validation; Python
cannot interrupt a hung CUDA operation or a slow incumbent prepare.

## Local validation

- `python3 scripts/test_decode_gate.py -v`: 12/12 pass.
- `python3 scripts/test_decode_gate_diag.py -v`: 9/9 pass.
- AST parsing of candidate, diagnostic, implementations, generators and tests:
  pass. Candidate size <1000000. `git diff --check`: pass.
- No local torch, Triton, nvcc, or GPU: compilation, exactness, liveness,
  speedup, null ratios, and <300 s prepare are **not GPU-verified**.

## Shared ledger handoff

`scripts/tried.py search gate overlap` returned zero matching rows. No checks
were spent. The installed helper's `add` always commits/pushes main, forbidden
to this lane; the caller should record/claim D1 before submitting its check.

Suggested TRIED row (replace after the caller's GPU result):

| D1 output-gate overlap; sm90a/C++20, speculative-ready race repair, strict two-pass null-gated admission + isolated ceiling | parked: CPU 21/21 pass; awaiting caller GPU | loop/decode-gate | - | kb/exp/decode_gate.py; kb/exp/LOOP_GATE_STATUS.md | Nov 04 |
