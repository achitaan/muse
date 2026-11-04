# Output-gate lane: pending quota reset

Worktree: `muse-gate`, branch `argon/gate-20261104`, clean base `bf9357d`.
Only this worktree was edited. No manual push or other lane changes.

## GPU checks

- Nov 4 sole remaining check: **3cbe4bc4bea04580**, source commit `3bb3ec9`.
  Conformance **13/13**, unchanged production score **2.2401195**.
  **No ceiling measurements**: optional placement-survey extension compiled for
  `sm_90` and FP8 WGMMA requires `sm_90a`. This is NOT a negative gate result.
  Prepare 146 s. Full result: `kb/runs/gate_diag.3cbe4bc4bea04580.json`.
- Evidenced repair commit **33888c7**: explicit `compute_90a/sm_90a` flags;
  optional survey build errors no longer discard independent projection data.
  Eight CPU diagnostic tests pass. Standalone diagnostic: **971962 bytes**.
- ONE repaired diagnostic check is scheduled for **Nov 5 00:01 UTC** in background
  shell **sh_1a86bc0b6001qtXjg1DgK4w0Wz**. It verifies the source SHA before running,
  waits for the job without polling, and will notify this session automatically.
  Its stdout/result goes to `kb/runs/gate_diag_repair.stdout.json`, stderr to
  `kb/runs/gate_diag_repair.stderr.txt`. Do not submit a duplicate or modify the
  scheduled `decode_gate_diag.py` before that shell runs.

## Built offline, not submitted

`kb/exp/decode_gate.py` (**968731 bytes**, readable source) starts from clean
`decode_v1.py`, never from an encoded experiment. Generator:
`scripts/build_decode_gate.py`; editable addition `kb/exp/gate_overlap_impl.py`.

Candidate is limited to native B=1, page=1, T=1/8 attention (S/F). Scheduling:

1. SK2 QKV prefix (54 of 96 CTAs; retains the boundary tile's 16 gate rows).
2. Separate SK2 gate suffix (42 CTAs, lossless existing stage-major stream and
   escape tables). Gate skips the input-independent predecessor wait and
   triggers its dependent immediately. Every gate CTA must have started before
   attention is released, avoiding gate starvation behind attention.
3. Native attention starts while the gate streams. Its query/KV wait consumes
   an acq_rel QKV-ready count, not gate completion. Existing PDL waits are placed
   immediately before each actual gate load in the finalizers. Thus the gate
   join is implicit in the existing attention kernel; o's normal wait covers
   both predecessor branches.
4. The QKV-ready count uses unused native workspace slot 32767. It self-resets
   at the native last-merge leader (all active splits have already consumed the
   QKV prefix). No new memset kernel, event join, or side stream in production.

All original MMA, correction, reduction, sigmoid and rounding operations are
unchanged. This is a correctness argument, **not GPU validation**. Admission
requires exact byte identity for the entire projection (including g), attention
output, both final outputs, expert ids, and KV writes. Default repeatability is
also strict: baseline nondeterminism rejects the arm rather than weakening the
user's requirement. Both two-pass read-flush ABBA/BAAB medians and paired ratios
must be <=0.992. Fallback is always the incumbent.

CPU tests: `python3 scripts/test_decode_gate.py -v` (6 pass),
`python3 scripts/test_decode_gate_diag.py -v` (8 pass), Python compile and diff
whitespace check pass. No torch, CUDA compiler, or GPU available locally.

## Resume after scheduled diagnostic completes

1. Parse its final `GATE_SUMMARY` in `result.log_tail`; report all 12 shapes,
   both passes, isolated QKV and whole-layer paired savings, A/A controls,
   exactness flags, and attention full/capped-SM measurements.
2. Run `python3 kb/muse_score.py --job f2ab6d4c --save-us SHAPE=US ...` for
   100% and 50% capture. Do not infer any savings from the unchanged board score.
3. If exposed savings are below ~2 us on d1/v1 and ~3 us elsewhere, STOP as
   requested. Otherwise use ONE gated full-layer check of `decode_gate.py`.
   At most two evidenced repairs. Post-reset total budget: 6 checks + 3 scored
   runs, including the scheduled diagnostic repair; no more Nov-4 quota exists.
4. Report adopted shapes/per-shape ratios, jobs, exactness and predictions.
5. Only at the VERY END call `python3 scripts/tried.py add` from the MAIN
   checkout (`muse`), per caller instructions. Never manually push. If the helper
   cannot push, provide its complete TRIED-style result row in the final report.

The ledger search `scripts/tried.py search output gate` found two unrelated
existing rows, not an output-gate critical-path experiment. No ledger writes
have been made yet (caller explicitly requires end-only writes).
