# Bounded dense consolidation

Branch: `loop/bf16-consolidate`, from `fa02f58`.

Check file: `candidates/dense_consolidate.py` (standalone).

## Scope and provenance

The default route is the board file `dense_kbv2_scored_1.5886.py`, before
its outer legacy ABBA tuner. Its native numerical kernels, packing, dispatch
tables and internal historical bakes are unchanged. Explicit output allocation
sites use the existing prepare-only borrowing helper; ordinary production
calls still allocate normally. No Torch/judge monkeypatching is involved.

Only these six optional arms are considered, with no search or timing retries:

| Cell | Arm | Source |
|---|---|---|
| router:8 | S8/WK8 `rt8w8` | `146ad32:kb/exp/bf16_v5.py` |
| router:16 | S8/WK16 `rt8w16` | same; only added native instantiation is `RSM(8,16)` |
| qkv_gate:32 | `t32s5curP-k7` | scored `bbde84195a7a40e8` board adoption |
| pooler:32 | `t32s6curP-k4` | same |
| qkv_gate:16 | `t16s5curP-k7` | same |
| qkv_gate:256 | `n256s5m1hzm-k7` | same |

The last cell is the historical RQ arm, not a new CURP variant. All six start
disabled. Rejected/unavailable/unconfirmed arms use the default dispatcher;
there is no inheritance of the board's previous outer-tuner wins. Nearby token
counts retain the board's existing routes. Router:32 is intentionally omitted.
Only a four-template RQ extension is built in the optional preparation thread.
C++20 and `compute_90a/sm_90a` flags are explicit for this extension and NB.

## Controlled admission

The measurement primitives are AST-identical to `fa02f58`: shared physical
input/output addresses aligned to 64 KiB, independent direct captures on one
side stream, symmetric cold warmups, fully counterbalanced event slots and
capture order, and pre/post BASE/BASE null controls. Every replay includes the
complete computation. No fixed replay cost is subtracted and no null is used
to normalize a measured ratio.

Each screening draw has four independent pre-null graph pairs, four candidate
graph pairs, and four independent post-null graph pairs. Null spread thus has
N=8 capture pairs, **not** 8 events from one capture pair. Each pair receives
32 balanced timing blocks, as in the repaired parent protocol.

For each estimator separately, compute median and **raw, unscaled MAD** across
the eight null-panel ratios. Use the larger marginal/paired MAD. The required
margin is `max(0.015, 3 * MAD)`. This is the requested robust noise-floor rule,
not a Gaussian confidence interval or a formal significance claim.

The candidate's median panel ratio and paired estimator must both be strictly
below `1 - margin`; both capture-order strata must agree. A negatively shifted
null center additionally lowers the admissible limit instead of making the
candidate look faster. Displaced null centers, phase aggregates or order
strata veto the draw outside the same adaptive band. Individual noisy null
panels are retained in the MAD calculation, not clipped or discarded.

A passing screen requires fresh-seed, fresh-input, fresh-capture confirmation
with another eight null pairs. The final margin is the maximum of screening,
confirmation and pooled MAD margins; the screen is rechecked under that final
margin too. A calmer confirmation cannot weaken screening's threshold.
Independent normal/outlier/one-hot/zero FP64/default correctness checks run
before each draw and after confirmation. No activation, graph or reference is
retained in returned state.

Final output contains exactly one `DENSE_ADMIT` line per cell, including
`null_med`, `null_mad`, `cand_med`, `paired`, `confirm`, `admitted`, panel N,
actual conservative limit, arm and rejection reason. A skipped/failed cell
reports unavailable statistics as `nan`, never synthetic successful values.

## Budget and verification

The absolute deadline is 270 s from prepare entry, including the default
prepare, optional build wait, packing, validation and measurement. Build waits
reserve 45 s for the audit. Every phase/block and each validation input checks
the deadline. An incomplete cell is never installed; fully confirmed earlier
cells may remain. This is a cooperative deadline: an incumbent synchronous
build/kernel/prepare cannot be preempted. **Actual <300 s duration needs H200
verification.**

Passed locally with CPU Torch 2.14.1:

- AST/Python compile and staged `git diff --check`.
- `scripts/verify_dense_consolidate.py`: exact controlled-primitive lineage,
  preserved base native math, fresh/nested/thread-local output allocation,
  16 counterbalancing seeds, adaptive-MAD policy/boundary/order/drift checks,
  full-computation graph model with false-win and confirmation veto tests,
  exact six-cell wiring, per-cell fallback and partial-budget failure.
- Independent inverse permutation of both router packs: 8,388,608 BF16
  payloads bit-exact. FP64 hi+lo router model relative error `2.50e-6`.
- Exact four RQ plan names, C++20/sm90a build flags, roundtrip rejection and
  off-scope plan rejection exercised through extracted actual helpers.
- Existing `verify_rotcodes_null.py`, `verify_dense_rotcodes2.py`, and
  `verify_dense_cg_rotcodes.py` all passed unchanged. Their CUDA-stubbed C++
  syntax checks do not constitute a native CUDA build of this candidate.

Evidence: `results/dense-consolidate-offline.json`.

No CUDA/nvcc/GPU or OKC operations were available/performed. There is no new
speedup claim. The H200 check must establish compilation, six actual gate
results, nearby correctness and prepare duration. The gate may retract all
four historical board arms, lowering the score honestly. The local TRIED row
must be published by the caller; no main commit/push was performed here.
