# PVT d32: fix the skipped observation, not the numerical threshold

Branch: `loop/attn-pvt-d32-probe`, directly from PVT **01323f0**.
Standalone check payload: **`candidates/attention_q1_pvt_d32probe.py`**.
No new check/submission, no waiting on other checks, no push or main commit.
The parent PVT/RQ/redesign branches are unchanged.

## Root cause established by the existing result

Evidence: `candidates/profiles/attention_q1_pvt.d526452be7674a5f.log`.

| Shape | Base reference rel-L2 | PVT reference rel-L2 | Judge/cache | Result |
|---|---:|---:|---|---|
| d32S (lines 15–18) | .0047349 | .0047351 | pass / exact | rejected before timing |
| d32F (lines 48–51) | .0032618 | .0032618 (printed) | pass / exact | rejected before timing |

The inherited `_at_shape.equiv` requires **both** `rel <= base_rel` and
`worst_row <= base_row` against **one** base replay, with **no slack**. This
extra requirement is not the judge's tolerance. For d32S, the printed global
metric alone proves the rejection: +2e-7, about 0.0042% of the base's error.
For d32F, at least one full-precision global/row comparison must have failed:
the log reports judge=1/cache=1, but the old `%.5g` print conceals its magnitude.
Equal printed numbers therefore do not establish an unexplained control-flow bug.

That establishes **why observation was skipped**, not the source of the output
difference. A transpose changes FP32 tensor-core/reduction scheduling, and its
new writer-lane mapping can change global atomic arbitration. A few BF16
rounding-boundary changes or base replay variability are plausible. They are
**not established** by aggregate reference errors alone. There are no saved
output tensors to inspect locally, and no local GPU to repeat this input.

Also, “32-token PV path” is misleading here: d32 is **32 requests x q_len 1**.
The PVT fragment helpers and `Cfg<L, FLAT>` are not specialized on batch 32.
d16S/d32S share the per-request family with different work per split;
d16F/d32F share the flat family. No batch-32-specific indexing defect was found.
The existing CPU fragment tests cover both KV heads, both key halves, multiple
tiles/splits, distinct row maxima, masking and final packing. They cannot prove
Hopper's accumulation order or rule out a GPU-only defect.

## Change: diagnostic timing and adoption are separate decisions

**No CUDA/host source, precision, split plan, judge bound or no-worse threshold
is changed.** PVT CUDA SHA256 remains:
`78a8482f8b197b8f8340a8ecfb6624fdf7188820443284f5ef96990cbb99c185`.

Only d32S/d32F get the new import-time diagnostic path:

1. Capture base, compute its reference errors, and clone its output so later
   graph replays cannot overwrite the diagnostic anchor.
2. Repeat base three times with poisoned new-cache slots. Print full-precision
   reference-error ranges and baseline-to-anchor differences. These ranges are
   **diagnostics, not enlarged acceptance tolerances**.
3. Execute three synchronized eager PVT probes. Report global and row errors
   with **12 significant digits**, their signed differences from base, exact
   cache status, changed-element count, max absolute/relative difference, and
   maximum BF16 ULP distance / count of elements farther than one BF16 ULP.
4. If all probes and the captured output pass the **unchanged judge bounds and
   whole-cache equality**, run the inherited 512-MiB-read-flushed ABBA timing
   even when the extra exact no-worse condition fails. Label it diagnostic.
5. Independently confirm a >=2% timing win even if numerical admission failed.
   **Adopt only if every original no-worse condition also passes**, including
   the extra probes, and both median/paired timing metrics pass 0.98 in both
   comparisons. A failed numerical admission can never publish a winner or
   update the reported production timing table.

Judge or cache failures before capture still prevent timing; failures detected
after timing prevent further acceptance. No epsilon was added to get a pass.
This deliberately preserves the original quality bar rather than deciding from
the rounded log that it is safe to relax it.

d32S/d32F run first. d1/d16 losers are controls only, not retried. The reproduced
v32S one-head arm is still re-gated. Runtime shape dispatch/fallback is unchanged;
all samples, comparisons and synchronization remain on independent synthetic
inputs at import, never on actual calls. No synthetic results survive tuning.

## Local evidence

With Python from the main checkout's read-only `.venv`:

```sh
PYTHONDONTWRITEBYTECODE=1 <python> scripts/validate_attention_pvt_d32probe.py
PYTHONDONTWRITEBYTECODE=1 <python> scripts/validate_attention_q1_pvt.py
```

- **13 new tests pass:** execute the actual old selector to reproduce d32S and
  hidden-precision d32F pre-timing rejection, then the new selector to prove
  diagnostic timing occurs without adoption. They cover all four performance
  gate comparisons, hard judge/cache failures, late failures, NaN timing,
  unchanged CUDA/guards and exact BF16 sign/zero/ULP diagnostics.
- One test runs the **actual CPU judge/scorer**, not supplied scalar metrics:
  one adjacent BF16-value change passes the judge but fails exact no-worse.
- **10 existing PVT tests pass**, including C++ helper syntax and independent
  dense/fragment/online-softmax CPU oracles. New/parent CUDA construction is
  byte-identical, so these tests remain applicable.
- Python AST, CPU Torch import and the parent/candidate whitespace delta pass.
  CPU import safely reports `skip no-cuda-or-capture`; no timings are invented.

**Not verified:** new GPU diagnostic flow/capture/timing, actual d32 output
differentials, base nondeterminism, or a >=2% d32 speedup. The parent check
established that the unchanged CUDA builds and the two d32 probes pass the
judge/cache checks; that is not a complete proof of numerical equivalence.

## Read the next result

`P32 numeric` exposes the precise failing comparison. `P32 differential` tests
the rounding hypothesis directly; `P32 base-envelope` shows whether base itself
crosses the original no-worse boundary. A large systematic, multi-ULP/head-like
difference merits a CUDA investigation, even if the judge bound passes. Small
candidate differences comparable with base replay spread support, but do not
automatically admit, a calibrated future gate. `P32 timing ... diagnostic_only=1`
is a measured unadmitted arm—not a production gain and not a speed negative.

Payload SHA256:
`2a3f1cd26ca805d36d66a587167a420590b6cbc2a8d9846bf3fb3fc46de1bd85`.
The branch-local TRIED row must be copied to the shared ledger by the caller;
the normal helper pushes to main, which this lane may not do.
