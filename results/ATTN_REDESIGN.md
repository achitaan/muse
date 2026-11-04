# Attention restart: fresh split-execution design, roofline stop

**Decision: do not build this redesign.** Its optimistic byte bound fails the
required **+3% board** gate, even before arithmetic and launch costs. This is one
bounded from-scratch analysis, not another split/knob sweep. No new GPU check or
submission; queued PVT/RQ jobs were not queried or awaited.

Branch `loop/attn-redesign`, directly from `63e253f`. Baseline provenance:
`board_best/BOARD_BEST.md:9,32`, scored job **8f53cd76d828427c**, **2.6536245x**,
CI [2.61382, 2.69343]. The main `results/jobs.json` only had its summary, so its
completed per-case records were retrieved read-only with `okc status ... --json`.
The compact source numbers are in `attention_redesign_baseline.json`.

## 1. Where a restart would have to win

Times below are the judge's corrected `median_ns`, not raw replay event times.
Time share is `weight*time / sum(weight*time)` over the 12 cases; it is not
itself a board scoring weight. The board is a geometric mean over six **loads**,
each load combining sliding/full with weights **46/16**.

| Case | Incumbent us | SGLang / incumbent | Weighted time share |
|---|---:|---:|---:|
| decode 1 sliding | 4.924 | 4.188x | 1.98% |
| decode 1 full | 10.714 | 2.954x | 1.50% |
| decode 16 sliding | 7.848 | 3.449x | 3.16% |
| decode 16 full | 86.489 | 2.317x | 12.11% |
| decode 32 sliding | 9.816 | 3.165x | 3.95% |
| decode 32 full | 165.734 | 2.314x | 23.20% |
| verify 1 sliding | 6.036 | 3.727x | 2.43% |
| verify 1 full | 14.133 | 2.552x | 1.98% |
| verify 16 sliding | 11.259 | 2.576x | 4.53% |
| **verify 16 full** | **98.178** | **2.082x** | **13.74%** |
| verify 32 sliding | 14.184 | 2.421x | 5.71% |
| **verify 32 full** | **183.659** | **2.123x** | **25.71%** |

Target **v16F + v32F**: lowest speedups, together **39.45%** of weighted time.
Their full phases occupy 75.20% and 81.83% of their respective loads. Reducing
both full-case times by **10.8067%** gives exactly +3% board (about **2.73323x**),
with the other ten cases unchanged. An isolated 3% shape win is insufficient.

## 2. Fundamentally different proposal: QK/softmax -> P -> PV

The current full kernels keep QK, softmax, V transpose, PV and merge within
one CTA pipeline; the ledger attributes the verify gap to consumer throughput
under the register budget (row 197). The fresh hypothesis is to **separate the
incompatible live ranges**, not transplant or tune that pipeline:

```text
kernel A: persistent (request, KV head, 64-key tile) tasks
          QK-norm / new-KV update -> FP8 QK with FP32 accumulation
          -> tile-local softmax -> global FP8 P + FP32 tile offsets
                                      |
                                      v
kernel B: independent persistent PV tasks, same logical-key tile order
          global P + paged FP8 V -> tensor-core PV + row sums
          -> stable partial merge -> output RMSNorm * sigmoid(gate)
```

- A consumer warpgroup owns 64 query rows (8 tokens x 8 GQA heads), but carries
  **no 64-dim output accumulator**. Kernel B carries output accumulators but no
  QK score/softmax bank. Both can use more independent CTAs to hide latency.
- P is E4M3 with a per-row, per-key-tile log2 offset `m`; store
  `P=round_e4m3(2^(score-m))`, shifting the range upward as in the incumbent.
  B folds scaled `(m, numerator, denominator)` with the stable max/rescale
  identity, rather than assuming independently normalized blocks can be added.
  Its denominator comes from P/ones, so no global l buffer is necessary.
- At B16/B32, allocate P for approximately **164/328 MB**, plus offsets. Every
  read slot is overwritten this call. New KV is written once by A; ordinary
  stream ordering makes it visible to B. No host length read or host sync, no
  output caching, no data-dependent captured allocations. No FP16-accumulation
  experiment, no physical-order traversal, no change to past KV layout.
- This could improve **consumer scheduling** and occupancy if temporary data
  were free. It is not the one-head split extension (rows 192–194), producer
  V transpose (177), private-output partials (191), or physical KV staging (200).
  Read-only ledger searches for Attention/material, two-pass and disaggregated
  found **zero rows**. This is not built on PVT or RQ.

## 3. H200 roofline: count the new bytes, not just removed registers

Let `N=sum(cached lengths)`. HBM peak is **4.8e6 bytes/us** (4.8 decimal TB/s).

- Actual unique past-KV payload: `2 allocations * 2 heads * 64 dims * N`
  = **256 N bytes**. **Do not multiply by q_len=8 or GQA=8**: the incumbent
  already reuses each loaded KV row for all queries.
- Intermediate P: `q_len * Hq * N` = **128 N bytes** at q_len 8.
- A global P write plus B's read adds **256 N bytes**: already another complete
  KV stream. One FP32 offset per query and 64-key tile adds about 16 N bytes
  over its write/read (4D/byte and end-tile rounding included by the script).
- Page-table reads, cache updates, Q/gate/output bytes, metadata traffic, compute
  and extra launch latency are all **omitted** from the optimistic stop bound.

The job does not log actual `seq_lens`. Therefore exact aggregate input bytes
cannot be recovered from its case labels. Nominal 80k gives KV **327.68/655.36
MB** at B16/B32. We also test **74k and 86k** mean lengths (the incumbent probe
models 80k +/-7.5%); these are explicit sensitivity points, not measured inputs.
The reported full-case SOL numbers exactly fit a 74k byte model plus 9,216 bytes
per new token, so they must not be mistaken for an observed traffic measurement.

Even granting a very generous **64 MiB cache entirely to P**, and granting that
credit on **both** P's write and read, sequential A/B have the lower bound:

`T_fresh >= [256 N + 2 max(128 N - 64 MiB, 0)] / 4.8e6` us.

| Mean cached length | B | KV MB | Ideal fused KV-only us | Fresh optimistic us | Incumbent us |
|---|---:|---:|---:|---:|---:|
| 74k | 16 | 303.104 | 63.147 | **98.331** | 98.178 |
| 74k | 32 | 606.208 | 126.293 | **224.625** | 183.659 |
| 80k | 16 | 327.680 | 68.267 | **108.571** | 98.178 |
| 80k | 32 | 655.360 | 136.533 | **245.105** | 183.659 |
| 86k | 16 | 352.256 | 73.387 | **118.811** | 98.178 |
| 86k | 32 | 704.512 | 146.773 | **265.585** | 183.659 |

At 80k, offsets add another **4.27/8.53 us** if streamed from HBM; no credit for
that cost is needed to kill the proposal. Without cache credit, just KV+P
traffic costs **136.53/273.07 us**, before offsets or any math.

**Latency floor:** the ledger's PAGE=1 diagnostic has a **4.88 us raw** floor;
this board run reports roughly **4.0–4.5 us replay overhead**. Those are different
measurements, not a calibrated 4.88-us device-launch constant. Do not add either
raw number to corrected case times or subtract it twice. The bound above uses
**zero extra latency** for A/B—more favorable than real sequential launches and
page-table -> KV dependency chains—yet it already loses. No TMA can make a
128-byte scattered PAGE=1 row into a contiguous 64-row bulk transaction.

**Result:** if forced, even the 74k/zero-metadata/64-MiB-cache model yields a
board upper bound **2.579997x (-2.775%)**; nominal 80k gives **2.516396x (-5.171%)**.
With the required incumbent fallback the result is **unchanged 2.653624x**, not
an improvement. **Build gate FAIL: no candidate or prepare-time gate is built.**
If this bound had passed, admission would require synchronized correctness,
whole-cache exactness, no-worse judge/row error, read-flush ABBA plus independent
confirmation, and confirmed times projecting >=1.03 board gain. No timing on
actual calls and no production dispatch change would be allowed before admission.

## 4. What "at the floor" can and cannot mean

Decode-full nominal useful rates are already **3.79/3.95 TB/s**, close to the
ledger's measured **3.96–4.15 TB/s** PAGE=1 gather stream. A gather-only reset is
not a credible large win; smaller sliding splits, no-merge head groups and DSMEM
merge redesigns already lost (168–170, 187, 189–191).

**Do not claim an absolute attention floor.** At nominal 80k, an unattainable
zero-overhead *fused* 4.8-TB/s kernel on v16F/v32F would still project
**2.882268x (about +8.6%)**. The byte roofline therefore does not prove universal
optimality. What it proves here is narrower: this fresh register-pressure reset
cannot reach +3%; removing its P traffic requires returning to on-chip fusion
or an unvalidated cross-SM bounded queue with a different throughput/progress
model, not pretending the intermediate is free. No such second attempt is made.

## Local validation / handoff

`PYTHONDONTWRITEBYTECODE=1 python3 scripts/attention_redesign_roofline.py`
passes **5 tests**, reproduces **2.653624497544015x** and score_us from the saved
completed job, verifies byte counts and board sensitivity, and enforces the
no-build conclusion. Python AST/JSON and whitespace checks pass. No GPU code
was generated or run; no check path exists for this analysis-only branch.
The branch-local TRIED row must be copied to the shared ledger by the caller;
its normal helper pushes to main, forbidden in this lane.
