# EXACTDIV: identical WGS indexing without general integer division

**Offline candidate; GPU correctness, compilation and speed are unverified.**

Branch: `loop/moe-second`, directly from **`63e253f`**. This branch does not
contain JOINFULL. It does not query or wait for JOINFULL check `8bf9cae70baa4192`.

Exact standalone candidate: **`experiments/moe_exactdiv/moe_exactdiv.py`**.
Base: `board_best/moe_m3_scored_2.3797.py`. CUDA is readable multiline source.

## Evidence, novelty and expected effect

The base's `wgs_cta_of` starts with a runtime **64-bit integer divide**, then
uses integer division again in its correction conditions. The producer's
range calculation and its cost-to-stage search use runtime **32-bit integer
divides** too. The base already comments that cold division subroutines can
cost about 1 us, but only uses its fast `muldiv_g` helper on other paths.

The supplied ledger reports slow MAIN startup (`:46`, `fe74255c`) and a
consumer-bound phase B (`:82`, `c967ffa3`). These do not prove that these
particular divisions are exposed. They motivate removing their latency
without changing scheduling or introducing another pipeline.

**Not a repeat:**

- Row 84 (`7bae45c5`) refitted **cost tables**, changing stage ownership. This
  candidate preserves every cost, range, stage and owner exactly.
- Row 78 (`523d3227`) changed cuts/private partials. This candidate does not
  snap cuts, move work, replace atomics or change fixup buffers.
- Row 52 (`7044c658`/`0414170f`) changed slot release. All TMA/barrier code is
  untouched here, unlike JOINFULL too.
- It does not change scales, readiness flags, PDL, epilogues or routing.

Searches of both the supplied ledger and the checkout ledger for division,
divide, reciprocal and `cta_of` found no previously closed equivalent idea.

**Pre-build hypothesis:** removing the inverse-owner 64-bit division from up
to one partial-unit callback per phase, and several producer-side u32 divides,
could save roughly 2–4% at M32 if those costs are exposed. Smaller or zero gains
are plausible if range calculation is hidden by gather and other warpgroups.
M128/256 are secondary targets; M1/8/16 should remain negative controls. This is
a conditional target, **not a measured prediction or a promised board win**.

## Algorithm and exactness proof

For each scheduler division `n / d`, where `d > 0`:

1. Convert n/d to FP32 and obtain a fast `__fdividef` estimate.
2. Truncate the estimate to an unsigned quotient q.
3. Form the exact **64-bit integer** product `p = q*d`.
4. While `p > n`, decrement q and subtract d from p.
5. While `p+d <= n`, increment q and add d to p.

At termination, `q*d <= n < (q+1)*d`, which uniquely defines exact floor
division. Float rounding cannot change any final index. The correction loops
use no integer division. Wide products avoid overflow during correction.

Inverse ownership simplifies algebraically, without changing the answer:

```text
floor(total*c/G) < xe  <=>  total*c < xe*G
largest such c = floor((xe*G - 1) / total)
```

All valid stage endpoints satisfy `1 <= xe <= total`. At the existing
MMAX=384 / JMAX=88 / maximum cost=255 / G<=160 bounds, the largest range or
inverse numerator is 1,838,284,800 (< 2^31); the largest scheduler quotient is
11,489,280 (< 2^24). Static assertions protect these bounds. H200 uses G=132.
The grid bound is the existing stream-K fixup capacity, not a newly expanded
support claim. All FP32 operands/results are finite and non-subnormal.

The previously kept zero-local-job guard (row 73, `2cba38a5`) is also ported,
so neither original nor replacement index code is called with zero total cost.
The gate's `OKC_EXACTDIV=0` control retains this same correctness fix.

## Changes intentionally excluded

Packing, dequantization, GEMMs, BF16 rounding boundaries, epilogues, norm,
global synchronization, shared-memory layout, launch geometry and host dispatch
are unchanged. The exact host AST matches the base apart from the CUDA string
and extension cache name. All eight non-WGS preprocessed variants are identical.

The inherited prepare-time build selector is unchanged too. Its closed/noisy
behavior is documented in ledger rows 40/42/44/252. A selected non-WGS build
does not exercise EXACTDIV; compare MAIN modules directly to avoid confounding.

## Local gates

```sh
python3 experiments/moe_exactdiv/test_exactdiv.py
git diff --cached --check
```

The Python runner extracts the **actual CUDA index function bodies** from
the candidate and base, then compiles/runs them as native C++17 with Clang.
Device attributes are stubbed; FP32 approximate division is modeled by native
FP32 division plus adversarial errors of -4/-2/-1/0/+1/+2/+4 ulps. Correction
counters are observation-only additions to the extracted helper.

Each enabled/control run verifies:

- 9,820,384 quotient cases against exact native integer division;
- 1,685,166 forward-range and 5,055,498 inverse-owner cases against the original
  functions, plus owner containment in the independent original range;
- 90,160 empty-job wrappers, including stale/poisoned ranges;
- all G=1..160 on selected plans, random/max-size plans, all n-tile counts,
  exact division boundaries and nearby-grid cases;
- at most five correction iterations observed under even +/-4-ulp error.

Removing the corrections and changing `xe*G-1` to `xe*G` each make the native
test fail. Additional source gates prove only scheduler helpers changed, the
host AST is unchanged, the eight non-WGS variants are identical after actual
preprocessing, and MAIN's enabled arm excludes its old general-division paths.

This is **native CPU compilation, not CUDA/NVRTC/ptxas compilation**. No local
GPU, CUDA toolkit, NVRTC, NVPTX Clang target or Linux container runtime exists.
No full MoE GPU tolerance/graph/deadlock/resource/timing gate was run. No OKC
checks/submissions were made. See `VALIDATION.txt` for exact local output/hash.

## Parent's one-check handoff

Check file: **`experiments/moe_exactdiv/moe_exactdiv.py`**.
Control: append **`-DOKC_EXACTDIV=0`** to MAIN's defines using `_ext` and run
that module's `okc_run` directly on the same state/inputs as enabled MAIN.

Require the normal correctness cases and all-remote/mixed graph replay;
compare registers, local/spill bytes and shared memory; then use many routing
draws with **read-flush** paired ABBA at M32/128/256 (nearby M33/257 too). Look
for elimination of the integer-divide helper calls in compiled PTX/SASS and
resource parity before interpreting latency. A few routing draws or a changed
prepare-selector choice are not adequate evidence. Stop if no repeatable >=2%
shape gain; do not turn this into another cost-table or scheduling sweep.

The local `TRIED.md` row must be merged by the parent into the shared ledger:
the prescribed helper is missing, and the available helper pushes main, which
this lane is not authorized to do. No other worktree/main/push was touched.
