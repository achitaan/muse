# SK2 rotcodes V2: calibrated gate, native resources, just-in-time decode

## Handoff and scope

- Branch: `loop/dense-m1632-v2`, created from
  `loop/dense-m1632-sk2rot` / `f6d27e586d6107d1dd7c2bf86163c08c446ff968`.
- **Exact check file:**
  `/Users/achita.anantachina/.local/share/opencode/worktree/e72615/loop-dense-m1632/candidates/dense_sk2_rotcodes_v2.py`
- No OKC calls, submissions, pushes, main commits, or writes to another
  worktree by this lane. The caller supplies the next GPU check.
- V1 candidate, tests and artifacts remain intact. V2 is a separate file.
  The original board incumbent's AST is still unchanged except the same
  helper exports in `_make_vjeux`'s final return. M1/M8/scalar paths and
  other lanes are untouched; their changes are not included in this file.
- Seven eligible cells: o, pooler, unpooler at M16/M32, and qkv_gate:M16.
  **Qkv_gate:M32 is explicitly disabled**, including a defensive runtime
  guard against its admission through a malformed state. Its weight
  geometry is compiled only for resource diagnostics, never admission.

## What check 4614986cedc447ed actually establishes

Read the complete caller-supplied log:
`/private/var/folders/pm/r8xm1mr525v913lct5tdg_5w0000gn/T/opencode/subs/profiles/dense_sk2_rotcodes.4614986cedc447ed.log`.

| Cell | Draw | Median ratio | Paired median | Exactness |
|---|---:|---:|---:|---:|
| o:32 | 0 | 0.9677 | 0.9737 | pass |
| o:32 | 1 | 0.9693 | 0.9770 | pass |
| o:16 | 0 | 0.9770 | 0.9773 | pass |
| unpooler:32 | 0 | 0.9841 | 0.9860 | pass |
| unpooler:16 | 0 | 0.9843 | 0.9886 | pass |
| pooler:32 | 0 | 0.9846 | 0.9920 | pass |
| pooler:16 | 0 | 0.9927 | 0.9951 | pass |
| qkv_gate:32 | 0 | 1.0356 | 1.0415 | pass |
| qkv_gate:16 | 0 | 0.9733 | 0.9808 | pass |

V1 adopted nothing because it stopped at the first draw exceeding .975
in either statistic. The o32 medians agree across its two observed draws;
its paired 0.9770 missed by .0020. This is good evidence for a prospective
follow-up, **not** a complete four-pass acceptance result. New thresholds
were requested after inspecting that result, so new independent validation
is required rather than retroactively calling V1 an admitted win.

**There is no base-vs-base null measurement or resource attribute anywhere
in this log.** Its median/paired discrepancy is not a null-spread estimate,
and no register/spill diagnosis can honestly be extracted from it. V2 adds
the missing controls and attributes instead of inventing those data.

Also, `DT ... qkv:32 keep-t32s5curP-k7` and the final DT summary show the
qkv32 incumbent was an admitted `rq` CURP route. V1's 1.0356 is **new SK2
versus CURP**, not new decoder versus old decoder. It cannot establish a
rotcodes register regression. The same caution applies whenever the full
incumbent is a different kernel family.

## Revised admission and why it remains conservative

The following thresholds are fixed before the next GPU run. They are not
widened, normalized to a favorable control, or retried after a bad draw.

Each eligible cell needs **three fresh input/graph draws plus a fourth
fresh-seed confirmation**. Confirmation uses its own explicit per-cell
generator seed, not the original generator's continuation. Every pass must
satisfy all of:

1. The unchanged correctness checks: bit-exact original SK2 twin;
   agreement with the complete incumbent; independent FP64 `x @ W.T`
   checks on normal, outlier, two one-hot batches and zero inputs; tighter
   probe tolerances; exact replay stability; post-timing validation on
   fresh inputs. The additional null output also agrees with the incumbent.
2. **Median candidate/base ratio <=0.980.**
3. **Median paired ABBA ratio <=0.985.**
4. **One-sided paired-population-median upper bound <=0.995.**
5. A separate base-vs-base ABBA control **before and after** the candidate
   measurement, with both median and paired ratios in **[0.9925, 1.0075]**.

Every measurement uses 40 ABBA blocks, the same 512 MiB vector **READ**
flush, and fresh independently captured graphs. The null arm is a second
capture of exactly the complete incumbent, on the same input buffer and
output shape—not a different kernel or timing-only surrogate. Each
successful cell thus requires eight passing null measurements. A bad null
immediately rejects the cell, even if the candidate otherwise wins.

The 2.0% median effect threshold is 2.67 times the allowed 0.75% null-band
half-width; the 1.5% paired threshold is twice that width. The bound adds a
directional consistency requirement, not just an effect-size cutoff:
for 40 independent pairs the **26th ascending observation** is a one-sided
upper bound on the population median with coverage
`1 - P(Binomial(40, 0.5) >= 26) = 0.9596547661`. It must remain below .995.

GPU temporal samples are not guaranteed independent. Consequently this
is **not** a claimed formal false-discovery probability, and probabilities
are not multiplied across passes. Flanking controls, fixed rejection
rules, fresh graph allocations, fresh-seed confirmation, and the absence
of a geometry search address drift/selection risks beyond that idealized
median bound. Actual null spread is still unknown until V2 runs; the new
code requires it to be small rather than assuming it. If controls fail,
the correct result is incumbent fallback, not threshold relaxation.

Ratios use raw event timings as V1 did. There is no subtraction of a guessed
graph-overhead constant inside the gate, which would amplify short-cell
noise. These are not directly the judge's replay-overhead-subtracted
per-cell ratios.

## Why the weaker cells need a different lever

The previous offline analysis used the actual 12-bit encoded weight bytes:
o is 12 MiB; pooler/unpooler are 48 MiB; qkv is 27 MiB. At the supplied
profile's times, pooler/unpooler already deliver **3.36–3.65 TB/s**, about
70–76% of the 4.8 TB/s useful-stream ceiling, versus o's **2.18–2.41 TB/s**.
Their stream-only floors are 10.486 us versus o's 2.621 us. Decoder work is
more readily hidden under the larger streams. This supports smaller
decode-only gains; it does not quantify the actual decode fraction.

There is also a route-confounding issue: o16/pooler16 and qkv may compare
different math families, whereas unpooler16/32 and pooler32 normally compare
SK2 variants. V2 therefore logs one **old-native-SK2 versus selected-candidate
ABBA diagnostic** on draw0, separately from the mandatory full-incumbent
gate. Those `SR twin` lines provide the missing within-family attribution.
They never determine admission by themselves.

The K-warp split is unchanged: WSK=2 for pooler/unpooler, UPW=2 except
pooler16's UPW=4. Unpooler32 has NT=4; the other weak cells have NT=2.
V1 eagerly decoded **8*NT B registers per lane** before issuing any of the
four k16 steps. Unpooler32 therefore carries 32 decoded B registers alongside
32 activation-fragment registers and its accumulators. V2's bounded change
decodes only the **current two B registers** immediately before use, while
retaining the original packed low/code registers until their last use.
That targets register lifetime/scheduling, especially NT=4, rather than
trying to remove more HBM bytes or changing the K split.

The source-order change is `j/mt/nt` -> `j/nt/mt`. It interleaves independent
accumulators differently but **each accumulator sees exactly the same
ordered MMA operations**, including NACC selection. Each high-byte mask,
rotate, base addition and PRMT reconstructs exactly the same BF16 pair.
All shared loads, slot-release barriers, escape correction, reduction,
FP32 accumulation and BF16 stores remain unchanged. No tensor precision,
traffic, tiling, epilogue communication or math work is reduced.

This does not promise a gain: NVCC can hoist the decoder again, and packed
operands also consume registers. Accordingly, just-in-time decode is
eligible only on the four pooler/unpooler cells and only if native queries
show **registers <= eager rotcodes, local bytes <= eager rotcodes, and
resident CTAs/SM >= eager rotcodes**, with all shared-memory/thread/stage
metadata equal. Otherwise that cell evaluates the original eager rotcodes
under the same strengthened gate. There is one candidate schedule per
cell, no geometry/parameter sweep, and no fallback performance retry after
a failing candidate. Ultimately it must beat the full incumbent in all
four passes; unchanged incumbents remain the default.

## Resource evidence and qkv32

V2 compiles native old-decoder and eager-rotcode twins for prepare-time
`cudaFuncGetAttributes`/`cudaOccupancyMaxActiveBlocksPerMultiprocessor`
queries. It also compiles the four just-in-time specializations. Attribute
vectors are logged at cell entry and again in a compact final summary:

```text
[registers, localBytes, staticSmemBytes, dynamicSmemBytes, threads,
 residentCTAsPerSM, stages, escapeWarps, SMcount]
```

The dynamic shared-memory formula is identical to the actual launcher,
including the real per-cell escape-table capacity. `localBytes` is total
local memory, **not an exact spill count**; that still requires ptxas/SASS
evidence. Old-decoder attributes are a same-flags, same-source-geometry
compile twin, not a direct query of the separately built parent module.

Static geometry already shows qkv32 has **96 CTAs for 132 H200 SMs**, a
24-row owner tile and 218,848 B (213.72 KiB) dynamic shared memory before its escape
table. It cannot reach a 132-SM one-wave stream ceiling with that grid.
Pooler32 and unpooler32 have 128 CTAs and 161,520/149,072 B
(157.73/145.58 KiB) respectively;
their residency is largely shared-memory constrained. Reducing registers
cannot magically double their CTA residency. These observations explain
why register-pressure or occupancy must be measured, not presumed.

Qkv32 remains off regardless of its new resource numbers. V2 prints its
old/rot attributes for diagnosis only. No attempt is made to beat the
faster admitted CURP by weakening precision, changing its route, or
repeating the closed geometry/CURP sweep (`83c46449`).

## Verification and expected effect

```sh
PYTHONDONTWRITEBYTECODE=1 \
  /Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse/.venv/bin/python \
  scripts/verify_dense_sk2_rotcodes_v2.py
```

Evidence: `results/dense-sk2-rotcodes-v2-offline.json` and `.log`.

- Original incumbent AST preserved; old/eager SK2 bodies unchanged after
  renames/dispatch. Late decode differs only in the decode/MMA block.
- All 20 native family specializations match original geometries: eight
  old-resource twins, eight eager rotcodes, four just-in-time decoders.
- 63,438,848 exhaustive payload/position/window cases, 1,000,034 random/basis
  words, actual Torch packing on 64,064 words, five malformed-pack rejections.
- Actual eager **and just-in-time** C++ decoder expressions compiled with
  clang++ C++20 and passed 7,929,856 word/window cases.
- Actual native-resource host dispatch compiled with CUDA API/kernel stubs:
  60 cases passed, including resource layout fields; negative capacity and
  unsupported qkv32 late decode rejected. This is **not CUDA compilation**.
- Per-accumulator instruction traces match across all eight geometries;
  native-K FP32 models still pass 32 cases with 2,752 exact escapes and worst
  relative L2 0.0017792 versus FP64. NumPy does not emulate actual GPU MMA
  rounding; exact GPU twin checks remain mandatory.
- Seven fresh-output wrapper pairs and 43 fallback cases passed, including
  forced qkv32 runtime rejection. Resource admission rejects register/local
  growth, residency loss and inconsistent layouts.
- Real event-loop code passed mocked 40-block ABBA/160-READ ordering tests.
  Boundary/invalid-event tests validate the statistic and exact binomial
  bound. Actual gate control passes three draws plus independently seeded
  confirmation, and rejects pre/post null drift, missing confirmation,
  fourth-pass regressions, marginal gains and pre/post correctness failures.
- Actual prepare fallback tests cover failed build, selective admission and
  deadline exhaustion after an earlier winner. No scratch graph/input/output
  or reference is retained in returned state.

**Expected:** the unchanged o32/o16 eager kernels now have a realistic path
to admission if V1's ~2.3–3.2% raw-event gains reproduce with clean controls;
qkv16 is another plausible adopter. Just-in-time decode is a hypothesis for
an additional fraction of a percent to a few percent on weak cells, most
plausibly unpooler32. It may do nothing or regress; pooler16 is particularly
unlikely to clear a 2% bar from its V1 starting point. There is no claim that
all weaker cells will pass and no further threshold concession for them.

**Not verified locally:** V2 CUDA compilation/registration, actual GPU
attributes/spills/residency, CUDA graphs/numerics, null spread, total prepare
duration or performance. nvcc/ptxas/NVRTC and a GPU are unavailable. More
diagnostic specializations increase compilation cost; the optional build
overlaps unchanged parent prepare and has a 285 s total deadline, but actual
wall time still needs the caller's GPU check. A failure keeps the incumbent.

TRIED is updated only in this scoped worktree because the shared helper is
absent; the caller must merge the V1 result and V2 result row into the shared
ledger. This is a user-requested follow-up, materially different from the
closed sub-12-bit codecs (`ee7d7b81`/`cbd91613`), geometry sweeps (`83c46449`)
and token-owned epilogue (`8ba14018`).
