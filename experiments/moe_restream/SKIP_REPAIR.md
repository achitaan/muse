# Repair of b3fe8be00fcb4666: TQ64 was never measured

## Root cause proved from the real log and incumbent code

The job logged MAIN **and** FALLBACK at M256, then `selected=[] s=41.7` and
`MOE_RESTREAM_SKIP reason=budget_or_nonPRMT`.

`_moe_valid(256, sfx)` has only **one** candidate without sfx: the mma.sync
fallback. Its extra FALLBACK arm is added only by `if M > 8 and sfx`. The
reported two arms therefore prove sfx was nonempty. The base sets this suffix
only to `('-DOKC_PRMTSC=1',)`. Thus **the >170 s elapsed-budget condition
fired**, not non-PRMT. That elapsed time included packing, all required builds
and the 41.7 s selector. The old log did not record exact total prepare time;
we cannot reconstruct that number honestly.

`PRMT` is the exact byte-permutation **scale-decode instruction path**, not
the physical weight layout's name. The supplied headroom log explicitly says
`layout=wgs` for M256, consistent with the MAIN arm and positive-normal scales.
No new weight-layout implementation is needed for this setup failure.

The 2.337x score is an **incumbent-only** result, not a TQ64 performance result.
There were no TQ64 compiler/resource/numerical/timing lines.

## Corrected order and bounded work

1. Clear prior build overrides and call `_moe_sel_original_prepare`: the
   original **pack/build core**, without `_moe_select`. The previous alias
   `_mh_original_prepare` called the wrapper that includes the full selector.
2. Run TQ64 admission, compilation, packing, numerical checks and its unchanged
   two-pass 16-draw ABBA gate. No acceptance threshold/sample reduction.
3. Only then optionally select small-shape incumbent modules at M16/M32,
   using **already-compiled** candidates. Keep the original 3 draws, 9 event
   samples, numerical bound and 3% selection rule. Abort incomplete cases.
   No fallback build can be started here. M128/M256 stay on their defaults,
   matching both supplied `selected=[]` runs and preserving TQ64's comparator.

TQ64's total deadline is now 285 s; compilation admission requires >=60 s
before that deadline. After compilation at least 20 s must remain, and after
packing at least 10 s. Packing is synchronized before reporting its completed
elapsed time. Optional cached selection has a 290 s deadline, leaving cleanup
margin under the 300 s limit. An unexpectedly slow native compiler cannot be
interrupted by this code; actual GPU-worker duration is still unverified.

## Diagnostics on the next check

Each `MOE_RESTREAM_PREP` / `MOE_RESTREAM_SKIP` line includes:

```text
stage=core_ready|pre_build|post_compile|post_pack|done
layout=wgs|unit|unit+fat scale_decode=prmt|lut prmt=1|0
elapsed_s=... remaining_s=... gate_remaining_s=...
```

`remaining_s` is 300 minus measured elapsed time. `gate_remaining_s` is 285
minus elapsed time. `layout` describes the actual incumbent M256 build.
Distinct skips are `reason=nonPRMT`, `insufficient_compile_budget`,
`insufficient_gate_budget` (with stage), and `unsupported_arch`. Exceptions
also print those fields plus `MOE_RESTREAM_ERROR type=... detail=...`.

Successful progression prints resource attributes, hot/empty checks, per-draw
timings, pass decisions, then `MOE_RESTREAM_READY shapes=[...]`. Optional legacy
selection has separate `MOE_SELECTOR_POLICY/CACHED/SKIP` messages, so it cannot
be mistaken for TQ64 admission or acceptance.

## C++20 dialect check

Previously neither builder supplied explicit `-std`; both inherited Torch's
defaults. We cannot inspect or claim the remote default from the local machine.
The repair instead guarantees equality: a **module-local** `load_inline`
adapter appends `-std=c++20` to both `extra_cflags` and `extra_cuda_cflags` for
**all original incumbent `_ext` calls and TQ64 calls**. Conflicting `-std=` /
`--std=` and two-argument forms are removed. The Torch module itself is not
modified. Every build prints:

```text
MOE_RESTREAM_DIALECT module=... host_std=c++20 cuda_std=c++20 policy=shared_incumbent_and_TQ64
```

The TQ64 module tag is v2. CPU tests execute the **real original `_ext` body**
and `_rs_ext` through this adapter with a recording builder, confirming one
identical C++20 flag in each host/CUDA invocation. Native metadata tests now
compile with `-std=c++20` and enforce `__cplusplus >= 202002L`.

The GPU kernel source `restream.cu` is byte-for-byte unchanged from b13447a.
No numerical/tiling/readiness workaround was stacked onto an unmeasured kernel.
Local tests prove the old failure condition, regression-test valid PRMT
admission at elapsed=171 s, distinguish both skip reasons/budgets, and prove
TQ64 runs before optional selection. Actual CUDA compilation, graph replay,
performance and <300 s wall time require the parent's next check; no local
GPU/Torch/CUDA toolkit is available and no check/submission was made here.
