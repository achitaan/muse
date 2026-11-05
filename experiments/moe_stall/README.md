# MoE warp-role stall diagnostic (prepare only)

Check file: `experiments/moe_stall/moe_stall.py`

Base: `board_best/moe_m3_scored_2.3797.py`, copied byte-for-byte as the
185,756-byte prefix (SHA256
`f72cdf2b5438c70369b0fb823e171f4022b22b97f0dff66c6170dfe3aaf845fb`).
The inherited `kernel_fn`, CUDA/CPP strings, packing, selector, and defines
are unchanged. No production optimization or selection is adopted.

The only shared build change is a `load_inline` adapter supplying **C++20 to
both host and nvcc**, including the incumbent's unchanged builder. Diagnostic
kernels are separately named extensions and use separate scratch/output
allocations. Production preparation/selection runs first. No candidate kernel
is reachable from the timed `kernel_fn`.

## Protocol

* bs16 MID, bs32 WGS MAIN, bs256 WGS MAIN; eight fresh draws each.
* Uses the file's own `_moe_routes`: skewed popularity, top-8, rotated to the
  busiest rank. Hidden states are generated first, just as in the selector.
* Compiles only the two necessary diagnostic define tuples. If the inherited
  selector actually picks a different build, that shape is explicitly skipped,
  rather than instrumenting one build and timing another.
* For each draw, capture three graphs: exact incumbent, instrumented binary
  with counters disabled, and the same binary with sampled counters enabled.
* Four symmetric `inc, disabled, sampled, sampled, disabled, inc` timing blocks.
  Every replay gets an independent **512 MiB READ flush** before its event
  interval. No trace copying/reset kernel or flush is included in that interval.
* Collect counters from the final sampled replay. Both eager and graph outputs
  must be finite and match the incumbent at relative L2 < 1e-2 (same admission
  tolerance as the inherited selector; small-batch atomic fixups are not
  bit-repeatable).
* Record lane 0 of **every warp** in sampled CTAs. Draw d samples
  `cta % 16 == 2*d`: 8 or 9 of 132 CTAs per draw. Samples span the grid and
  rotate, but deliberately do not claim a full-grid tail census.
* Counters use `%globaltimer` nanoseconds. Entry/exit `clock64` cycles and
  entry SM IDs are also retained. Clock cycles are never compared across SMs.
* Counters have one writer per `(CTA, warp, phase, metric)`; each replay resets
  its sampled slab. No new shared memory, atomics, or synchronization is added.
  The incumbent's ring/barrier/atomic-publication/MMA instruction sequence is
  statically checked unchanged.

## Log meanings

`MOE_STALL_BUILD` / `MOE_STALL_RESOURCE`: actual defines and diagnostic
resources. Resource vector order is in the field name: launched threads,
dynamic shared bytes, then registers/local bytes/static shared bytes/occupancy
for the `<false>` and `<true>` kernel instantiations. Both must be launchable.

`MOE_STALL_DRAW`: histogram, selected CTA count, output error and three graph
times for one routing draw. `overhead_pct` is sampled/incumbent - 1.

`MOE_STALL bs=... phase=... role=... metric=... median=... p90=...`:
medians/interpolated p90 over sampled **warp records**, not weighted by number
of stages. Values are cumulative per launch, in **microseconds**, except
`elapsed_clock`, which is in cycles. Roles:

* WGS `weight_producer` and `activation_producer`: lane-0 issuer warps;
  `consumer` pools all 16 consumer warps (their individual samples are retained
  through collection, and used to calculate intra-CTA skew).
* MID `consumer_refill`: compute warp 0, whose thread 0 refills weights;
  `consumer`: the other seven compute warps; `epilogue_helper`: ninth warp.

Metrics:

| Metric | Meaning |
|---|---|
| `span` | Role's phase start-to-end wall time. FINAL spans helper join, final barrier, cleanup and finalizer. |
| `empty_wait` | WGS producer's explicit ring-empty barrier call, only when a slot is reused. High means consumer back-pressure. |
| `implicit_empty_wait` | **MID warp-0 per-unit CTA barrier**, the implicit prerequisite for refill. Not an explicit ring-empty wait; includes compute skew/partial stores. Alias of its `barrier_wait`, not additive. |
| `weight_wait` | Consumer full-slot (weights ready) wait-call time. |
| `activation_wait` | WGS consumer activation-full wait-call time, after its weight wait. |
| `dqmma` | Stage/unit loads, dequant, MMA issue and in-flight MMA waits; deferred fixup/epilogue calls subtracted. Not pure tensor-core busy time. |
| `retire_wait` | WGS segment's final `wg_wait<0>`. |
| `dependency_wait` | MID job doneA; WGS activation producer gather/doneAs; WGS designated split-unit finisher wait (consumer). |
| `partial_empty_wait` | MID compute waits for helper to release a partial buffer. |
| `partial_ready_wait` | MID helper waits for compute partials. |
| `barrier_wait` | MID compute per-unit CTA barrier. |
| `epilogue_total` | Direct epilogue/partial publication plus deferred WGS flush time, including any wait inside a flush. |
| `transition_wait` | First B dependency wait (MID, WGS activation producer), or first B activation-full wait (WGS consumer). **Subset/alias** of another wait, not additive. |
| `weight_issue` | MID/WGS weight copy issue work. MID steady-state issue belongs to compute/refill warp, not helper. |
| `finalizer_wait` | Consumer waits from pre-grid CTA sync through post-grid CTA sync. |
| `helper_join_wait` | MID final compute/helper rendezvous. |
| `consumer_end_skew` | Within a sampled CTA, latest minus earliest consumer-warp phase end. |
| `sampled_cta_end_lag` | Sampled CTA's latest consumer end minus earliest sampled CTA's latest end, separately for each draw. **Not full-grid skew.** |
| `elapsed_clock` | Role's entry-to-exit `clock64` delta. |

`MOE_STALL_SUMMARY bs=...` is an **approximate, exclusive consumer-warp-time
accounting**, from consumer A start to kernel exit. Within each draw it averages
warp shares, then prints median/p90 over draws. Four buckets:

1. `producer_starved`: consumer weight-full waits.
2. `dependency_bound`: consumer activation-full, doneA/fixup and partial-empty
   waits, plus the A-to-B function/boundary gap.
3. `consumer_bound`: remaining measured A/B time, **including epilogues,
   boundaries, timer/counter overhead and unclassified work**. This is a
   residual/upper bound, not a pure compute-utilization measurement.
4. `tail`: consumer B end to kernel exit (helper join, grid wait and finalizer).

These sum to 100% for each individual warp; independent per-bucket medians need
not sum exactly to 100%. Producer waits overlap consumer work and are **never
added to this partition**. Wait attribution follows the actual wait order, so
simultaneously missing weights and activations cannot be causally separated.
`accounting_overflow_ns` reports any per-phase over-count before conservative
clipping; a nonzero value needs investigation.

Overhead fields distinguish sampled/incumbent (`total_overhead_pct`), disabled
binary/incumbent (`codegen_overhead_pct`), and sampled/disabled
(`counter_overhead_pct`). Instrumentation can change register allocation and
WGMMA scheduling even when counters are off. Sampling can also perturb unsampled
CTAs through expert dependencies. Large overhead makes the shares qualitative;
do not treat them as an unperturbed causal profile or a predicted speedup.

All raw role metrics precede the three summaries so summaries survive tail
truncation. Fetch the full log for a complete role census.

`MOE_STALL_SKIP` records budget/unsupported build/no-local-route/diagnostic
errors. Added diagnostic exceptions and logging failures are contained in
prepare. No zero-local-job draw is launched, avoiding the inherited all-remote hang.
`MOE_STALL_DONE` reports total preparation elapsed time.

## Budget and limitations

Soft deadline is **270 s from entry to prepare**, leaving a 30 s margin. New
compilation requires at least 90 s remaining initially and at least 45 s before
each distinct build. Replay loops stop at the deadline and partial draw counts
are explicit. Native extension compilation is blocking/non-preemptible; pathological
compiler delays can exceed the soft deadline. Runtime <300 s cannot be verified
without the caller's H200 check. The inherited preparation itself is unchanged
and can also consume the budget.

No local Torch/CUDA toolkit/nvcc/GPU was available. Thus sm90a compilation,
resource usage, actual overhead, graph correctness and wall-clock setup budget
remain **unverified on GPU**, not claimed as passing.

## Offline validation

```sh
python3 -m unittest discover -s experiments/moe_stall -p 'test_*.py' -v
python3 -m unittest discover -s experiments/moe_headroom -p 'test_*.py' -v
```

The new suite includes a real host-C++20 compile/run of the counter helpers,
substituting only hardware clock/SM reads; replay reset, alias duration equality,
sampling, non-overlap, address guards, null counters, and non-lane-0 behavior
are executed. Python tests cover exact prefix/AST, source drift, source
synchronization/arithmetic preservation, exports, build dialect, graph timing
order, budget/error paths, manufactured accounting and skew/overhead reporting.

Branch-local TRIED row only: the caller explicitly prohibited submitting,
pushing, or committing to main. Caller must publish the shared-ledger claim and
record the eventual GPU result/job ID.
