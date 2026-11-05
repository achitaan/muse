# Clean same-routing MoE counterfactual

CHECK file: `experiments/moe_cf/moe_cf.py` (standalone).
Branch: `loop/moe-cf`, parent `d966ccd` (`loop/moe-stall`).

The production file is an exact byte prefix, including line endings. The only
production setup differences are C++20 build-flag compatibility and bypassing
the optional selector; production dispatch, kernels, weights and defaults stay
unchanged. Diagnostics never publish ablation modules or scratch to production.

Prepare measures bs32 MAIN, bs16 MID, then optional bs256 MAIN. Each case uses
six fresh draws from the incumbent `_moe_routes`, with the **same x/ids/weights**
for all arms and isolated persistent scratch per arm. BASE directly invokes the
incumbent extension. A separate capture of the identical BASE launch supplies
the null. Each draw/arm has 12 ABBA rounds, one single-kernel graph replay per
event interval and a 512 MiB read flush before every replay. Ratios are var/base.

Compile-time source specializations:

* NODQ: shared code/scale loads feed garbage BF16x2 register fragments directly;
  FP4 decode and scale arithmetic are removed; MMA retained.
* NOMMA: code/scale/activation loads and decode retained; MMA issue/fence/commit/
  wait removed. Rotating-XOR register sinks feed one private end-of-segment store;
  opaque finite accumulators preserve downstream epilogue work.
* NOEPI: removes phase-A SwiGLU/activation stores and phase-B epilogue work,
  including MID sum-of-squares atomics, while retaining accumulator sinks and
  **all readiness publications/partial-counter protocols**.
* NOFIN: returns after barrier/generation/flag housekeeping, before finalization.
* STREAM: no decode/MMA/epilogues/finalizer, but all global producers, shared
  code/scale reads, MID activation reads and slot-release/partial protocols
  retained. Register sinks prevent load elimination by ptxas.

MAIN's five ablations build concurrently on import-time daemon threads. MID
starts after all MAIN builds finish. Only bounded build waits occur in prepare;
the diagnostic soft deadline is 270 seconds from prepare entry, with ten seconds
reserved for reporting. Incomplete rounds are not reported; missing arms/draws
are explicit. Native compiler threads cannot be interrupted, but cannot hold
prepare's return. There are no explicit process launchers or custom graph APIs.

`MOE_CF_PAIR` records per-draw results; `MOE_CF` reports medians of draw medians
with completed counts and the null ratio. `MOE_CF_SUMMARY` reports individual
saved-us deltas, NOMMA-minus-NODQ, STREAM gap, and STREAM-minus-NOFIN as a rough
recoverable-consumer **proxy**. They are nonadditive interventions, not exclusive
profiler subcosts. Sink overhead is not subtracted; NOEPI/STREAM intentionally
read zero/stale activation/output data from private scratch. Register-allocation
changes and altered dependencies can also affect timing. These caveats are why
the interpretation says “candidate,” not proof.

Offline validation: `python3 -m unittest discover -s experiments/moe_cf -p test_cf.py -v`.
No local Torch, CUDA compiler or GPU exists; CUDA compilation, GPU timing,
full six-draw completion and prepare wall time require the caller's H200 check.
No check was submitted and nothing was pushed or committed to main.
