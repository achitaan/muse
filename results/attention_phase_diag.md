# Attention CTA phase diagnostic

Check file: `candidates/attention_phase_diag.py` on `loop/attn-phase`.
Base commit: `4cd65186da5259d3ec8b1654924cbbd00331376c`.
Production prefix SHA256: `6b65dec7c1aaeb4675815555ddbb0ee8a0de17ab93e56523c3d1d386899998ca`.

## Scope and execution

The complete `board_best/attention_kb1h2_scored_2.6536.py` is an exact byte
prefix. Its CUDA literals, builders, tuner, dispatch table and `kernel_fn` are
not replaced or patched. An appended idempotent `prepare()` runs after the
incumbent tuner, through a module-tail adapter (this stateless board has no
public prepare hook). No added diagnostic work executes in `kernel_fn`.

Prepare builds separate paired-head, one-head and VP2 extensions with the
original source-specific flags plus explicit C++20, SM90a. Builds run in
cancellable subprocess groups; ninja/nvcc descendants are killed on a deadline,
not left behind as late background compilers. Total deadline is 280 seconds
from the original import start, reserving 45 seconds after build cutoff.
All diagnostic execution is best effort and exception guarded.

The six sliding cases and d1F/v1F use independent synthetic inputs with the
incumbent's 80K/scattered-page distribution and its actual selected route.
All CTA starts/exits are recorded; full phases are sampled for CTA 0,
middle, last, and all splits of request 0/head 0. This covers that group's
dynamic last-arriving merge leader. Diagnostic workspaces/caches are isolated
and original cache object identities are restored, including on failure.

Each case performs four ABBA blocks: eight cold replays of the **actual
incumbent** preceded by the marker, eight of the instrumented copy preceded by
the same marker, and eight incumbent-only controls. Every timed graph is
preceded by a genuine 512 MiB HBM READ reduction outside the event interval.
Trace resets and host trace copies are outside timing. The instrumented copy
must reproduce the incumbent output within rel-L2 0.002 and byte-identical
new K/V rows after those rows have been poisoned.

## Printed lines

- `ATTN_PHASE_BUILD kind ready/failed/budget_cancelled`: build availability.
- `ATTN_PHASE case cta=... role=first/middle/last/leader n=8 ...`:
  median timestamps in microseconds **since that CTA's own entry**. `entry_us`
  is zero. `start_offset_us` is relative to the first CTA of the replay.
  The dynamic leader is aggregated by role across all eight replays; a `|`
  list identifies the different actual CTA IDs. Missing phases are omitted,
  never treated as zero (e.g. an empty split, or a non-finalizing CTA).
  `gate_kcycles` is the same-CTA clock64 entry-to-gate interval in thousands
  of cycles, as a cross-check independent of globaltimer's nanosecond units.
- `ATTN_PHASE_OVERHEAD case ...`: median gross CUDA-graph event times for
  raw incumbent, marked incumbent and marked instrumented copy. Absolute
  instrumentation cost is instrumented-minus-marked-incumbent; its percent
  denominator is the **marked graph**, not a CUPTI kernel duration.
  `marker_envelope_us` exposes the extra marker/graph-node cost separately.
- `ATTN_PHASE_SUMMARY case ...`: median all-CTA start skew and observed grid
  span, plus phase durations/shares for the latest fully sampled output owner
  in each replay. Phase percent denominator is that owner's entry-to-exit
  residence; start-skew percent denominator is the observed whole-grid span.
  Neither percentages nor the start skew are claimed as removable savings.
- `ATTN_PHASE_ERR ...`: failed or skipped measurements; no missing case is
  imputed. No check/submit command is run by this candidate.

Phases are entry; Q/K normalization, RoPE and FP8 cache-store **issue** complete;
first KV tile **observed ready by the consumer**; thread-0 mainloop-end
observation; partial's release/acquire arrival ticket complete; post-ticket
wait start; post-CTA-barrier wait end; post-CTA-barrier output-store issue end.
Thread 0 also records an exit observation. Pending global stores/CTA teardown
are not force-drained, and this observed span is **not** a CUPTI kernel span.
Inactive cooperative CTAs with no output slice are not called output owners.

VP2 is structurally different: its existing barrier jointly observes completed
Q/K preparation and KV arrival, so these timestamps are identical, not an
artificial serialization. Mainloop/push timestamps are written by WG1 thread
128, wait/gate by WG0 thread 0 (same CTA/SM for clock64). Push is an asynchronous
DSMEM **issue** observation; wait end observes the real local mbarrier complete.
The signed `wait_setup_us` gap can be negative because push and wait overlap.
VP2 has distributed output owners, **no last-arriving CTA leader**. An added
WG0-only named barrier makes the gate timestamp cover all WG0 output stores.

## Launch-timing limitation

`launch_marker_to_first_us` is the globaltimer difference between a preceding
GPU marker and the first observed CTA entry. It includes the marker's tail
and the device inter-kernel scheduling gap. It is explicitly a **proxy**, not
host CUDA API launch-to-execution latency. Measuring that literal host/device
quantity requires correlated CUPTI activity/API timestamps; it cannot be
honestly obtained by subtracting host time from an uncalibrated globaltimer.
The matched marker controls quantify its perturbation but do not remove it.

## Local verification

`python3 scripts/validate_attention_phase_diag.py`: passed. Checks exact
production byte prefix/AST and addition-only diff, all source transformation
anchors, timestamp coverage/units/huge-integer precision, dynamic leader
aggregation, signed VP2 overlap, output formulas, real replay-loop cold-flush
ordering and failure-injected cache identity restoration. Simulated eight-case
transcript is about 13 KiB before INFO/build lines.

`python3 scripts/validate_attention_sliding_diag.py`: passed (existing tests).
Appended diagnostic lines are whitespace-clean (CPU assertion). Full staged
`git diff --check` reports only the incumbent's inherited trailing whitespace,
which is deliberately preserved for byte identity. The check with
`core.whitespace=-blank-at-eol` passes; no new-tail whitespace is waived.

No local GPU or nvcc. CUDA compilation, device correctness, instrumentation
overhead, absolute timings and the 280-second runtime ceiling require the
caller's GPU check. No GPU job has been submitted by this branch. The ledger
entry is branch-local only because the caller forbids all main-branch writes.
