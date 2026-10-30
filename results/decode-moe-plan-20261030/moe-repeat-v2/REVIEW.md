# Bounded reporting repair of d563fa2d

Source `moe_repeat_diag_v2.py`, SHA256
`b493dc7b9046bf559c25daa70544a2781fe69caf9aafc3a500283c09e7850d12`,
242,413 bytes. One additional unscored diagnostic check at most; the total
question budget is two checks including `d563fa2d0a1b47e6`. No scored run,
algorithm selection, parameter change or performance claim.

The previous run passed the official tests and reached MR_DONE, but its
per-case JSON exceeded the result's 16,000-character log-tail window. Only
nine final M32 parent-control rows were retained. That is a diagnosed
instrumentation defect; it leaves the small-token root cause unresolved.
The prior source, result and incomplete evidence remain preserved.

This version keeps the same parent CUDA, observed TR8 CUDA, native compiler
flags, routing fixtures, graph order, scratch snapshots and timed function.
It collects records during prepare and prints a compact zlib/base64 summary
at the end. All comparison/field/shape aggregates are lossless. Four first
chronological differences provide samples. Full records are included only
when their compressed representation fits 4,000 characters. The summary has
a hard 9,000-character ceiling; optional examples are removed before any
aggregate would be discarded. Summary columns and field order are embedded.

Diagnostic error metrics now use FP64 after snapshotting, so extreme finite
FP32 values cannot overflow the report's norm. This changes no timed arithmetic.
Nonfinite input samples are explicitly tagged strings in JSON.

CPU tests verify the unchanged parent prefix and function ASTs, snapshot bank
offsets, valid prefixes, signed zero, NaN serialization and route fixtures.
A 75-case/35-group adversarial reporting test round-trips every aggregate
through a retained 16,000-character tail plus a 1,260-character profiler suffix.
Its footer is 5,700 characters. This tests retention, not GPU kernel behavior.

Admission must record the previous job, exact repair, fresh quota/active-job
state and any spacing exception before upload. A diagnosed logging-only repair
is the sole reason for an exception; it is not a repeat to seek a better point.
Keep four or more MoE checks after the retry and preserve the final Decode
check. A second incomplete diagnostic does not authorize another blind retry.
