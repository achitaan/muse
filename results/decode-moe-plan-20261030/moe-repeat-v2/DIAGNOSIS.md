> Scope correction2026-10-30: when WGS runs at M32,the old snapshot reads unused row-major inter and SS regions. Those byte observations do not establish live SwiGLU xB equality or WGS shared norm statistics. The recorded ybuf/output observations remain valid. Tiny/MID live inter/atomic-SS observations and the MoE-only authorization are unaffected. See ../wgs-live-snapshot/REVIEW.md. Raw evidence is preserved.

# H200 MoE bit-stability diagnosis

Job `9dc89929ea384b7d`, source `b493dc7b`; all 75 fixture records are represented by the retained compact summary.

| M | Parent SS changes /15 | Parent output changes /15 | TR8 inter/ybuf changes | TR8 SS changes /15 | TR8 output changes /15 |
|---|---:|---:|---:|---:|---:|
| 1 | 12 | 1 | 0 | 11 | 1 |
| 3 | 12 | 6 | 0 | 12 | 6 |
| 8 | 12 | 8 | 0 | 12 | 8 |
| 16 | 12 | 7 | not run | not run | not run |
| 32 | 0 | 0 | not run | not run | not run |

The table uses consecutive replays of the same parent graph, with unchanged inputs and tracked scratch-bank generations. All intermediate/SwiGLU and FP32 ybuf comparisons were byte-identical across every parent control and TR8 comparison. All observed outputs were finite. M32 controls were byte-identical in every field.

Maximum final-output relative L2 was 7.8388557062e-05 for parent controls and 7.83981253937e-05 for TR8 versus parent. Norm-sum relative L2 maxima were 6.88660651445e-07 and 5.16495488584e-07. These are maxima in this protocol, not confidence bounds or whole-domain guarantees.

The data localizes observed differences to normalization statistics. Combined with the unchanged sourceâ€™s FP32 atomicAdd norm accumulation, this supports a summation-order explanation. It does not prove every TR8 route correct, reproduce the unavailable historical failed weight/input tensors, or establish a speedup.

Official check: 5/5 tests, 6/6 benchmarks, integrity/no flags; prepare 138.934373 s. Timed kernel was unchanged parent; its point 2.343913173x is not a new algorithm result.

The strict user byte gate remains in force. TR8 stays rejected: no timing comparison, scored upload, transfer or publication follows from this diagnosis. No further diagnostic retry is scheduled.
