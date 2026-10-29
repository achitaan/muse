# Factor scratch/cache rejected

Check92577a6426f74605, exact source85a317b7, completed October29. The candidate
compiled with96registers, zero local/static shared bytes,231168dynamic shared
bytes and one resident576-thread block per SM. Both BIG variants agree.
Preparation took218.051seconds in the candidate's diagnostic timer.

The new kernel passed all18 fixed synthetic paired probe cases, two fresh-input
graph replays, four all-remote exact-zero probes and a duplicate-route comparison.
Reported error was zero except M385 (relative2.85e-6, worst-row2.99e-5), within
the fixed limits. This is real GPU probe evidence for the new path.

It failed the predeclared performance gate:

| Normal tokens | Median paired new/old time | Change |
| ---: | ---: | ---: |
| 32 | 1.007734641 | 0.77% slower |
| 128 | 1.008412964 | 0.84% slower |
| 256 | 1.003610915 | 0.36% slower |

Skewed M128 was1.30%slower. Other tested nearby shapes ranged from about0.36%
faster to0.90%slower; there is no material-gain signal. These paired repetitions
come from one job and synthetic routing, not independent jobs or official judged
route samples. No statistical significance is claimed.

Consequently selected_main=false. The official full5tests/6benchmarks/integrity
suite ran the retained paths, with no factor namespace in benchmark profiles.
Its2.325355016x result is an incumbent measurement, not a speedup or official
full-suite validation of the new factor kernel. Less logical scratch traffic
did not earn an improvement after the added reconstruction/cache work in these
probes. The run does not identify a single isolated hardware cause.

The controller discarded trial220eb61 and reverted atc24010d. Original33114871
and all44 protected source/metadata files are exact. No scored submission,
source promotion, Decode transfer or parameter retry follows. Immutable phase
history is autoresearch-results/archive/20261029-032818/.

One check used. Derived from the fresh prelaunch quota minus this job:
Dense19, MoE16, Decode20, Attention20, Prefill20checks; ten scored submissions
per track. Total95checks/50scores. This is not a later live quota query.
No new check before2026-10-29T04:22:28.724212Z; preserve the20:00Toronto window
and reserves. CPU routing/rotation/Decode-binding screens are in cpu-screens.md
and their source-bound JSON files. They do not authorize another GPU run alone.

Reproduce this terminal review with
`python experiments/prepared/review_moe_factor_y_result.py`. Raw launch, quota,
submission, result and all paired samples remain beside this review. The
predeclared mechanism, CPU proofs and gates are in
experiments/prepared/moe_factor_y_review.md and the preflight packet.
