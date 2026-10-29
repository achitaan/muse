# Paired epilogue stores: compiled, correct probes, no material gain

Official check `5262c2c560c94197`, source `9fc2993a`, finished October 29.
The resource-query namespace repair compiled successfully. Both BIG variants
report 96 registers, zero runtime local/static shared bytes, 231168 dynamic
shared bytes, 576 threads and one resident block per SM. Preparation took
218.748 seconds. This is actual GPU resource evidence, unlike the earlier
NVRTC proxy.

The new kernel passed all 18 fixed synthetic comparisons with reported zero
relative and worst-row error. Both changing-input/route graph replays also
reported zero error. All four all-remote probes returned exact zero, and the
duplicate-route comparison passed with zero error. These probes executed the
new kernel. They do not establish its official whole-suite coverage.

It failed the predeclared performance gate:

| Normal tokens | Median paired new/old time | Change |
| ---: | ---: | ---: |
| 32 | 1.002975822 | 0.30% slower |
| 128 | 1.003882508 | 0.39% slower |
| 256 | 0.992302996 | 0.77% faster |

Skewed M128 was 0.48% slower. The largest nearby point improvement was about
1.87% at M65, below the fixed material threshold and not a reason to construct
a selector after seeing the data. Ratios are medians of paired ratios, not
ratios of the two marginal medians. All raw samples and arm orders remain in
the result. Repeated graph measurements within this one job are not independent
jobs, and no statistical significance is claimed.

Consequently `selected_main=false`. The official five tests, six benchmarks and
integrity checks passed using the retained paths. No benchmark profile contains
the new namespace. The official 2.335583633x point belongs to those retained
paths; it is neither a speedup nor full-suite correctness evidence for the new
kernel. Halving this helper's static store-instruction count did not earn the
required overall improvement after charging its complete work. This result
does not isolate a single hardware cause.

The controller discarded trial `850345e` and reverted as `d228b6e`. All 44
protected source and metadata files remain exact, including MoE `33114871`.
No scored submission, source promotion, Decode transfer or parameter retry
follows. The namespace defect is resolved and this performance hypothesis is
closed. The original failed source and result remain separately preserved.

One check was used, zero scores. From the fresh prelaunch balances minus this
accepted check: Dense 19, MoE 14, Attention 19, Decode 20, Prefill 20 checks;
ten scores per track, 92 checks and 50 scores total. This is a derived balance,
not a later live quota observation. No new check before
`2026-10-29T07:28:00.763398Z`. The authorized window ends at 8 p.m. Toronto,
with normal hourly spacing and reserves unchanged.

Raw launch, source, quota, complete result, profiles and immutable controller
history are beside this review. Reproduce the terminal summary with
`python experiments/prepared/review_moe_epair_store_repaired_result.py`.
The independent CPU Attention rejection packet saved during the wait is
`results/attention-l8-nvrtc-screen-20261029/`; it consumed no GPU checks.
