# Current handoff - October 28, 2026

Goal: approach top 50 overall, main chat only, no deadline. Preserve the selected
model and conserve official checks/scores. No subagents, paid GPUs, external
model providers, global changes, credential output, blind sweeps, repeated
baselines or cosmetic resubmissions. The ordered MoE -> Decode -> Attention ->
Prefill pass has full passing coverage. Top 50 overall remains unmet; Goal active.

| Track | Published speedup | Last observed place | Job |
| --- | ---: | ---: | --- |
| Dense | 1.527771327x | 99/1245, 17:04 UTC | 6131bc96427e4486 |
| MoE | 2.348299935x | 53/994, 17:58 UTC | b6c70c9b83724d2e |
| Decode | 1.993956388x | 146/869, 17:58 UTC | 441eb580f3074669 |
| Attention | 2.562402309x | 82/951, 17:04 UTC | 20524d06e3894595 |
| Prefill | 1.663469245x | 123/576, 17:04 UTC | 23e7c8b4c3e54869 |

Overall: **146/1436 at 17:58:08 UTC**. Decode's 50th-place score was 2.1438x;
MoE's was 2.3539x, about 0.24% above ours. Placements change. Prefill does not
count toward overall. No job or controller is pending.

## Decoder publication and rejected trials

Exact 09badc9b source is both best/decoder_verified.py and
best/decoder_published.py, 988314 bytes. Primary full check 09b34f4e921e49a6:
1.994319167x. Scored confirmation 441eb580f3074669: 1.993956388x, interval
[1.964047042,2.023865734], full 13 tests/12 benchmarks/integrity. It transfers
the corrected 33114871 MoE, adds implicit small-token FFN zero padding and early
FFN loads, preserving all BF16/FP32 norm boundaries and full FP32 small routing.
This replaces a61735ee's 1.965324622x publication; frozen history remains.

Phase7 skipped new projection setup at its budget gate. Phase8 compared three
fixed SK2 small-token projections: all tied or lost. Phase9 compared one output
projection: 4.9% slower. Both explicit choice tuples were empty and the guard
discarded them despite higher noisy aggregate points. No rejected projection
source was scored. Those full trials reused incumbents for timed calls, so do
not claim full official correctness of unselected new kernels. Evidence:
results/resumed-decoder-phase7-20261028/ through phase9, including full sources,
profiles, audits and immutable controller histories. Observed setup varied from
215 to 257 seconds; stay below the official 300-second prepare and 1 MB limits.

Publication used one confirmation of the retained update after its primary
check and three incumbent-only observations. Individual intervals overlap;
report measured points and placements without claiming formal significance.

Profile correction: m16trace_ is the 16-token output projection (~5.6 us).
The ~14.6 us dd_cap2_b_cpssm stage is unpooling. Resolve dispatch and symbols
before attributing truncated profile names. Do not repeat the mistaken O16
bottleneck attribution or the rejected projection comparisons.

## Next: MoE lossless scale layout

MoE 33114871 remains checked and published at 2.348299935x, full 5/6/integrity.
Paired BF16 epilogues and exact DQ8 placement preserve original GEMM rounding
and FP32 norm arithmetic. The prior mixed skinny/fat selector also chose none.

Read experiments/prepared/moe_scale6_review.md. A new CPU prototype packs
eligible FP8 scale bytes into six bits without changing values, reducing WGS
stage weight bytes by 2.7778%. Its audit round-tripped 1,310,720 scale values,
exhausted adjacent six-bit field pairs and all 256 eligibility byte values,
and verified two-word load bounds and alignment. The GPU kernel is NOT yet
implemented or measured. Added decoding can erase the bandwidth gain.

Implement only the WGS scale tail and corresponding exact CUDA load/decode,
with original fallback for any scale outside 0x40..0x7e. Keep non-WGS layouts,
NVFP4 code bits, all BF16/FP32 math, routing and norms unchanged. Audit actual
full-layout inversion, compile variants, transaction sizes and factory bindings
before one full MoE check. No geometry sweep. Require evidence the compressed
path actually ran; fallback-only timing cannot establish a new optimization.

Read STATE.json, per-track notes, official TASK.md/AGENTS.md and notices. All
controllers are archived; initialize one scoped phase with cached baseline
before trials. Preserve handles across observation failures and push complete
evidence at boundaries. Query quotas only when needed. The last MoE observation
before phases8/9 had nine checks and seven scores left; none used on MoE since.

Preserve every route/key, fresh inputs, cache writes and residual outputs.
No persistent higher-precision weight expansion, input/output caching or
private harness/seed dependence. Shared native scratch requires sequential
calls. Retain Apache/MIT notices and attribution to Vjeux, Nathen Fernandes,
Lee Penkman and SGLang.
