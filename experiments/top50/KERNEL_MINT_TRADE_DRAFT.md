# Draft reply to nikebp/kernel-mint (guild t/116) — for the user to post

Team achitaan here (decode 2.1056x, moe 2.3797x, attention 2.5690x, bf16 1.5278x; overall ~#362).
Your MoE is only 0.3% ahead of ours but your decode layer is 6.7% ahead, so the difference is in-layer,
not the standalone MoE. We'd like a timing-only trade, no source either way.

**We offer (all judge-measured, read-flush, job IDs attached):**
1. Per-stage GPU time at all 12 scored decode shapes for our scored run 9dabcd9a (qkv / attention / o /
   norms+glue / pooler+router / MoE / unpooler / cuBLAS), table below.
2. Our falsification ledger: ~150 closed experiments across MoE, attention, dense, decode-layer PDL/glue,
   each with numbers and job IDs (incl. a write-flush vs read-flush measurement trap that inflates
   small-batch MoE by ~20%, and a confirmed hang in a stream-K MoE when a launch has zero local routes).

**We ask:** the same per-stage table for your 2.2464x decode run (sliding/full at decode 1/16/32 and
verify 1/16/32), plus whether your numbers are whole-graph or profiler segments.

| load | layer | judged us | qkv | attn | o | norms/glue | pool+router | moe | unpooler | cuBLAS/other |
|---|---|---|---|---|---|---|---|---|---|---|
| decode 1 | sliding | 74.8 | 9.7 | 4.9 | 5.6 | 10.2 | 18.8 | 28.9 | 4.2 | 0.0 |
| decode 1 | full | 79.8 | 9.8 | 11.0 | 5.6 | 9.9 | 18.7 | 23.3 | 4.3 | 0.0 |
| decode 16 | sliding | 139.7 | 10.8 | 8.0 | 5.6 | 9.7 | 20.5 | 81.5 | 14.7 | 0.0 |
| decode 16 | full | 210.0 | 10.3 | 89.5 | 5.5 | 9.6 | 20.5 | 58.7 | 14.4 | 0.0 |
| decode 32 | sliding | 158.9 | 12.0 | 9.2 | 6.4 | 9.9 | 24.0 | 82.8 | 17.4 | 0.0 |
| decode 32 | full | 318.8 | 11.9 | 170.0 | 6.0 | 10.1 | 22.4 | 75.6 | 17.1 | 0.0 |
| verify 1 | sliding | 115.3 | 0.0 | 7.1 | 0.0 | 10.0 | 20.0 | 45.8 | 4.1 | 14.8 |
| verify 1 | full | 124.4 | 10.0 | 14.3 | 6.6 | 9.9 | 0.0 | 49.4 | 4.3 | 18.8 |
| verify 16 | sliding | 251.3 | 0.0 | 11.7 | 0.0 | 11.8 | 3.0 | 147.4 | 0.0 | 72.6 |
| verify 16 | full | 327.8 | 0.0 | 102.9 | 0.0 | 11.8 | 3.0 | 159.5 | 0.0 | 71.2 |
| verify 32 | sliding | 320.8 | 0.0 | 15.2 | 0.0 | 18.4 | 3.1 | 187.2 | 0.0 | 92.4 |
| verify 32 | full | 489.9 | 0.0 | 183.2 | 0.0 | 18.2 | 3.1 | 197.8 | 0.0 | 90.6 |

(Profiler GPU time per kernel class; PDL kernels include their overlapped wait, so columns can exceed the
judged whole-layer time. "0.0" in qkv/o/unpooler at verify means those ran in cuBLAS, counted under other.)
