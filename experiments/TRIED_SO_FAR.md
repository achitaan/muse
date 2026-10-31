# What has already been tried (all sessions, as of Oct 31)

## Sources
- MOE_NOTES.md, DECODE_NOTES.md, HANDOFF.md, experiments/prepared/*.md and results/* (main worktree)
- experiments/attn_next/DEAD_ENDS.md and experiments/moe_next/RESULTS.md (this branch)
- the other session's `.slim/worktrees/decode-moe-plan/results/decode-moe-plan-20261030/` (CURRENT_STATE.json, */REVIEW.md, */outcome.json)

Job ids are given where the notes have them. **Kept** means it's in the published decoder (2.0755x, `452fbd9d`).

## MoE kernel (`moe_fused`)

| Idea | Outcome | Number | Job | Where |
|---|---|---|---|---|
| One persistent fused kernel (TINY / TINY2 / MID / MAIN builds) | kept | MoE board 2.3797x | `33114871` and later | MOE_NOTES |
| Small-M FP32 finalizer | kept | 1.1496x aggregate | `74ba5356` | MOE_NOTES |
| Single-token CUDA NVFP4 GEMV | reverted | 148 us vs 54 us | `67f2f03d` | MOE_NOTES |
| Mixed skinny/fat backend for 9-32 tokens | closed | 6.1% slower at 16, 7.1% at 32 | `ca76bcd1` | MOE_NOTES |
| WGS32 four-tile build for 17-32 tokens | closed | 2.165-2.231x vs 2.365x | `33114871` base | MOE_NOTES, daf5c04 |
| MID vs MAIN at 32 tokens | tie | MID 410.5 us vs MAIN 413.4 us | `63c17c9c` | decode-moe-plan/observed-tail-63c17 |
| WGS finalizer row-read reuse | closed | 0 of 12 comparisons above 3% | `5a9d8a25` | decode-moe-plan/wgs-final-reuse |
| Removing the grid barrier at 1-16 tokens | closed | tail only 1.6-2.5 us | `63c17c9c` | moe_next/RESULTS.md |
| Coarser work splits; dedicated 16-token tile | closed | slower | - | MOE_NOTES |
| TR8 routing change | closed | failed the byte gate | `9dc89929` | decode-moe-plan/moe-repeat-v2 |
| Paired cluster activation multicast | closed | no standalone gain | `6627d540` | DECODE_NOTES |
| Paired epilogue store | closed | no gain | `5262c2c5` | MOE_NOTES |
| Norm group / norm register / whole-row tiny norm | closed | slower on GPU | - | decode-moe-plan CURRENT_STATE |
| Stream-K cut / fixup / tail variants | parked | not run | - | decode-moe-plan/wgs-fixup-screen |
| Tiny routing-warp ownership | not run on GPU | no exposed routing bound | - | decode-moe-plan/route-warps |

## Pooler + router

| Idea | Outcome | Number | Job | Where |
|---|---|---|---|---|
| R2f fused pooler + FP32 router at 1/8 tokens | kept | 2.0510x when published | - | DECODE_NOTES |
| `sk2` fused pooler/router at 9-32 tokens | kept | - | - | DECODE_NOTES |
| FP32 router kernel at 1-8 tokens | kept, then folded into R2f | 11.7 us at M=1, 14.3 at M=8 | - | DECODE_NOTES |
| Shared router input | closed | 12.7/11.0/17.0/18.3 vs 11.6/11.6/14.9/17.9 us | - | DECODE_NOTES |
| Wide router for 128/256 tokens | kept | 6.3-8.6 us | - | DECODE_NOTES |
| Router sidecar (separate concurrent router kernel) | closed | 30-34% slower | `18022301` | decode-moe-plan/router-sidecar |
| Router M=1 specialization | closed | no gain | `8efb0121` | decode-moe-plan/router-m1 |
| Fused routing + `_pn`; router load tweaks | closed | - | - | earlier sessions |

## qkv / o / unpooler projections

| Idea | Outcome | Number | Job | Where |
|---|---|---|---|---|
| Custom 12-bit lossless (c12) GEMVs at 1-32 tokens | kept | - | - | DECODE_NOTES |
| Shared-build qkv | not selected | tied or slightly slower | `50eae88b` | DECODE_NOTES |
| qkv warpgroup at M=16/32 | closed | 18.1 vs 13.9 us; 18.5 vs 15.6 us | `ec1ea264` | decode-moe-plan/o32-resource-fixed |
| Sliced unpooler, load-hoist repair | kept | 3.8 us at M=1 (was 5.5) | - | DECODE_NOTES |
| Shared-build sliced unpooler | not selected | 3-5% slower | `50eae88b` | DECODE_NOTES |
| M=32 warpgroup unpooler | closed | 22.3 vs 20.6 us | `ec1ea264` | decode-moe-plan/o32-resource-fixed |
| Fixed o projection for 9-16 tokens | prepared, never measured | - | - | experiments/prepared/decoder_output_projection_review.md |
| cuBLAS at 128/256 tokens | kept; custom kernels lost to it | - | - | dense-top50 track |

## Norms and glue kernels

| Idea | Outcome | Number | Job | Where |
|---|---|---|---|---|
| Fused pre-expert norm + expert-id rotation (`_pn`) | kept | 1.6-2.6 us, replacing 4.4-6.1 | `90e0e5d1` | DECODE_NOTES |
| Sliced FFN boundary `_sfn` at 1-8 tokens | kept | - | - | DECODE_NOTES |
| Cluster boundary norm (spread over 4 CTAs) | closed | 1.65-4.57x slower | `545f0b52` | results/decoder-cluster-norm-dual-order-check-20261029 |
| Cluster norm, first version | setup failure | FP32 order differed: 5.5e-5 vs 2e-5 limit | `63acbd4d` | results/decoder-cluster-norm-reduction-diagnosis-20261029 |
| `_sfn` warp variants | closed | - | - | earlier sessions |
| M=1 norm PDL integration (`decoder_pdls1_draft.py`) | **parked, never run on GPU** | unit gain alone below the 3% layer gate | - | decode-moe-plan CURRENT_STATE |
| Boundary-norm follow-up | parked | single-CTA and gates already kept | - | decode-moe-plan/boundary-norm-source-review |

## Attention

| Idea | Outcome | Number | Job | Where |
|---|---|---|---|---|
| Native FP8 split-KV kernel | kept | Attention board 2.569x | - | board_best |
| A3 split change in Decode | closed | no change on the target shape | `9171012a` | attn_next/DEAD_ENDS.md |
| Fewer sliding splits | closed | every changed shape slower | `543fe60e` | attn_next/DEAD_ENDS.md |
| Launch-plan sweep (45 variants) | closed | all slower | `a7c4e6e6` | attn_next/DEAD_ENDS.md |
| `nc` workspace copies 2/4/8 | closed | +2-10% slower | `0b48e4f5` | attn_next/DEAD_ENDS.md |
| decode1 timeline | diagnostic | 6 steps of 0.4-1.1 us | `eaebaa8f` | attn_next/DEAD_ENDS.md |
| `nc1` builds | closed | no family enabled | `1d606d43` | decode-moe-plan/attention-nc1 |
| Page-1 constants | rejected on CPU | instruction ratio 0.955 misses the 0.95 gate | - | decode-moe-plan/attention-page1-constants |
| One-query FP8 QK | closed | no usable gain | - | HANDOFF |

## Across kernels

| Idea | Outcome | Number | Job | Where |
|---|---|---|---|---|
| PDL handoffs for qkv -> attention -> o, norm -> R2f, MoE -> unpooler at 1/8 tokens | kept | 2.0755x | `452fbd9d` | board_best/decoder_pdli_scored_2.0755.py |
| L2 weight prefetch during attention and the norm | closed | best -0.8 us; +20-25 us with pooler weights | `94c68965`, `bad2bc21` | attn_next/DEAD_ENDS.md |
| PDL for MoE | **never tried** | switch exists; `_native_moe_PDL = False` | - | decoder source |
| PDL for stock norm / `_sfn` / `_pn` / `routing_kernel` | **never tried as a bundle** | - | - | `_PDLI_NORM_THREADS = []` |
| Folding the norm into the producer's last CTA | **never tried** | - | - | RESEARCH_20261031.md |
| Top-k folded into the router at 16-256 tokens | **never tried** | - | - | RESEARCH_20261031.md |
| MoE readiness flags at 32+ tokens | **never tried** | tail 5.8-8.8 us | - | RESEARCH_20261031.md |

## Noise
The same attention file scored 1441.7 and 1490.9 us (`8347b4ca`, `804f8d7c`), about 3.4% apart. Single checks cannot settle changes below about 3-4%.
