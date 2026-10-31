# TRIED: every optimization idea tried on Muse 2 (shared ledger, all sessions)

**Before trying anything:** `python ".tools/tried.py" search <words>` (from the repo root `C:\Users\Achita\Documents\ChatGPT\Muse 2`).
If a row says closed, don't retry it unless the new idea is materially different; say how in the new row.

**Claim before spending checks:** `tried.py add --area <Area> --idea "..." --outcome "in progress" --number "session/branch"`.

**After every result** (kept, closed, parked, diagnostic): `tried.py add --area <Area> --idea "..." --outcome closed --number "key number" --job <id> --source <file> --replace "<text of the in-progress or open row>"`.

Outcomes: kept | closed | parked | diagnostic | in progress | open (not tried). Areas: MoE kernel (`moe_fused`), Pooler + router, qkv / o / unpooler projections, Norms and glue kernels, Attention, Across kernels, Open ideas, Process.
The helper commits only this file to `main` (fast-forward, with retries) and never touches a checkout. Read-only fallback: `git fetch origin main; git show origin/main:TRIED.md`.

## MoE kernel (`moe_fused`)

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| One persistent fused kernel (TINY / TINY2 / MID / MAIN builds) | kept | MoE board 2.3797x | `33114871` and later | MOE_NOTES | <= Oct 31 |
| Small-M FP32 finalizer | kept | 1.1496x aggregate | `74ba5356` | MOE_NOTES | <= Oct 31 |
| Single-token CUDA NVFP4 GEMV | reverted | 148 us vs 54 us | `67f2f03d` | MOE_NOTES | <= Oct 31 |
| Mixed skinny/fat backend for 9-32 tokens | closed | 6.1% slower at 16, 7.1% at 32 | `ca76bcd1` | MOE_NOTES | <= Oct 31 |
| WGS32 four-tile build for 17-32 tokens | closed | 2.165-2.231x vs 2.365x | `33114871` base | MOE_NOTES, daf5c04 | <= Oct 31 |
| MID vs MAIN at 32 tokens | tie | MID 410.5 us vs MAIN 413.4 us | `63c17c9c` | decode-moe-plan/observed-tail-63c17 | <= Oct 31 |
| WGS finalizer row-read reuse | closed | 0 of 12 comparisons above 3% | `5a9d8a25` | decode-moe-plan/wgs-final-reuse | <= Oct 31 |
| Removing the grid barrier at 1-16 tokens | closed | tail only 1.6-2.5 us | `63c17c9c` | moe_next/RESULTS.md | <= Oct 31 |
| Coarser work splits; dedicated 16-token tile | closed | slower | - | MOE_NOTES | <= Oct 31 |
| TR8 routing change | closed | failed the byte gate | `9dc89929` | decode-moe-plan/moe-repeat-v2 | <= Oct 31 |
| Paired cluster activation multicast | closed | no standalone gain | `6627d540` | DECODE_NOTES | <= Oct 31 |
| Paired epilogue store | closed | no gain | `5262c2c5` | MOE_NOTES | <= Oct 31 |
| Norm group / norm register / whole-row tiny norm | closed | slower on GPU | - | decode-moe-plan CURRENT_STATE | <= Oct 31 |
| Stream-K cut / fixup / tail variants | parked | not run | - | decode-moe-plan/wgs-fixup-screen | <= Oct 31 |
| Tiny routing-warp ownership | not run on GPU | no exposed routing bound | - | decode-moe-plan/route-warps | <= Oct 31 |

## Pooler + router

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| R2f fused pooler + FP32 router at 1/8 tokens | kept | 2.0510x when published | - | DECODE_NOTES | <= Oct 31 |
| `sk2` fused pooler/router at 9-32 tokens | kept | - | - | DECODE_NOTES | <= Oct 31 |
| FP32 router kernel at 1-8 tokens | kept, then folded into R2f | 11.7 us at M=1, 14.3 at M=8 | - | DECODE_NOTES | <= Oct 31 |
| Shared router input | closed | 12.7/11.0/17.0/18.3 vs 11.6/11.6/14.9/17.9 us | - | DECODE_NOTES | <= Oct 31 |
| Wide router for 128/256 tokens | kept | 6.3-8.6 us | - | DECODE_NOTES | <= Oct 31 |
| Router sidecar (separate concurrent router kernel) | closed | 30-34% slower | `18022301` | decode-moe-plan/router-sidecar | <= Oct 31 |
| Router M=1 specialization | closed | no gain | `8efb0121` | decode-moe-plan/router-m1 | <= Oct 31 |
| Fused routing + `_pn`; router load tweaks | closed | - | - | earlier sessions | <= Oct 31 |

## qkv / o / unpooler projections

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| Custom 12-bit lossless (c12) GEMVs at 1-32 tokens | kept | - | - | DECODE_NOTES | <= Oct 31 |
| Shared-build qkv | not selected | tied or slightly slower | `50eae88b` | DECODE_NOTES | <= Oct 31 |
| qkv warpgroup at M=16/32 | closed | 18.1 vs 13.9 us; 18.5 vs 15.6 us | `ec1ea264` | decode-moe-plan/o32-resource-fixed | <= Oct 31 |
| Sliced unpooler, load-hoist repair | kept | 3.8 us at M=1 (was 5.5) | - | DECODE_NOTES | <= Oct 31 |
| Shared-build sliced unpooler | not selected | 3-5% slower | `50eae88b` | DECODE_NOTES | <= Oct 31 |
| M=32 warpgroup unpooler | closed | 22.3 vs 20.6 us | `ec1ea264` | decode-moe-plan/o32-resource-fixed | <= Oct 31 |
| Fixed o projection for 9-16 tokens | prepared, never measured | - | - | experiments/prepared/decoder_output_projection_review.md | <= Oct 31 |
| cuBLAS at 128/256 tokens | kept; custom kernels lost to it | - | - | dense-top50 track | <= Oct 31 |

## Norms and glue kernels

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| Fused pre-expert norm + expert-id rotation (`_pn`) | kept | 1.6-2.6 us, replacing 4.4-6.1 | `90e0e5d1` | DECODE_NOTES | <= Oct 31 |
| Sliced FFN boundary `_sfn` at 1-8 tokens | kept | - | - | DECODE_NOTES | <= Oct 31 |
| Cluster boundary norm (spread over 4 CTAs) | closed | 1.65-4.57x slower | `545f0b52` | results/decoder-cluster-norm-dual-order-check-20261029 | <= Oct 31 |
| Cluster norm, first version | setup failure | FP32 order differed: 5.5e-5 vs 2e-5 limit | `63acbd4d` | results/decoder-cluster-norm-reduction-diagnosis-20261029 | <= Oct 31 |
| `_sfn` warp variants | closed | - | - | earlier sessions | <= Oct 31 |
| M=1 norm PDL integration (`decoder_pdls1_draft.py`) | **parked, never run on GPU** | unit gain alone below the 3% layer gate | - | decode-moe-plan CURRENT_STATE | <= Oct 31 |
| Boundary-norm follow-up | parked | single-CTA and gates already kept | - | decode-moe-plan/boundary-norm-source-review | <= Oct 31 |
| Fold boundary norm into producer GEMV last CTA (o / unpooler) | closed (inspection, no GPU run) | last-block norm serializes the row on one SM; full-grid spin needs unproven residency; ceiling marginal after fences | - | .slim/worktrees/decode-fresh/experiments/prepared/decoder_projection_norm_fusion_review.md | Oct 31 |

## Attention

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| Native FP8 split-KV kernel | kept | Attention board 2.569x | - | board_best | <= Oct 31 |
| A3 split change in Decode | closed | no change on the target shape | `9171012a` | attn_next/DEAD_ENDS.md | <= Oct 31 |
| Fewer sliding splits | closed | every changed shape slower | `543fe60e` | attn_next/DEAD_ENDS.md | <= Oct 31 |
| Launch-plan sweep (45 variants) | closed | all slower | `a7c4e6e6` | attn_next/DEAD_ENDS.md | <= Oct 31 |
| `nc` workspace copies 2/4/8 | closed | +2-10% slower | `0b48e4f5` | attn_next/DEAD_ENDS.md | <= Oct 31 |
| decode1 timeline | diagnostic | 6 steps of 0.4-1.1 us | `eaebaa8f` | attn_next/DEAD_ENDS.md | <= Oct 31 |
| `nc1` builds | closed | no family enabled | `1d606d43` | decode-moe-plan/attention-nc1 | <= Oct 31 |
| Page-1 constants | rejected on CPU | instruction ratio 0.955 misses the 0.95 gate | - | decode-moe-plan/attention-page1-constants | <= Oct 31 |
| One-query FP8 QK | closed | no usable gain | - | HANDOFF | <= Oct 31 |
| Senior advice chain HYA (HYA_ROUTE) -> AX1 -> KS1/KS2/KS3 | diagnostic | names absent from all refs; QK-norm/RoPE/FP8 cache write, 8-way GQA packing, 64-key tiles, length-balanced splits already in attn_kernel_fp8pv | - | researcher ses_e6f618865ffeGiiwqHHpNWnD1i; git grep all refs | Oct 31 |

## Across kernels

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| PDL handoffs for qkv -> attention -> o, norm -> R2f, MoE -> unpooler at 1/8 tokens | kept | 2.0755x | `452fbd9d` | board_best/decoder_pdli_scored_2.0755.py | <= Oct 31 |
| L2 weight prefetch during attention and the norm | closed | best -0.8 us; +20-25 us with pooler weights | `94c68965`, `bad2bc21` | attn_next/DEAD_ENDS.md | <= Oct 31 |

## Open ideas

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| PDL for MoE | open (not tried) | switch exists; `_native_moe_PDL = False` | - | decoder source | Oct 31 |
| PDL for stock norm / `_sfn` / `_pn` / `routing_kernel` | open (not tried) | - | - | `_PDLI_NORM_THREADS = []` | Oct 31 |
| Top-k folded into the router at 16-256 tokens | open (not tried) | - | - | RESEARCH_20261031.md | Oct 31 |
| MoE readiness flags at 32+ tokens | open (not tried) | tail 5.8-8.8 us | - | RESEARCH_20261031.md | Oct 31 |
| Consumer-side redundant exact norm: attention-boundary norm inside the R2f prologue, pooled-input norm inside tiny MoE staging (no global barrier; Hazy-style) | open (not tried) | est. 2-4 us at T=1/8, 0.7-1.4% Decode; different from closed producer-last-CTA and cluster norms | - | researcher ses_e6f618865ffeGiiwqHHpNWnD1i | Oct 31 |

## Process

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| Single-check keep/revert on the Attention board | closed as a method | same file 1441.7 vs 1490.9 us (3.4%); use paired setup-time A/B | `8347b4ca`, `804f8d7c` | experiments/attn_next/DEAD_ENDS.md (decode-next) | Oct 30 |
| Global TRIED ledger + tried.py helper | kept | search before trying; add after every result | - | TRIED.md, .tools/tried.py | Oct 31 |
| Gate for small changes: paired in-prepare A/B, >=1% per change with both arm orders agreeing | kept | user decision Oct 31; replaces 3% per-component gate | - | experiments/PLAN_TOP50_20261031.md (decode-next) | Oct 31 |
