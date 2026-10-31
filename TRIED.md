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
| Large-M (32-256) tail 5.8-8.8 us as entitlement for readiness flags | diagnostic: unverified | phase trace resources differ from incumbent; needs resource-matched trace + actual touched-expert counts (judge floor uses 0.75 x router_distinct) | - | results/moe-phase-diagnostic-20261029/diagnostic-verdict.json | Oct 31 |
| Researcher R1 (small-batch MoE gap vs board leaders 2.76x) | diagnostic | A/B already overlap per job; B refills vacated A slots; TINY prefetches A before routing. Top idea: publish skinny-job completion at expert boundaries (0.5-2 us at bs8/16); then diagnose short-phase latency vs dequant; whole-B smem caching low value (A and B share HBM) | - | researcher ses_e6f1573b2ffebmhXzecQ9Pkmcx | Oct 31 |
| PNFOLD v3: phase-A x via nc (LDG.CONSTANT) again | closed | isolated PNF MoE still 4.5-5 us slower than _pn+parent (M1 30.0 vs 25.4; M8 80.7 vs 75.2); layer +5..+16%; cause unexplained (not flags, not loads); PNF line closed | `97c2ab82` | experiments/top50/decoder_p7_pnfold.py (decode-next) | Oct 31 |

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
| Own top-8 PDL routing kernel (_RF finalize math) replacing SGLang routing at 9-256 tokens, judge-semantics admission | closed | admitted (weights rel<=4.8e-7, ids equal except 1 exact tie) but producer+routing unit chain slower by 2.4-3.1 us at T=16/32/128/256 in both orders; SGLang's routing kernel is already fast; none of 12 candidate formulas reproduce SGLang weights bit-exactly | `e057dc9d` | experiments/top50/decoder_p8_topk.py (decode-next) | Oct 31 |

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
| Early stock-norm PDL trigger (pdls1 port) at M1, bit 32, on top of p2 | closed | 2.0503x (noise); unit norm->R2f -1.9/-2.3 us (bytes ok) but whole layer only -0.5/-0.1 us (0.993/0.999, fails 1%); M1_full layer rel_l2 3.1e-3 (594 elems) rejected | `f82e5ab8` | experiments/top50/decoder_p3_norm.py (decode-next) | Oct 31 |
| Early stock-norm PDL trigger extended to M8 (bit 32 at M1/M8, on p3) | kept | 2.0891x (CI 2.058-2.120); selected M1 sliding mask 63 (layer -1.4/-1.4 us) and M8 full mask 52 (-2.1/-2.1 us); unit M8 norm->R2f -1.2/-1.4 us | `d543da40` | experiments/top50/decoder_p4_norm8.py (decode-next) | Oct 31 |

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
| Researcher R2: full-context attention gap (contiguous-run TMA, V transpose on producers, lookahead, 128-key tiles) | diagnostic | top idea (contiguous 64-row runs -> swizzled tensor TMA) is dead: reference.py:261 scatters pages with randperm; verify-only idea (V transpose to producers + QK/softmax overlap) est. V 10-25% = ~0.4-1% Decode; d1 attention-board loads are dominated by sliding latency | - | researcher ses_e6ef050e4ffexFxquZ68Ytipts | Oct 31 |

## Across kernels

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| PDL handoffs for qkv -> attention -> o, norm -> R2f, MoE -> unpooler at 1/8 tokens | kept | 2.0755x | `452fbd9d` | board_best/decoder_pdli_scored_2.0755.py | <= Oct 31 |
| L2 weight prefetch during attention and the norm | closed | best -0.8 us; +20-25 us with pooler weights | `94c68965`, `bad2bc21` | attn_next/DEAD_ENDS.md | <= Oct 31 |
| PDLI selector: 1% gate + tolerant whole-layer probe (rel_l2<=1e-3 AND <=16 mismatching elems) | closed | 2.0434x (CI 2.013-2.074); tolerant probe still rejected: MoE run-to-run diffs reach 18-731 elems, rel_l2 up to 1.3e-3; even M1_sliding lost its selection -> selected={} | `7abd8d0d` | experiments/top50/decoder_p1_gate.py (decode-next) | Oct 31 |
| PDLI selector: 1% gate + tolerant whole-layer/moe_unpool probe rel_l2<=2e-3 (no elem cap) | kept | 2.0975x (CI 2.066-2.129); selects M1 sliding+full mask 15 (whole-layer -4.0/-4.7 us both orders); M8 r2f-only fails 1% | `0ca71861` | experiments/top50/decoder_p1b_tol.py (decode-next) | Oct 31 |
| PDL on _pn/_sfn Triton twins (gdc_wait/launch_dependents, launch_pdl) + MoE launch PDL scope, mask bit 16 | kept | 2.0813x (CI 2.050-2.113, noise vs p1b); in-prepare whole layer M1 sliding -4.9/-5.1 us, M1 full -6.4/-6.9 us, M8 sliding -1.5/-2.8 us (selected); M8 full -1.2 us (0.990, just misses gate) | `ac41a480` | experiments/top50/decoder_p2_pdl.py (decode-next) | Oct 31 |
| Glue PDL (bit 16: _pn twin + MoE launch) at 16/32-token decode, batch-m probe data, on p2 | kept | 2.0900x (CI 2.059-2.121); M16 sliding selected (-2.1/-1.8 us, 0.987/0.990); M16 full, M32 both -1.2..-2.3 us but miss 1% gate | `831de9a6` | experiments/top50/decoder_p5b_glue16.py (decode-next) | Oct 31 |
| Integrated PDL bundle p6 (gate 1% + tol probe + glue PDL 1/8/16/32 + early norm 1/8 + review fixes) | kept | checks: p5 2.0960 (6553b299), p6 2.0682 (689fd93e, prepare 273.7 s); all 13/13 tests; check-level gain within ~3% noise; scored submit 0aa30c92 | `6553b299, 689fd93e, 0aa30c92` | experiments/top50/decoder_p6_final.py (decode-next) | Oct 31 |
| M8 qkv->attention / attention->o PDL with tolerant probe (attention output nondeterministic at M8; cache bytes strict) | kept | 2.0942x (CI 2.063-2.126); M8 sliding mask 18 (layer -2.3/-3.5 us), M8 full mask 19 (qa+o+glue, layer -4.7/-5.0 us vs ~-2.5 before) | `c5f34d4f` | experiments/top50/decoder_p9_tolattn.py (decode-next) | Oct 31 |

## Open ideas

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| PDL for stock norm / `_sfn` / `_pn` / `routing_kernel` | open (not tried) | - | - | `_PDLI_NORM_THREADS = []` | Oct 31 |
| Top-k folded into the router at 16-256 tokens | open (not tried) | - | - | RESEARCH_20261031.md | Oct 31 |
| MoE readiness flags at 32+ tokens | open (not tried) | tail 5.8-8.8 us | - | RESEARCH_20261031.md | Oct 31 |
| Consumer-side redundant exact norm: attention-boundary norm inside the R2f prologue, pooled-input norm inside tiny MoE staging (no global barrier; Hazy-style) | open (not tried) | est. 2-4 us at T=1/8, 0.7-1.4% Decode; different from closed producer-last-CTA and cluster norms | - | researcher ses_e6f618865ffeGiiwqHHpNWnD1i | Oct 31 |
| MoE PDL as a per-call okc_run argument instead of the process-global okc_set_pdl scope (reviewer major; thread-safety) | open (not tried) | needs C++ ABI change in packed MoE source; current scope safe while MoE runs on one thread | - | experiments/top50/decoder_p6_final.py ~478, ~7111 (decode-next) | Oct 31 |

## Process

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| Single-check keep/revert on the Attention board | closed as a method | same file 1441.7 vs 1490.9 us (3.4%); use paired setup-time A/B | `8347b4ca`, `804f8d7c` | experiments/attn_next/DEAD_ENDS.md (decode-next) | Oct 30 |
| Global TRIED ledger + tried.py helper | kept | search before trying; add after every result | - | TRIED.md, .tools/tried.py | Oct 31 |
| Gate for small changes: paired in-prepare A/B, >=1% per change with both arm orders agreeing | kept | user decision Oct 31; replaces 3% per-component gate | - | experiments/PLAN_TOP50_20261031.md (decode-next) | Oct 31 |
| Oracle-lite (Gemini 4 Argon medium) strategy opinion | diagnostic | rejected all: W4A8 violates BF16 activations; unpooler 'red flag' wrong (50 MB c12); c12-vs-BF16 loss refuted by dense-top50 P3 a61fe75d (c12 3.9 TB/s); L2 weight persistence across replays = judge gaming | - | ses_e6ee10543ffeXBsTTjH424VIRP | Oct 31 |
