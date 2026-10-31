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
| Lesson from PNFOLD: with PDL, MoE prologue/weight prefetch already overlaps the small _pn kernel, so removing _pn saves nothing and in-kernel norm adds its latency to the critical path (+4.5-6.5 us) | diagnostic | explains PNF regression | - | PROGRESS.md (decode-next) | Oct 31 |
| Publish skinny-job phase-A completion at expert boundaries (EBPUB) for TINY/TINY2/MID | closed | correct (probes pass), no gain: isolated MoE M1 30.6->30.1 (1.7%, passes) but layer M1 sliding/full within +-1%; M8 79.4 vs 79.7; M16 109.4 vs 109.2; nothing selected | `c1d5a6f1` | experiments/top50/decoder_p12_ebpub.py (decode-next) | Oct 31 |
| Latent bug: WGS (>16 tok) with zero local jobs (all routes non-local) reads uninitialized S.jC[0] in range calc; may stall producers | open (not fixed in published kernel) | found by hard-fixer during WGF timeout diagnosis; judge tests have not hit it | - | experiments/top50/decoder_p13_wgsfin.py report (decode-next) | Oct 31 |
| WGS finalizer segmented (variant A, identical reduction order) with bounded waits | closed | bit-identical at M32/128/256 (probes bits=True) but isolated MoE not faster: M32 136.96 vs 137.66, M128 242.8 vs 244.7, M256 245.2 vs 244.8 us; the finalizer isn't the tail bottleneck | `f35cb9c2` | experiments/top50/decoder_p15_prebuild.py (decode-next) | Oct 31 |
| TINY/TINY2 skinny ring NS 3->4 (one more weight-load lookahead slot, same smem) | closed | isolated MoE slower: M1 ratio 1.056/1.066, M8 1.030/1.033 (6-round medians); outputs not bit-equal but within tol | `482d0e7c` | experiments/top50/decoder_p18_moeb.py (decode-next) | Oct 31 |
| WGS: immediate doneAs release after phase-A epilogue + no WGSREORD for small stage counts (oracle-lite #2 change 2) | closed | V1 slower 0.6-1.5%, V2 neutral (0.997-1.001), V3 slower; at M32/128/256; nothing selected; scored 2.0698x (noise) | `15e19cf8` | experiments/top50/decoder_p22_wgsrel.py (decode-next) | Oct 31 |
| In-layer per-shape MoE build selection (TINY/TINY2/MID/MAIN) by robust whole-layer A/B with real-router routes (M=8: TINY2 vs MID; M=16: MID vs TINY2 vs MAIN; M=32: MAIN vs MID) | in progress | decode-next pass 26, decoder_p24_moesel.py; earlier MID vs MAIN at 32 = tie on MoE board (63c17c9c) | - | - | Oct 31 |
| MoE board: background prebuild of MoE exts + robust per-M build selection (M8 tiny2/mid/tiny, M16 mid/main, M32 main/mid) | closed | check 2.3442x (noise vs 2.3797 best); prepare 59 s; no override adopted: M8 mid 1.003/1.000, tiny 0.992/0.998; M16 main 1.084/1.088; M32 mid 1.037/1.038 -> current builds are best | `88569ac0` | experiments/boards/moe/moe_b1.py (decode-next) | Oct 31 |
| Researcher (MoE board next steps) | diagnostic | tail marker DBG(3) is thread 0 only (may misattribute tail); ranked: (1) per-warpgroup end-time diagnostic; (2) WGS M<=32 build with X slots capped at 32 rows -> 4th weight slot (NS=4), est. 3-10% bs32; (3) MID helper-warp epilogue overlap 2-6% bs16 (MID already uses OKC_WS helper: check); no public W4A16 MoE >=4.4 TB/s at 16-32 tokens; FP4->FP8 weight path impossible with BF16 activations | - | researcher ses_e6d44e8edffeRJaWBvjNq01iZF | Oct 31 |
| WGS build for M<=32 with X slots capped at 32 rows freeing smem for a 4th weight-ring slot (NS=4), selected by robust A/B; distinct from closed WGS 4-tile and TINY NS4 | in progress | experiments/boards/moe/moe_b2.py (also applies to decode d32) | - | - | Oct 31 |

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
| Enable sk2 (fused pooler/router at 9-32 tok) PDL: 2 weight stages before griddepcontrol.wait (existing sk2_set_pdl, default 0), mask 512 with glue PDL at M16/M32 | in progress | decode-next pass 27, decoder_p25_sk2pdl.py; queued after quota reset | - | - | Oct 31 |

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
| Dense board: admit dense-top50 near-misses (o M8 v18s (4,8,2,1) NT1024; router M16 routertail S16/WK4) with robust 6-round gate instead of ABBA | closed | robust gate says both are SLOWER: o m8 cand/base 1.016/1.013, router m16 1.018/1.011 (earlier ABBA 'wins' were noise); WG M16/M32 variants also slower (1.19/1.14/1.017); check 1.5226x (noise); prepare 87.6 s | `3a5df127` | experiments/boards/dense/dense_b1.py (decode-next) | Oct 31 |
| Researcher (dense board next steps) | diagnostic | M128/256 qkv/o/unpooler are the weak shapes (20% cut there = +4.4% board). Ranked: (1) noncluster CUTLASS persistent/Stream-K vs cuBLAS at M128 unpooler, 0-1.5%; (2) new c12 producer-decode->smem->SS-WGMMA path, 0-3%, high effort (needs >=4.4 TB/s reconstructed BF16); (3) TMA bulk-load diagnostic for small-M streams, 0-1.5%. Register-A c12 pipeline tweaks and cuBLASLt searches are closed (DENSE_LEDGER) | - | researcher ses_e6d4521c7ffesRmwAuqXi2DgAv | Oct 31 |
| Dense board: CUTLASS 3.x sm90 GEMMs (pingpong / cooperative / stream-K persistent, headers from flashinfer/data/cutlass) vs current routes at M=128/256 for qkv/o/pooler/unpooler, robust 6-round gate | closed | CUTLASS 4.5.0 builds OK (78-87 s, background); all 10 variants slower than current routes at every (layer, m): qkv m128 29.1 vs 19.3 us, o 13.7 vs 11.7, pooler 34.5 vs 30.8, unpooler 30.1 vs 29.0; m256 best pooler 1.06x, unpooler 1.016x slower; check 1.5002x (noise) | `df119bda` | experiments/boards/dense/dense_b2.py (decode-next) | Oct 31 |

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
| R2f consumer-side attention-boundary norm at M1 (R2n: every R2f CTA recomputes post-norm/residual/pre-norm; removes the stock norm kernel) | kept | bit-identical (rel 0, same ids); whole-layer M1 full 0.974/0.979 selected (mask 159); M1 sliding 0.989/0.997 not selected; scored 2.0718x (noise) | `9b254bbb` | experiments/top50/decoder_p19_r2norm.py (decode-next) | Oct 31 |

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
| Verify (q_len 8) attention: FP8 V transpose moved from consumers to producer warps (ldmatrix/prmt/stmatrix after cp.async) | closed | correct (cache exact) but slower: b1 q8 sliding 1.003/1.009, b1 full ~1.0, b16 q8 full 1.279/1.275 (layer 1.085) | `0898ae00` | experiments/top50/decoder_p20_attnv.py (decode-next) | Oct 31 |
| Attention board: in-prepare robust per-shape launch-plan autotune (first eager kernel_fn call; no prepare hook on this board) | closed (inconclusive) | all 12 shapes kept the planner plan (ratio 1.000); 2-4 of 4-6 candidates rejected by correctness (q_len 8 FP8 split rounding) and the stage took ~0.1 s per shape, so the robust timing likely never ran; tuning inside kernel_fn is also questionable under the rules; check 2.5324x (noise) | `07f498fd` | experiments/boards/attn/attn_b1.py (decode-next) | Oct 31 |
| Researcher (attention board next steps) | diagnostic | 1 us off every sliding shape = +4.6% board; ranked: (1) partial cubic exp2 emulation (FA4) for 12.5-25% of q8-full score exps, est +1.3-3.3%; (2) small-shape direct-finalize specialization (one CTA per query-head subgroup, no inter-CTA merge), +2.2-4.6% if 0.5-1 us saved, high effort; (3) deeper gather pipeline only after copy-only ceiling diagnostic. Packed ex2 f16x2 lowers to 2 MUFU ops on sm90 (no 2x) | - | researcher ses_e6d455c85ffeUXfAh3oIhyfZz2 | Oct 31 |
| q_len-8 attention: emulate 12.5-25% of score exp2 with FA4-style Cody-Waite + cubic polynomial on FMA units to offload MUFU (static per build, measured by checks) | in progress | experiments/boards/attn/attn_b2.py | - | - | Oct 31 |

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
| Selection stability: tolerant attention probes at all M + whole-layer tol 5e-3 | kept | 2.1043x (CI 2.073-2.136), best check so far; selected M1 30/31, M8 18/19, PDLS M1 sliding 62; M16/M32 not selected this run | `f9845bc9` | experiments/top50/decoder_p10_tol5.py (decode-next) | Oct 31 |
| MoE (TINY/TINY2) issues L2 bulk prefetch of the sliced-unpooler c12 weights (6.3 MB) during its barrier wait/finalize so the PDL unpooler streams from L2 (M1/M8); differs from closed prefetch during attention/norm | in progress | decode-next pass 25, decoder_p23_upf.py | - | - | Oct 31 |

## Open ideas

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| PDL for stock norm / `_sfn` / `_pn` / `routing_kernel` | open (not tried) | - | - | `_PDLI_NORM_THREADS = []` | Oct 31 |
| Top-k folded into the router at 16-256 tokens | open (not tried) | - | - | RESEARCH_20261031.md | Oct 31 |
| MoE readiness flags at 32+ tokens | open (not tried) | tail 5.8-8.8 us | - | RESEARCH_20261031.md | Oct 31 |
| MoE PDL as a per-call okc_run argument instead of the process-global okc_set_pdl scope (reviewer major; thread-safety) | open (not tried) | needs C++ ABI change in packed MoE source; current scope safe while MoE runs on one thread | - | experiments/top50/decoder_p6_final.py ~478, ~7111 (decode-next) | Oct 31 |

## Process

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| Single-check keep/revert on the Attention board | closed as a method | same file 1441.7 vs 1490.9 us (3.4%); use paired setup-time A/B | `8347b4ca`, `804f8d7c` | experiments/attn_next/DEAD_ENDS.md (decode-next) | Oct 30 |
| Global TRIED ledger + tried.py helper | kept | search before trying; add after every result | - | TRIED.md, .tools/tried.py | Oct 31 |
| Gate for small changes: paired in-prepare A/B, >=1% per change with both arm orders agreeing | kept | user decision Oct 31; replaces 3% per-component gate | - | experiments/PLAN_TOP50_20261031.md (decode-next) | Oct 31 |
| Oracle-lite (Gemini 4 Argon medium) strategy opinion | diagnostic | rejected all: W4A8 violates BF16 activations; unpooler 'red flag' wrong (50 MB c12); c12-vs-BF16 loss refuted by dense-top50 P3 a61fe75d (c12 3.9 TB/s); L2 weight persistence across replays = judge gaming | - | ses_e6ee10543ffeXBsTTjH424VIRP | Oct 31 |
| Layer timeline via graph-captured CUDA events | closed (diagnostic failed) | event stage times inside graphs meaningless; M32 timeline hit illegal address (run failed); graph totals: M1 sliding PDL 66.0 vs plain 69.3 us, M1 full 75.8 vs 82.3, M8 91.9/93.3, 105.9/108.0, M16 149.9/152.0 | `2cf7e6bf` | experiments/top50/decoder_p11_diag.py (decode-next) | Oct 31 |
| Prepare-time breakdown (load_inline wrapper) | diagnostic | 4 MoE ext builds serial on MainThread 90.6->256.5 s (51/39/38/38 s), then fpr 13.5 s + fp32 router 13.4 s serial; 46 builds total, background ones done by ~65 s; PDLI+PDLS selection only 1.4 s; prepare 274.7 s | `182d084e` | experiments/top50/decoder_p14_prepdiag.py (decode-next) | Oct 31 |
| Prebuild the 4 MoE exts + fpr + fp32 router in background threads at import (were serial on MainThread in prepare) | kept | prepare 274.7 s -> 115.5 s; scored run 2.0926x (valid); frees ~160 s for robust selection | `f35cb9c2` | experiments/top50/decoder_p15_prebuild.py (decode-next) | Oct 31 |
| Robust in-prepare A/B: 6 rounds alternating order, median per order < 0.99 | kept | ratios now tight (+-0.3%): M1 0.92/0.91, M8 0.98/0.97, M16 sliding 0.988 selected; M16 full 0.990, M32 0.989-0.995 consistently ~1% but below gate; PDLS M8 not selected (earlier selections were noise); scored 2.0814x (noise), prepare 107.6 s | `d23cc6a2` | experiments/top50/decoder_p16_robust.py (decode-next) | Oct 31 |
| Oracle (GPT-6 Astra High) final strategy consult | diagnostic | no credible +8% in 3.5 days; realistic +1-3% (2.12-2.16x). Megakernel not recommended (est. +1.2-3.2% if it works, high regression risk). Ranked: (1) recover sub-1% gains with confirmation 0.3-0.8%; (2) MoE small-M phase-B load lookahead 0.4-1.2%; (3) R2f consumer-side norm at M1 0-0.7%; (4) verify attention V transpose to producers 0.3-1% | - | ses_e6e84b2aaffekTI96HEIowlfDS | Oct 31 |
| Sub-1% whole-layer admission (<0.993 both orders) with independent 6-round confirmation | kept | scored 2.0717x (noise); adds M16 full (confirm 0.9921/0.9922); M16 sliding confirmed 0.989/0.991; M32 not admitted; prepare 99.8 s | `848a03ac` | experiments/top50/decoder_p17_gate.py (decode-next) | Oct 31 |
