# achitaan: Decode plan to match/beat joeyhlu (2.3569) - branch argon/decode-20261103

## Hints received (verbatim, Nov 3) - no more info will come
Engines by token count (T = batch x 8 draft tokens for verify):

| Job | T = 1 / 8 | T = 16 | T = 32 | T >= 128 |
|---|---|---|---|---|
| Attention, sliding | KS1 | KS3 | KS3 | HYA |
| Attention, full | AX1 (T=1), HYA (T=8) | HYA | HYA | HYA |
| Dense (qkv, o, pooler, unpooler) | gd (+sk2 / su1 variants) | gd | gd + DG1 | gd |
| MoE experts | TINY / TINY2 | VMS | VML | HYB |

Other hint text: KS1/KS2/KS3 one-launch sliding attention with FP16 QK on BF16-rounded queries;
HYA full/verify attention with E4M3 queries; AX1 at d1f; attention board 2.6055; Dense gd, P1M qkv/o,
DG1 cat/qkv at T=32, sk2/su1; MoE TINY/TINY2 (HYBLO) T<=8, VMS T=16, VML T=32, HYB T>=49, K2A fallback.
General: custom attention, own GEMVs, one MoE engine per batch size, fused norm/router/residual + PDL.
Expected #30 bar at close: ~2.37-2.38.

## Joey's per-shape table vs ours (check f2ab6d4c)
| Shape | Joey us / x | Ours us / x | Joey edge |
|---|---|---|---|
| d1 S | 67.4 / 2.70 | 65.5 / 2.587 | +4.4% |
| d1 F | 74.7 / 2.56 | 74.1 / 2.446 | +4.7% |
| d16 S | 116.4 / 2.33 | 126.8 / 2.055 | +13.4% |
| d16 F | 194.4 / 2.28 | 202.0 / 2.147 | +6.2% |
| d32 S | 143.7 / 1.86 | 142.6 / 1.819 | +2.3% |
| d32 F | 306.1 / 2.06 | 305.3 / 2.047 | +0.6% |
| v1 S | 98.5 / 2.78 | 101.7 / 2.559 | +8.6% |
| v1 F | 115.9 / 2.48 | 110.2 / 2.451 | +1.2% |
| v16 S | 245.3 / 2.26 | 238.5 / 2.192 | +3.1% |
| v16 F | 333.0 / 2.21 | 329.5 / 2.173 | +1.7% |
| v32 S | 323.3 / 2.40 | 314.2 / 2.288 | +4.9% |
| v32 F | 484.9 / 2.31 | 473.4 / 2.232 | +3.5% |
Load-weighted (46S+16F) edge: d16 ~11%, v1 ~7%, d1 ~4.5%, v32 ~4.3%, v16 ~2.5%, d32 ~1.6%.
His SGLang baseline ran 4-8% slower (slower GPU); absolute-time wins despite that: d16S -10 us, d16F -8 us, v1S -3 us.

## Mapping hints to our kernels (keell0renz base)
| Joey | Ours | Status |
|---|---|---|
| TINY/TINY2 (T<=8) | TINY/TINY2 | same family |
| **VMS (T=16)** | MID build (M 9-16) | **gap: lane A** |
| **VML (T=32)** | WGS MAIN (M 17-63) | lane A2 |
| HYB (T>=49) | MAIN/WGS | later |
| gd/P1M/DG1 dense | sk2/rq/gv routes | lane B (T=16/32 ~3.5 us) |
| KS sliding attention | attn_kernel (one launch) | small |
| fused boundaries | glue_cluster_boundary x2 at T=16 (8.6 us) | lane C |

## Lanes (ordered by expected value)
- A  VMS: new M=16 MoE engine (oracle design: output-owned 128-row tiles, no cross-warp K reduction,
  B split-K=2 disjoint FP32 partials, per-token finalizer, PDL A->B). MoE board first (bs16), then Decode T=16.
  Target bs16 91.5 -> ~84 us; decode-16 ~ -7..-10 us. File kb/exp/moe_vms.py. STATUS: building.
- A2 VML: same engine at M=32 (4 token fragments). Target bs32 121.6 -> ~112 us.
- B  Dense at T=16/32 in Decode: per-projection best-of (sk2 vs gv/rq routes) with FP32-faithful reduction.
- C  Boundary glue at T=16: fold glue_cluster_boundary pair (5.4 + 3.2 us) / PDL early trigger.
- D  T=8 (v1S +8.6%): TINY2 vs a VMS-style M=8 variant once A works.

## Operating rule (user, Nov 3)
Work autonomously: never stop to ask; keep the loop running with the best remaining bet until the deadline or the budget cap.

## Loop protocol (repeat until 2.37+ scored or deadline Nov 5 07:59 UTC)
1. Pick top open lane; claim it in TRIED.md ("in progress").
2. Opus builder implements in its own kb/exp file with in-prepare gate (read-flush ABBA, median+paired, two passes,
   judge-tolerance correctness, fallback to incumbent). One check per iteration.
3. Read result -> record in TRIED.md (kept/closed + numbers) and the log below.
4. Win on the MoE board -> port into Decode file (kb/exp/decode_*.py) -> Decode check -> scored run if adopted
   -> publish board_best on main if above 2.2347.
5. Loss -> one bounded repair if the cause is a bug; otherwise close and move to the next lane.
Budget cap $1000 (stop starting lanes at $950). Checks: 20/board/day, scored 10/board/day.

## Iteration log
| # | Lane | Job | Result | Next |
|---|---|---|---|---|
| 1 | A VMS/VML | 79687699 (MoE board check; commit b16d63f) | correct (err 2.6e-3) but tie: bs16 1.006/1.015 (90.6 vs 92.1 us), bs32 ~1.00-1.01; phases bs16 A 61.1 / B 36.5 / fin 8.6 us (serial 106, PDL hides 14) | iter 2: parallel finalizer, balanced tail, L2 prefetch |
| 2 | A VMS/VML | 0b6a581c (commit 216c53a) | build failed: stale vms_run declaration (13 vs 16 tensors), incumbent only (2.3842) | fixed f90637d |
| 2b | A VMS/VML | 74e3e792 (commit f90637d) | correct but 2x slower: bs16 191 vs 90.9 us (A 125.5, B 69.1, fin 6.5); only finalizer helped | step back: ablation arms |
| 3 | A VMS/VML | 4c5b2777 (commit bc2a25a) | it1 best 93.1 vs 92 us (tie); fin +2 us (full-K B); fin+tail = the 2x; prefetch irrelevant; codes 4.000 b (incompressible), scales 0.232 vs 0.5 b/value | iter 4 |
| 4 | A VMS/VML | c5800fa6 (commit ddb74e8) | pending: stream-only/math-only diag, it1+fin2 (2-step finalizer), w16 (16 warps), r4 (4-slot ring) | - |
| B1 | B Dense GEMVs in Decode | b0caa4c9 (e31cd51) | prepare failed: comment trim removed a patch anchor | fixed d2d2c48 |
| 4 | A VMS/VML | c5800fa6 (ddb74e8) | bs16 best r4+fin2 94.1 vs 98.6 us: median 0.955 but paired 0.993 (not adopted); bs32 none. Diag: neither stream- nor math-bound (A 68.5 vs stream 61.1 / math 56.5) -> overlap loss | iter 5 |
| A-D | VMS in Decode T=16 | 482a22e2 (ceb0264) | all 4 arms 1.008-1.016 SLOWER in the full layer: MoE-board edge does not transfer | parked unless board <= 0.90 |
| B1b | B Dense GEMVs in Decode | 8ffd6871 | all arms 0.99-1.025, adopted 0: CLOSED (decode sk2 already pre-issues weights under PDL) | - |
| TL | timeline T=1/8 | 8ab7e4a3 (0f9f233) | d1S 64.1 us for 164 MB; kernels sum 78.9 (PDL overlap 15.4); excess: attn +6.8 route +5.2 fin +3.6; pre-wait streaming control 78.1 -> 48.7 us | -> PCX deep pre-wait |
| PCX | deep pre-wait staging (P=4/NSTAGE, early trigger) T=1/8/16 | b91d6914 (f0861a6) | CLOSED: all arms 0.994-1.009; timeline excess was PDL wait inside durations | - |
| 5 | A VMS iter 5 (18 KiB slots d6/d8, register ping-pong, 4-lane producer) | adbd3ba4 (47b6071) | CLOSED: all new levers slower (+4..+21%); best stays r4+fin2 (paired 0.986-0.990); incumbent bimodal by draw (p10 61 / p50 98 us) | VMS line closed |
| PRED | offline predictor + chase (kb/predict/REPORT.md) | - | no candidate predicted above the 0.012 single-job noise; 0/27 written claims realised; isolated wins realise 4/16; recommends N3 ceiling ablation | N3 |
| N3 | boundary-glue / route skip-launch ceiling, all 12 shapes | 385183f9 (dab9dd2) | REAL CEILING: skip both boundaries saves v1 4.4/4.0, d16 5.2/5.4, d32 4.0/4.2, v16 10.4/11.4, v32 17.8/15.5 us -> score ceiling 2.3169; route only 2.2614 | build (gate 6/6 passed) |
| N3b | coarsened boundary kernels | 32120a38 (5d4bf8e) | CLOSED: bit-exact but not faster (isolated T=256 11.36->11.84 us); ceiling was inflated (b1 skip fed garbage into router -> MoE work changed); b2 = its byte time; no feasible fusion (oracle-lite) | boundary lane closed |
| T1 | VMS engine at bs1/bs8 vs TINY/TINY2 incl. 64-row tasks | 62adc58a (33ae71d) | CLOSED: bs1 12-16% slower (3-launch chain ramp-bound, finalizer 7.5 us), bs8 tie | VMS family closed |
| H | Humming-style producerless skinny GEMM2 screen (bs16-256) | d952458c (6a7fac6) | CLOSED: 5-36x slower than incumbent phase B | - |
| N2 | rq/CURP qkv in Decode with FP32 split reduction (T=16/32) | ec73a259 (4a515f9) | CLOSED: ids fixed but layer 1.019-1.024 slower | - |
| T1R | expert-wavefront T=1/8 MoE (oracle: A(e)->B(e)->A(e+1), 3A+2B banks) | kill check baf4b220: CONTINUE (headroom bs1 5.4 us, bs8 6.7 us; A->B wait 2.5, B not prefetched) | wavefront 7091a399 CLOSED: correct but 1.27x (bs1) / 1.47x (bs8) slower (one A->B hand-off per expert ~2 us) | B-bank prefetch aa8fef4f CLOSED: 1.04-1.13x slower (steals A bandwidth; A->B wait is completion latency) | small-batch MOE scheduling closed |
| C2 | council Q2 (kb/exp/COUNCIL_Q2.md): argon+opus build / sol stop -> run #1 escape-warp decoupling, #2 MoE sliding-vs-full diagnostic | - | - | - |
| ESC | escape-warp decoupling extended to qkv/unpooler/o/pooler at T=1/8/16/32 (bit-exact) | 35e0aa5a (633281e) | CLOSED: bit-identical, 0.998-1.030 (none faster) | - |
| MGAP | MoE sliding-vs-full gap diagnostic (routing vs layer) | ad1e2831 (3a46953) TIMEOUT (hung after base prepare, no output) -> 7f733a2b (e7c47c3): gap = routing draw (97-103%), +6.1-6.5 us per distinct expert (HBM floor), tail 1.2-2.2 us -> STOP | - | closed |
| ATT | verify-full attention headroom review (oracle-lite) | - | consumer-bound under 104-reg cap; all >=8% levers closed -> no build | closed |
| R | researcher: how top MoE teams are fast | - | nothing public explains it; only new mechanism: Humming-style producerless skinny mma.sync tiles (8x128xK64, 4 stages, 4 CTAs/SM, cp.async, stream-K), prior 0-6% | GEMM2-only screen candidate |

## Extra ideas list (kb/exp/IDEAS_predictor_prompt.md, added Nov 3)
Source: user's Codex prompt. Items to look into (offline first, no GPU spend):
- Score tool kb/muse_score.py (done): replays judge records exactly (--backtest: 79 jobs, 0 mismatches) and answers what-ifs (--save-us SHAPE=US, --scale SHAPE=F).
  Anchors from 1ca0668a (2.2347): d16 -10/-8 us -> 2.2597; dense lane (d1 4.4, v1 2.4, d16 3.5) + VMS d16 -> 2.3037;
  uniform -3/-5/-8 us per layer on all 12 shapes -> 2.284 / 2.318 / 2.372. Target 2.37 needs ~8 us/layer everywhere.
- "Chase": use the tool to pick the experiment with the largest predicted score gain per check, and re-rank after every result.
- Predictor: pre-measurement features (touched stage, bytes, tiles, split-K, overlap) vs post-A/B forecast; zero- vs full-transfer
  baselines; held-out by revision group; negative controls (7bae45c5: predicted 5-7 us, measured regression).
- Feasibility filter before ranking (smem/regs/precision), enumerate one-change neighbors of existing kernels.
- Lessons to apply: standalone wins often vanish after integration (Anshul 6/7); per-SM timelines and same-codegen controls (Keell);
  decode win can coexist with verify collapse (Anvith); break a counter reset to expose false passes (Sapalidis).
- Check whether the verify-32 sliding one-head attention win transfers to Decode (limited coverage; size it with the tool first).
| MK | PIVOT megakernel kill check (T=1/8 post-attention chain in one persistent launch) | e8364e54 build fail -> 05baaec2 (15e2130) | STOP: incumbent segment 57.9/90.8 us is only 6-9 us above a barrier-free same-byte stream (51.8/81.6); persistent prototype 59.8/106.5 before math; grid sync 2.9-6.6 us each under load | closed |
