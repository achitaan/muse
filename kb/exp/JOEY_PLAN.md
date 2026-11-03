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

## Loop protocol (repeat until 2.37+ scored or deadline Nov 5 07:59 UTC)
1. Pick top open lane; claim it in TRIED.md ("in progress").
2. Opus builder implements in its own kb/exp file with in-prepare gate (read-flush ABBA, median+paired, two passes,
   judge-tolerance correctness, fallback to incumbent). One check per iteration.
3. Read result -> record in TRIED.md (kept/closed + numbers) and the log below.
4. Win on the MoE board -> port into Decode file (kb/exp/decode_*.py) -> Decode check -> scored run if adopted
   -> publish board_best on main if above 2.2347.
5. Loss -> one bounded repair if the cause is a bug; otherwise close and move to the next lane.
Budget cap $800 (stop starting lanes at $750). Checks: 20/board/day, scored 10/board/day.

## Iteration log
| # | Lane | Job | Result | Next |
|---|---|---|---|---|
| 1 | A VMS/VML | 79687699 (MoE board check; commit b16d63f) | correct (err 2.6e-3) but tie: bs16 1.006/1.015 (90.6 vs 92.1 us), bs32 ~1.00-1.01; phases bs16 A 61.1 / B 36.5 / fin 8.6 us (serial 106, PDL hides 14) | iter 2: parallel finalizer, balanced tail, L2 prefetch |
| 2 | A VMS/VML | 0b6a581c (commit 216c53a) | build failed: stale vms_run declaration (13 vs 16 tensors), incumbent only (2.3842) | fixed f90637d |
| 2b | A VMS/VML | 74e3e792 (commit f90637d) | pending | if bs16 wins -> submit decode_vms.py (55f3402, T=16/32 switch, full-layer gate); todo: PDL-attach vms_A (gdc_wait before reading x/ids/tw) to keep route->MoE overlap in Decode |
