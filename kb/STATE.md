# STATE: Decode push (branch argon/decode-20261103) - read this first

Last updated: Nov 3 2026. Kernel submissions close **Nov 5 07:59 UTC**. Budget cap $1000 (stop new lanes at $950).

## 1. The problem
- Board `muse_decode_layer` (sets our overall rank). One decoder layer of Muse Spark Open on one H200 under TP8/EP8:
  qkv_gate proj -> gated attention (sliding window or full context, FP8 KV) -> o proj + residual/norm boundary ->
  pooler + FP32 router (top-8 of 256) -> routed NVFP4 MoE (32 local experts) -> unpooler + residual/norm boundary.
- Rules: kb/ref/muse_decode_layer/TASK.md, reference.py. Weights keep their precision (lossless repack/compression only),
  BF16 activations, queries may be FP8 E4M3, routing may differ only on near-ties (we do NOT exploit that), CUDA-graph
  replay, cold L2 every replay, no caching, readable source, file < 1,000,000 bytes, prepare() < 300 s.
- Score = geomean over 6 loads of sum(w*SGLang)/sum(w*ours); each load = 46 sliding + 16 full layers.
  Loads: d1, d16, d32 (decode T = batch) and v1, v16, v32 (verify, T = 8 x batch = 8, 128, 256).
- Tool: `python kb/muse_score.py --job 1ca0668a --save-us d16S=10 --table` (exact replay; --backtest 79/79 jobs).
  User variant with history/controls: kb/muse_score_ref.py. Uniform savings on all 12 shapes: 3/5/8 us -> 2.284/2.318/2.372.

## 2. Where we stand (Nov 3)
| Board | Ours | Rank | keell0renz (same judge) |
|---|---:|---:|---:|
| Decode | 2.2347 | 232 / 1215 | 2.2588 |
| MoE | 2.3797 | 229 | 2.3874 |
| Attention | 2.6536 | 133 | 2.6421 |
| Dense | 1.5886 | 71 | 1.5598 |
| Prefill | 1.9257 | 51 | 1.9070 |
Overall rank 232 (= Decode rank). Target: joeyhlu 2.3569 (#33); #30 bar expected ~2.37-2.38.
Unchanged-path Decode noise band (7 checks): 2.2318-2.2607 (median 2.2433).

## 3. Our Decode file and its time (profile kb/ref/decode_curp.f2ab6d4c6ba74769.txt)
Base = keell0renz's decode (kb/v1/decode_v1.py) + J0 guard + ERL/HINT/FINREG. Kernels per T:
| T | qkv | attention | o + boundary | pooler+router | MoE | unpooler + boundary |
|---|---|---|---|---|---|---|
| 1 | sk2 10.2 | attn 5.5 (S) / 11.8 (F) | sk2_gle 6.2 | sk2_gle fused 17.3 | TINY 29.7 | sk2 4.4 + _sl_cluster_finish 2.0 |
| 8 | sk2_narrow 9.3 | attn 7.2 / 14.9 | sk2 5.9 + glue 3.2 | gle fused 18.7 | TINY2 71.4 | sk2 5.0 + finish 1.8 |
| 16 | sk2 14.9 | attn 7.5 / 88.4 | sk2 6.1 + glue 5.4 | sk2f fused 18.5 | MID 74.3 (S) / 56.5 (F) | sk2 10.8 + glue 3.2 |
| 32 | sk2 | attn 9.6 / 171 | sk2 | rq/sk2 | WGS MAIN 75.7 / 106 | sk2 |
| 128/256 | rq / cuBLASLt | attn 12-16 / 108-187 | Lt | rq + rk | MAIN 162-192 | Lt |
MoE = 50-64% of sliding layers at T>=16; full attention = 31-49% of full layers at T>=16 (already near the random-gather ceiling).

## 4. Gap to Joey (his speedup / ours, per shape)
d16S +13.4%, v1S +8.6%, d16F +6.2%, v32S +4.9%, d1S/d1F +4.5%, v32F +3.5%, v16S +3.1%, d32S +2.3%, v16F +1.7%, v1F +1.2%, d32F +0.6%.
By load: d16 ~11%, v1 ~7%, d1 ~4.5%, v32 ~4.3%, v16 ~2.5%, d32 ~1.6%. Real absolute wins despite his slower GPU: d16S -10 us,
d16F -8 us, v1S -3 us. His engines: MoE TINY/TINY2 (<=8), VMS (16), VML (32), HYB (>=128); Dense "gd" (+DG1 at 32);
attention KS1/KS3/HYA/AX1. His component boards are below ours, so his edge lives inside his Decode file.

## 5. Active lanes (details + iteration log: kb/exp/JOEY_PLAN.md)
| Lane | Idea | File | Status |
|---|---|---|---|
| A | VMS/VML: own MoE engine at T=16/32 (oracle design, kb/exp/VMS_BRIEF.md) | kb/exp/moe_vms4.py | iter1 tie (93 vs 92 us); iter2 2x slower (tail split); iter3 ablation; iter4 bound diagnosis c5800fa6 pending |
| A->Decode | wire VMS into Decode T=16/32 with full-layer gate | kb/exp/decode_vms.py | ready, waits for a MoE-board win; todo PDL-attach stage A |
| B | Dense-board GEMVs (cgemv m1, gv m8/16) in Decode for qkv/o/unpooler | kb/exp/decode_gd.py | 8ffd6871 pending (first run hit a patch-anchor bug) |
| Predictor | offline ranking/"chase" with muse_score | kb/predict/ | agent running |
Queued: A2 VML at 32, C boundary-glue at 16 (8.6 us), D T=8 variant, E lossless scale coding (scales 0.232 vs 0.5 bit/value; codes incompressible).

## 6. Closed (do not repeat; full list in main:TRIED.md, snapshot kb/ref/TRIED.md)
MoE: new single-WG phase-B core (1.5x slower), refit stream-K tables (slower), pre-swizzled xA (0 us exposed), private partials,
L2 hints, build selection by T, producer decoupling. Decode: CURP/rq routes (BF16 split reduction flips 1.6-2.3% expert ids),
routing fused into pooler tail (2x slower), top-k into router (<=1.2% ceiling), finish-kernel fold, attention ports,
gle deferred norm at T=8, glue PDL, early norms. Dense: sub-12-bit packing (1.6-2.4x slower), simple GEMVs, CUTLASS, rk2 router.
Policy: routing-tolerance expert dropping - dropped by user as gaming.

## 7. How every change ships
In-prepare gate: read-flush (512 MiB) ABBA CUDA-graph timing, both orders, two passes; median AND paired <= 0.988 (MoE board
<= 0.97); correctness within judge tolerance (Decode: identical expert ids); else fall back. Then: check -> scored run ->
publish board_best on main (byte-identical, BOARD_BEST.md row) if above the board score. Record every result with
`python "C:\Users\Achita\Documents\ChatGPT\Muse 2\.tools\tried.py" add ...`.

## 8. File map
- kb/STATE.md (this), kb/exp/JOEY_PLAN.md (hints, lanes, loop protocol, iteration log), kb/exp/VMS_BRIEF.md (MoE design brief),
  kb/exp/IDEAS_predictor_prompt.md (predictor/chase ideas)
- kb/ref/: rules (muse_decode_layer, muse_moe_w4a16), TRIED.md snapshot, KEELL_LESSONS.md, ALYGG_NEGATIVE.md, H200_FACTS.md, profiles
- kb/v0, kb/v1: keell base files (MoE board, Decode). kb/exp/*.py: experiments. kb/runs/*.json: saved judge results.
- kb/muse_score.py, kb/muse_score_ref.py: score replay / what-if.
- kb/exp/archive/, docs/archive/oct23-sprint/: superseded plans and the Oct 23 Dense sprint docs (historical only).
