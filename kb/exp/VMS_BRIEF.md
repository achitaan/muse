# Brief: a dedicated MoE engine for M=16 (and M=32) - "VMS/VML"

## Goal
Design a new NVFP4 W4A16 routed-MoE kernel specialised for M=16 tokens (and a variant for M=32) that is
substantially faster than keell0renz's current MID (M 9-16) and WGS-MAIN (M 17-63) builds, near the HBM floor.
It will first be measured on the MoE board (muse_moe_w4a16, shapes bs16/bs32) and then dropped into the Decode
layer's MoE call at T=16/32 (same inputs: x [M,4096] bf16, local ids/weights, NVFP4 w13/w2 + scales, post-norm).

## Evidence
- Another team (decode 2.3569, #33) beats our decode (2.2347) by ~11% on the decode-16 load and is ~10 us faster
  absolute on decode-16 sliding (116.4 vs 126.8 us). Their engine table: T<=8 TINY/TINY2 (same as keell),
  T=16 "VMS", T=32 "VML", T>=128 "HYB". So the win is a dedicated M=16 engine (and a smaller one at 32).
- Our decode-16 sliding layer: MoE 74.3 us of 142.7 us GPU time (kb/ref/decode_curp.f2ab6d4c6ba74769.txt profile 2);
  decode-16 full MoE 56.5 us (profile 3).
- MoE board (kb/ref/moe_p2.08c529571a4b4672.txt): ours vs reference speed-of-light
  bs1 25.3/11.4, bs8 65.6/37.3, **bs16 91.5/52.5**, **bs32 121.6/70.8**, bs128 179.9/101.3, bs256 215.5/108.9 us.
  SOL counts 0.75 x distinct experts at 4.8 TB/s (kb/ref/muse_moe_w4a16/reference.py), so a pure-streaming floor is
  roughly 1.39 x SOL (~73 us at bs16, ~98 us at bs32). We are at 1.74 x SOL: ~18-23 us of headroom per call.
  Top MoE board team: 2.76x overall vs our 2.38x.
- Earlier ledger notes: MID "streams ~4.6 TB/s at bs16/32" in-phase; large-batch phase B is consumer-bound
  (producer blocked 83% on a full ring). A 2-CTA/SM single-consumer-WG phase-B core was 1.5x slower.

## Code
- keell MoE board file: kb/v0/moe_v0.py (builds TINY/TINY2/MID/MAIN, lines ~54-165 for the variant selection;
  the CUDA source is inside the file). Decode file: kb/v1/decode_v1.py (same MoE source embedded; MoE ~5500-6200).
- Rules: kb/ref/muse_moe_w4a16/TASK.md + reference.py (NVFP4 weights stay NVFP4; BF16 activations; per-expert
  post-RMSNorm on y*(1+w); routing skewed so this GPU is the busiest; cold L2 each replay; CUDA-graph replay).
- What failed before: kb/ref/TRIED.md (search MoE), kb/ref/KEELL_LESSONS.md, kb/ref/ALYGG_NEGATIVE.md,
  H200 numbers: kb/ref/H200_FACTS.md.

## Question
1. Where do the ~20 us at bs16 go in the MID build (ramp-up before weights flow, the A->B dependency per expert,
   per-expert RMSNorm/finalize tail, load imbalance across 132 SMs with ~10-25 touched experts, duplicate reads)?
   Reason from the code and numbers; say what a cheap in-check diagnostic should measure to confirm.
2. Propose ONE concrete engine design for M=16 (and how it changes for M=32) that removes that waste:
   CTA/SM mapping, how w13 and w2 work is split (N-split per expert across many CTAs, split-K for GEMM2,
   ownership of the post-norm), the weight load path (TMA / cp.async.bulk, ring depth, L2 hints), the tensor-core
   path at N=16 (mma.sync m16n8k16 vs wgmma), and the A->B handoff (global flags vs a second launch with PDL).
   Include the predicted time at bs16/bs32 with arithmetic.
3. A build plan an Opus implementer can finish in ~6-8 hours, in a new file based on kb/v0/moe_v0.py, with an
   in-prepare gate (read-flush ABBA vs the incumbent per bs, exact-tolerance correctness) and fallback.
Return <= 600 words, concrete, with file/line anchors where relevant.
