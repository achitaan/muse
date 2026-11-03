# Council question (Nov 3, 14:30 UTC): what should the last ~41 hours buy?

Read first (all in this worktree): kb/STATE.md, kb/APPROACH.md, kb/exp/JOEY_PLAN.md (iteration log), kb/predict/REPORT.md,
kb/ref/TRIED.md (ledger snapshot; the newest rows are on main and summarised below), kb/ref/decode_curp.f2ab6d4c6ba74769.txt (profile).

Goal: raise our Decode score (2.2347, rank ~232; overall rank = Decode rank) toward 2.36-2.38 (#30). Real kernel speedups only
(no routing-tolerance exploitation, no lower precision, lossless weights). Deadline Nov 5 07:59 UTC; ~$380 of agent budget left;
8 Decode checks + 9 MoE checks left today, 20 each tomorrow. Implementer: Opus "fixer" agents (~4-10 h per build).

## Everything measured today (all closed unless noted)
- VMS/VML own small-batch MoE engine (3 launches A->B->fin): best bs16 93-94 vs 92-98 us (median 0.95, paired 0.99), never
  passes; in the Decode layer 1.01-1.016 slower; at bs1 1.12-1.16 slower; iteration levers (tail split 2x, deeper ring +8%,
  register ping-pong +4%, 4-lane producer ~0) all lost. Diagnostics: A/B neither stream- nor math-bound (overlap loss).
- Dense-board GEMVs inside Decode (cgemv/gv): 0.99-1.025, no transfer (sk2 already PDL-prefetches).
- Deep PDL pre-wait staging (P=4/NSTAGE) + early triggers: 0.994-1.009.
- Timeline T=1/8: PDL already overlaps 15 us; "excess" was wait time inside durations.
- Boundary glue: skip-ceiling looked like 2.317 but was inflated (skipping b1 changed routing); bit-exact coarsened kernels
  not faster; b2 = its byte time. Closed.
- rq/CURP qkv with FP32 split reduce: ids fixed, layer 1.02 slower.
- Humming-style producerless GEMM2: 5-36x slower.
- TINY/TINY2 (bs1/bs8) kill check: incumbent 26.2/63.9 us vs same-address stream 19.3/55.6 -> headroom 5.4/6.7 us;
  critical path bs1 = ramp 3.1 + A 9.6-13 + A->B wait 2.4 + B 5-6 + drain 0.5 + barrier 2 + final 0.7.
  Expert-wavefront rewrite 1.27x/1.47x slower (per-expert hand-offs); B-bank prefetch 1.11x/1.04x slower (steals A bandwidth;
  the A->B wait is completion latency).
- Predictor: 0/27 written claims realised; isolated wins realise 4/16; gate wins reach the judge at ~0.45x; noise sd 0.53%.
- Joey (2.3569) beats us per shape: d16S +13.4%, v1S +8.6%, d16F +6.2%, v32S +4.9%, d1 +4.5%; we are faster in absolute us
  at v16/v32. His hint names per-T MoE engines (TINY/TINY2 <=8, VMS 16, VML 32, HYB >=49), dense "gd" (+DG1 cat/qkv at
  T=32), sliding attention KS1/KS3 with FP16 QK on BF16-rounded queries, HYA full attention with E4M3 queries, AX1 at d1F.
- Score value per us: d1S 18.4 bp, v1S 11.5, d16S 8.5, d32S 6.7, d1F 6.4, v16S 4.8, v1F 4.0, v32S 3.5, d16F 2.9 ...

## Question
Propose the best plan for the remaining time. Give at most 3 experiments, ranked, each with: mechanism (concrete, different
from everything closed above), target shapes, expected us and score (state your assumed transfer factor), the cheapest kill
check (1 check, diagnostic or gated), and build size. You may also recommend spending nothing further if you think no
experiment has positive expected value; justify it. Also say whether the unused scored-run quota has any legitimate use
(we do not resubmit unchanged code to farm noise). <= 400 words.
