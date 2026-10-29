# Other tracks (Dense, MoE, Prefill) — investigation in lane `omos/other-tracks-20261029`

Branch created October 29, 2026 ~20:20 UTC from `6b7209fd89fc859ca658e3148b60b05d1052b44e`
(worktree `.slim/worktrees/other-tracks`). No `best/` source changed. No GPU job from this lane yet.
All numbers below come from archived official runs in this repository.

## Dense (`best/verified.py`, scored 1.5277713x, job `6131bc96427e4486`)

Per-load speedups: decode 1 1.672, decode 16 1.593, decode 32 1.391, verify 1 2.107,
verify 16 1.302, verify 32 1.252.

| Idea examined here | Evidence | Decision |
|---|---|---|
| Lossless-12-bit (c12) `tt_mw` wgmma GEMM for M=128 pooler/unpooler (cuBLAS BF16 today, memory-bound at M=128) | The retained prepare-time selector already timed c12 `tt_mw` against the incumbents (log of `6131bc96427e4486`): QKV M16 best 17.98 vs 14.85 µs, QKV M32 18.02 vs 15.46 µs, unpooler M32 22.30 vs 20.26 µs. Decode cost grows with token tiles, so M=128 is worse | not checked |
| Unused qb config `(64, 36, 256, 3, 4, 219, 5)` for M=256 QKV | timed in-prepare at 38.9–40.5 vs 18.0–18.2 µs (jobs `1f6e5d17cf8c4c75`, `f873dc6fd6e948a2`) | not checked |
| Fixed BF16 cluster families at M=128 | lost 4–7% (HANDOFF.md) | closed earlier |

Remaining measured gap: small-M latency kernels (router 3.89–7.91 µs vs 1.75–2.68 µs floor;
sliced unpooler 3.91/4.09 vs 1.75; output projection 4.99 vs 3.50). A structural candidate
for a future window: a TMA-bulk GEMV for M<=8 where each CTA issues one `cp.async.bulk`
for its contiguous weight slice at kernel start (≈64 KB/SM for 8.4 MB), FP32 K-split partials
with a self-cleaning last-CTA reduction. Estimated 1.0–1.3 µs per small projection
(≈3–5% on the decode 1 / verify 1 loads); needs GPU iteration before a check.

## MoE (`best/moe_verified.py`, checked 2.3654374x, job `d600231c8a8244cb`)

Per-load: decode 1 3.146, decode 16 2.003, decode 32 1.642, verify 1 2.273, verify 16 2.591,
verify 32 2.875. Weakest load decode 32 runs MAIN/WGS (231,168 B dynamic smem, 96 regs,
one CTA/SM). The FT_T=32 WGS specialization for 17..32 tokens was already checked and lost
(`47f44d0ff8704a44`, 2.3537x; 32-token 1.638 vs 1.642). Extending the MID split build to
M<=32 is capacity-safe (<=64 jobs < JMAX 88) but re-streams any expert with more than 8
tokens; measured efficiency is equal (MID 2.6 TB/s at decode 16, MAIN 2.7 TB/s at decode 32),
so it is expected to lose. No configuration-level candidate found.

## Prefill (`best/prefill_verified.py`, scored 1.6634692x; excluded from overall)

Full-attention shapes are the weakest (turn 1 1.426, turn 4 1.408, first-turn ctx 65536
1.419). FA3 is already in FP8 mode (`q.to(FP8)`), `num_splits` is left to FA3's occupancy
heuristic (turn 1: ~64 m-blocks -> ~2 splits, ~128 CTAs), page size 1 forces gathered
KV. First-turn shapes: MoE 37–40%, cuBLAS GEMMs ~77% of BF16 peak, two FP32-residual
norms at ~4 TB/s. No configuration-level candidate found.

## Candidates built and frozen (offline, not yet checked)

Both use the pattern "time on the judge's GPU during prepare, keep the incumbent unless
another option wins by >=3% with matching output", so a check either finds a gain or
returns timing evidence without regressing a shape.

| Id | Track | Candidate bytes | Parent | Mechanism | Gates |
|---|---|---|---|---|---|
| P1 | Prefill | `6d1f43575637910e3d047c496b5833f28ac45c6f10e6f1558b63f4c3531665e0` (265,601 B) | `93e57ab4?` checked 1.6393x (`9ca79c7f676f4b63`) | FA3 `num_splits` x `pack_gqa` per scored attention shape (graph replay, cold L2, reference-shaped page layout) | `results/prefill-fa-select-check-20261029/gates.json` |
| M1 | MoE | `95d82f5b552e6266a6f8a7826db59a9533b1b19c0fc5133c1ccde21df0860e4d` (184,785 B) | `33114871?` checked 2.3654x (`d600231c8a8244cb`) | compiled build (TINY/TINY2/MID/MAIN) per token count 1/8/16/32, three routing draws shaped like the task inputs | `results/moe-select-check-20261029/gates.json` |

Dense: the source already contains the cuBLASLt algorithm/layout search for the M=128/256
shapes (`tt_lt__LT_H200_FIXED`, `tt_lt__LT_SEARCH_SEEDS`, transposed layouts) and the c12
selector results above, so no new Dense selector was built; the remaining Dense gap needs a
new small-M kernel.

## Checks run October 29 (official, free checks only; no scored submissions)

| Id | Job | Result | Parent | Outcome |
|---|---|---|---|---|
| M1 MoE build selector | `d746cb32bcc045f4` | 2.3763335x pass | 2.3654374x | nothing selected (incumbent builds fastest at 1/8/16/32 tokens); noise |
| P1 Prefill FA3 selector | `bb9ce14d3ad8490b` | 1.6564053x pass | 1.6392908x | nothing selected (FA3 heuristic within 2.1%); noise |
| M2 MoE TINY2 for 1..4 tokens | `3158fdc490ed46d1` | 2.3593403x pass | 2.3654374x | decode 1 25.77 vs 24.85 us (0.962 normalized); rejected |

No Dense/MoE/Prefill change is retained; `best/` is unchanged. Remaining gaps need new kernels.
