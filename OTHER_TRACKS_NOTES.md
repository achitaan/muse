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
| D1 Dense row-parallel Triton GEMV (router / sliced unpooler, 1 and 8 tokens) | `4edf30f7aa6142e7` | 1.5048449x pass | 1.5277713x | nothing selected: incumbent 7.97/8.48/8.45/8.51 us vs best Triton 8.26/12.90/8.61/16.90 us |
| P2 Prefill MoE schedule (large x fine) per token count | `b5556364aff148f9` | 1.6647948x pass | 1.6392908x | nothing selected: thresholds 8192/2048 best or within 0.9% at T=753/3012/16384 |
| M3 MoE WGS MAIN vs mma.sync fallback at 16/32/128/256 | `4ed3dd65ae5341a0` | 2.3376805x pass | 2.3654374x | nothing selected: fallback 13-38% slower; MID vs MAIN at 32 tokens flips sign between runs (noise) |
| D3 Dense cuBLASLt algorithms for (128, 'o') | `7287935eb7d34f7b` | 1.5170751x pass | 1.5277713x | nothing selected: incumbent 10.75 us beat all 32 algorithms (4 layouts x 8; best 11.10 us) |
| P3 Prefill GEMM layout (x@W^T vs pre-transposed W) | `f2241332a476452e` | 1.6497240x pass | 1.6392908x | nothing selected: best alternative +0.1%; qkv_gate at 753 tokens 46.0 vs 42.0 us |
| D5 Dense olayer configs for O at 1/8/16/32 tokens | `3b13fab22b2b4480` | 1.5196641x pass | 1.5277713x | nothing selected: incumbent 9.25/9.44/9.82/10.11 us vs best config 10.14/10.24/10.18/10.14 us |
| D6 Dense olayer configs for O at 128/256 tokens (weakest Dense shape: O m256 1.028x) | `4e8fda34635040e6` | 1.4949033x pass | 1.5277713x | nothing selected: m256 best olayer 12.90 us ties fixed cuBLASLt 12.90; m128 incumbent 10.75 vs 10.82 |
| D7 Dense cuBLASLt neighborhood (restore + expand: split-K, stages, swizzle x custom x cluster, + 16 exhaustive heuristics) for unpooler M=128/256 | `96bc324b987a4c44` | 1.5094800x pass | 1.5277713x | nothing selected: m128 best 22.24 vs 22.69 us (2.0%, custom option 3->4), m256 27.81 vs 27.84; fixed descriptors are local optima |
| M4 MoE ring depth OKC_NS 3->4 (MID at 16, MAIN at 32 tokens) | `0e1634cf0bc245a1` | 2.3391587x pass | 2.3654374x | M16 MID+NS4 0.5% faster (kept incumbent); M32 skipped by the 200 s prepare guard (one compile ~38 s, parent prepare ~167 s) -> M5 |
| P4 Prefill FA3 on a contiguous KV copy (gathered by page table) for full layers | `37a3e45b4e4b431d` | 1.6531207x pass | 1.6392908x | **selected and applied** for first turn full ctx 32768 / 65536 (FA3 4474.3->4304.9 us, 8071.5->7766.6 us in prepare); slower at 753 tokens (423->516 us, kept paged). Affected shapes normalized vs same-night P1/P2/P3: 1.011-1.056 |
| P5 confirmation of P4 (identical code + comment) | `c4d2e6ef68674473` | 1.6648694x pass | 1.6392908x | same selections, applied; FA3 4499.5->4341.3 and 8013.7->7730.6 us. Mean of P4/P5 normalized vs P1/P2/P3: ctx 65536 1.026/1.032/1.052 (meets 1.02), ctx 32768 1.002/1.029/1.046 (fails vs P1). Frozen rule needs both shapes: **not retained** |
| M5 MoE ring depth OKC_NS 3->4 at 32 tokens (MAIN WGS) | `940d22a6cc7642d5` | 2.3382191x pass | 2.3654374x | +0.4% (383.52 vs 385.15 us over 3 draws, bit-identical); NS5 skipped by compile guard (prepare 219 s). Ring depth is not the decode-32 bottleneck |

D5's first upload (22:59:10 UTC) failed with `tls: bad record MAC`; the jobs list showed no
Dense job after 22:52:45, so the identical content was relaunched (`upload-failed-tls.json`).

Candidate hashes (append-only add-ons over the retained bytes, parent SHA asserted at build):
D1 `6776a4c41152fd6378625dd967bcd50816d7e44aad4ff26dfd31841364810a54`,
P2 `245a5d4cc417dd1d0640a274b0e3e595173ced336ca537c8467f21aa550504fa`,
M3 `23008c1b004a6cc4b6bb0384e9ef9bc180d398c901edc309faf656b26301e795`,
D3 `38aeb4867237bc3d927a3334bebada7c011cc952183fcc38e0f0c14e3f61fd7f`,
P3 `a7289eb1465e81590cdc7c9a1e57f934f57e263e85a7a59e481bb0bd376236a9`,
D5 `a56d8d33da2a9a0d2d6ebcd084145236aab6026992b3e396d999300e61038c39`,
D6 `af3a3c2c1ba7df10500d4323a2e53bee95966420a9a5417609edb5124652ebb7`,
D7 `f2bb94589ab113e9a9816b4e5e8032afac4682e3c1d42ff10577621e25abb1ef`,
M4 `8ec4dc7d2f7dfe055d6b131aecb7c0d73495e969c877b1e16f82bfe4b8c64c50`,
M5 `6ee2ac8447775985328659015c3cee13a4aa81b30bc62700cf0a85164e85d8fc`,
P4 `5d08eb926b36b190ae789d26db849340b2744fb8d5ad4f4d70f9f906487b5e53`,
P5 (P4 + trailing comment) `15cc2a942f13312b3a5f92fcd15d22b6d5807df7331ef672501aecfe51049992`.
Each result directory holds `gates.json` (frozen before launch), `result.json` and `verdict.json`.

Track-level moves in these runs (-1.5% to +1.6%) occur with no device-path change (every
selector kept the incumbent), so single-run track deltas below ~2% are run variance here;
tiny Dense shapes move 5-11% between runs.

No Dense/MoE/Prefill change is retained; `best/` is unchanged. Every configuration lever
examined (MoE build per token count, MoE fallback build, MoE ring depth at 16/32 tokens, Prefill
FA3 splits, Prefill MoE schedule thresholds, Prefill GEMM layout, Dense cuBLASLt algorithms at
(128,'o'), Dense olayer configs for O at 1-256 tokens, Dense unpooler cuBLASLt neighborhood at
128/256, Triton GEMV for router/unpooler) was measured on the judge GPU and the incumbent was
fastest or within 3%. The one lever that won in-process is Prefill's contiguous-KV FA3 for long
full first-turn shapes (FA3 -3.5..-3.8% in two runs; ctx 65536 passed its shape gate, ctx 32768
did not against one reference). It is the best starting point for a future Prefill change, ideally
with a cheaper gather (P6 tests an 8-byte-wide row gather). Remaining gaps need new kernels.
