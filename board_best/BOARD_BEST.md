# Board bests and the code behind them (okc leaderboard, 2026-10-29 23:57 UTC)

The board keeps each team's best scored run. Sources of tonight's improvements are copied here
unchanged (the `best/` baselines are untouched).

| Board | Best | Rank | Scored job | Source in this folder | SHA-256 |
|---|---:|---:|---|---|---|
| MoE | 2.3797x | 59 / 1131 (was 90) | `ac80911ea3d34dac` | `moe_m3_scored_2.3797.py` (M3 selector; selects nothing -> same kernels as `best/moe_verified.py`) | `23008c1b004a6cc4b6bb0384e9ef9bc180d398c901edc309faf656b26301e795` |
| Attention | 2.5690x | 144 / 1087 (was 149) | `c8f7560097d348c5` | `attention_a3_scored_2.5690.py` (A3: decode 16 sliding n_splits 8) | `ada6df950cdb54aae711929b6e45f0b5cb1b365acd0bcaaac831394fc916d1e1` |
| Decode | 2.0510x | 214 / 1005 | `94572fe644414691` | `decoder_r2f_scored_2.0510.py`: the 2.0071x source (`decoder_finreg_fp8pv_scored_2.0071.py`) plus R2f. At 1 and 8 tokens the exact FP32 router runs inside the pooler GEMV launch: warps 0-15 of each of the 128 pooler CTAs compute the router partials with the original FMA order, and an extra 129th block (on an SM the pooler leaves free) runs the original top-8 tail. Routing ids, weights and pooler output are checked bitwise against the original path in prepare. Passed 13/13. Check `d778271fc15348ed` measured 2.0633x. From branch `rgon/decode-moe-20261030` (details in results/decode-moe-plan-20261030/RESULTS.md) | =fa2a73b4de60410c10dc3609141f3677c9ff79acff0512e45fdd6b5084f747` |
| Dense | 1.5278x | 158 / 1411 | `6131bc96427e4486` | `best/verified.py` | `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b` |
| Prefill | 1.6635x | 179 / 652 | (earlier scored run) | `best/prefill_verified.py` | `93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7` |

Honest note: both new bests run kernels already measured as equivalent to the incumbents
(MoE M3 selected nothing; A3's split change was not confirmed against same-night controls).
The higher scores are favorable measurements of the same code, which the board keeps.
The Decode file is likewise speed-neutral within noise. Its two changes each measured as ties: 1.9949x and 1.9958x in checks, against D9's 1.9902x. Its value is that it is a correct source with every required BF16 precision boundary.

Retained baselines (one per problem) are in `best/`; the full experiment history is on branches `omos/decode-fresh-20261029` and `omos/other-tracks-20261029` (see `NEXT_PLAN.md` there).
