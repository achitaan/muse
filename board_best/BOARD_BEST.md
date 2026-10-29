# Board bests and the code behind them (okc leaderboard, 2026-10-29 23:57 UTC)

The board keeps each team's best scored run. Sources of tonight's improvements are copied here
unchanged (the `best/` baselines are untouched).

| Board | Best | Rank | Scored job | Source in this folder | SHA-256 |
|---|---:|---:|---|---|---|
| MoE | 2.3797x | 59 / 1131 (was 90) | `ac80911ea3d34dac` | `moe_m3_scored_2.3797.py` (M3 selector; selects nothing -> same kernels as `best/moe_verified.py`) | `23008c1b004a6cc4b6bb0384e9ef9bc180d398c901edc309faf656b26301e795` |
| Attention | 2.5690x | 144 / 1087 (was 149) | `c8f7560097d348c5` | `attention_a3_scored_2.5690.py` (A3: decode 16 sliding n_splits 8) | `ada6df950cdb54aae711929b6e45f0b5cb1b365acd0bcaaac831394fc916d1e1` |
| Decode | 2.0071x | 221 / 1000 | `646774bad23e4c7f` (other lane) | on branch `argon/top50-20261029`, not in this folder | - |
| Dense | 1.5278x | 158 / 1411 | `6131bc96427e4486` | `best/verified.py` | `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b` |
| Prefill | 1.6635x | 179 / 652 | (earlier scored run) | `best/prefill_verified.py` | `93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7` |

Honest note: both new bests run kernels already measured as equivalent to the incumbents
(MoE M3 selected nothing; A3's split change was not confirmed against same-night controls).
The higher scores are favorable measurements of the same code, which the board keeps.
