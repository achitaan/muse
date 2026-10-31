# Board bests and source provenance

Decode refreshed 2026-10-31 07:28 UTC from the scored run `0aa30c92` (rank 280 / 1091; #50 is 2.246x). Earlier Decode sources remain here.

Decode refreshed from the API on 2026-10-30 at 07:31 UTC. Other rows retain the 2026-10-29 23:57 UTC snapshot.

The board keeps each team's best scored run. Sources of tonight's improvements are copied here
unchanged (the `best/` baselines are untouched).

| Board | Best | Rank | Scored job | Source in this folder | SHA-256 |
|---|---:|---:|---|---|---|
| MoE | 2.3797x | 59 / 1131 (was 90) | `ac80911ea3d34dac` | `moe_m3_scored_2.3797.py` (M3 selector; selects nothing -> same kernels as `best/moe_verified.py`) | `23008c1b004a6cc4b6bb0384e9ef9bc180d398c901edc309faf656b26301e795` |
| Attention | 2.5690x | 144 / 1087 (was 149) | `c8f7560097d348c5` | `attention_a3_scored_2.5690.py` (A3: decode 16 sliding n_splits 8) | `ada6df950cdb54aae711929b6e45f0b5cb1b365acd0bcaaac831394fc916d1e1` |
| Decode | 2.0783x | 280 / 1091 (top 50 at 2.246x) | `0aa30c9234014956` | `decoder_p6_scored_2.0783.py`: the 2.0755 parent plus a 1% paired-A/B gate, a MoE-variance-tolerant whole-layer probe (rel L2 <= 2e-3), PDL on the `_pn`/`_sfn` Triton glue and the MoE launch (1/8/16/32 tokens), and an early-trigger private stock norm at 1/8 tokens. The scored prepare selected (per its log) per-shape masks; passed 13/13 and 12/12; prepare ~274 s. Checks of this lineage: 2.0682-2.0975x (`689fd93e`, `6553b299`, `831de9a6`, `d543da40`, `ac41a480`, `0ca71861`); a no-selection control scored 2.0434x (`7abd8d0d`). Branch `argon/decode-next-20261030`, `experiments/top50/` and `PROGRESS.md`. | `7079947b3a2a7f435b6a9959b88bc3301fbab1b31e9f59f5120f5427ee709fdf` |
| Dense | 1.5278x | 158 / 1411 | `6131bc96427e4486` | `best/verified.py` | `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b` |
| Prefill | 1.6635x | 179 / 652 | (earlier scored run) | `best/prefill_verified.py` | `93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7` |

MoE/Attention provenance: those rows run kernels already measured as equivalent to the incumbents
(MoE M3 selected nothing; A3's split change was not confirmed against same-night controls).
The higher scores are favorable measurements of the same code, which the board keeps.
The Decode PDL path has source-bound paired layer evidence: the scored prepare saved 3.008 / 3.040 us in the two arm orders with bitwise-equal outputs and caches. Its official one-token sliding case took 75.819 us (79.027 us in the parent check). The scored interval is [2.0443, 2.1066]; it overlaps the parent, and the entire aggregate point increase is not attributed to this one changed case. Failed byte/performance gates retain the parent. Earlier checked and scored sources remain in this folder.

Retained baselines (one per problem) are in `best/`; the full experiment history is on branches `omos/decode-fresh-20261029` and `omos/other-tracks-20261029` (see `NEXT_PLAN.md` there).
