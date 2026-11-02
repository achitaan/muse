# Board bests and source provenance

Attention refreshed 2026-11-03 from scored run `baedd9d7` (2.6461x): keell0renz's attention file plus achitaan's import-time self-tuner; in this run the tuner adopted 0/12 shapes, so the scored path is their default (a tie with their file, not a speedup).

Decode refreshed 2026-11-03 from scored run `f79f3879` (2.2452x): kb v1 plus a gated chain-scheduling experiment (boundary preload, rq PDL, PDL at T=256); the gate adopted 0/8 classes, so the scored path equals kb v1 and the change from 2.2442x is noise.

Decode refreshed 2026-11-02 from scored run `26331ae5` (2.2442x): kb v1 plus a gated port of keell0renz's newer attention kernels; the gate adopted 0/12 shapes, so the scored path equals kb v1 (2.2341x row below) and the gain over it is noise.

Prefill refreshed 2026-11-02 from scored run `a0c18fe3` (1.9257x): keell0renz's public Apache-2.0 prefill file with our prepare-time A/B diagnostic; the scored path is their default (license/notice in `keell0renz_LICENSE` / `keell0renz_NOTICE`).

Decode refreshed 2026-11-02 from scored run `2ecb3de5` (2.2341x): keell0renz's public Apache-2.0 decode file plus our changes (license and notice copied as `keell0renz_LICENSE` / `keell0renz_NOTICE`).

Decode refreshed 2026-11-01 from scored run `9dabcd9a` (2.1056x).

Decode refreshed 2026-10-31 09:04 UTC from scored run `e1af1dd0` (rank 272 / 1093). Earlier Decode sources remain here.

Decode refreshed from the API on 2026-10-30 at 07:31 UTC. Other rows retain the 2026-10-29 23:57 UTC snapshot.

The board keeps each team's best scored run. Sources of tonight's improvements are copied here
unchanged (the `best/` baselines are untouched).

| Board | Best | Rank | Scored job | Source in this folder | SHA-256 |
|---|---:|---:|---|---|---|
| MoE | 2.3797x | 59 / 1131 (was 90) | `ac80911ea3d34dac` | `moe_m3_scored_2.3797.py` (M3 selector; selects nothing -> same kernels as `best/moe_verified.py`) | `23008c1b004a6cc4b6bb0384e9ef9bc180d398c901edc309faf656b26301e795` |
| Attention | 2.6461x | (pending refresh) | `baedd9d76cfe43f3` | `attention_kbtune_scored_2.6461.py`: keell0renz's attention file (Apache-2.0, kernels unchanged) plus achitaan's import-time per-shape plan tuner (137 candidates, read-flush ABBA, confirm pass); this run adopted 0/12 (check `129ffe91` had adopted d16S PDL and v16S nc=2 at ~2%, which did not reproduce). Scored CI [2.6064, 2.7014]; keell's unchanged check `6ae275da` 2.6421x. Branch `argon/keell-base-20261102`, `kb/exp/attn_tune.py` (c9413d3). | `46bef77f73453f2328b793b5bffaa9360a84280789e7b477b1723bc2ce7a4515` |
| Attention | 2.5690x | 144 / 1087 (was 149) | `c8f7560097d348c5` | `attention_a3_scored_2.5690.py` (A3: decode 16 sliding n_splits 8) | `ada6df950cdb54aae711929b6e45f0b5cb1b365acd0bcaaac831394fc916d1e1` |
| Decode | 2.2452x | (pending refresh) | `f79f387976824c34` | `decoder_kbchain_scored_2.2452.py`: kb v1 plus achitaan's per-class gated chain arms (A: boundary operands loaded before griddepcontrol.wait + early trigger; B: PDL on rq; C: PDL at T=256); gate adopted 0/8 (best v16f A+B 0.992, d32s A 0.994 not equivalent), so production = kb v1. Scored CI [2.2116, 2.2789]; keell's unchanged check `cde2084a` 2.2588x. Branch `argon/keell-base-20261102`, `kb/exp/decode_chain.py` (d493ab5). | `b2ae009e8321d4c0acc77694f9de0439221e0c80f3d9111e4cc20a75ee5a3b82` |
| Decode | 2.2442x | (pending refresh) | `26331ae5768b4471` | `decoder_kbatp_scored_2.2442.py`: kb v1 (keell0renz decode + J0 guard + ERL/HINT/FINREG) plus achitaan's gated port of keell0renz's standalone v298bulk / wst / vp2 attention; full-layer read-flush ABBA gate adopted 0/12 shapes (ratios 0.985-1.041, verify-16/32 not equivalent), so production = kb v1. Scored CI [2.2106, 2.2779]; keell's unchanged check `cde2084a` 2.2588x. Branch `argon/keell-base-20261102`, `kb/exp/decode_attnport.py` (8ebcc1e). | `f63506c2dfa0400dc5167609909ff627329c67006503ac8fb4145162cd9cc569` |
| Decode | 2.2341x | (pending refresh) | `2ecb3de53d7f44e9` | `decoder_kb1_scored_2.2341.py`: keell0renz's decode file (github.com/keell0renz/muse-open-kernel-challenge, Apache-2.0) plus achitaan changes: J==0 stream-K hang guard, MoE ERL + HINT + FINREG. Same-judge check `2a45f597` 2.2589x vs their unchanged file `cde2084a` 2.2588x (tie). Branch `argon/keell-base-20261102`, `kb/v1/decode_v1.py`. | `65b108fdf1744b4d9e949b66401809928947b9fdf5ec30028579b63ca0763982` |
| Decode | 2.1056x | (pending refresh; was 368 / 1175 at 2.0970; top 50 at 2.2980) | `9dabcd9a13554bad` | `decoder_p36_scored_2.1056.py`: the 2.0970 lineage plus sk2 PDL at 16/32 tokens, early-trigger norm at 16/32, MoE HINT+ERL+FINREG (bit-identical MAIN tweaks), and in-prepare A/B gates that flush L2 by reading (judge-like) instead of writing; the trigger-after-wait variants it carries were rejected by the gate. Check `22240ad3` 2.1178x; scored CI [2.0740, 2.1372]. Branch `argon/decode-next-20261030`, `experiments/top50/decoder_p36_trig.py`. | `449ef16c1e3af46254d6297497d658d6dd7f931d597aef4dd2ed2f62bc854f44` |
| Dense | 1.5278x | 158 / 1411 | `6131bc96427e4486` | `best/verified.py` | `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b` |
| Prefill | 1.9257x | (pending refresh) | `a0c18fe320e9404a` | `prefill_kbab_scored_1.9257.py`: keell0renz's prefill file (Apache-2.0, unchanged kernels) plus achitaan's prepare-time A/B diagnostic (7 variants, none adopted; production path = their default). Scored CI [1.8968, 1.9547]; same-judge check of their unchanged file `9d666aa1` 1.9070x, so this is a tie with their file, not a speedup. Branch `argon/keell-base-20261102`, `kb/exp/prefill_ab.py` (1367fc3). | `1c60aa5a2840586b61cb20ba56e597b832801a05c0522d14905bf86af0a4550b` |
| Prefill | 1.6635x | 179 / 652 | (earlier scored run) | `best/prefill_verified.py` | `93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7` |

MoE/Attention provenance: those rows run kernels already measured as equivalent to the incumbents
(MoE M3 selected nothing; A3's split change was not confirmed against same-night controls).
The higher scores are favorable measurements of the same code, which the board keeps.
The Decode PDL path has source-bound paired layer evidence: the scored prepare saved 3.008 / 3.040 us in the two arm orders with bitwise-equal outputs and caches. Its official one-token sliding case took 75.819 us (79.027 us in the parent check). The scored interval is [2.0443, 2.1066]; it overlaps the parent, and the entire aggregate point increase is not attributed to this one changed case. Failed byte/performance gates retain the parent. Earlier checked and scored sources remain in this folder.

Retained baselines (one per problem) are in `best/`; the full experiment history is on branches `omos/decode-fresh-20261029` and `omos/other-tracks-20261029` (see `NEXT_PLAN.md` there).
