# Lane handoff — `omos/decode-fresh-20261029` (draft, updated as results arrive)

Independent lane opened October 29, 2026 from `codex/h200-2-5x-20261028` at
`6b7209fd89fc859ca658e3148b60b05d1052b44e`, in the worktree
`Muse 2/.slim/worktrees/decode-fresh` (git-local exclude only; the main checkout
and its tracked files were not modified). All five `best/` sources are unchanged:

| Track | Retained file | SHA256 (unchanged) |
|---|---|---|
| Dense | `best/verified.py` | `edc61a1e…765b` |
| MoE | `best/moe_verified.py` | `33114871…0d50` |
| Decode | `best/decoder_verified.py` | `d9fe7558…9988` |
| Attention | `best/attention_checked.py` | `9e922c35…412f` |
| Prefill | `best/prefill_verified.py` | `93e57ab4…aae7` |

## Finding: the overall board follows the whole-layer Decode rank

Read-only analysis of the archived leaderboard snapshots (no API calls):

| Snapshot (UTC) | Account overall rank | Account Decode rank | Overall rows identical to Decode order |
|---|---:|---:|---|
| Oct 29 00:21 | 155 | 155 | first 256 rows |
| Oct 29 05:17 | 162 | 162 | first 268 rows |
| Oct 29 18:56 | 207 | 207 | first 314 rows |

`muse_decode_layer/TASK.md` states the overall board "ranks by how much your best
runs together would speed up a decode step". The first deviation in every snapshot
is a team without a Decode entry. For this account, Dense/MoE/Attention gains do not
move the overall rank unless they move the Decode speedup. Decode rank-50 cutoff at
18:56 UTC: 2.1836x (2.148 at 00:21, 2.158 at 05:17).

## Experiments in this lane

### E1 — idle-DRAM L2 weight prefetch (Decode) — rejected

- Add-on `experiments/prepared/decode_l2pf_addon.py`; builder
  `experiments/prepared/build_decode_l2pf.py`; candidate bytes
  `650b53a7498ca259e54933bfcf57790ea74774818077c994eac5c86ad3518abe` (997,990 B).
- First upload `30a63b2e…` (1,003,591 B) was rejected at upload with
  `422 submission is larger than 1000000 bytes`; no job, no quota, no ledger row.
- Check `2a9eba681e444075`: passed 13 tests, **1.9708513x** (CI 1.9413–2.0004)
  vs parent D9 1.9902249x (`9560c915e85f4f50`).
- Frozen gates (`results/decode-l2pf-check-20261029/gates.json`): affected GM
  0.9816 (needed ≥1.02), control GM 1.0218 (in range), track 1.9709 (needed ≥2.0101).
- Mechanism from the profile: the next GEMVs became 0.4–1.7 µs faster (L2 hits),
  but the bandwidth-paced prefetch kernel stayed resident ~10 µs and slowed
  critical-path kernels (attention 9.2→11.6 µs at T=32 sliding; route_fp32<8>
  15.2→17.6 µs). Net loss; line closed.

### A1 — attention sliding split sweep — not retained (one shape won)

- Candidate `0e74765beedf96d82224b147dfd2e8cb68015de670123da94860ce9c7f8f7258`
  (`_TUNED` overrides only); check `d18f02619ca44b65`: passed 16 tests,
  **2.4237347x** (CI 2.3874–2.4693) vs parent 2.5370005x (`8b767ea7b2b74de6`).
- Controls (6 full shapes + verify 32 sliding) GM 1.0040. Normalized per shape:
  decode 1 sliding 32→16 splits 0.883; **decode 16 sliding 16→8 splits 1.055 (win,
  9.31→8.79 µs)**; decode 32 sliding 8→4 0.869; verify 1 sliding 33→16 0.775;
  verify 16 sliding 8→4 0.854.
- Decode transfer (D2) skipped: the winner would move Decode's 8 µs decode-16 sliding
  attention by ~0.4 µs (~0.05% of the Decode track).
- Follow-up A2 frozen, not launched: `candidates/attention_split_a2.py`
  (`513af2884589912f9d04b31c34d26eff39e9bffc5639ad0e42240bc24dca851f`) confirms
  decode16=8 and probes 16 splits for decode 32 and verify 16 sliding;
  gates in `results/attention-split-a2-20261029/gates.json`.

## Dense, MoE and Prefill: candidate assessment (offline, no checks spent)

Evidence gathered from the retained per-shape profiles before choosing where the
paced checks go (60 min spacing leaves three checks after E1 in this window):

- **Dense** (`6131bc96427e4486`, 1.5278x): weakest loads verify 16/32 (1.30/1.25)
  and decode 32 (1.39). M=128/256 pooler/unpooler and M=256 QKV run cuBLAS nvjet
  (18.4–25.6 µs, 1.3–1.8x floor). The only unused M=256 QKV family config,
  `(64, 36, 256, 3, 4, 219, 5)`, was already timed in-prepare on the judge at
  38.9–40.5 µs vs 18.0–18.2 µs (jobs `1f6e5d17cf8c4c75`, `f873dc6fd6e948a2`), and
  both fixed M=128 cluster families lost 4–7%. Remaining gaps (router 2.2–2.95x,
  sliced unpooler 2.2x floor) are single-kernel latency in tuned families; no
  untested configuration-level candidate with a credible gain was found.
- **MoE** (`d600231c8a8244cb`, 2.3654x checked): one persistent kernel; all builds
  use `MAIN_NS = 3` with 36 KB units, already ~14 MB in flight at M=1, so deeper
  stages do not raise bandwidth; two-resident-block, cache-priority, skinny16,
  epair and cluster variants are closed. Extending the MID split build to M<=32 is
  capacity-safe (jobs <= 64 < JMAX 88) but re-streams hot experts with more than 8
  tokens (~15–25% more traffic at equal measured efficiency: MID 2.6 TB/s at
  decode 16, MAIN 2.7 TB/s at decode 32), so it is not expected to win.
- **Prefill** (`9ca79c7f676f4b63`, 1.6393x checked; excluded from overall): first-turn
  sliding (weight 138) is MoE 40%, cuBLAS GEMMs ~77% of BF16 peak, and two FP32
  residual norms at ~4 TB/s (bandwidth-bound on the FP32 stream); no
  configuration-level lever.

Conclusion: clear gains on these three tracks need structural kernel work
(e.g. a lossless-compressed MMA GEMM for M>=128, a fused small-M MoE phase
schedule), which cannot be validated in this window without spending checks on
low-prior guesses. The remaining paced checks go to the attention split sweep
and its Decode transfer, where per-shape leverage is measured.

## Second lane: `omos/other-tracks-20261029` (user request: work the other tracks)

Worktree `.slim/worktrees/other-tracks`, branched from the same handoff commit
`6b7209fd89fc859ca658e3148b60b05d1052b44e`. Notes: `OTHER_TRACKS_NOTES.md` there.
Both candidates use prepare-time timing on the judge's GPU and keep the incumbent unless
another option wins by >=3% with matching output.

### M1 ? MoE build selector by token count ? measurement only, not retained

- Candidate `95d82f5b552e6266a6f8a7826db59a9533b1b19c0fc5133c1ccde21df0860e4d`
  (184,785 B; parent `33114871?`). Check `d746cb32bcc045f4`: passed, **2.3763335x**
  (CI 2.3407?2.4120) vs parent checked 2.3654374x (`d600231c8a8244cb`).
- Selector log (?s, sum of medians over three routing draws shaped like the task inputs):
  M=1 TINY 85.98 (incumbent) / TINY2 84.42 / MID 87.97 / MAIN 129.66;
  M=8 TINY2 189.47 (incumbent) / TINY 191.55 / MID 196.99 / MAIN 224.42;
  M=16 MID 244.38 (incumbent) / MAIN 268.58; M=32 MAIN 357.60 (incumbent) / MID 380.58.
  Nothing selected, so the timed path equals the parent; the +0.46% is run noise.
- Finding: the retained per-token-count build choice is already the fastest existing
  build on the judge's H200 (TINY2 at M=1 is 1.8% faster on synthetic routing, below gate).

### P1 ? Prefill FA3 num_splits / pack_gqa selector ? measurement only, not retained

- Candidate `6d1f43575637910e3d047c496b5833f28ac45c6f10e6f1558b63f4c3531665e0`
  (265,601 B; parent `93e57ab4?`, checked 1.6392908x `9ca79c7f676f4b63`);
  gates `results/prefill-fa-select-check-20261029/gates.json` (other-tracks lane).

- P1 check `bb9ce14d3ad8490b`: passed, 1.6564053x (CI 1.6316?1.6813) vs 1.6392908x; FA3's
  heuristic was within 2.1% of every alternative on all eight shapes, nothing selected; noise.

### M2 ? MoE TINY2 for 1..4 tokens ? rejected

- Candidate `6e1643db357ad2f862b7e5a9c7bcec4a6fd88046777dec8bee179c64aa09e352`; check
  `3158fdc490ed46d1`: passed, 2.3593403x; decode 1 25.77 vs 24.85 ?s (0.962 normalized,
  controls 1.002). The synthetic-routing edge did not transfer; TINY<=4 stays.

### A2 / A3 ? attention decode 16 sliding n_splits 16?8 ? retained in this lane

| Run | Job | Track | decode16 sliding ?s | normalized |
|---|---|---:|---:|---:|
| A1 (sweep) | `d18f02619ca44b65` | 2.4237347x | 8.79 | 1.055 |
| A2 (confirm + probes) | `a359cf6f0b8b4760` | 2.5113048x | 8.62 | 1.085 |
| A3 (single change) | `8347b4ca3b1e42e5` | **2.6123215x** (CI 2.5731?2.6685) | 8.44 | 1.096 |

- A3 candidate `candidates/attention_split_a3.py`,
  `ada6df950cdb54aae711929b6e45f0b5cb1b365acd0bcaaac831394fc916d1e1`: one `_TUNED` entry
  `(16, 1, True, 1): dict(n_splits=8, nc=1)` over `best/attention_checked.py` (9e922c35).
- A2 probes closed the other direction: decode 32 sliding 16 splits 0.691, verify 16 sliding
  16 splits 0.927 (A1: halving lost 0.869/0.854); 8 splits are local optima there.
- Most of A3's +3.0% track change is run variance (unchanged full-context shapes ran
  2.8?3.9% faster than in the parent run); attributable gain ? +0.25?0.5%.
- Not published (scored submissions were not authorized); `best/attention_checked.py` unchanged.
  Decode transfer not checked: ?0.7 ?s of 138 ?s on one Decode shape, below resolution.

### A4 - attention probes on top of A3 - probes rejected, A3 setting won a fourth time

- Candidate `3e8ff2b3e8676c1756ad4072bc178ede9bc816e7802d8de228d299fa1b5ae260`; check
  `1ff88c24d1ef4932`: passed, 2.3880791x; control GM 0.9871.
- Normalized per shape: decode 16 sliding n_splits=8 (A3 setting) 1.0992 (4th win: 8.58 us vs
  9.31); probes decode 1 full nc 8->4 0.9409, decode 16 full flat 264->396 0.8332, verify 1 full
  252->132 0.9157, verify 1 sliding coop 1->0 0.8565. All four probes rejected.

### D2 - Decode transfer of the A3 split - not retained

- Candidate `58931ccd297e6bba3fa8a593b37edb63494ac9ee9676138bf3c781ccdf0af424` (989,060 B,
  `_native_attn__TUNED` gains `(16, 1, True, 1): dict(n_splits=8, nc=1)` over D9); check
  `1245828eda2e4d54`: passed, 1.9729413x vs parent 1.9902249x.
- decode 16 sliding 138.35 -> 133.18 us (1.039), but unchanged full-context controls moved
  +0.3..+4.5% and other sliding loads -3.7..+10% in the same run: not separable from drift.

## Authorization and quota record

- 60-minute pacing applied until 22:11 UTC; afterwards the user authorized more frequent free
  checks until 00:00 UTC (no scored submissions, no merges). One job at a time throughout.
- Live quota read 22:11 UTC: checks left Dense 18, Attention 15, MoE 11, Decode 16, Prefill 20;
  scored 10 each unused. After this session's later checks: Attention 13, MoE 10, Prefill 19.
- Checks used by these lanes on October 29: E1, A1, M1, P1, A2, A3, M2 (seven). No scored run.

## Status against the objective

- Decode (overall-rank driver): no improvement; top 50 needs ?+9.7% (2.1836x cutoff) and the
  remaining gap is structural (small-kernel latency, MoE phase schedule), not configuration.
- Attention: one confirmed shape-level improvement (decode 16 sliding ?9%), small at track level.
- MoE, Prefill: existing configuration choices measured as already optimal on the judge GPU.
- Dense: no candidate; cuBLAS algorithm/layout search and c12 variants were already covered.
