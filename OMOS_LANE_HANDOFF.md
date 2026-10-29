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

### A1 — attention sliding split sweep — pending

See `results/attention-split-sweep-20261029/`.

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
