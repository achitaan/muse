# HANDOFF - Muse Open Kernel Challenge (team achitaan), as of Nov 3 2026 ~16:00 UTC

Kernel submissions close **Nov 5 07:59 UTC**. Read this file first, then kb/STATE.md (problem + conclusions) and
kb/APPROACH.md (how we choose work). Every experiment ever run is a row in `TRIED.md` on `main` (the ledger).

---------------------------------------------------------------------------------------------------------------------

## 1. Objective and ground rules (from the user)
- Make our entries outperform the public repos of **keell0renz** (github.com/keell0renz/muse-open-kernel-challenge)
  and **alygg77** (github.com/alygg77/okc-muse-spark-kernels); the overall rank equals our **Decode** rank, so Decode
  is the board that matters. Later target: match/beat **joeyhlu** (Decode 2.3569, #33; #30 bar ~2.37-2.38).
- Build on keell0renz's Apache-2.0 files; keep LICENSE/NOTICE; mark our edits `achitaan`. Never submit their unchanged
  code for score. Readable source only (no base64/zlib/encoded code; lossless *weight* compression is fine).
- **Real kernel speedups only.** The user explicitly rejected "metagaming": no exploiting the routing near-tie
  tolerance to drop experts, no lower precision, no caching / L2 persistence across calls, no noise farming
  (no re-submitting unchanged code to roll the dice).
- Work autonomously; never stop to ask; pivot freely ("one step back, two steps forward").
- Budget cap **$1000** (check `py -3.12 tools/oc_cost.py`; ~$660 spent at handoff). Use Opus (`fixer`) to build;
  let "smart models" (oracle / oracle-lite / council / researcher) plan.
- Record every result in the ledger: `python tools/tried.py add --area ... --idea ... --outcome ... --number ...
  --job ... --source ... [--replace "<row text>"]` (commits TRIED.md to main only). `tried.py search <words>` first.

## 2. How to run things
- CLI: `wsl -d Ubuntu -- "/mnt/c/Users/Achita/Documents/ChatGPT/Muse 2/.tools/okc" check|submit|status <id> --wait --json`
  (`okc problems --json` = daily quota; `okc leaderboard <problem> --json`). Retry on `tls: bad record MAC` / HTTP 530.
- Daily limits per board: 20 checks, 10 scored runs; reset 00:00 UTC. A run has 10 min total; prepare() <= 300 s;
  file < 1,000,000 bytes; only the last 16 KB of stdout survives -> print summaries LAST.
- Score tool: `python kb/muse_score.py --backtest` (exact replay, 79/79 saved jobs) and
  `python kb/muse_score.py --job 1ca0668a --save-us d16S=10 --table` (what-ifs). User variant with history/controls:
  `python kb/muse_score_ref.py --backtest` / `--save-us ...`.
- Predictor / chase tool: `python kb/predict/predict.py all` (report kb/predict/REPORT.md).
- Spend: `py -3.12 tools/oc_cost.py`. Ledger helper: `tools/tried.py` (copy of `.tools/tried.py`).
- Gate harness pattern (copy from kb/exp/decode_pcx.py / decode_wgs4.py / decode_bnd.py): in prepare(), capture the
  real layer as a CUDA graph, restore inputs/stream/KV before every replay, 512 MiB read flush, ABBA both orders,
  two passes, adopt only if median AND paired <= 0.988 (MoE board <= 0.97), identical expert ids / KV, else fall back.
  Decode files near 1 MB: use the comment trim + anchor-preservation scan from decode_pcx.py (a trim once deleted a
  patch anchor and broke prepare: b0caa4c9).

## 3. Repos, branches, worktrees
- Repo `achitaan/muse`. `main`: TRIED.md ledger, board_best/ (byte-identical published bests) + BOARD_BEST.md.
- `argon/keell-base-20261102` (worktree .slim/worktrees/kbase): keell base + Nov 2 work.
- **`argon/decode-20261103`** (worktree .slim/worktrees/decode-1103): everything from Nov 3, all tools, saved runs
  and profiles (this file). Old Oct 23 Dense-sprint docs are archived in docs/archive/oct23-sprint/.

## 4. Where we stand (leaderboard Nov 3 03:20 UTC)
| Board | Our best | Rank | Job | keell0renz same judge | Real gain of ours |
|---|---:|---:|---|---:|---|
| Decode (= overall) | 2.2347 | 232 / 1215 | 1ca0668a | 2.2588 | none (keell kernels; noise) |
| MoE | 2.3797 | 229 / 1339 | ac80911e | 2.3874 | none |
| Attention | **2.6536** | 133 / 1283 | 8f53cd76 | 2.6421 | one-KV-head kernel at verify-32 sliding (~3% on that shape, ~+0.2% board) |
| Dense | **1.5886** | 71 / 1664 | bbde8419 | 1.5598 | CURP 12-bit routes qkv@16/32, pooler@32 (~+0.3%); one-wave routertail router@8/16 (~+0.3%) |
| Prefill | 1.9257 | 51 / 826 | a0c18fe3 | 1.9070 | none (keell default; noise) |
Overall rank 232 / 1888. Unchanged-path Decode noise: 2.2318-2.2607 (7 controls), score sd ~0.53%.
Published on main board_best/: decoder_kbv3_scored_2.2347.py, dense_kbv2_scored_1.5886.py,
attention_kb1h2_scored_2.6536.py (main 1ede07f) and earlier rows.

## 5. Intelligence gathered
- keell0renz's README (Nov 2): their #1 leaderboard entry was "judgemaxxed"; the public repo files are their fair kernels.
- alygg77 builds on vjeux/okc-2026; their Decode (2.2058) loses to keell per shape everywhere except v32S (+1.4%).
- joeyhlu (Joey) shared (kb/exp/JOEY_PLAN.md): engines by T (TINY/TINY2 <=8, VMS 16, VML 32, HYB >=128 for MoE; dense
  "gd" (+DG1 at 32); attention KS1/KS3 sliding with FP16 QK on BF16 queries, HYA full with E4M3 queries, AX1 at d1F),
  fused norm/router/residual + PDL, and his per-shape table. His lead vs ours by load: d16 ~11%, v1 ~7%, d1 ~4.5%,
  v32 ~4.3%, v16 ~2.5%, d32 ~1.6%. His component boards are all BELOW ours (MoE 2.3121, Attention 2.6055, Dense 1.1155),
  so his edge lives inside his Decode file. His earlier "dead ends" list (routing-tolerance policy etc.) was what he
  did NOT keep. Our names map: TINY/TINY2 = same; VMS = keell MID; VML = WGS MAIN; gd = sk2/gv/rq family.
- Top public writeups (Rohan, Anshul, Sapalidis, Naklecha) give process advice only; nothing public explains the
  MoE-board leaders (2.76). Humming (vLLM) small-M policy was the only public mechanism we had not tried (it lost).
- Judge routing seeds change every run: per-shape Decode times move ~6 us per touched expert between runs
  (our d16S across 21 runs: 117.0-132.5 us). Single-run per-shape comparisons carry several % of draw noise.

## 6. Everything tried (chronological). Details, numbers, job ids: TRIED.md on main.
### Nov 2 (keell base, before Joey)
- Decode: v3/v4 (WSI8, CURP routes) tie; gle at T=8 slower; o-weight L2 prefetch 0; routing fused into pooler tail
  2x slower; graph launch-completion edges error; CURP port (f2ab6d4c) flips 1.6-2.3% expert ids; keell newer
  attention port (26331ae5) 0/12; chain preload (f79f3879) 0/8; MoE refit tables (ff8f8f9c..7bae45c5) slower.
- MoE board: partials/snapping/ordering/L2/small-batch, producer decoupling, P2-RS64 phase-B core (1.5x slower).
- Prefill: phase specialization, tile orders, FMHA knobs, A/B harness: keell defaults hold.
- Attention: tuner, BULKRED, wst32, lazy softmax, splits, latency bundle closed; **one-head kernel kept** (v32S).
- Dense: v3 wider CURP, ZipServ 3-bit (11.39 bpw, 2-2.4x slower), simple GEMVs (1.2-10x slower); **CURP routes kept**.
### Nov 2-3 Dense follow-ups
- ZS2 sub-12-bit (ee7d7b81): escapes dominate; 1.6-1.7x slower, closed. v4/v5 (19b6b942, 54fa5e33, 7f045a96,
  85d020de): **one-wave routertail router@8/16 kept** (0.92-0.95); rk2 router@128/256 slower; cg8 slower.
### Nov 3 Decode loop (branch argon/decode-20261103), all closed with measurements
| # | Lane | Checks | Result |
|---|---|---|---|
| A | VMS/VML own MoE engine at 16/32 (A->B->fin, PDL) iters 1-5 | 79687699, 0b6a581c (build bug), 74e3e792, 4c5b2777, c5800fa6, adbd3ba4 | best tie (bs16 paired 0.99, median 0.95); tail split 2x slower; deeper ring/ping-pong slower |
| A-D | VMS inside Decode T=16 (+PDL stage A) | 482a22e2 | 1.008-1.016 slower |
| T1 | VMS at bs1/bs8 | 62adc58a | bs1 12-16% slower, bs8 tie |
| B | Dense-board GEMVs in Decode T=1/8/16 | b0caa4c9 (anchor bug), 8ffd6871 | 0.99-1.025 |
| TL | timeline T=1/8 + byte floor | 8ab7e4a3 | no exposed gap (PDL already overlaps) |
| PCX | deep PDL pre-wait staging + early triggers | b91d6914 | 0.994-1.009 |
| N3 | boundary skip-launch ceiling | 385183f9 | looked like 2.317 but inflated (routing side effect) |
| N3b | coarsened boundary kernels (bit-exact) | 32120a38 | not faster |
| N2 | rq/CURP qkv with FP32 split reduce | ec73a259 | ids fixed, 1.02 slower |
| H | Humming-style producerless GEMM2 | d952458c | 5-36x slower |
| T1R | TINY/TINY2 kill check -> expert wavefront -> B-bank prefetch | baf4b220, 7091a399, aa8fef4f | headroom 5.4/6.7 us found; wavefront 1.27-1.47x slower; B prefetch 1.04-1.13x slower |
| ESC | escape-warp decoupling extended (bit-exact) | 35e0aa5a | 0.998-1.03 |
| MGAP | MoE sliding-vs-full gap | ad1e2831 (timeout), 7f733a2b | gap = routing draw; +6.1-6.5 us per touched expert (HBM floor); tail 1.2-2.2 us |
| ATT | verify-full attention review | (oracle-lite) | consumer-bound under 104-reg cap; no lever |
| MK | persistent megakernel kill check | e8364e54 (build), 05baaec2 | keell chain only 6-9 us above a sync-free byte floor; grid sync 2.9-6.6 us each -> slower |
Planning/consults: oracle (VMS design, boundary design, T1R design), oracle-lite (Decode ideas, boundary fusion,
attention), researcher (Humming / top MoE teams), two councils (kb/exp/COUNCIL_Q2.md, earlier kb/exp/archive/),
offline predictor (kb/predict/REPORT.md: 0/27 written claims realised; isolated wins realise 4/16; gate wins reach
the judge at ~0.45x).

## 7. Conclusions
- keell0renz's Decode is close to the hardware floor for these shapes: small-T MoE runs at the HBM floor per touched
  expert; the post-attention launch chain is within 6-9 us of a sync-free stream; boundaries run at byte time;
  full attention is gather-bound (decode) or consumer-bound (verify); projections are tuned.
- Isolated (board) wins rarely transfer into the Decode layer; always gate in the full layer.
- Our real, reproduced gains are small: Dense (~+0.6%), Attention (~+0.2%).

## 8. If you continue
- Only new hypotheses with a measured pool are worth a check (run a kill-check diagnostic first; timing-only arms).
- Untested ideas left (low expected value): CURP qkv gated by the judge's routing tolerance instead of identical ids
  (f2ab6d4c measured <=1.5% at d32 only); lossless 6-bit scale coding for MoE (~3% bytes, decode cost risk);
  T=16 WGS build with FAST1 pre-issue (council Q2 #3).
- Scored runs: only for materially new, gated files; publish to main board_best/ (byte-identical) + BOARD_BEST.md row
  only if above the board score.

## 9. File map (this branch)
- HANDOFF.md (this), README.md, kb/STATE.md, kb/APPROACH.md, kb/exp/JOEY_PLAN.md (hints + iteration log),
  kb/exp/VMS_BRIEF.md, kb/exp/COUNCIL_Q2.md, kb/exp/IDEAS_predictor_prompt.md, kb/exp/archive/ (old plans).
- kb/v0/moe_v0.py, kb/v1/decode_v1.py: keell base files (+J0 guard, ERL/HINT/FINREG). kb/exp/*.py: every experiment.
- kb/runs/*.json: saved judge results (99 files); kb/exp/profiles, kb/profiles: judge profiles/logs.
- kb/ref/: rules (muse_decode_layer, muse_moe_w4a16), TRIED.md snapshot, KEELL_LESSONS.md, ALYGG_NEGATIVE.md,
  H200_FACTS.md, profiles.
- kb/muse_score.py, kb/muse_score_ref.py, kb/predict/: score replay, what-ifs, predictor.
- tools/: tried.py (ledger helper), oc_cost.py (spend), install-okc.sh, okc-llms.txt, TRACK_NOTES.md.
