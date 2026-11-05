# Kernel track final + serving track plan (written Nov 5, 2026)

Covers the opencode loop session of Nov 4 (about 17:00 PT to the kernel close at 00:00 PT Nov 5) and the
plan for the serving track. Every experiment row is also in `TRIED.md`; the minute-by-minute log is
`notes/loop-2026-11-04/LOOP_LOG.md`.

## 1. Kernel track: final (provisional) standings

The kernel track closed Nov 4 23:59 PT. The organizers say the scores are not final: they will review
submissions, rerun the top entries several times (median of reruns decides), and post final standings
before the serving track opens.

| Board | Score | Rank at close | Change on Nov 4 |
|---|---:|---:|---|
| **Overall kernel board** (follows Decode) | - | **136** | was 127 earlier that day; others passed us |
| Decode layer | 2.288x | 136 | none (best run `bc0b306e`, incumbent path) |
| Dense (bf16_proj) | 1.598x | 71 | none |
| Prefill | 1.926x | 91 | not worked on (user instruction) |
| Attention decode | **2.679x** | **150** | **up from 2.6536x / rank 185** |
| MoE w4a16 | 2.380x | 309 | none |

Ahead of joeyhlu on every board (his decode 2.277x rank 161, attention 2.606x, MoE 2.312x, dense 1.116x).
Top 50 overall get serving-queue priority; we are not in it.

### The one board gain of Nov 4
Attention scored run `3a16a9026b05414d`: **2.679x** (CI 2.638-2.719). Change: the one-KV-head sliding routes
(v16S `1h-s8`, v32S `1h-s4`) are now the default for those two shapes, guarded by a judge-tolerance and
KV-cache safety check instead of the noisy timing gate that kept dropping them. Across Nov 4 those routes
timed 0.950-0.969 (3-5% faster) in every run. Source: `board_best/attention_kb1hfix_scored_2.6790.py`
(branch `loop/attn-1hfix`, commit e489fa1). Real effect is small (v16S ~2.4% faster); part of the score is run luck.

## 2. What Nov 4 established (details in TRIED.md)

### Where the decode layer's time goes (in-layer ablation, check `7dc880bcc6544992`)
Component critical-path cost with routing held fixed (us): MoE d1S 19.7, d16S 46.1, d32S 84.3, v1S 65.6,
d32F 86.8; pooler+router 13.9-16.8 at every shape; attention 2-7 on sliding shapes, 82-169 on d16F/d32F;
qkv 7-11; unpooler 2-12. If a component were 25% faster: MoE +8.0%, attention +2.4%, pooler+router +2.3%,
qkv +1.3%, unpooler +0.8% on the decode board.

### MoE is mostly weight-stream bound
An instrumented census said consumers were 81-87% of the time, but the counters inflated runtime 76-111%.
The clean, uninstrumented counterfactual (check `6d2c4a40f52548f3`) showed a loads-only kernel saves only
7.4 / 12.8 / 25.2 us at bs16 / bs32 / bs256 (of ~105 / 130 / 220 us): at most ~7-11% of MoE time is
recoverable on the consumer side. A perfect consumer would give roughly +2-3% on decode, not +8%.

### Attention
One fused kernel per case. Full-context cases run at 71-83% of HBM peak. Sliding kernels are latency-bound
(2-36% of peak); inside them prep (QK-norm, RoPE, FP8 cache write) is 30-45% and publication 13-30%, but
every trim tried (warp RoPE, barrier removal, RoPE table, float4 publication exchange, q8 gate reorder) was
noise or slower. The one-head route lock-in is the only attention win.

### Closed on Nov 4 (all lost or tied; see TRIED.md)
Decode: output-gate overlap (1.5-4% slower), KV tile-0 pre-wait, qkv16t_r200 config, full-attention pre-wait
(incumbent already pre-waits; QKV and attention can't co-reside), pooler router packing, combo of near-misses,
VMS MoE at T=32 (0.5-5% slower), publication exchange port, one-head port (failed correctness in decode:
likely a port bug), full-wave qkv/unpooler configs (qkv 1.8-3.2% slower).
MoE: JOINFULL, EXACTDIV, mma.sync verify, TQ64, RSTREAM restart (still 1.8x slower after 4 rounds),
VMS at bs32 (0.6%, did not reproduce 1.3%). Dense: owner epilogue, redesign, slabs, rotcodes (M=1 gain was
mostly harness bias), consolidation (router@8 tail real but ~0.12 us).

### Process lessons (apply to every future check)
- The judge rejects any source that starts processes (subprocess JIT included). Build with `load_inline` in-process.
- The judge runs CUDA 13 (torch cu130): the `_v2`/`_v3` graph APIs do not exist.
- Every `load_inline`: host and nvcc `-std=c++20`, `-gencode arch=compute_90a,code=sm_90a`.
- Prepare is ~140 s for decode; diagnostics must be built at import time or they get skipped for budget.
- Base-vs-base null arms are mandatory: the ABBA harness has a 1-5% bias on small cells.
- The incumbent is not run-to-run bit-repeatable at many shapes: bitwise gates falsely reject; use judge tolerance + identical expert ids.
- Act on results the moment they land (twice ~1 h was lost waiting).

### Spend
About $214 of a $400 cap for the session and its subagents (`python3 .loop/spend.py`). Most went to
hard-fixers on GPT-6.1 Sol xhigh before the switch to cheaper models.

## 3. Serving track: what it is

The second and final part of the challenge, and **the only part with prizes**.

- **Dates:** opens **Nov 10, 9 am PT**; closes **Nov 19, 11:59 pm PT**.
- **Prizes:** $100,000 total: 1st $50,000 + a signed H200, 2nd $30,000, 3rd $10,000, 4th $3,000, 5th $2,000, 6th-10th $1,000 each.
- **Submission:** a git diff against SGLang commit `a55ab7d5` (image `lmsysorg/sglang@sha256:8cfc5a72...`).
  Only files under `python/sglang/`, not `python/sglang/srt/model_loader/`; only `.py .cu .cuh .h .hpp .c .cc .cpp .json`;
  no renames/binaries; at most 400 KB; no added line over 400 chars. `sgl-kernel` binaries are not rebuilt.
- **Model/flags:** `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`, drafter `RadixArk/Muse-Spark-Open-DFlash2`
  (DFLASH, 8 draft tokens), TP8, EP8, `modelopt_fp4`, `--moe-runner-backend marlin`, max 32 running requests,
  CUDA graphs up to bs 32, chunked prefill 16384. The patch may change what the flags do, not the flags.
- **Scoring:** whole 8xH200 node, up to 90 min; multi-turn agentic coding conversations at concurrency
  **1, 4, 16, 32** (3 reps each for 1/4/16, 1 rep for 32); throughput at stock SGLang's interactivity vs stock.
- **Must hold:** exactly 220 tokens per request; answers must match stock nearly as well as stock matches
  itself; time to first token at most 2x stock at each load; model unchanged (8 experts/token, 2048 window,
  every norm); BF16 activations; FP8 KV; DFlash drafter may be made faster but not changed.
- **Limits:** 1 scored run + 3 check runs per day (reset 00:00 UTC). A check runs only the concurrency-16 rung,
  once, rescored over 8 conversations, 35-minute limit.
- **Leaderboard is private** until the end (you see your own scores).
- **Mergeability counts:** SGLang maintainers will review the top patches; sprawling or workload-specific
  patches can be disqualified or asked to be cleaned up.
- Local copy: `okc serve init` was run into `can/serving/sglang/` (outside this repo) with `RULES.md`.

## 4. Serving work so far

### S1: small-batch fused post-expert norm (ready)
- Patch: `serving/s1_postnorm_v2.patch.diff` (7,375 bytes, 3 files; SGLang branch `serve/s1-postnorm`, commit cdabc2c).
  PR text: `serving/S1_PR.md`.
- Change: `MuseSpark1xFp4Experts._forward_marlin` sends batches of fewer than 32 tokens through the existing
  Triton post-expert-norm kernel instead of a ~10-op PyTorch fallback (keyword-only cutoff override; default unchanged).
- Evidence (H200 op level, earlier diagnostic `cd2e95e7`): 3.9-5.3x faster at T=1..31 (33.8-48.3 us -> 6.8-9.2 us
  per MoE layer per step), outputs within 1 BF16 ulp of stock, all-remote rows exactly 0.
- Applies at concurrency 1 every step and at concurrency 4 sometimes; inactive at 16/32, so checks (C=16 only)
  can show correctness but not the gain; only a scored run shows it.
- Unknowns: server-level gain unmeasured; a BF16-ulp difference could flip a generated token vs stock.

### MoE kernel port: design (not built)
Full design: `serving/MOE_PORT_DESIGN.md` (file:line citations into SGLang and our kernel). Key points:
- **Memory is the blocker.** Our board MoE keeps four weight layouts: 3.375 GiB per layer per GPU vs stock's
  0.84 GiB. All layers + stock Marlin for prefill = +202 GiB per GPU (impossible; H200 has 141 GB). Two layouts
  for all layers is still +101 GiB.
- **Bounded proof is feasible:** our kernel on N of ~60 layers, two layouts, ~1.7 GiB per layer (N=8 ~13.5 GiB,
  taken from KV cache). Expected ~1-2% at N=4, ~2-5% at N=8 at concurrency 16.
- **Big gain needs a single-layout redesign** (our kernel reading one shared layout, or replacing Marlin's
  layout everywhere including prefill). Substantial kernel work.
- **Missing safety guard:** the board MoE file lacks the zero-local-work (J==0) hang guard; a server would hit
  it. Must be added in any port.
- **Token matching:** stock discards the up-projection global scale and rounds scales through BF16; the port
  must reproduce stock's numerics, not the ideal ones.
- DFlash verify uses 8 tokens per request, so decode/verify T <= 256: within the shapes our kernel is tuned for.
- Proposed integration point: inside `_forward_marlin`, after the pre-expert norm, before `_marlin_expert_outputs`;
  auto-selected for eligible Muse/Hopper decode/verify (a new runner flag is not allowed).

## 4b. MoE serving idea (the plan for a large serving gain)

**What the kernel track taught us about MoE**
- MoE is mostly limited by streaming expert weights from memory, not by math: the clean counterfactual
  (`6d2c4a40`) found only ~7-11% of MoE time recoverable on the compute (consumer) side.
- Our kernel-board MoE still beats SGLang's Marlin path by 1.6-3x per call (decode1 26 vs 80 us, decode16 93 vs 188,
  decode32 127 vs 207, verify 8/128/256 tokens 65 vs 147, 183 vs 474, 216 vs 618 us), because it streams the weights
  far more efficiently. So the serving win is replacing Marlin's decode/verify MoE with a better streaming kernel,
  not tuning compute.

**The obstacle:** our kernel keeps four copies of the weights (one layout per variant: TINY, MID, MAIN, WGS).
For all ~60 layers that does not fit next to Marlin's copy (see `serving/MOE_PORT_DESIGN.md`: +202 GiB/GPU).

**Options, in order of preference**
1. **Fast kernel that reads Marlin's existing weight layout directly (preferred).** Our streaming design, rewritten so
   its inner loop consumes SGLang's own Marlin-packed NVFP4 weights and scales. Zero extra memory; prefill stays on
   Marlin untouched; most mergeable (a faster small-batch Hopper path for the same weights, auto-selected, no new
   flag); reuses stock's exact scale handling, which makes matching stock's tokens far easier. Cost: rewriting the
   kernel's load/dequant path for Marlin's tile and bit layout.
2. **One shared layout of our own, replacing Marlin's.** Repack once after load into a single layout every variant
   reads, free Marlin's copy (memory = stock). Needs a prefill kernel that reads our layout too: more work, harder to merge.
3. **Bounded proof (stepping stone).** Current kernel on 4-8 layers with two layouts (~1.7 GiB/layer). ~2-5% at
   concurrency 16; proves integration, startup time and token matching on the judge.

**Required in every option:** the zero-local-job (J==0) hang guard (missing in the board MoE file); reproducing
stock's numerics exactly (stock drops the up-projection global scale and rounds scales through BF16); CUDA-graph
capture per batch size; decode/verify T <= 256 (DFlash verifies 8 tokens per request), prefill stays on Marlin.

**Rough upside if option 1 fully works:** MoE is ~30-60% of a decode step at concurrency 16/32 and our kernel was
1.6-2.9x faster per call, so roughly **+15-30% throughput at 16/32**, less at concurrency 1. An estimate from
kernel-board ratios, not a measurement.

**What it takes:** a 1x H200 (~$2.5-4/GPU-hour; ~$100-250 for a week of kernel work) to compare one MoE layer
against stock Marlin output-for-output and time it; then the judge's 3 daily checks (concurrency 16) for the full
server, or a short 8x H200 session (~$20-35/hour) to confirm. Without a GPU, option 1 is a blind rewrite.

**Why it matters beyond the prize:** SGLang maintainers (consulting with Meta) review the top serving patches for
mergeability. A small, measured, mergeable patch (S1 now, option 1 later), plus a public write-up and an actual
SGLang PR after the track, is the strongest visible signal for a Meta ML-infra/inference interview. Confirm the
okc account opted in to sharing email with Meta (`GET /me` -> `rules`).

## 5. Options (decision pending)

1. **Minimal:** submit S1 as the first scored run on Nov 10, look at the score, stop. Low cost, small gain.
2. **Serious attempt:** S1 -> bounded MoE proof (4-8 layers) -> single-layout MoE redesign if the proof
   passes cleanly. Build can start now (submissions only from Nov 10). Realistically needs a rented GPU
   (ideally 8xH200, at least 1xH200 for kernel-level parity) to test token equivalence before spending the
   3 daily checks; without one it repeats the blind loop of Nov 4. Spend would exceed what remains of the $400.
3. **Stop** after the kernel track.

Recommendation recorded at the time: option 1 first, decide on option 2 after S1's first score; if option 2,
start building now so day one is not wasted.

## 6. Where things are

- Ledger: `TRIED.md` (main). Loop contract and log: `notes/loop-2026-11-04/LOOP.md`, `LOOP_LOG.md`.
- Board sources: `board_best/` (+ new `attention_kb1hfix_scored_2.6790.py`), `board_best/BOARD_BEST.md`.
- Experiment branches pushed to origin: `loop/*` (one per experiment, Nov 4).
- Serving: `serving/` on main (S1 patch, PR text, MoE design, log). SGLang checkout + worktrees live outside the
  repo in `can/serving/` (`sglang/`, `wt-s1`, `wt-moe`).
- Spend counter: `.loop/spend.py` (reads the local opencode DB).
