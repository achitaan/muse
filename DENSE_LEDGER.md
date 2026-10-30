# Dense (`muse_bf16_proj`) Anti-Loop Ledger

- **Board best:** `1.5278x` (`best/verified.py`, SHA `edc61a1e…765b`, job `6131bc96427e4486`, rank 158, `score_us = 3162.0`)
- **Dense top-50 cutoff (snapshot, not refreshed):** `1.5613x` → needs about +2.19% overall
- **Branch:** `omos/dense-top50-20261030` (worktree `.slim/worktrees/dense-top50`, from `origin/main` `fcd0e50`)
- **Rules**
  1. Log every experiment here **before** launch (hypothesis, gate, frozen artifact) and **after** (job ID, raw verdict, RETAINED or CLOSED).
  2. Never retry anything in Section 1.
  3. Check budget is a ceiling, not a schedule. An inconclusive result keeps the incumbent and does **not** trigger more configs.
  4. No automatic scored submission and no automatic `main` update. Each needs its own user decision.
  5. Timing predictions and L2/HBM byte counts are **hypotheses** until measured. Logical repeated requests ≠ HBM transfers. Byte counts must include escape metadata, padding and activation reloads.
  6. **Every result that slowed anything down goes into Section 1b ("Slowed it down") immediately**, with how much slower, the job ID and raw paired numbers. Losers are never re-run.

Score model: `G = (Π_i T_i,old / T_i,new)^(1/6)` over 6 loads. 1 µs saved ≈ decode1 +0.47%, verify1 +0.46%, decode16 +0.36%, decode32 +0.30%, verify16 +0.25%, verify32 +0.20%.

---

## 1. Closed — DO NOT REPEAT

| ID | Idea | Shapes | Evidence (job) | Result | Why closed |
|---|---|---|---|---|---|
| D3 | cuBLASLt heuristic search | (128,o) | `7287935eb7d34f7b` | `ol_projection` 10.75 µs beat all 32 algos (best 11.10) | Custom TMA kernel already faster |
| D7/D8 | cuBLASLt descriptor neighborhoods (split-K, stages, swizzle, cluster, exhaustive heuristics) | unpooler/qkv_gate/pooler M128,256; o M256 | `96bc324b987a4c44`, `e793260539cb4877` | 527 variants; best +2.0% (128,unpooler), +1.3% (128,pooler); 0 passed the 3% gate | `tt_lt__LT_H200_FIXED` is the cuBLASLt optimum. **All cuBLASLt tuning closed.** |
| D5 | olayer tile sweep | o M1/8/16/32 | `3b13fab22b2b4480` | 9.25/9.44/9.82/10.11 incumbent vs ≥10.14 | c12 GEMV reads 25% less |
| D6 | olayer tile sweep | o M128/256 | `4e8fda34635040e6` | 25 configs; none beat (128,128,4,3,1,1) or `rl_early_projection` | Exhausted |
| D1 | Whole-row Triton GEMV (no split-K, uncompressed) | router M1/8, unpooler M1/8 | `4edf30f7aa6142e7` | 8.26–16.90 vs 7.97–8.51 µs | No split-K / no c12 |
| WG | Unmodified `m32w_gemm` K-partition sweep (`_achita_wg`) | (16,qkv_gate),(32,qkv_gate),(32,unpooler) | Every run of `best/verified.py` | 18.50 vs 13.89, 18.50 vs 15.52, 22.05 vs 20.26 → WG_KEEP | `wait_group 0` + 2-barrier `ld.shared::cluster` epilogue. Dead sweep removed in S1 candidates. |
| CL | Fixed uncompressed BF16 cluster GEMMs (Vjeux ports) | (128,qkv_gate),(128,pooler) | `f026c3b14a864be0` | 3.7–7.0% slower | 16-bit at 1 CTA/SM cannot win |
| QB | qb config `(64,36,256,3,4,219,5)` | (256,qkv_gate) | `1f6e5d17cf8c4c75`, `f873dc6fd6e948a2` | 38.9–40.5 vs 18.0 µs | Severe regression |
| S1 | v18s_gemv RED3 `(1,8,4,2,3)` and RED0 `(1,16,4,1,0)` for sliced unpooler | (1,unpooler),(8,unpooler) | `f03ec089ee8d451b` | **SLOWED DOWN 5–10%** vs incumbent smtrace, 0–1/16 paired wins | KU=4 / fewer warps does not beat smtrace's KU=2 S=4/S=2 at this size; do not retry v18s geometries for the sliced unpooler |
| S2 | Unmodified m32w c12 WGMMA at BN=128 (WK 2/4, split=1, no cluster) | (128, unpooler) | `a5bd287a677c4fa9` | **SLOWED DOWN 18% (WK2) / 3% (WK4)** vs cuBLASLt | per-warpgroup `wait_group 0` serializes decode and MMA; do not retry unmodified m32w at M≥128 |
| S2b | Pipelined m32w (wait_group 1, 2 A banks), BN=128, WK 2/3 | (128, unpooler) | `825b34dd178a4138` | **SLOWED DOWN 5% (WK3) / 16% (WK2)** | MMA serialization is not the limiter; do not retry m32w pipelining variants at M=128 |
| LOCK | pooler M1/8/16, unpooler M16 | 4 shapes | profile `e793260539cb4877` | 13.14–13.56 µs, already below the 13.99–14.06 BF16 floor | **LOCKED** |

## 1b. Slowed it down — results that made Dense SLOWER (never retry)

Ratio = incumbent time / candidate time; below 1.00 means the candidate **slowed it down**.

| When | Experiment | Shape(s) | Incumbent | Candidate | How much slower | Job |
|---|---|---|---|---|---|---|
| 2026-10-30 | **S2b** pipelined c12 WGMMA (Section 3b), BN=128, WK=3, Q=4 | (128, unpooler) | cuBLASLt ≈ 24.4 µs graph / 18.7 µs kernel | ≈ 25.5 µs graph | **SLOWED DOWN ~5%** (ratio 0.954, 4/16 wins) | `825b34dd178a4138` |
| 2026-10-30 | **S2b** pipelined c12 WGMMA, BN=128, WK=2, Q=6 | (128, unpooler) | same | ≈ 29.0 µs graph | **SLOWED DOWN ~16%** (ratio 0.845, 0/16 wins) | `825b34dd178a4138` |
| 2026-10-30 | **S2** m32w c12 WGMMA, BN=128, WK=2, Q=3 (split=1, no cluster) | (128, unpooler 8192×4096) | cuBLASLt `nvjet 128x64` ≈ 24 µs graph / 18.6–19.4 µs kernel | ≈ 29.7 µs graph | **SLOWED DOWN ~18%** (ratio 0.822, 0/16 paired wins) | `a5bd287a677c4fa9` |
| 2026-10-30 | **S2** m32w c12 WGMMA, BN=128, WK=4, Q=2 (split=1, no cluster) | (128, unpooler) | same | ≈ 24.5 µs graph | **SLOWED DOWN ~3%** (ratio 0.9725, 5/16 paired wins) | `a5bd287a677c4fa9` |
| 2026-10-30 | **S1** v18s RED3 `(1,8,4,2,3)` (S=2 cluster, intra-CTA reduce + 1 DSMEM push) | (1, unpooler 1024×4096) / (8, unpooler) | smtrace ≈ 7.8 / 8.1 µs graph (3.2 / 3.3 µs kernel) | ≈ 8.7 / 8.8 µs graph | **SLOWED DOWN ~10% / ~8%** (ratio 0.899 / 0.916, 0/16 wins) | `f03ec089ee8d451b` |
| 2026-10-30 | **S1** v18s RED0 `(1,16,4,1,0)` (no cluster) | (1, unpooler) / (8, unpooler) | same | ≈ 8.3 / 8.4 µs graph | **SLOWED DOWN ~7% / ~5%** (ratio 0.934 / 0.951, 0–1/16 wins) | `f03ec089ee8d451b` |
| earlier | Unused qb config `(64,36,256,3,4,219,5)` | (256, qkv_gate) | 18.0 µs | 38.9–40.5 µs | **SLOWED DOWN ~2.2×** | `1f6e5d17cf8c4c75`, `f873dc6fd6e948a2` |
| earlier | Unmodified m32w K-partition sweep (`_achita_wg`) | (16,qkv_gate) / (32,qkv_gate) / (32,unpooler) | 13.89 / 15.52 / 20.26 µs | 18.50 / 18.50 / 22.05 µs | **SLOWED DOWN 33% / 19% / 9%** | every run of `best/verified.py` |
| earlier | Whole-row Triton GEMV (D1) | router M1/M8, unpooler M1/M8 | 7.97 / 8.48 / 8.45 / 8.51 µs | 8.26 / 12.90 / 8.61 / 16.90 µs | **SLOWED DOWN 2–99%** | `4edf30f7aa6142e7` |
| earlier | Uncompressed BF16 cluster GEMMs (Vjeux ports) | (128, qkv_gate), (128, pooler) | qb / cuBLASLt | — | **SLOWED DOWN 3.2–7.0%** | `f026c3b14a864be0` |
| earlier | olayer tile sweep (D5) | o M1/8/16/32 | 9.25 / 9.44 / 9.82 / 10.11 µs | ≥ 10.14 µs | **SLOWED DOWN up to ~10%** | `3b13fab22b2b4480` |
| earlier | olayer tile sweep (D6) | o M128 / M256 | 10.75 / 8.78 µs | 10.82 / 12.90 µs | **SLOWED DOWN 0.7% / 47%** | `4e8fda34635040e6` |
| earlier | cuBLASLt heuristic search (D3) | (128, o) | 10.75 µs | 11.10 µs best | **SLOWED DOWN ~3%** | `7287935eb7d34f7b` |

Not slower but **too small to keep** (below the 3% gate): cuBLASLt neighborhoods D7/D8 (best +2.0% on (128,unpooler), +1.3% on (128,pooler)).

What S2 taught us (evidence, not a retry license): going from 2 to 4 warpgroups moved the ratio from 0.822 to 0.973. Overlapping decode with MMA is what limits this kernel: each warpgroup's `wait_group 0` serializes it. The unmodified m32w kernel at M=128 is closed. Only a genuinely pipelined kernel, with an explicit register/smem lifetime schedule, would be a new hypothesis.

What S2b taught us: the pipelined kernel (verified in SASS) moved WK2 only from 0.822 to 0.845, and WK3 reached 0.954. **Per-warpgroup MMA serialization is NOT the main limiter.** Performance tracks the number of warpgroups, i.e. weight bytes in flight: WK2 ≈ 24 KB/SM, WK3 ≈ 36 KB, WK4 ≈ 48 KB. It does not track x stages: WK2/Q6 has the most x in flight and is still slow. Do not retry pipelining tweaks of m32w at M=128. Next: **measure** which stream binds (S2c ablation diagnostic) before building any new mechanism.

## 2. Open targets (from profile `e793260539cb4877`)

| Shape | Kernel today | µs | c12 floor | Status |
|---|---|---:|---:|---|
| (1, unpooler 1024×4096) | `smtrace_gemv` (1,8,2,4) RED2 | 4.04 | 1.31 | **S1 (active)** |
| (8, unpooler 1024×4096) | `smtrace_gemv` (1,16,2,2) RED2 | 4.30 | 1.33 | **S1 (active)** |
| (128, unpooler 8192×4096) | cuBLASLt fixed BF16 | 18.77 | 11.14 | S2: c12 WGMMA reuse, split=1, no cluster |
| (128, pooler/qkv_gate/o) | cuBLASLt / qb / ol (BF16) | 19.82 / 13.81 / 6.97 | — | Deferred until S2 shows evidence |
| (32, unpooler/o/qkv_gate/pooler) | m32w split=2 / ol BF16 / m32x | 16.51 / 5.81 / 11.93 / 16.23 | — | Deferred |
| (1/8/16, o) | smallm2 / v18s / m16trace | 5.31 / 5.07 / 5.29 | ~2.65 | Deferred (small-O sweep) |
| router all M | rr / routertail / rwdl (FP32 / hi-lo) | 3.70–7.98 | — | **Deferred:** FMT=3 needs confirmed FP32 router contract first |
| M256 all layers | cuBLASLt / rl_early | 8.78–24.11 | — | Deferred (shared-weight architecture) |

## 3. Narrowed order (after review)

1. **S1:** sliced unpooler M1/M8 only, two `v18s_gemv` geometries:
   - `(WR,WK,KU,S,RED)=(1,8,4,2,3)`: S=2 cluster, intra-CTA reduction, then one DSMEM push per CTA
   - `(1,16,4,1,0)`: no cluster, single shared-memory reduction
2. **S2 (only after S1 is resolved):** M128 unpooler only. Compressed-weight reuse across 2×64-token tiles, split=1, **no inter-block cluster reduction**, no `st.async` owner-slice epilogue. Prerequisites before any check:
   - Explicit register/smem lifetime schedule for any `wait_group 1` overlap. No register bank may be rewritten while an in-flight WGMMA reads it (undefined behavior otherwise).
   - Register budget recomputed: 2×256 threads × R ≤ 65,536 → R ≤ 128. An extra 16-register decoded bank (132–140) would **not** fit.
   - `m64n64k16` covers K=16, so K=64 × 2 token tiles = **8** WGMMA instructions per K-step.
   - Query active-cluster/CTA occupancy; do not assume 128 CTAs = one wave.
3. **Expand** a winning mechanism to another projection, or add the cluster epilogue (with a full receiver-init / completion / visibility / safe-exit protocol), only after evidence.

### Correctness gates (both required)
- **Representation:** pack→unpack bit-exact vs original BF16 bits, including signed zero and escapes; unsupported encodings fall back (`v18s_pack_format` returns None → incumbent).
- **Computation:** one-hot exactness and FP64 random/spike rows (`v18s_plan`), plus the full official checker on the job (per-row, finite, dtype, shape).

### Timing gate
Balanced ABBA paired cold-L2 CUDA-graph replays, 16 pairs, raw samples printed. Select only if the median ratio is ≥ 1.03 **and** ≥ 75% of pairs favor the candidate. A selection must be confirmed by a separate run of the frozen selection.

---

## 3c. Direction review (2026-10-30, after S1/S2/S2b/S2c)

**Evidence base:** per-shape medians of 4 identical-runtime checks; leader 1.6498; top-50 1.5613; ours 1.528.

- **M ≥ 128 c12 WGMMA: DEPRIORITIZED.** S2c shows the decode + register-A WGMMA path alone costs ≈ 14.3 µs at (128, unpooler), against cuBLASLt's 18.7 µs total. Even perfect memory overlap would leave at most ≈ 1% of score, with high effort. Do not start new M≥128 c12 kernels without a fundamentally cheaper decode/MMA path.
- **Next: decode-32 load (weakest small-M load, 1.38×).**
  - **Excess over the memory model:** qkv +3.7 µs, unpooler +3.2, pooler +2.8, router +2.2 (c12 hi+lo), o +1.4. That's ≈ 3.9% of score in total.
  - **Hypothesis H32:** v18s-family GEMV kernels reload x per warp via `ld.global.nc` (no smem staging). At M=32 that is 74 MB (qkv) and 128 MB (pooler) of x versus 28–50 MB of weights.
  - **Test first, at zero routing risk:** the existing v18s `DIAG & 2` ("no x loads") on the M32 pooler/qkv configs. Only if no-x approaches the weight-only model (≈ 13.5 / 8.3 µs) build an x-in-smem / larger-row-tile GEMV.
- **Parallel low-risk item: router c12 hi+lo (V18S FMT 3) at M ≤ 32.**
  - **Contract:** TASK.md allows a hi+lo split "at any m" plus lossless compression, and M8/M16 already run hi+lo on the board.
  - **Expected:** −25% router bytes, ≈ 0.5 µs per shape, ≈ 0.8% of score. The probe-row (0.1%) and router (0.01%) limits are checked in prepare.
- **Unpooler M32 (m32w split 2) is not x-bound** (82 MB of traffic); it shares S2c's decode/WGMMA cost. Treat it with the M≥128 family.

## 3b. S2b — pipelined c12 WGMMA lifetime schedule (required before any check)

Kernel: `m32w_gemm` body with `wait_group 0` removed from the MMA helper. Per warpgroup, with step `i`, A-register bank `b = i & 1` (`PF = 2`, so `b` is the compile-time unrolled index `p`) and x stage `s = i % Q`:

| t | Action at step i | Precondition (why it is safe) |
|---|---|---|
| t0 | decode `raw[p]` → `A[b]` (16 regs) | last reader of `A[b]` was WGMMA group `i-2`; it is complete because step `i-1` ended with `wait_group 1` |
| t1 | global load of step `i+PF` into `raw[p]` | `raw[p]` was consumed at t0; plain registers, not read by WGMMA |
| t2 | escape patch of `A[b]` | before the WGMMA that reads it is issued |
| t3 | `mbarrier` wait `full[s]` (parity `(i/Q)&1`) | TMA for step `i` landed |
| t4 | `wgmma.fence`; 4× `m64n128k16` (K=64 = 4 instructions per 64-row tile); `commit` → group `i` | `acc` is touched only by WGMMA from here until the final `wait_group 0` |
| t5 | `wait_group 1` | afterwards group `i-1` is complete, so `A[b^1]` and x stage `(i-1)%Q` are free |
| t6 | TMA for step `i-1+Q` into stage `(i-1)%Q` | that stage's last reader (group `i-1`) completed at t5 |
| end | `wait_group 0`, then `__syncthreads`, `mbarrier.inval`, smem epilogue | all groups complete before smem is reused for the reduction |

- **Overlap gained:** the decode of step `i+1` (t0) runs while group `i` is in flight.
- **Cost:** x prefetch distance drops by one stage, so Q ≥ 3.
- **Registers:** +16 (second A bank). Measured with ptxas (sm_90a, NVRTC 13.0) before any check; an instance is excluded if it spills or if ptxas reports wgmma serialization (C7515).
- **Register budget:** 1 CTA/SM. WK=4 means 512 threads, so at most 128 registers; WK=3 means 384 threads, at most 170; WK=2 means 256 threads, at most 255.
- **Unchanged from the S2 baseline:** split=1, no cluster, no `st.async`, and grid 128 × 1. Occupancy is reported by the job, not assumed.

## 4. Experiment log

| Step | Date | Hypothesis / change | Offline evidence | Official job | Verdict | Status |
|---|---|---|---|---|---|---|
| S0 | 2026-10-30 | Worktree + ledger; drop dead `_achita_wg` sweep from candidates | SHA of parent verified | — | — | done |
| S1-pre | 2026-10-30 | Two unpooler geometries compile cleanly at exact NTMAX | NVRTC sm_90a rc 0: RED3 `(1,8,4,2,3)` 72 regs, 0 spill; RED0 `(1,16,4,1,0)` 64 regs (ptxas heuristic, no cap flag), 8 B stack / 8 B st / 16 B ld spill. `results/s1-offline/s1-resources.txt` | — | compile OK | done |
| S1 | 2026-10-30 | Lower reduction/coordination overhead at sliced-unpooler M1/M8 via v18s RED3 `(1,8,4,2,3)` / RED0 `(1,16,4,1,0)`. Candidate `candidates/dense_s1.py` sha `4394ff32…a760` | pack bit-exact (455 escapes), one-hot + FP64 pass, agreement vs incumbent ≤ 6.8e-7 | check `f03ec089ee8d451b` passed, 1.5315x (CI 1.509–1.555), all S1_KEEP | paired base/cand: M1 0.899 (0/16 wins), 0.934 (0/16); M8 0.916 (0/16), 0.951 (1/16). **SLOWED DOWN: both layouts 5–10% slower** | **CLOSED** |
| S2 | 2026-10-30 | M128 unpooler: exact-c12 m32w_gemm WGMMA, BN=128 tokens/CTA (weight decode reused for 128 tokens), split=1, no cluster; primary `(128,64,2,2,3)`, alt `(128,64,4,2,2)`. Candidate `candidates/dense_s2.py` sha `d87ab665…da33`, gates `results/dense-s2-check-1/gates.json` | NVRTC: WK2 135 regs/256 thr, WK4 125 regs/512 thr, 0 spill. Target scenario 18.6 → ≤ 16 µs (hypothesis only) | check `a5bd287a677c4fa9` passed, 1.5299x (CI 1.507–1.553), S2_KEEP | pack bit-exact (3329 escapes), one-hot + FP64 pass; paired base/cand: WK2 0.822 (0/16), WK4 0.9725 (5/16). **SLOWED DOWN 18% / 3%** | **CLOSED** |
| S2b | 2026-10-30 | M128 unpooler: S2 kernel **pipelined** per Section 3b (no-wait MMA, 2 A banks, `wait_group 1`, lagged x refill). Primary `(128,64,3,2,4)`, alt `(128,64,2,2,6)`; `(128,64,4,2,3)` excluded (spills at the 128-register cap). Candidate `candidates/dense_s2b.py` sha `cba61449…6d5d`, gates `results/dense-s2b-check-1/gates.json` | NVRTC: 151 regs, 0 spill, 0 serialization warnings. SASS: `DEPBAR.LE gsb0, 0x1` per step, banks R88–103 / R104–119 | check `825b34dd178a4138` passed, 1.5200x (CI 1.497–1.543; incumbent path unchanged), S2B_KEEP | pack bit-exact (3296 escapes), one-hot + FP64 pass; paired base/cand: WK3 0.954 (4/16), WK2/Q6 0.845 (0/16). **SLOWED DOWN 5% / 16%** | **CLOSED** |
| S2c | 2026-10-30 | **Diagnostic only** (no routing change): S2b kernel WK3 with compile-time ablations (1 = no weight loads, 2 = no x TMA, 4 = no WGMMA; masks 0–6) timed round-robin vs incumbent, to find the binding stream at (128, unpooler). Candidate `candidates/dense_s2c_diag.py` sha `e45dd8be…b3c4`, gates `results/dense-s2c-diag-1/gates.json` | 0 spill all masks; SASS confirms ablations (LDG 14→2, UTMALDG 6→0, HGMMA 8→0, PRMT 39 kept) | v1 check `cc7bc18615274276` passed 1.5291x but the diagnostic aborted: `cudaErrorStreamCaptureInvalidated` (unlocalized, no timings). v2 (sha `7846c837…a79e`, `results/dense-s2c-diag-2/`): sequential builds, eager rc+sync per kernel, thread_local capture, eager fallback | v2 check `2a940471cf4646b0` passed 1.5075x (noise; runtime unchanged): incumbent and abl0 rc 0, **abl1–6 rc 1** (cudaErrorInvalidValue). Root cause: identical `launch_t`/`m32w_gemm` instantiations in 7 modules share GCC's process-wide unique function-local static `ready`, so modules 2–7 skipped `cudaFuncSetAttribute`. v3 (sha `1a03ec01…b632`, `results/dense-s2c-diag-3/`) renames both per module | v3 check `44254e1c65d94cf1` **timed out** (passed false, check only): abl0/abl1/abl2 rc 0, then abl4 (no WGMMA) hung on first launch. v4 (sha `6f0b1196…b602`, `results/dense-s2c-diag-4/`): masks 0,1,2,3 only | v4 check `f5749b996f2e4fe0` passed 1.5228x (runtime unchanged). Graph medians (16 rounds, cold L2): incumbent cuBLASLt 22.98, full 24.48, no-W 20.96, no-x 22.53, **no memory (decode+WGMMA only) 18.56 µs**, i.e. ≈ 14.3 µs kernel. Removing x saves ≈ 2.0 µs and removing W saves ≈ 3.5 µs, but the compute floor alone is 14.3 µs vs cuBLASLt's 18.7 µs total | **x-traffic hypothesis REFUTED as the main M128 limiter**; c12 decode + register-A WGMMA is itself too slow | **CLOSED (diagnostic done)** |
| S3d | 2026-10-30 | **Diagnostic only** (no routing change), testing hypothesis H32 (Section 3c): at M=32, per-warp x reloads limit the GEMV kernels. Times incumbent vs v18s proxy at the incumbent splits, qkv `(1,4,8,4)` and pooler `(1,8,16,1)`, FMT 1 MT 4, with DIAG 0 (full, verified by v18s_plan) and DIAG 2 (no x loads), ABBA paired cold-L2 graph replays, 16 pairs, raw samples. Interpretation: if DIAG0 ~ incumbent (proxy valid) and DIAG2 reaches the weight-only model (pooler ~13.5 us, qkv ~8.3 us), H32 is confirmed; else REFUTED. NVRTC: 4 instances 56–101 regs, 0 spill. Candidate `candidates/dense_s3d_diag.py` sha `9ef78914…36d4` | first upload failed (`tls: bad record MAC`); job list showed it was not registered, so the intent was renamed and the check relaunched as `b9e1212b1514476b` (02:57Z) | check passed 1.5096x (runtime unchanged). Graph medians, ABBA 16 pairs: **qkv** incumbent 15.81, proxy full 17.28 (0.918, proxy 9% slower than m32x_q), proxy no-x 14.82 (1.067 vs incumbent); **pooler** incumbent 22.14/19.74 (drift between windows), proxy full 22.24 (0.991, valid proxy), proxy no-x 18.24 (1.085) | **H32 PARTLY CONFIRMED.** x reloads cost ≈ 1.5 µs (8.5%) at pooler M32 and ≈ 2.5 µs in the qkv proxy. qkv without x is still ≈ 2.7 µs above the weight-only model, so its excess is mostly not x. Upper bound if x were free: pooler ≈ 0.44% + qkv ≈ 0.74% ≈ 1.2% score; a real smem-staged fix captures only part of it | **CLOSED (diagnostic done)** |
| S4 | 2026-10-30 | Router c12 hi+lo (v18s FMT 3, 24-bit lossless pack of the BF16 hi/lo pair) at M 1/8/16/32: about 25% fewer router bytes. Geometries `(1,4,4,8)` RED3 (FMT-3 twin of the existing hi+lo table entry) and `(1,8,4,4)` RED3. Contract: TASK.md allows the hi+lo split at any m plus lossless compression. Gates: pack bit-exact (unpack == hi/lo bits), one-hot vs hi+lo exact, FP64 random/spike at router tol/4, official checker (router 1e-4, probe rows 1e-3); ABBA paired cold-L2 graph, 16 pairs, select iff median >= 1.03 and wins >= 75%; marker `S4_USE`. Target scenario (hypothesis): ~0.5 us/shape | NVRTC: 6 instances 125–174 regs, 0 spill. Candidate `candidates/dense_s4.py` sha `62ee3cc2…2d8f`, gates `results/dense-s4-check-1/gates.json` | pending | | launching |
