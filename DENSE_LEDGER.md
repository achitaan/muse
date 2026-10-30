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
| LOCK | pooler M1/8/16, unpooler M16 | 4 shapes | profile `e793260539cb4877` | 13.14–13.56 µs, already below the 13.99–14.06 BF16 floor | **LOCKED** |

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

## 4. Experiment log

| Step | Date | Hypothesis / change | Offline evidence | Official job | Verdict | Status |
|---|---|---|---|---|---|---|
| S0 | 2026-10-30 | Worktree + ledger; drop dead `_achita_wg` sweep from candidates | SHA of parent verified | — | — | done |
| S1-pre | 2026-10-30 | Two unpooler geometries compile cleanly at exact NTMAX | NVRTC sm_90a rc 0: RED3 `(1,8,4,2,3)` 72 regs, 0 spill; RED0 `(1,16,4,1,0)` 64 regs (ptxas heuristic, no cap flag), 8 B stack / 8 B st / 16 B ld spill. `results/s1-offline/s1-resources.txt` | — | compile OK | done |
| S1 | 2026-10-30 | Lower reduction/coordination overhead at M1/M8 (target scenario: 4.0→≤3.5 µs; hypothesis only). Candidate `candidates/dense_s1.py` sha `4394ff32…a760`, gates `results/dense-s1-check-1/gates.json` | AST ok, 633,111 B, last `prepare`/`kernel_fn` = S1 wrappers, dead sweep removed | pending | | launching |
