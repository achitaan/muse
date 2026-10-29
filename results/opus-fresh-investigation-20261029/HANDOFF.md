# Fresh investigation & critical review handoff (October 29, 2026)

Branch `opus/fresh-investigation-20261029`, worktree `.slim/worktrees/opus-fresh`,
rooted at `6b7209fd89fc859ca658e3148b60b05d1052b44e` (`codex/h200-2-5x-20261028`).

This investigation is strictly offline (CPU math, AST/source inspection, and
CPU-only NVRTC compilation via `.tools/cuda-compiler-13.0`). **Zero official
checks, scored submissions, GPU jobs, okc/API calls, network fetches, subagents,
model switches or merges were performed.** All five `best/` kernels and
`candidates/` files are byte-for-byte unchanged:

| Track | Retained file | SHA256 | Source-bound result |
|---|---|---|---|
| Dense | `best/verified.py` | `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b` | Scored **1.5277713270942548x**, job `6131bc96427e4486` |
| MoE | `best/moe_verified.py` | `331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50` | Scored **2.348299935218348x**, `b6c70c9b83724d2e`; check **2.365437416088433x**, `d600231c8a8244cb` |
| Decode | `best/decoder_verified.py` | `d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988` | Checked **1.9902248796281297x**, `9560c915e85f4f50` (unscored); pred. `1fb5d05d...` scored **1.986659677x**, `75f2c7f464f34c61` |
| Attention | `best/attention_checked.py` | `9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f` | Scored **2.5370005142373167x**, job `8b767ea7b2b74de6` |
| Prefill | `best/prefill_verified.py` | `93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7` | Scored **1.6634692445639083x**, job `23e7c8b4c3e54869` |

---

## Part 1: Critical review of our initial investigation (`HANDOFF.md` v1)

Re-examining our first-pass investigation against the task contracts
(`muse_*/TASK.md`, `reference.py`, `baseline.py`) and all saved profile packets
exposed four substantive weaknesses in our initial analysis:

### 1. Why the initial 4-weight overall-board fit had tail errors up to 120 ranks (`exact_overall_formula.py`)
- **What v1 assumed:** `overall_model.py` fitted a function of the track-level
  summary speedups `(s_dense, s_attn, s_moe, s_decode)` (`composite =
  1/(0.25/s_dense + 0.11/s_attn + 0.48/s_moe + 0.16)`), getting mean error 1.53
  but max error 120 in the tail and leaving 113 adjacent Decode-rank inversions
  unexplained.
- **What the deeper audit establishes (derived from archived snapshots):**
  - Partitioning the 977 `muse_decode_layer` entrants by whether any later
    Decode entrant has a better Decode rank separates them into **843 backbone
    teams** (whose `decode.rank` is **100% strictly increasing with zero
    inversions**) and **134 jumped Decode teams** whose non-Decode component
    submissions beat their Decode submission.
  - Using the 843 backbone teams as exact interval bounds for all **754
    non-backbone teams** (620 without a Decode entry + 134 jumped Decode teams),
    a track-level harmonic linear fit `(w_0, w_dense, w_attn, w_moe) =
    (0.1494, 0.2558, 0.1097, 0.4871)` places **700 of 754 teams** inside their
    exact backbone interval (`<= 5e-4`).
  - **Why the remaining 54 tail teams violate any track-level formula:** We
    found pairs of single-track entrants on `muse_bf16_proj` (`solved: 1`)
    whose overall order is inverted relative to their `muse_bf16_proj` speedup:
    e.g., `Kingisback01` (`muse_bf16_proj = 1.1423x`, overall rank `1591`) ranks
    *below* `tomhaverford624` (`muse_bf16_proj = 1.0285x`, overall rank `1579`).
    No monotone function of track-level summary speedups can rank `1.1423 <
    1.0285`. This proves that the overall board combines a team's best runs
    **at the per-load / per-shape decode-step level** (as stated in `TASK.md`:
    *"ranks by how much your best runs together would speed up a decode step"*),
    where `bf16_proj` is 34.3% of `decode 1` SGLang time (`3697 / 10784 us`)
    but only 12.4% of `verify 32` (`6336 / 51016 us`), and `router` is 20% of
    `bf16_proj`'s geometric mean but `< 2%` of a decode step.
  - **Corrected implication:** Because `best/decoder_verified.py` already
    incorporates our fastest Dense, Attention and MoE kernels plus whole-layer
    fusions (`sk2`, `_pn_inputs_kernel`, `_sfn_kernel`), our whole-layer Decode
    submission (`~1.99x`) beats our component-wise decode step (`~1.75x`) on
    every load. Reaching top 50 overall (`~2.184x` at the snapshot, `+9.5%`
    over `1.9940x`, or `~14.3 us` uniform saving per shape) requires speeding
    up `muse_decode_layer` (including the MoE, Dense and Attention kernels it
    runs).

### 2. False equivalence between standalone MoE routing (`muse_moe_w4a16`) and Decode MoE routing (`muse_decode_layer`) (`routing_comparison.py`, `moe_regime_analysis.py`)
- **What v1 missed:** v1 noted that `_native_moe_CUDA_SRC` is byte-identical in
  `best/moe_verified.py` and `best/decoder_verified.py`, and treated MoE as a
  uniform ~50%-of-floor black box. It failed to compare `muse_moe_w4a16/reference.py`
  and `muse_decode_layer/reference.py`.
- **Calculated routing difference (32-seed CPU simulation of both official
  `reference.py` distributions):**
  - `muse_moe_w4a16` draws scores from `rand(bs, 256)` plus `POPULARITY = 0.03`
    Student-t(2) noise.
  - `muse_decode_layer` draws `stream` with a strong shared mean (`STREAM_MEAN =
    0.7`) and rank-4 subspace (`STREAM_SHARED = 0.9`) aligned with the router
    (`ROUTER_ALIGN = 0.07`).
  - **Resulting expert concentration at `T = 16` (`decode 16`) and `T = 32`
    (`decode 32`):**

| Load (`T`) | `muse_moe_w4a16` routes / distinct experts / max tok per expert | `muse_decode_layer` routes / distinct experts / max tok per expert | Consequence in `_native_moe` |
|---|---|---|---|
| `decode 1` (`1`) | `2.6` / `2.66` (`15.7 us` floor) / `1.0` | `2.7` / `2.47` (`14.6 us` floor) / `1.0` | Same regime (`TINY`) |
| `verify 1` (`8`) | `13.6` / `8.84` (`52.2 us` floor) / `4.4` | `15.8` / `7.31` (`43.1 us` floor) / `5.2` | Same regime (`TINY2`, 0 split jobs) |
| **`decode 16` (`16`)** | `26.1` / **`12.06`** (`71.1 us`) / `7.8` | `34.2` / **`6.38`** (`37.6 us`) / **`12.1`** | `MID` (`-DOKC_SPLIT=1`) splits experts with `> 8` tokens and **re-streams their 28.3 MB weights**: only **`+2.8%` (`+2.0 us`)** in MoE track, but **`+18.1%` (`+6.8 us` ideal, `~12 us` actual)** in Decode! |
| **`decode 32` (`32`)** | `47.8` / **`15.62`** (`92.2 us`) / `11.3` | `68.1` / **`7.38`** (`43.5 us`) / **`23.5`** | `MAIN/WGS` has only **`14.3` Phase-B stages (`0.89` units) per CTA** in Decode (`92.2%` of Phase-B units and `55.5%` of Phase-A units Stream-K split!). Measured D9 `moe_fused` takes **`84.8 us`** for `7.38` experts vs **`64.6 us`** for `7.31` experts in `TINY2` at `verify 1`! |
| `verify 16` (`128`) | `197.3` / `22.25` (`131.2 us`) / `62.5` | `231.3` / `16.94` (`99.9 us`) / `68.5` | `MAIN/WGS` (`22.7%` A / `45.3%` B split in Decode) |
| `verify 32` (`256`) | `395.6` / `24.53` (`144.7 us`) / `114.2` | `449.4` / `19.16` (`113.0 us`) / `133.3` | `MAIN/WGS` (`17.9%` A / `36.0%` B split in Decode) |

### 3. Flaws in v1's Possibilities 1 and 2
- **v1 Possibility 1 (`%globaltimer` phase trace in `moe_fused`):** Already
  screened and rejected in `results/moe-trace-resource-screen-20261029/review.md`
  — enabling `DBG` timestamps in `WGS` increases offline stack/spill bytes from
  `16 / 56 / 124` to `24–32 / 156–204 / 300–356`, perturbing the kernel being
  measured.
- **v1 Possibility 2 ("Small-T routing is known before the pooler finishes; fold
  `_pn_inputs_kernel` into MoE prologue with PDL"):** Contradicted by the saved
  D9 profile (`submission.9560c915e85f4f50.txt`):
  - At `T = 8` (`verify 1`), `route_fp32<8>` takes **`15.2–16.8 us`**, which is
    **longer** than the pooler `smallm2_gemv` (`13.6–14.7 us`)!
  - At `T = 1` (`decode 1`), the raw graph median (`87.41 us`) equals the sum
    of all 10 kernel durations (`87.4 us`), because the pooler (`13.3–14.4 us`
    vs `14.0 us` floor) saturates HBM and all 132 SMs, starving `route_fp32`'s
    128 CTAs on `side_stream` and stretching its 128-CTA barrier across the
    pooler.
  - At `T = 16, 32`, `_fpr_bfp_fused_pooler_router` (`sk2_kernel`) fuses pooler
    and router into one kernel on `main_stream`, followed sequentially by
    `routing_kernel`, so routing finishes *after* pooler.

### 4. Missed `verify 16` (`M = 128`) and `verify 32` (`M = 256`) Dense projection fallback to uncompressed 16-bit cuBLASLt
- Inspecting `submission.6131bc96427e4486.txt` (Dense) and
  `submission.9560c915e85f4f50.txt` (Decode) revealed that while `M = 1..32`
  projections use **12-bit lossless exponent-window compression (`c12`, `1.5
  bytes/weight`, 25% less HBM traffic)**, at `M = 128` (`pooler`, `unpooler`)
  and `M = 256` (`qkv_gate`, `pooler`, `unpooler`) both tracks fall back to
  **uncompressed 16-bit cuBLASLt (`nvjet_sm90_tst_*`)**, achieving only
  `1.041x–1.284x` over SGLang (`43.0–45.5 us` combined `pooler + unpooler` in
  Decode at `M = 128`, and `75.0–76.3 us` combined `qkv_gate + pooler +
  unpooler` in Decode at `M = 256`).

---

## Part 2: Concrete, independently developed kernel alternatives (with CPU-only NVRTC & mathematical verification)

None of the four alternatives below repeats prior rejected experiments; each
is backed by source-level analysis, exact byte/FLOP calculations, and CPU-only
NVRTC 13.0.88 `sm_90a` compilation (`experiments/opus_fresh/nvrtc_screen_alternatives.py`,
saved in `results/opus-fresh-investigation-20261029/nvrtc_and_cpu_verification.json`).
GPU runtime speedups remain **untested hypotheses**.

### Alternative 1: `m128w_c12` — Extend 12-Bit Lossless RS-WGMMA (`m32w_gemm`) to `BN = 128` for `verify 16` (`M = 128`) `pooler`, `unpooler`, and `qkv_gate`
- **Mechanism:**
  - In `best/verified.py` and `best/decoder_verified.py`, `m32w_gemm`
    (`tt_mw_CUDA_CORE`) is an RS-WGMMA kernel (`wgmma.mma_async.sync.aligned.m64n{BN}k16.f32.bf16.bf16`)
    that loads 12-bit lossless `c12` weights (`1.5 B/weight`) via
    `ld.global.nc.L1::no_allocate.v4.u32`, decodes 64 weight rows × 64 K into
    `uint32_t a[16]` in registers (`4 * dec` + sparse `patch`), and multiplies
    against `x` staged in shared memory via 2D TMA.
  - Crucially, the weight load, 12-bit decode (`dec`), escape patch (`patch`),
    and the host weight packer `tt_mw_pack_weight(w, r=64, wk, split,
    compressed=True)` are **100% independent of `BN` (`M`)**!
  - Adding `128` to `(16, 32, 64)` in `tt_mw_cuda_source` (either via direct
    `wgmma.m64n128k16` or two `wgmma.m64n64k16` calls reusing the same decoded
    `a[0..15]`) enables `m32w_gemm` at `M = 128` (`BN = 128, grid.y = 1`).
  - Furthermore, because `packs` in `dd_decodediet_tt_mw_prepare_m32w` is keyed
    by `(layer, R, WK, split, compressed)` (independent of `BN`), `M = 128` can
    **reuse the exact same packed weight tensor as `M = 32` (`split = 1` or
    `split = 2`) with zero additional weight memory and zero extra packing time
    in `prepare()`**.
- **Calculated traffic & compute:**
  - At `M = 128`, `4096 x 8192` (`pooler` / `unpooler`) requires only **`8.69 us`
    of Tensor Core compute** at 989 TFLOP/s, so it is **HBM weight-bandwidth
    bound** (`16`-bit weight floor `13.98 us`; cuBLASLt takes `19.8–23.1 us` on
    `pooler` and `18.4–22.4 us` on `unpooler`).
  - `c12` cuts weight traffic by **25%**: `67.11 MB -> 50.33 MB` (`10.49 us`
    weight floor at 4.8 TB/s) on both `pooler` and `unpooler`, and `37.75 MB ->
    28.31 MB` (`5.90 us` weight floor) on `qkv_gate` — removing **`43.0 MB` of
    HBM reads per layer** at `verify 16`.
  - Dynamic shared memory for `BN = 128, R = 64, WK = 2, Q = 2` is
    `max(67584, 69632) = 69,632` bytes (`68 KB`, well within `227 KB`), or
    `34,816` bytes for `WK = 1` (allowing **2 CTAs/SM**).
- **CPU-only NVRTC 13.0.88 `sm_90a` & bitwise verification (measured offline):**
  - Both `direct_mma128` and `dual_mma64_reuse_a16` compile cleanly (`rc = 0`)
    for `<128, 64, 2, 2, 2, true, 1>` and `<128, 64, 1, 2, 2, true, 2>` at
    **`141` registers, `0` bytes stack frame, `0` bytes spill stores, `0` bytes
    spill loads (`fa.localSizeBytes == 0`), and `0` warnings (no `C7512`)**.
  - `tt_mw_sm_c12_bytes` / `tt_mw_sm_c12_unpack` uses `torch.equal` on `int32`
    bit patterns and passed our CPU bitwise exactness check including `-0.0`,
    `+0.0`, subnormals, and escaped outliers (`17` escapes).
- **Hypothesized gain (untested on GPU):** `~3–5 us` per projection (`~8–12 us`
  total on `verify 16` in both `muse_bf16_proj` and `muse_decode_layer`).

### Alternative 2: `wgs_finisher_inreg` — In-Register Phase-B Designated Finisher in MoE `WGS` (`M = 32, 128, 256`)
- **Problem identified in `moe.cu` (lines 1939–1985 and 2074–2117):**
  - In `WGS` Phase B (`PB == true`), `OKC_WGSDFIN = 2` designates the share with
    `s0 == 0` (`s1 < 16`) as the unit's finisher (`pd.kind = 3`).
  - Across all **20,303 Phase-B split units** in the 192 saved synthetic draws
    (`results/moe-unit-ownership-screen-20261029/screen.json`), `s0 == 0, s1 < 16`
    is **100% of the time the final segment of its CTA's Phase-B range**
    (`gs + s1 == g1`), while the other contributor (`s0 > 0`) runs at the
    **very start** of CTA `c+1`'s Phase-B range.
  - Currently, when the finisher CTA (`s0 == 0`) completes `wg_wait<0>()` on its
    last Phase-B segment (line 2071) with `float d[NTT][4]` already in
    registers:
    1. It writes `d[nt]` out to global memory `fb` via `red.global.add.v4.f32`
       (line 2078: `~16 KB/CTA = ~2.0 MB` across ~120 CTAs simultaneously right
       at the end of Phase B),
    2. Exits the loop and immediately calls `wgs_flush<true>` (line 2117),
    3. Executes `fence.acq_rel.gpu` (`MEMBAR.ALL.GPU`, line 1963), **stalling on
       its own just-issued `red.global.add.v4.f32` round-trip to L2**, and
    4. Reads `fb` back from L2 (`__ldcg`, line 1972) in a dynamic `nt0 < ntt`
       loop into a second inlined copy of `wgs_epi_tile<true>`.
- **Mechanism (`wgs_finisher_inreg`):**
  - In `wgs_seg`, when `DF && (PB ? s0 == 0 : s1 == 16)`, keep `d[nt][0..3]` in
    registers at line 2074, wait on `fc` + `bar.sync (3 + wg), 128` +
    `fence.acq_rel.gpu` (with **zero** self-issued atomic writes in flight),
    load the other contributor's partial `v = __ldcg(fb + nt * 512)`, zero
    `fb`, add `d[nt] += v` in registers, zero `*fc = 0`, and **fall straight
    through into the existing compile-time-unrolled `#pragma unroll for (int
    nt = 0; nt < NTT; ++nt) wgs_epi_tile<PB>(p, S, j, r16, nt, d[nt]);` on line
    2087**.
  - Unlike `moe-private-fixup-preflight-20261029` (which stored both shares to
    20 MB of new scratch and increased stack/spills to `24–184 / 100–768 /
    204–1216`), `wgs_finisher_inreg` needs **0 extra scratch bytes**, **0 extra
    registers**, and **removes** the entire `DF` readback + duplicate
    `wgs_epi_tile` body from `wgs_flush`.
- **CPU-only NVRTC 13.0.88 `sm_90a` & invariant verification (measured offline):**

| Metric (`moe_fused<false>` & `moe_fused<true>`, `MAIN` WGS) | Retained baseline (`1e60cf46`) | `wgs_finisher_inreg` | Delta |
|---|---:|---:|---:|
| Registers / thread | `96` | `96` | `0` |
| Offline stack frame (bytes) | `16` | **`8`** | **`-50%` (`-8 B`)** |
| Ptxas spill stores (bytes) | `56` | **`52`** | **`-4 B`** |
| Ptxas spill loads (bytes) | `124` | **`84`** | **`-32.3%` (`-40 B`)** |
| Compiled cubin size (bytes) | `3,859,728` | **`3,695,760`** | **`-163,968 B` (`-4.25%`)** |
| Serialization warnings (`C7512`) | `0` | `0` | `0` |
| Phase-B finisher global atomic write bytes (`M = 32..256`) | `1.0–2.7 MB` | **`0 MB`** | **`-100%`** |

### Alternative 3: Decode-Specific MoE `T = 16` / `T = 32` Dispatch & Early Per-Job Phase-A Release (`skinny_ring_ws` + `WGS` Phase-A Reorder)
- **Problems identified in `moe.cu`:**
  1. **Early per-job `doneA` release in `skinny_ring_ws` (`M = 1..16`, `TINY` /
     `TINY2` / `MID`):** In `ws_helper` (`moe.cu` line 1561), a CTA credits
     `p.doneA + jj` only when `q == nA - 1` (after *all* of the CTA's Phase-A
     units finish). When a CTA straddles job `jj` and job `jj + 1` (`jf < jl`),
     it holds back the last units of job `jj` until job `jj + 1` also finishes
     on that CTA. Crediting `p.doneA + j` immediately when `rb == RB_A - 1 ||
     q == nA - 1` (with `__threadfence()` before `atomicAdd`) releases each
     completed job to Phase B as soon as its 512th unit finishes.
  2. **Decode `T = 16` / `T = 32` concentration mismatch:** In
     `muse_decode_layer`, `decode 32` (`T = 32`) activates only **`7.38` local
     experts** (`43.5 us` weight floor, virtually identical to `verify 1`'s
     `7.31` experts at `T = 8`), yet `WGS` splits `118` Phase-B 256-row units
     across `132` CTAs (`92.2%` split units) and takes `84.8 us`. Meanwhile, in
     Phase A of `WGS`, `OKC_WGSDFIN = 2` leaves Phase A on `atom.acq_rel`
     (`DF = false`) because `wgs_split` runs the `s0 == 0` share first.
     Reversing the Phase-A reorder so the non-finisher share (`s0 > 0, s1 == 16`)
     runs first and the `s0 == 0` share runs last allows **Phase A to also use
     `wgs_finisher_inreg` (`OKC_WGSDFIN = 3`)**, eliminating half of Phase A's
     `red.global.add.v4.f32` writes (`1.0–3.4 MB`) and the atomic `atom.acq_rel`
     round-trip in Phase A as well!

### Alternative 4: `rf_vec4_smem` — Shared-Memory `x` Staging + `float4` Loads + Smaller CTA Grid for Small-Token FP32 Router (`route_fp32`, `T = 1..8`)
- **Problem identified in `_RF_CUDA` (`experiments/opus_fresh/scratch/rf.cu`):**
  - `route_fp32<1>` and `route_fp32<8>` launch `128` CTAs (`16` row tiles × `8`
    K-splits) on `side_stream` concurrently with the pooler (`13.3–14.7 us`),
    using scalar `float` loads where all 8 warps in each CTA redundantly load
    the same `x[m, ks*1024 .. ks*1024+1023]` slice from global memory (`16 * 8 =
    128x` logical reads of `x`), taking `11.2 us` at `T = 1` and `15.2–16.8 us`
    at `T = 8` (slower than the `67.1 MB` pooler at `T = 8`!).
- **Mechanism (`rf_vec4_smem`):**
  - Cooperatively stage `x[TM, 1024]` (`4 KB` for `TM = 1`, `32 KB` for `TM = 8`)
    into shared memory `sx[TM][1024]` once per CTA using vectorized `float4`
    loads, and read `sx[m][j * 32 + lane]` in the inner loop.
  - Because each lane `lane` still accumulates `k = ks * 1024 + j * 32 + lane`
    in the exact same `j = 0..31` order with `__fmaf_rn(v, w0, a[m])`, the
    FP32 FMA sequence, warp shuffle reduction, and 8-split reduction are
    **100% bit-identical** to `route_fp32`, while cutting per-CTA global `x`
    requests by **87.5% (`8x -> 1x`)**.
  - Optionally pairing 4 router rows per warp (`64` rows/CTA, `4 * 8 = 32` CTAs
    instead of `128` CTAs) reduces `side_stream` SM occupancy to 1/4 wave so
    `route_fp32` can complete alongside the pooler without waiting for 128 CTAs
    to schedule across a saturated GPU.
- **CPU-only NVRTC 13.0.88 `sm_90a` verification (measured offline):**
  - `route_fp32<1>` and `route_fp32<8>` with `smem_vec4_x` compile cleanly
    (`rc = 0`) at the **exact same `48` registers, `0` stack bytes, `0` spill
    bytes**, and `4,100 B` / `32,772 B` static shared memory.

---

## Part 3: Changed files & SHA256 hashes on `opus/fresh-investigation-20261029`

All added files live under `experiments/opus_fresh/` and
`results/opus-fresh-investigation-20261029/`. No tracked file outside those
directories was modified.

- `experiments/opus_fresh/overall_model.py`
- `experiments/opus_fresh/decode_sensitivity.py`
- `experiments/opus_fresh/list_strings.py`
- `experiments/opus_fresh/exact_overall_formula.py`
- `experiments/opus_fresh/routing_comparison.py`
- `experiments/opus_fresh/moe_regime_analysis.py`
- `experiments/opus_fresh/nvrtc_screen_alternatives.py`
- `results/opus-fresh-investigation-20261029/HANDOFF.md`
- `results/opus-fresh-investigation-20261029/overall_model_output.txt`
- `results/opus-fresh-investigation-20261029/decode_sensitivity_output.txt`
- `results/opus-fresh-investigation-20261029/exact_overall_formula_output.txt`
- `results/opus-fresh-investigation-20261029/routing_comparison_output.txt`
- `results/opus-fresh-investigation-20261029/moe_regime_analysis_output.txt`
- `results/opus-fresh-investigation-20261029/nvrtc_and_cpu_verification.json`

---

## Part 4: Tests actually run

1. **Retained baseline integrity check:** Recomputed SHA256 of all five `best/`
   files and verified `git diff 6b7209f HEAD -- best candidates` is empty.
2. **Archived leaderboard backbone & interval analysis (`exact_overall_formula.py`):**
   Verified 0 inversions across all 843 backbone Decode teams; solved interval
   bounds for all 754 non-backbone teams; identified single-track inversion
   counterexamples proving per-load decode-step aggregation.
3. **Official reference routing comparison (`routing_comparison.py`,
   `moe_regime_analysis.py`):** Simulated 32 seeds per shape of the exact
   `muse_moe_w4a16/reference.py` and `muse_decode_layer/reference.py` routing
   distributions on CPU, quantifying active local experts, `-DOKC_SPLIT=1`
   weight re-streaming (`+18.1%` at Decode `T = 16`), and `WGS` Stream-K unit
   splitting (`92.2%` in Phase B at Decode `T = 32`).
4. **Uncached CPU-only NVRTC 13.0.88 `sm_90a` compilation (`nvrtc_screen_alternatives.py`
   with `CUDA_CACHE_DISABLE=1`):**
   - Compiled `m32w_gemm` at `BN = 128` (`direct_mma128` and
     `dual_mma64_reuse_a16`): `141` regs, `0` stack/spills, `0` warnings.
   - Compiled `moe_fused<false>` and `moe_fused<true>` with
     `wgs_finisher_inreg`: `96` regs, stack `16 B -> 8 B`, spill stores
     `56 B -> 52 B`, spill loads `124 B -> 84 B`, cubin `3,859,728 B ->
     3,695,760 B` (`-163,968 B`), `0` warnings.
   - Compiled `route_fp32<1>` and `route_fp32<8>` with `smem_vec4_x`: `48` regs,
     `0` stack/spills.
5. **CPU mathematical & bitwise invariants (`cpu_math_checks`):**
   - Verified 12-bit `c12` pack/unpack bitwise exactness (`torch.equal` on
     16-bit patterns) across normal, `-0.0`, `+0.0`, subnormal, and 17 escaped
     outlier values.
   - Verified across all 192 synthetic draws (`20,303` Phase-B split units) that
     every Phase-B split unit has a unique `s0 == 0` finisher and that `s0 == 0`
     is always the final segment of its CTA's Phase-B range.

---

## Part 5: Remaining uncertainties

- **No GPU execution or timing:** Offline NVRTC `ptxas` register/spill/cubin
  improvements and HBM byte reductions do not prove H200 runtime speedups or
  CUDA-graph replay gains.
- **Preparation time (`<= 300 s`) and source size (`<= 1 MB`) budgets:**
  `best/decoder_verified.py` is 989,018 bytes with historical prepare times of
  215–257 seconds. Any integration into Decode must reuse existing weight packs
  (as `m128w_c12` does) or replace in-place rather than adding new prepare
  sweeps.
- **Evaluation window:** `EVALUATION_POLICY.json`'s window ends
  `2026-10-30T00:00:00Z` (8 p.m. America/Toronto October 29, 2026); any future
  GPU check requires fresh human authorization.
