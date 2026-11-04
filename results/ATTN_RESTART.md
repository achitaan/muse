# Attention restart — offline candidate, not a measured win

**Superseded after parent GPU checks:** v1/v2 built correctly but admitted no restart arms; full verification was 3.3–3.9× slower. See `ATTN_RESTART_V3.md` for measured root-cause accounting and the sliding-only v3 candidate. The initial performance hypotheses below are preserved as history, not current recommendations.

**Recommended check file (v2):** `/Users/achita.anantachina/.local/share/opencode/worktree/e72615/loop-attn-restart/candidates/attention_restart_v2.py`

**Preserved v1 file:** `/Users/achita.anantachina/.local/share/opencode/worktree/e72615/loop-attn-restart/candidates/attention_restart.py` (unchanged, SHA256 `519527b6325373d56751ecd9f8a224860d27550078d958e46b1a14710688c4c1`).

Branch: `loop/attn-restart`, starting at `63e253f`. No check, scored run, push, or change to another worktree was made.

## What was rebuilt

The deliverable is one self-contained Python file. Its prefix is the **unchanged** 2.6536× incumbent, retained only as fallback; its final section contains an independent attention engine:

* `mma.sync.m16n8k32` FP8 QK, **not** WGMMA, with FP32 accumulation. Q is quantized after normalization/RoPE, before folding in the model's query scale, as explicitly permitted by the spec.
* Warp-local softmax, FP16 probabilities, exactly representable FP8-to-FP16 V conversion, FP32 PV accumulation. No FP8 probability rounding, lazy maxima, fixed logit reference, approximate attention, or skipped values.
* A CTA owns **both KV heads and every query token/head**, streaming disjoint 64-key tiles; decode also measures a 128-key variant. Three-stage cooperative `cp.async` gathers, no producer warpgroup, mbarrier protocol, full V transpose, or warpgroup barriers.
* One page-table dereference per token, broadcast to its eight copy lanes; each K/V row is globally fetched once. Each split constructs the new cache rows locally, and split zero writes the required cache entries. Attention never races another CTA's new-cache writes.
* Decode uses two key stripes per head, **merges them in shared memory first**, then writes only one immutable output per split, avoiding doubled global partial payload in this new warp mapping. Verification's four warps/head own separate query rows and share the same K/V tile. Private-partial merges themselves are not claimed as new: they previously lost inside the incumbent; the new QK/PV execution core is the materially different experiment.
* Stable two-level `(max, denominator, numerator)` reduction: last-arriver single-launch tail, or a separately launched parallel row merge. No floating-point atomics, fixed score bound, spin loop, or grid-residency requirement. Acquire/release arrival counters are reset by the last CTA. Attention is rounded to BF16 before norm/gate, honoring the real activation boundary.

This is not a modification of the incumbent's `segment()` engine, a one-head port, a launch-plan sweep over that engine, producer-side transpose, or physical-cache staging. The new CUDA source is about 550 lines; the large file size comes from embedding the unchanged fallback.

## Re-reading the accounting

Sources re-read: fresh `okc show` and `okc init` TASK/reference/checks/baseline, plus the closed-idea ledger. The published 4.8 TB/s floor is a useful-byte lower bound, not a latency guarantee.

For one TP8 token, K plus V is `2 * 2 * 64 = 256` bytes. The minimum unique cache traffic is:

* full: `256 * sum_b(seq_len[b] + L)`;
* sliding, with sufficiently long contexts: `256 * B * (2048 + L - 1)`.

These formulas count generated new rows as required cache **writes**, rather than requiring their data to be read back. Historical rows are reads. The sliding union is 2,055 positions for verification, **not** eight times 2,048. The causal/window mask still changes for every query token.

Add approximately four page-table bytes per historical token, 4,608 input-QKV bytes and 2,048 output bytes per new token, positions/cache-location metadata, and any implementation-specific split workspace. The 256-byte new cache write is already included above. GQA's eight query heads **do not** multiply the minimum K/V traffic; neither do the eight verification tokens. The incumbent already achieves this sharing. Re-reading the raw cache eight times is not the explanation for its gap.

`PAGE=1` means each logical page is a scattered **128-byte row in each of two allocations**. A 64-row tile is not contiguous globally, so a conventional tensor TMA load cannot replace it. Random rows do not imply an automatic 8× sector amplification: the full 128-byte row is used. Bulk-loading one 128-byte row at a time adds its own instruction/barrier costs. The new engine retains direct row gathers rather than assuming contiguous runs.

Scales do not change byte counts. Keys/values are encoded as `E4M3(value / scale)` and decoded by multiplying the device scale. The new engine uses runtime K scale in scores and V scale after reduction; CPU tests also cover 1/64. Conservative production admission currently uses the unit-scale/page-1 regime and leaves other cases to the incumbent.

### Nominal 80K payload floors, excluding auxiliary traffic and latency

| Case | Unique K/V MB, decimal | 4.8 TB/s floor, µs |
|---|---:|---:|
| d1S / v1S | 0.5243 / 0.5261 | 0.109 / 0.110 |
| d16S / v16S | 8.3886 / 8.4173 | 1.748 / 1.754 |
| d32S / v32S | 16.7772 / 16.8346 | 3.495 / 3.507 |
| d1F / v1F | 20.4803 / 20.4820 | 4.267 / 4.267 |
| d16F / v16F | 327.6841 / 327.7128 | 68.268 / 68.273 |
| d32F / v32F | 655.3682 / 655.4255 | 136.535 / 136.547 |

Batched lengths are actual device values around 80K, not uniformly 80K. The public SOL helper deliberately multiplies nominal tokens by 0.925 for agentic spread; its d16F/d32F figures around 63.2/126.4 µs are conservative lower bounds, **not** expected payload times. It applies the same loose factor even to sliding windows, whose visible lengths remain 2,048. Nominal useful bandwidth is therefore not an actual HBM-counter measurement.

The earlier useful-bandwidth estimates of 3.79/3.95/3.34/3.57 TB/s leave respectively **21.0% / 17.7% / 30.4% / 25.6% time** above a perfect 4.8-TB/s transfer, before accounting for auxiliary bytes and random-gather limits. “79–82% of peak” does not mean “only 5% left.” It also says nothing about the much smaller sliding/batch-1 cells.

### Correct score weighting

The score is a geometric mean of **six load ratios**, where each load costs `46*S + 16*F`. It is not the sum of twelve shape times or an HBM-traffic-weighted average. Therefore each batch-1 load has the same outer scoring weight as batch 32.

Representative incumbent-family paired profile `c300e99c653946ab` (not the exact scored incumbent run) gives:

| Load | S, µs | F, µs | S time share | F time share |
|---|---:|---:|---:|---:|
| d1 | 5.21 | 10.93 | 57.8% | 42.2% |
| d16 | 7.85 | 86.55 | 20.7% | 79.3% |
| d32 | 9.69 | 166.00 | 14.4% | 85.6% |
| v1 | 6.02 | 13.88 | 55.5% | 44.5% |
| v16 | 11.34 | 98.23 | 24.9% | 75.1% |
| v32 | 14.62 | 182.39 | 18.7% | 81.3% |

The mean of those six sliding shares is about **32%**. The sliding share of an incorrectly flattened sum is only about 22%, explaining why a global full-context roofline overstated how constrained the score is. Raw graph medians additionally include roughly 4 µs replay overhead; the judge subtracts it. Hardware/graph launch floors are a separate quantity, not `bytes / bandwidth`. No arbitrary replay-cost subtraction is made in admission timing.

## Why this could close a meaningful gap

The 1,419→1,128 µs display change is a **20.5% time reduction / 25.8% speedup**, not a claim that every full-context cell is 20.5% faster. Such an aggregate improvement does not violate the byte floor: sliding and batch-1 cells can contribute disproportionately, and verification-full still has materially more bandwidth-equivalent slack than decode-full.

For decode, native FP8 QK halves the normal FP16 QK instruction count and removes K conversion, while this core has a smaller prologue/control protocol. For verification, warp-local register MMA removes the incumbent's warpgroup synchronization, probability-to-FP8 packing, and whole-tile V transpose; it pays extra FP16 PV tensor-core work instead. That trade is physically plausible: nominal v32F QK+PV is about 84 GFLOP, approximately 85 µs at 989 TFLOP/s, below the ~137 µs K/V transfer floor. Achieving adequate issue efficiency is **not** proven by this bound.

There is no honest evidence yet that this implementation reaches the leaders, or even wins a cell. In particular the last-arriver private-partial tail can be slower than the incumbent's atomic/cluster tail, and synchronous warp MMA can lose to well-pipelined WGMMA. The point of this restart is to test a structurally different execution tradeoff rather than repeat exhausted local tuning. The admission gate prevents a demonstrated loser from being selected.

## Prepare-time admission and fallback

All building and paired tuning happens at module preparation/import, never on call inputs or inside `kernel_fn`. The incumbent's own setup/admission is retained, followed by a bounded 240-second restart preparation phase. For each case, independently generated probes are used to:

1. launch synchronously before capture;
2. compare entire cache bytes to the incumbent after poisoning new slots;
3. check the public BF16-query tolerance **and** a tighter 0.5% relative-L2/row bound against the precise FP8-query reference;
4. capture the actual kernel, then verify again after poisoned-cache replay;
5. measure 16 cold-L2 read-flushed ABBA blocks;
6. recapture the winning arm and independently repeat ABBA and correctness checks.

Both median and paired ratios must beat **0.975 in both rounds**. Otherwise that cell keeps the full incumbent dispatcher, including its already-gated one-head wins. Compilation failure keeps the incumbent. Runtime device faults propagate rather than pretending a poisoned CUDA context is recoverable. The retained state consists of scalar launch plans and permitted scratch/constants, never synthetic graphs, outputs, cache data, or partials. Scratch is cleared after preparation. Runtime dispatch only inspects shapes/config/device metadata and covers nearby 60K–100K table capacities; lengths, positions, scales, cache contents and masks remain device inputs.

## Validation actually performed

* `python -m py_compile` on the candidate and validation scripts: pass.
* `scripts/validate_attention_restart.py` with local offline-installed CPU Torch 2.14.1/NumPy: **15 cases pass the pristine public judge**, including 80K full decode/verify, 80K sliding edge keys, mixed lengths, page 1/64, scale 1/64, empty context, too many splits, partial tiles, and sliding/causal boundary cases. Worst L2 against FP8-query reference: **0.0028755**.
* All 32 lanes' actual FP8 QK / FP16 PV fragment addresses and V byte permutations reconstructed independently and compared to dense matrix products for d64, d128, v64. Worst PV absolute difference: **1.91e-6**.
* Four manufactured constant-V / zero-V solutions, including scale 1/64: exact expected BF16 outputs.
* Shared-address bijections/alignment and unique output ownership: pass. Three-stage ring logical reuse: **1,807 iterations** pass. FP32 Cody–Waite reduction versus independent sin/cos: max error **1.82e-7** through position 2,097,151 (does not test GPU SFU approximation).
* Clang CUDA **host and device syntax checks** of all new template instantiations with explicit API stubs: pass. This is **not** real CUDA-header compilation or PTX validation.
* CPU-only candidate import: pass, no restart route selected, no GPU compilation attempted.
* NVRTC, nvcc, ptxas, GPU execution, race/sanitizer testing, and performance: **unavailable/unverified** on this ARM macOS machine. No OKC check spent.

Evidence: `attn_restart_cpu.json`, `attn_restart_cuda_syntax.json`, `attn_restart_accounting.json` in this directory. Reproduce the numerical/fragment tests with `scripts/validate_attention_restart.py`; it assembles the single check file from the unchanged incumbent and `.tools/attention_restart_tail.py`. The latter is development-only and is **not required** by the candidate at runtime.

## Remaining risks / parent action

The next action is one parent-owned OKC **check** of the exact file above. Inspect `RR search/confirm/summary` diagnostics, compiler errors, spills, and actual selected cells. There may be **zero adoptions**; no GPU speedup is claimed. Real-header/SM90a compilation, acquire/release visibility, resource occupancy, FP8 MMA instruction support, graph-capture behavior, rounding and performance remain GPU validation risks. First use on a previously unseen stream during capture may add scratch-counter initialization; the performance impact is not known. As with the incumbent's persistent scratch, concurrently replaying graphs sharing a scratch allocation is not supported; distinct preparation/current streams receive distinct restart workspaces.

Ledger: a local branch-only `TRIED.md` row records this attempt. The discovered ledger helper pushes directly to main, so it was deliberately **not used to add a row**, respecting the task's explicit no-main/no-push constraint. The parent should reconcile the shared ledger before spending a check.

## V2: researcher follow-up, separate candidate

V2 keeps v1 intact and adds two independently gated design arms, taking advantage of the clarified reduced-precision-partial rule. It retains FP32 PV accumulation and does **not** touch the other session's FP16-PV work.

1. **256-key ping-pong gather, 64-key compute microtiles.** Two 64-KiB K/V slots occupy ~140.1 KiB with Q/new-KV/RoPE storage; this permits one CTA/SM. Four consecutive 64-key score/PV steps consume each gather without intervening CTA barriers. The score array remains at most 32 FP32 registers/thread, rather than expanding to 128 score registers for a 256-key computation. The next 256-key gather runs asynchronously in the other slot. The planner accordingly measures one-wave and two-wave split populations for this larger footprint. Random pages are still gathered individually; no contiguous-physical-page assumption was introduced.

2. **Normalized FP16 split means, FP32 statistics.** Store `mean_i = fp16(O_i / l_i)`, with `(m_i,l_i)` retained in FP32. Merge using `w_i = exp2(m_i-m) * l_i`, `O = sum(w_i*mean_i) / sum(w_i)`. This halves numerator writes and reads and reduces total partial-row traffic from 264 to 136 bytes (**48.5%**). Unlike casting raw numerators, means stay near the finite FP8 value range even for very long/uniform-attention splits; no clamping or fixed logit bound is needed. Specifically, nonnegative FP16 rounding obeys `round(P) <= 2*P`, including subnormals: the ideal-accumulation mean is bounded by twice the maximum encoded V magnitude (896), comfortably below FP16 overflow. Manufactured overflow and probability-bound tests exercise this reasoning. This targets exactly the last-arriver merge bottleneck in small/sliding cases rather than claiming fewer required cache bytes.

The original 64/128-key, FP32-partial arms remain available in v2; both merge protocols and both partial formats are gated against the full incumbent. The correctness gate is unchanged (public BF16-query bound plus 0.5% against the precise FP8-query reference), with two independent ≥2.5% cold-L2 ABBA wins required. The bounded preparation allowance is 300 seconds, and sliding d1/v1/d16/v16/d32/v32 are tested before full contexts to prioritize aggregate-score latency sensitivity.

**Why not short-fragment FP16 PV accumulation here?** Merely adding FP16 fragment accumulators while consolidating into the existing live FP32 output would increase, not reduce, live storage. Actually eliminating those FP32 registers requires another shared-memory checkpoint/flush design or a different rescaling scheme. That is not essential to these two changes and would duplicate the owned FP16 lane. The 256-key design instead bounds score-register pressure explicitly. PDL was not added: standalone attention has no preceding projection producer to overlap, and the new last-arriver arm already avoids a merge launch; the parallel arm is measured including its second launch.

**V2 offline results:** 52 numerical variants (13 input families × two gather sizes × two partial formats) pass the pristine public judge; worst relative L2 against the FP8-query reference is **0.0028581**. All **32 analytical constant/zero-V variants** pass exactly. Fragment reconstruction includes the 256-row shared layout (worst PV absolute error **7.63e-6**). Two-/three-slot logical pipeline reuse passes **3,800 iterations**. Clang CUDA host/device syntax with explicit API stubs passes all **16 attention template instantiations**. Real CUDA compilation, races, occupancy, graph capture, and speed remain unverified; zero adoption remains possible. No OKC check was spent.

Evidence: `attn_restart_v2_cpu.json`, `attn_restart_v2_cuda_syntax.json`. Reproduce with `scripts/validate_attention_restart_v2.py` and `scripts/check_attention_restart_cuda.py --v2`; the single check file has no runtime dependency on those scripts or `.tools/attention_restart_v2_tail.py`.
