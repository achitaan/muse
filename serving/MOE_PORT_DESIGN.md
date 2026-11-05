# Muse Spark serving MoE port: phase-1 design

Date: 2026-11-05. Status: **design only; no GPU validation, checks, submissions, or implementation**.
Target: SGLang `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`, worktree `wt-moe`, branch `serve/moe-port`.
Judge configuration is unchanged: TP8/EP8, ModelOpt NVFP4, Marlin, DFlash/8, C<=32, decode graph BS<=32, prefill chunk<=16384.

## Decision / executive answer

**The mathematical operation and integration boundary fit. The supplied board implementation is NOT a deployable drop-in.**
It can replace the *local routed-experts plus post-expert norm/combine* portion, not the router, pre-expert norm, pool/unpool projections, or TP all-reduce. There are three gates before enabling it:

1. **Resident weight memory is the main architectural blocker.** The board stores four full layouts: **3.375 GiB per NVFP4 layer per rank**, versus stock's 0.84375 GiB. Keeping Marlin for prefill and adding the unchanged board bank adds approximately **202.5 GiB/rank for 60 layers**, before scratch. Even keeping only its two necessary fast-path layouts adds 101.25 GiB/rank. Neither fits H200.
2. **The stated J==0 fix is missing from this exact artifact.** Its WGS range constructor indexes a nonexistent first job for zero local work. It must be fixed and explicitly tested; see section 5.
3. Board correctness tolerances and its 2.3797x score do **not** establish serving token equivalence. Compare against this exact stock Marlin path, including its scale collapse and BF16 boundaries, not an ideal FP32 MoE or Blackwell FP4-activation backend.

Recommended next decision: authorize a **bounded, selected-layer integration proof**, not an all-layer four-layout copy. Keep the existing fast arithmetic, retain only unit-order + WGS banks, and impose a model-wide bank budget. A full-payoff all-layer port needs a separate **single shared weight-layout architecture** (new kernel consumes resident Marlin layout, or Marlin prefill consumes the new layout); do not assume that rewrite preserves the board timings. That architectural choice is an explicit follow-up for the caller, not hidden implementation scope.

No paid check should be spent until memory arithmetic, zero-local behavior, and a startup differential harness are ready. No `okc` command was run in this phase.

## Citation map and artifact identity

All SGLang paths below are relative to `wt-moe/python/sglang/`:

| Short name | File |
|---|---|
| `MODEL` | `srt/models/muse_spark.py` |
| `FP4` | `srt/layers/muse_spark_fp4_moe.py` |
| `QUANT` | `srt/layers/quantization/muse_spark_nvfp4_moe.py` |
| `MODELOPT` | `srt/layers/quantization/modelopt_quant.py` |
| `REPACK` | `srt/layers/quantization/marlin_utils_fp4.py` |
| `FINAL` | `kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py` |
| `NORMS` | `srt/layers/muse_spark_layers.py` |
| `ROUTE` | `kernels/ops/muse_spark_v12/fused_muse_spark_routing.py` |
| `GRAPH` | `srt/model_executor/runner/decode_cuda_graph_runner.py` |
| `RUNNER` | `srt/model_executor/model_runner.py` |
| `DFLASH` | `srt/speculative/dflash_worker_v2.py` |
| `CONFIG` | `srt/configs/muse_spark.py` |
| `BOARD` | `/Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse/board_best/moe_m3_scored_2.3797.py` |

`BOARD`: 185756 bytes, SHA256 `f72cdf2b5438c70369b0fb823e171f4022b22b97f0dff66c6170dfe3aaf845fb`.
Its CUDA is a **single escaped Python line, BOARD:396**. For readable precise sub-citations, `BOARD:396 [CUDA:N]` means line N of `ast.literal_eval(CUDA_SRC).splitlines()`, counting from 1. It contains 2693 logical CUDA lines / 139873 characters. Comments in this artifact are occasionally stale; executable code takes precedence.

## 1. Exact integration boundary and tensor contract

### Actual call chain (important: not the generic Marlin runner)

`MuseSpark1xMoE._forward_tp` -> `_forward_experts` -> `MuseSpark1xFp4Experts.forward` -> `_forward_marlin`.

* `MODEL:839-856`: project model hidden BF16 `[T,8192]` into expert hidden BF16 `[T,4096]`; this is **not** the router input. The router still sees model hidden states in FP32 (or the stock split-BF16 router implementation).
* `MODEL:866-875`: compute router and join its side stream; routing gives `topk_scores`, `topk_ids`, and separately packed routing.
* `MODEL:899-909`: passes expert input, IDs, scores, post norm, pre norm, and optional AR workspace into the FP4 expert module.
* `FP4:697-704`: Marlin bypasses FlashInfer `prepare_input` entirely.
* `FP4:595-599`: stock pre-expert RMSNorm, then `_marlin_expert_outputs`.
* `FP4:600-610`: effective norm binding and **per-route post norm then weighted local sum**.
* `MODEL:748-774`: TP all-reduce of local expert sums, then post-expert projection `[T,4096] -> [T,8192]`.
* `MODEL:1088-1100`: existing FFN boundary post norm/residual/next pre norm remains untouched.

**Proposed dispatch:** in `_forward_marlin`, after the existing pre-expert norm at `FP4:595-596`, before `_marlin_expert_outputs` at `FP4:597`. Return the custom already-normalized/combined local result directly; do not call the stock finalizer again. Bind the exact same effective post-norm weight via `set_post_expert_norm` before launch. An explicit decode/verify eligibility bit must be passed from the parent (see section 4); shape alone cannot distinguish short prefill.

The generic `srt/layers/moe/moe_runner/marlin.py:115-208` -> `fused_moe_triton/fused_marlin_moe.py:149-443` is **not this model's executed path**. In particular the latter applies route weights in GEMM2 (`:399-400`) and sums without Muse's intervening norm. Replacing that generic function alone would miss Muse; using it as the correctness oracle would be wrong. Preserve both generic files unchanged.

### Device tensor ABI

| Tensor / metadata | Required contract |
|---|---|
| `x` | CUDA BF16 contiguous `[T,4096]`, **after the existing pre-expert norm**, no activation quantization |
| `ids` | CUDA int32 contiguous `[T,8]`, global expert IDs `0..255`; `-1` denotes graph padding; WGS requires 16-byte pointer alignment |
| `scores` | CUDA FP32 contiguous `[T,8]`, stock routing outputs, already normalized and multiplied by `route_scale`; do not renormalize per rank |
| local offset | `32 * moe_ep_rank`, not necessarily a guessed CUDA ordinal; local expert index `global_id - offset` when `0 <= index < 32` |
| weight bank | **32 local experts only**, physical index `0..31`, gate then up logical row halves, FP4 E2M1 codes + E4M3FN scales per 16 K values; per-expert FP32 global-scale containers |
| norm | effective BF16 `[4096]`, exactly `FP4.set_post_expert_norm`'s cached weight; weight-first norm with checkpoint epsilon `1e-8` |
| result | CUDA BF16 contiguous `[T,4096]`, this rank's local expert partial **after** per-expert norm and weighted sum; no communication inside kernel |

EP ownership comes from `MODEL:529-539` and the stock map from `FP4:174-181,471-475`. Every rank processes the same token rows, owns a contiguous block of 32 experts, and contributes zero for other routes. TP=EP means **no expert-TP reduction before the norm** (`FP4:134-138`); the sum all-reduce remains `MODEL:754-760`. The Hopper path has no Blackwell finalize/AR workspace (`MODEL:1173-1182`); still require `ar_workspace is None` for custom dispatch to avoid a future double/missing reduction.

`ROUTE:213-229` stores separate int32 IDs / unrounded FP32 route scores; `ROUTE:259-262` rounds weights to BF16 **only for packed routing**. Feed the separate tensors like stock Marlin; do **not** unpack `packed_topk_ids` to obtain weights. Padded rows can have uninitialized scores (`ROUTE:119-135`), so never read scores for invalid routes. The board final pass already predicates those reads on a valid local position.

Porting detail: `BOARD:586,599-603,615-625` currently combines `LOCAL_START=0` with indexing a global expert bank. Serving tensors are already local. Change the packer loop to `e=0..31`, pass the real offset only as runtime `lo`, and slice globals locally. Merely changing `LOCAL_START` to rank*32 would index beyond SGLang's local tensors on ranks 1..7. The CUDA routing subtraction already accepts runtime `p.lo` (`BOARD:396 [CUDA:305-319]`).

## 2. Semantics and numerical differences

Let `Q1_e`, `Q3_e`, `Q2_e` mean the logical FP4 matrices with their FP8 group scales applied. Let `a_e` be **the gate column** of the loaded w13 global scale and `b_e` the down global scale. The operational stock computation is:

1. Existing BF16 pre-norm: `x = pre_expert_norm(expert_input)`.
2. Marlin BF16-output GEMM1: gate/up use the **same** `a_e`, including stock's accumulator/global-scale rounding boundaries.
3. `h_e = BF16(silu(FP32(gate_e)) * FP32(up_e))`.
4. Marlin BF16-output GEMM2 using `b_e` gives `z_e`.
5. FP32 `v_e = FP32(z_e) * FP32(effective_norm_weight)`.
6. `n_e = v_e * rsqrt(sum(v_e**2)/4096 + eps)` in FP32.
7. BF16 `local_out[t] = sum(local routes in stock slot order) scores[t,s] * n_e`.
8. Stock TP all-reduce, unpool projection, and layer norms/residuals.

Evidence: `FP4:513-574` sets `mul_topk_weights=False` in **both** GEMMs and uses BF16 caches with FP32 reduction. `FP4:541-546` uses **filtered ordinary** SiLU-and-multiply, not the generic activation-rounding/clamped branch. `FINAL:41-60,85-99` does weight-first norm and FP32 weighting, with a torch path below T=32 and Triton path at/above T=32. Norm class selection is `MODEL:600-609`; effective zero-centered weight is `NORMS:86-98`, cached BF16 by `FP4:231-243`.

### Match / difference table

| Item | Assessment / required action |
|---|---|
| Per-expert norm before weighting | Matches board algorithm (`BOARD:227-234`, `:396 [CUDA:431-442,2459-2559]`). Do not replace with norm after the sum, or apply route weights before the norm. |
| Weight-first vs ordinary norm | Board supports only weight-first with hardcoded epsilon `1e-8`. Require `_post_expert_norm_weight_first` and matching epsilon; otherwise use stock. `NORMS.forward` is not itself the finalizer oracle: the executed FP4 fused finalizer performs FP32 weight-first arithmetic, whereas the standalone module can BF16-round its weight*x product (`NORMS:120-132`). |
| Norm zero centering | `BOARD:617` actually packs `BF16(FP32(raw_w)+1)`, and CUDA loads it directly; stale comments say CUDA adds 1. Supply the already-effective stock BF16 weight directly to the port. Do not add 1 twice or subtract/re-add to reconstruct a raw parameter. |
| SwiGLU | Same unclamped SiLU(gate)*up and BF16 intermediate. Board uses `__expf`/`__fdividef`; stock `activation.cuh:25-28,88-101` uses expf/division but Hopper compiles with `--use_fast_math` (`kernels/jit/utils/arch.py:184-189`). Likely similar lowered math, not a proof of bit identity. No extra BF16 rounding of SiLU alone in this call path. |
| Weight formats | Both W4A16 NVFP4; do not copy the Blackwell FP4-activation input-quantization path (`FP4:618-681`). Codes/scales are permutations, not requantizations. Preserve scale bytes and FP32 global-scale parameters. |
| Global scales | Stock **discards the up-column global scale** (`MODELOPT:2685-2711`), converts global scales through BF16 plus exponent rebias (`REPACK:47-59,514-515,563-567`). The board already has explicit BF16 accumulator then BF16-global-scale multiply boundaries (`BOARD:396 [CUDA:407-413]`). Feed the same raw gate scale for gate and up, and raw down scale; validate rebias algebra and saturation/underflow against stock. Do not restore a separate up scale merely because it is more mathematically faithful. |
| Scale-collapse rationale | `FP4:584-593` describes a gain cancellation by post norm. That is a checkpoint-specific approximation once epsilon and BF16 rounding are included: `norm(c*z;eps)=norm(z;eps/c^2)` for positive c, not exact at nonzero eps. Matching stock means matching its collapsed-scale operation, not relying on cancellation to forgive a different one. |
| Accumulation | Both BF16 operands/FP32 accumulators, but Marlin split-K and the board mma.sync/WGMMA/stream-K partition/reduction trees differ. Different sums may cross BF16 rounding ties at either GEMM boundary. WGS split fixups can add run-to-run scheduling-dependent FP32 differences; measure, don't assume determinism. |
| Final norm/reduction | Board forms route sum-of-squares via its own tree (and possibly atomic partial sums outside WGS) versus torch/Triton reduction trees. Board computes `(score * rsqrt) * v`; stock computes `(v * rsqrt) * score`. Contraction and association can differ even with identical BF16 GEMM outputs. Route slots remain in original order, result rounded to BF16 once. |
| Invalid/nonlocal routes | Must contribute exactly zero without touching stale NaNs in unused output/scores. Include mixed local/nonlocal, all-nonlocal and all-padding rows in tests. |

**No numerical equivalence claim is established locally.** No GPU or actual checkpoint is available. The correct acceptance target is custom-vs-stock error **on the scale of measured stock-vs-stock variation**, plus unchanged generated token sequences, not a loose board-relative-L2 threshold. If stock repeated output is bit-identical, that is a zero baseline; do not replace it with an arbitrary 1% allowance. Tensor non-bit-identity is a risk to investigate with logits/margins and generation tests, not automatic permission to ship.

Disable/remove the board's final `prepare` override and `_MOE_SEL` tuner (`BOARD:701-835`): it times synthetic routes and admits alternate outputs with relative error `<1e-2` (`:786-789`), far too weak for this serving gate. It also runs once **per layer**, clears global choices on each `prepare`, and can trigger extra compilation. Use fixed reviewed thresholds for the first port.

## 3. Post-load weight conversion, lifetime and memory

### Where to convert without touching model_loader/

Hook `MuseSparkNvFp4FusedMoEMethod._process_weights_for_marlin`, **before** the base processing at `QUANT:131`, while the canonical tensors below still exist. Preserve small global-scale metadata; create only approved derived packed banks; then run the existing Marlin conversion and native-name cleanup (`QUANT:128-140`). Do not inverse-permute already prepared Marlin tensors: their names/layouts have changed and the up global scale is gone.

| Canonical loaded attribute | Shape/dtype for this rank | Board logical source |
|---|---|---|
| `gate_up_proj` | `[32,8192,2048]` uint8, low nibble first, `[gate;up]` row order | `w13` |
| `down_proj` | `[32,4096,2048]` uint8 | `w2` |
| `gate_up_proj_weight_scale` | `[32,8192,256]` E4M3FN | `w13_sf` |
| `down_proj_weight_scale` | `[32,4096,256]` E4M3FN | `w2_sf` |
| `gate_up_proj_weight_scale_2` | `[32,2]` FP32 | **column 0 only**, contiguous FP32 `[32]` |
| `down_proj_weight_scale_2` | `[32]` FP32 | contiguous FP32 `[32]` |

Allocation evidence: `QUANT:98-115`; packing low-nibble convention: `BOARD:471-478`. Row order for custom bank interleaves 8 gate rows then corresponding 8 up rows (`BOARD:592-603`), **not** the FlashInfer gate/up swap at `QUANT:36-39`. Block scales stay FP8 bytes; Marlin's encoded block-scale format (`REPACK:29-44`) is **not** the custom bank's FP8 input format. Per-projection input calibration scales are not applied in this W4A16 Marlin path.

At `MODEL:1695-1709`, model-side `post_load_weights` freezes effective norm weights; quant postprocessing may be a separate outer loader phase. Do not assume both have completed inside one hook. Repack matrices in quant postprocessing, bind norm from `FP4.set_post_expert_norm` once loaded, and explicitly initialize extensions/scratch before graph capture. `RUNNER:673-693` establishes load -> post-load transforms; `RUNNER:912-925` allocates KV pools later. Integration must verify actual loader ordering with a test, not move loading code into a forbidden directory.

### Memory ledger (binary units, excluding tiny globals/norm/LUT)

Per expert: three 4096x4096 FP4 matrices. Per rank/layer:

* FP4 codes: `32 * 3 * 4096^2 / 2` = 805306368 B = **0.75 GiB**.
* FP8 group scales: `32 * 3 * 4096^2 / 16` = 100663296 B = **0.09375 GiB**.
* One complete code+scale layout = **0.84375 GiB**.
* `BOARD:615-625` retains base mma, fat, unit and WGS layouts = **3.375 GiB**.
* `BOARD:638-651` scratch = **164668960 B = 157.04 MiB = 0.15336 GiB**. This includes the generously sized WGS fixups/x buffers. It is allocated **per prepared state** in the board file.

The model defaults to 62 layers (`CONFIG:25-31`); first/last expert BF16 exclusion is noted in `MODEL:1029-1030`. For the expected 60 NVFP4 layers, verify the actual checkpoint exclusions at startup:

| Storage choice | Additional bank / GPU for 60 layers | Verdict |
|---|---:|---|
| unchanged board bank + stock Marlin | 202.5 GiB, plus 9.20 GiB per-layer scratch | Impossible |
| unit + WGS only + stock Marlin | 101.25 GiB | Impossible |
| one additional layout + stock Marlin | 50.625 GiB | Not a safe all-layer plan; almost exhausts static allowance after model weights, before KV/draft/graphs |
| two layouts on N layers | `1.6875*N` GiB + shared 0.15336 GiB scratch | Bounded proof possible, subject to measured KV/draft/graph headroom |
| one shared resident layout, Marlin and new kernel both read it | near-zero duplicate weights | Full-model deployment target; requires a new layout-reading kernel path |

The raw 816B/8 FP4 estimate is 51 GB/GPU **before** FP8 scales, BF16-excluded experts, dense weights, draft model, and workspaces. Stock NVFP4 experts alone are about 50.625 GiB/rank plus ~6 GiB for two BF16 expert layers; other weights add more. The advertised 141 GB is not all available for banks: judge `mem_fraction_static=.85` leaves a runtime reserve, and stock KV allocation consumes remaining static space. Query actual CUDA total bytes rather than assuming decimal GB equals GiB. One extra full-model layout can consume essentially all remaining static space even if bare weights still fit physically.

### Implementable bounded first port

1. For PRMT-compatible selected layers keep **only unit-order** (TINY/TINY2/MID) and **WGS-order** (MAIN) banks. Existing selected build paths use these, not the retained base/fat copies. Remove unreachable layout dependencies cleanly; validate all four builds. Unsupported scales/layers stay wholly Marlin (do not drag in fallback layouts for the first version).
2. Select at most N layers by a deterministic model-wide byte budget, same layer indices on all ranks. Example: N=4 costs 6.75 GiB banks; N=8 costs 13.5 GiB. These are sizing examples, **not an assertion of free capacity**. Start with one or two layers for a diagnostic correctness proof; choose a serving candidate only after stock residency/KV capacity is known. Do not silently allocate arbitrary duplicates until OOM.
3. Share one scratch arena across sequential custom MoE calls on one execution stream, including across layers and graph batch sizes. The board already requires sequential replay (`BOARD:6`). Eager calls and graph captures/replays must use the same serialized ownership. Do not support PDMux, two-batch overlap, cross-stream concurrent forwards, or parallel graph replay with that arena; fall back unless separately allocated per execution lane. A fresh scratch allocation **per captured batch/layer** would destroy the budget.
4. Repack one expert at a time into preallocated final banks, avoiding `list + torch.cat`'s full extra destination copies and huge int64 expanded-nibble temporaries. Free temporaries between experts/projections. Keep Marlin parameters for all layers/prefill; delete native aliases exactly as stock already does.
5. Allocate/account banks before KV sizing; report resulting full/SWA KV token capacity and runtime reserve. `RUNNER:917-925` sizes the pool from current memory. Bank memory thus trades against KV cache and can regress C=32/long-context throughput or TTFT even when the kernels are faster. No-regression requires evidence for that trade, not merely successful allocation.
6. For mutable weight updates, invalidate/rebuild derived banks outside capture or reject that mode; do not keep stale weight-derived state. These judge runs are static-weight inference.

Concretely, tag experts with their parent layer index at construction (`MODEL:627-645`) so quant postprocessing can apply the same selection on every rank. Bind a small model-owned bank-budget/scratch-owner object to eligible experts after model layer construction (`MODEL:1146-1163`), without registering parent modules as children. Allocate its arena once during setup, not in `kernel_fn`, and hold it alive for the lifetime of captured graphs. Compile/eligibility failures and any memory-derived coverage decision must be resolved consistently across ranks before capture; never change the selected set in response to runtime tensor values or replay timing.

**Not recommended:** repack all weights on every prefill/decode mode switch, CPU-offload the custom bank and upload per step, or drop stock weights and call a new prefill implementation while claiming prefill stayed Marlin. Those alternatives change the cost model drastically and are not this plan.

For full coverage, the promising direction is teaching the persistent kernel to read **Marlin's already resident tiles/scales**, eliminating duplicate weights and canonical bank retention. That requires deriving Marlin tile/scale addressing and adapting mma/WGMMA tile loads; timings and precision must be re-established. Conversely modifying Marlin to consume a custom layout changes a substantial generic GEMM implementation. Choose one explicitly before that larger phase-2 rewrite.

## 4. Token counts, phase dispatch and CUDA graphs

**Eligibility:** Muse NVFP4 + resolved Marlin + exact SM90 Hopper + TP8/EP8/expert-TP off + square 4096 experts/8 routes/32 local experts + BF16 input + compatible weight-first norm/epsilon + resident validated bank + no fused AR/overlap/SP + **decode OR target-verify** + `1 <= T <= 288`.

* T=0: return empty local zeros before launch. All other unsupported cases use unchanged Marlin.
* T=1..4 -> TINY; 5..8 -> TINY2; 9..16 -> MID; 17..288 -> WGS MAIN. These are `BOARD:332-360,711-718`, **not** the final autotuner's synthetic choices.
* Keep original 384-token scratch bound initially; 288 fits in one launch. Avoid a simultaneous scratch-shape rewrite before proving correctness. Validate 17/31/32/33/63/64/65 and the upper bound; board timing points do not cover every T or routing skew.
* **Every prefill/extend stays Marlin**, including a 1-token prompt or short final chunk. T>288, including chunks to 16384, stays Marlin. No heuristic `T<=288 => decode`.
* Thread a keyword-only `allow_decode_kernel=False` (or equivalent forward mode) from `MODEL._forward_tp` through `_forward_experts` and `FP4.forward` to `_forward_marlin`. `_forward_dp_attention`, SP, missing `forward_batch`, and other callers default to stock. A context-based phase getter is possible, but the explicit bit makes capture behavior/test coverage clearer and is only a small model-local signature change.

**Correction to the conservative 32*9 envelope:** this checkout's normal DFlash verify has width **8**, not 9. `DFLASH:2765-2774` builds an anchor plus `block_size-1` proposals, and `:2785-2792` passes exactly `block_size` tokens into target verify. `srt/speculative/dflash_info.py:60-64` sets `num_tokens_per_req=draft_token_num`. Thus C=1/4/16/32 normally produces T=8/32/128/256. Supporting through 288 is harmless defensive coverage, not a claim that the judge ordinarily verifies 9 rows/request. Do not change draft count to exploit a different kernel variant.

The graph system already gives fixed shapes:

* `GRAPH:299-313` gets request width from `RUNNER.decode_num_tokens_per_req` (`RUNNER:827-848`) and uses TARGET_VERIFY capture under speculation.
* `GRAPH:331-378` builds request buckets and max token capacity. Default speculative BS buckets through 32 are `[1..8,10,12,...,32]` (`srt/arg_groups/cuda_graph_hook.py:641-654`), subject to existing alignment/pool filtering (`srt/model_executor/runner/base_cuda_graph_runner.py:64-95`).
* Each captured shape has `T=BS*captured_req_width` (`GRAPH:1174-1182`). An odd live batch can be padded to the next bucket; kernel selection uses **captured** T, and device routing handles changing IDs/weights per replay. Padded routes `-1` must be safe even when all routes on a rank are nonlocal.
* `GRAPH:1060-1069` warms up before capture; still compile extensions / bind norm / allocate and initialize scratch explicitly before this point. Do not compile, synchronize, tune, read a device scalar, or resize persistent banks during capture/replay.
* Preserve the board's double-buffered sums-of-squares and generation-counter reset protocol. Replaying multiple graph sizes sequentially with the same scratch is legal only if every launch fully completes its reset, including zero-local launches. Capture executes kernels too, so do not initialize counters by assumptions about which graph was captured first.
* The C++ launcher uses `at::cuda::getCurrentCUDAStream` (`BOARD:396 [CUDA:2671]`); preserve it, add a tensor-device guard and launch-error checks, validate resource/occupancy bounds. No default-stream launch or device-wide sync in forward. Preinitialize dynamic shared-memory attributes and check CUDA errors; the board currently ignores these errors (`[CUDA:2663-2669]`). Persistent-grid barriers require the grid to be resident/progress-capable on the target device.

## 5. Zero-local-work hang: artifact discrepancy and required repair

The task says the J==0 guard is already fixed. **It is not in the supplied SHA256 artifact.** In `BOARD:396`:

* `[CUDA:2146-2150]`: `nj = S.njF + S.njS`, then unconditional prefix and both `wgs_range` calls.
* `[CUDA:2140-2143]`: unconditional `wgs_stage_at(S,nj,...)`.
* `[CUDA:2133-2138]`: when `nj==0`, the binary-search loop skips but reads **uninitialized `S.jC[0]`**, then divides by `wgs_cost` of that value.
* `[CUDA:1665-1666]`: `wgs_cost` selects a byte using shift `8 * (ceil(cnt/8)-1)`; a nonexistent zero/garbage count is outside its valid 1..64 domain and can give a zero divisor or invalid shift.
* `[CUDA:1777-1792,1811-1836]`: garbage nonempty stage ranges can issue invalid TMA or wait for nonexistent ready/done flags. This is a concrete unsafe control path; no GPU reproduction was possible here.

Minimal proposed repair: in `wgs_ranges`, after zero-job prefix initialization and **before calling `wgs_range`**, if `nj==0`, set all four `S.wgr` bounds to zero and return **from the helper**, not from the main kernel or producer before its required joins. Consumers/producers must still reach the established CTA joins, global generation barrier, flag resets, and final zero output. A defensive `nj<=0 -> 0` in `wgs_stage_at` is additional hardening, not a substitute for testing barrier participation.

Test zero-local work for every variant and graph shape, then alternate zero/nonzero calls and graph sizes for >=1000 replays in a watchdog-bounded GPU subprocess. Include all `ids=-1`, all experts owned by another rank, all eight EP offsets, and actual T=17..288 WGS. Low-batch EP zero work is ordinary: under independent uniform ownership, a rank receives no route at T=1 with probability `(7/8)^8 ~= .344`; correlated routing can create zero-local ranks at larger T too. Never infer nonzero work from T alone.

## 6. JIT compilation, cache, startup and patch rules

### Packaging

Put source under `python/sglang/kernels/...` or a model-local subdirectory:

* one CUDA `.cu` with the decoded board source, documented provenance and retained license;
* a tiny `.cpp` PyBind wrapper exporting the launcher (or use equivalent generated bindings);
* a Python module for bank conversion, guarded dispatch and extension cache;
* small edits to `MODEL`, `FP4`, `QUANT`; tests as allowed `.py` files under `python/sglang/`.

**Yes: standalone, sgl-kernel-free `.cu` sources are permitted by the supplied rules.** `RULES.md:52-55` allows `.cu/.cpp/.h/...` beneath `python/sglang/`, forbids edits beneath `srt/model_loader/`, and caps the unified diff at 400 KB with no added line >400 characters. No binary, rename, mode change, external download, or dependency on the local muse checkout in the patch. The external design Markdown is not part of the submitted diff.

`BOARD:396` cannot be pasted literally: its one line is ~144 KB; decoded CUDA also has **nine lines >400 characters** (maximum 552). Split strings/source and wrap those declarations/asm operands without changing executable expressions. The decoded `.cu` is ~140 KB, so even including Python conversion/bindings/license and diff prefixes a clean port should target **<220 KB**, leaving room for tests and other serving patches. Measure `git diff --binary a55ab7d5 -- python/sglang` byte count and added-line lengths before any check; do not count just source bytes.

Keep original Apache-2.0 license/notices. The supplied file explicitly attributes Vjeux/okc-2026 at `BOARD:2-3` and Lee Penkman/lee101 at `:697-699`. The brief names keell0renz lineage; preserve/verifiably reconcile upstream notices during implementation rather than replacing actual artifact attribution with only that name. A license can be retained in source comments to obey the allowed-extension rule.

### Build once, load per rank; not compile once per layer

`BOARD:413-441` currently calls `load_inline` and caches extensions per define tuple per Python process. It explicitly targets `compute_90a/sm_90a`; it **does not explicitly set C++20** in this file despite the brief. In the port pin the intended standard in host and CUDA flags (avoid duplicate/conflicting `-std` flags), arch, ABI, and build definitions. Do not enable global fast math as a performance shortcut for norm/GEMMs; preserve the board's explicit intrinsics and stock BF16 boundaries.

Prefer checked-in `.cu/.cpp` plus existing `srt/utils/cpp_extension_loader.py:116-158` (`load_extension_with_recovery`) rather than a giant Python `load_inline` string. Its interprocess flock handles dead-process stale Torch lock files (`:94-113`). A suitable identity is hash(source + flags + Torch/CUDA ABI + architecture + variant), stable across TP ranks, **no rank suffix**. Respect `TORCH_EXTENSIONS_DIR`, otherwise Torch's normal cache root (`cpp_extension_loader.py:25-45`); do not assume it survives between remote jobs. Every TP process imports the same library; the first process builds a variant, the others wait/load it. There is no reason to compile eight identical copies on one node. If maintaining `load_inline`, it needs the same crash-safe serialization and full cache key; vanilla persistent lock files are a startup-hang risk.

Compile the four fixed variants once per node/source version, load once per rank, and reuse across all selected layers and graph sizes. If non-PRMT layers are supported later, they may require separate variants; first version can stay stock on those layers. Do **not** run `_moe_select` per layer. Limit compiler parallelism and avoid an eight-rank compiler-memory stampede. No distributed collective should sit inside an unbounded compilation wait; validate cold-start against distributed initialization timeouts too.

**Compile wall time is unknown here.** There is no nvcc/GPU in this environment; neither seconds nor minutes can be claimed for this large source/four heavy templates. Operational target: incremental cold-start build+pack well below ~5 minutes, measured separately from existing load/capture; if it approaches that target, reduce variant count or reconsider packaging before a scarce evaluation. The 90-minute scored-node reservation is not proof of a 90-minute readiness allowance. `entrypoints/http_server.py:2376-2396` uses warmup timeouts (defaults 600/1800 seconds when the environment override is absent), and `srt/environ.py:334` defaults `SGLANG_WARMUP_TIMEOUT=-1`; external judge startup deadlines are **not specified in the provided RULES.md**. `SGLANG_TIMEOUT_KEEP_ALIVE=1800` is HTTP keepalive, not a compiler/readiness budget. Cannot promise startup acceptance without first remote logs.

Compile failures should make one logged, rank-consistent stock fallback before graphs are built; kernel launch/illegal-memory/correctness failures must not be swallowed after CUDA state is poisoned. Never try an unbounded automatic retry or silent precision relaxation.

### Backend selection / mergeability

Do **not** add a required `--moe-runner-backend board_moe` value: judge pins `marlin` (`RULES.md:25,31`). Keep `use_marlin=True`, stock weight preparation, and a narrowly scoped **automatic subpath for eligible Muse/Hopper decode/verify**. Default-on for proven supported layers/shapes; a debug-only default-preserving environment opt-out could assist local comparison but must not be required in judge flags. All other models, GPUs, prefill, BF16-excluded experts, norm forms and unsupported execution modes keep their original code. Avoid touching enums, generic Marlin, scheduler, attention, or the drafter. This makes eventual cherry-picking to other serving changes localized.

## 7. Expected speedup: kernel gain is not server gain

The supplied board per-call timings are inputs to this design, not measurements repeated here. Under **full coverage with the original fast layouts**, normal DFlash maps as follows:

| Serving concurrency | Target T | Stock/custom board us | Expert-call ratio R |
|---:|---:|---:|---:|
| 1 | 8 | 147 / 65 | 2.26 |
| 4 | 32 | 207 / 127 | 1.63 |
| 16 | 128 | 474 / 183 | 2.59 |
| 32 | 256 | 618 / 216 | 2.86 |

C=4 reuses the T=32 board point; routing/load balance may differ from the named decode32 workload. Non-speculative T=1/16/32 ratios are 3.08/2.02/1.63 from 80/26, 188/93, 207/127 us, but DFlash spends most target decode iterations verifying blocks, not doing 32 independent single-token expert calls.

Let `p` be the fraction of **whole-server wall time** in replaceable routed-expert work, including the fraction of layers actually ported. If everything else and DFlash acceptance remain fixed, `S = 1 / (1 - p*(1 - 1/R))`. Draft time, attention/KV reads, dense projections, router, pre-norm, all-reduce, first/last BF16 experts, prefill, and host overhead do not get R. Do not multiply per-call speedup by TP size; collectives wait for the slowest expert-owning rank, whereas board routes were tuned for a busy rank.

Illustrative **full-coverage**, decode-heavy whole-server scenarios (not observed profiling):

| C | Assumed whole-server replaceable fraction p | Amdahl speedup |
|---:|---:|---:|
| 1 | .10-.30 | 1.06-1.20x |
| 4 | .18-.40 | 1.07-1.18x |
| 16 | .26-.55 | 1.19-1.51x |
| 32 | .26-.60 | 1.20-1.64x |

The supplied layer ablation (MoE 20-90 us within layers 67-470 us) supports *material but non-total* MoE share, not a unique p: those endpoints are not a paired C-specific serving profile and exclude draft/prefill/host costs. If end-to-end p is only .2 at C=16, even R=2.59 gives **1.14x**. If long-context attention/prefill dominates, gains are smaller. Throughput at stock's interactivity also depends on TTFT and scheduling, not only the inverse of a target step duration.

**Memory-bounded proof expectations are much lower.** With N of ~60 similarly costly NVFP4 layers accelerated, replace p by `p*N/60`. At C=16 the table's p range gives roughly 1.01-1.02x for N=4 and 1.02-1.05x for N=8. At C=1/4 N=4 is roughly 1.00-1.01x; at C=32 roughly 1.01-1.03x. A one/two-layer correctness proof may have immeasurable server gain. Any KV-capacity or startup regression can erase that benefit. These small returns are why an all-layer **memory-neutral layout** is the real optimization opportunity, not a license to allocate four layouts everywhere.

## 8. Staged validation and use of daily checks

Opening is Nov 10 at 09:00 PT. **No remote actions now.** On an explicit implementation go, do CPU/static work first; no local GPU performance claim.

### Stage A: implementation gates before any remote check

* Choose bounded proof vs memory-neutral rewrite explicitly; fix source identity/provenance and architecture scope. Small proof should use unchanged arithmetic, fixed four variants, and a deterministic selected-layer bank cap.
* CPU/static tests: pack/unpack byte permutations on exhaustive nibble values and distinctive row/K patterns, all EP offsets, canonical-shape/dtype checks, unsupported-mode fallback, phase selection (including tiny prefill), memory byte ledger, cache-key identity/locking, patch path/size/line rules. Expected values come from the canonical NVFP4 encoding and original board packers, not from self-generated custom outputs.
* Write an opt-in startup differential harness under allowed source paths. GPU portion runs **once on bounded representative loaded layers**, before graph capture, using the actual stock `_forward_marlin` as reference. It must not be a permanent per-request comparison or a synthetic performance-based selector. Do not add judge flags to invoke it; make the first diagnostic patch run the bounded gate during normal startup and fail visibly if it fails.
* Cases: T=0/1/2/3/4/5/7/8/9/15/16/17/31/32/33/63/64/65/127/128/255/256/287/288, realistic routing plus skew (all local, one expert hot, eight hot experts, all nonlocal, mixed, all padding), offset0..224. Gate/up scales deliberately unequal; scale bytes at normal/subnormal/zero boundaries; BF16 ties; zero/tiny activations and near-epsilon norms. Unsupported scale-byte banks fall back rather than silently run PRMT.
* On GPU use repeated stock outputs, repeated custom outputs, normalized L2/max absolute/ULP/NaN counts, and where feasible GEMM1/SwiGLU/GEMM2 diagnostic taps to locate the first rounding mismatch. Compare against stock's own repeated and eager/graph variations for the same shape, not against a different finalizer or different T.
* Watchdog-bounded hang tests and alternating graph-shape replays; initialize extensions/attributes once before capture. Check all eight ranks and slowest-rank time. Source errors/faults are failures, not automatic fallback successes.

### Day-one 3-check allocation (all checks fixed C=16)

1. **Correctness/no-regression check first.** Minimal selected-layer implementation with a strict startup differential gate and logs: compile/build seconds, pack seconds, resident bank/scratch bytes, loaded layer/scale/norm eligibility, KV capacity, captured T buckets, custom/stock dispatch counts, peak GPU memory, output/acceptance metrics available from the evaluator. Require official answer correctness, exactly 220-token requests, no hangs/replay faults, no throughput regression outside noise, and TTFT <=2x stock. A tensor test alone is not a pass. If only a one/two-layer proof is ready, call it diagnostic, not a performance candidate.
2. **Only after check 1 passes:** one controlled change, normally bank coverage under a measured capacity budget (or the shared-layout kernel if separately ready). Same arithmetic and graph settings. Look for C=16 target-step improvement *without* lower DFlash acceptance, KV eviction pressure, readiness problems or TTFT regression. If check 1 failed, use check 2 for one evidenced correctness/resource fix instead; no speculative tuning pile-on.
3. **Reproducibility/final candidate:** repeat the retained candidate or validate one specifically justified repair. Reserve this check rather than spending it on four-way variant autotuning. If correctness remains unresolved, stop; do not spend the daily scored run hoping that tolerance passes.

Only after those gates should the caller consider the one scored run/day to observe C=1/4/16/32. A C=16 check does **not** exercise the ordinary C=1 TINY2 or C=4 WGS T=32 serving shape sufficiently, nor establish C=32/long-context capacity. Cover them with the startup GPU harness first and label real C=1/4/32 end-to-end behavior unknown until the scored load. No score prediction should be reported as measured.

## Remaining risks / explicit follow-ups

* **Architecture decision:** memory-bounded proof is straightforward but probably only a few percent server benefit. A memory-neutral Marlin-layout persistent kernel is a materially different implementation requiring another design/validation pass; caller should consult the numerical/kernel architecture owner before authorizing it.
* **Missing J==0 guard:** critical even if historical notes say fixed; reconcile exact file/version first.
* **Token equivalence:** rounding order, scale encoding, weight-first epsilon and WGS split sums may change generated tokens; passing a permissive kernel-board check cannot establish serving acceptance.
* **Resource headroom:** KV residency, draft allocations, large-prefill workspaces and capture memory are workload-dependent. Neither advertised GPU capacity nor a successful single-layer pack proves full-server feasibility.
* **Cold start/toolchain:** build time, NVCC/host C++20 support, compiler memory and external readiness/distributed timeouts remain unmeasured. Cached builds are not guaranteed across jobs.
* **Concurrency:** shared scratch is correct only with serialized model execution and stable graph replay ownership; default overlap scheduling is not itself permission for two simultaneous GPU forwards sharing counters.
* **Source-only deliverable:** no runtime imports from `muse/board_best`, no binaries, no flag changes, no edits under `model_loader/`, preserve every expert and norm and BF16 activation boundary.

## Phase-1 verification performed

Read the full 61-line serving RULES.md and the requested model, FP4/Marlin, quantization and runner/graph paths at the pinned commit. Parsed the board file with Python AST without importing/executing it; inspected decoded CUDA, established the missing guard path, measured source/line sizes, and independently recomputed tensor-memory byte counts. `nvcc` is absent; no CUDA compilation, GPU execution, model loading, serving checks or submissions were attempted. Worktree implementation files remain unchanged. No commit/push was made; this external design document is not a legal submitted source patch.
