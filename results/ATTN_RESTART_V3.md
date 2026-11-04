# Restart v3: diagnose the loss, retire full-context warp MMA

**Check file:** `/Users/achita.anantachina/.local/share/opencode/worktree/e72615/loop-attn-restart/candidates/attention_restart_v3.py`

Branch `loop/attn-restart`. V1/v2 files remain unchanged. No check or push was made by this session.

## Measured evidence and conclusion

Parent-owned checks `2847448242174b71` and `c36537ede3ff45c1` compiled/executed v1/v2 correctly but adopted no restart arms. Their paired setup measurements are stronger evidence than the initial roofline hypothesis. `scripts/analyze_attention_restart.py` parses those logs into `attn_restart_slowdown.json`, using the **exact reproducible synthetic sequence lengths**, not an assumed fixed 80K length:

| v2 case/control | incumbent gross µs | restart gross µs | useful KV TB/s |
|---|---:|---:|---:|
| v16F, T64, split32, parallel, FP32 partial | 100.51 | 333.70 | 0.970 |
| v32F, T64, split16, parallel, FP16 partial | 186.51 | 635.47 | 1.030 |
| d16F best | 89.95 | 103.81 | 3.145 |
| d32F best | 172.10 | 189.20 | 3.512 |

These are useful bytes / **gross graph time**, not HBM hardware counters. The fixed graph cost is included on both sides. V2's saved log is truncated before its sliding measurements; the complete v1 log supplies the small-case controls: d1S 13.87 vs 8.61 µs, d16S 16.90 vs 11.55 µs, v1S 22.11 vs 9.54 µs (best listed restart choices).

**Decision:** incremental copy-pipeline tuning cannot credibly recover the required 3.3–3.9× verification improvement. A competitive full-verify kernel needs native warpgroup compute/operand reuse and overlap like the incumbent, not this legacy warp-MMA consumer. Merely recreating those mechanisms would recreate the already-optimized incumbent. Batched full decode was closer, but still 10–15% behind its best controls with only modest remaining random-gather headroom; d1F was ~3× slower. V3 therefore **does not gate or dispatch any full-context production arm**. All full cases use the unmodified incumbent.

## Why it loses: what is established, and what is not

### 1. Consumer issue/operand construction, not an 8× HBM reread

The restart globally gathers each KV row once and uses all 128 bytes of each PAGE=1 row in each allocation. Eight copy lanes cover each aligned row. There is no eight-query-head multiplication of HBM bytes, no missing coalescing factor, and no contiguous-page assumption.

But verification's four warps/head each construct their own K/V operands from shared memory. Per **64 keys / both heads**, the new source issues:

* 128 warp QK MMA operations and 256 warp FP16-PV MMA operations;
* 256 scalar warp shared K loads, reading 32 KiB versus the 8-KiB raw K tile;
* 128 uint2 warp V loads, reading another 32 KiB versus the 8-KiB raw V tile;
* 256 byte-permutation and 512 FP8-to-half2 warp conversion operations, plus repeated address construction;
* explicit row-max/denominator reductions and FP32 output rescaling on every compute microtile.

These are **source/PTX operation counts**, not SASS counts or exact cycle attribution. In contrast, v32F's actual incumbent `vj` path uses two 64-query-row warpgroups, native FP8 WGMMA QK/PV, one V transpose shared by each head's warpgroup, fused denominator columns, lazy rescaling, and four independent producer warps. It needs eight warpgroup MMA operations per equivalent 64-key tile (32 collective warp issue events versus 384 warp MMA operations), and overlaps producer copies with consumer work.

**Important route correction:** v16F does **not** use that generic `vj` path. `_zh_route` dispatches batch16 full verification to `wst_ct14b<4,2,4,1>`: a balanced cooperative grid, 128-key tiles, 512 producer threads plus 256 consumer threads, four 32-KiB raw K/V slots, 128-key QK WGMMA with packed FP16 scores, approximate packed FP8 probability construction, native FP8 PV WGMMA, and fused denominator columns/fold. It uses about six warpgroup MMA operations per equivalent 64-key tile and avoids the restart's precise scalar exp2/packing path. Both incumbent routes have 24 resident warps/SM by their intended layouts; v2's 256-key variant has only eight.

This explains why a theoretical FP16-PV FLOP floor was misleading: arithmetic peak does not account for tiny warp instruction shapes, operand conversion, address/control issue, and dependency chains. **The exact machine-instruction mix and any additional spilling are still unknown**; I do not claim measured cycle percentages or unverified FP8-mma lowering behavior.

### 2. Little's law: capacity versus actual offered traffic

At 4.8 TB/s, 1 µs latency and 132 SMs, the requirement is **36,364 bytes = 35.51 KiB/SM**. Future-copy *capacity upper bounds* are:

| geometry | intended CTA/SM | raw pipeline | maximum future bytes/SM |
|---|---:|---|---:|
| v2 verify T64 | 2 | 3 × 16-KiB stages/CTA | 64 KiB |
| v2 verify T256 | 1 | 2 × 64-KiB stages/CTA | 64 KiB |
| incumbent v32F | 2 | 4 × 16-KiB stages/CTA | 96 KiB |
| incumbent v16F WST | 1 | 4 × 32-KiB stages/CTA | 96 KiB |
| v3 T64 | 2 | 4 × 16-KiB stages/CTA | 96 KiB |
| v3 verify T128 | 2 | 3 × 32-KiB stages/CTA | 128 KiB |

The current stage is excluded. These are **not measurements of bytes actually in flight**. Copies that have landed no longer hide HBM latency. V2's 256-key change did not increase the maximum per-SM future-copy capacity at all: the lost CTA canceled it. It also lost consumer/producer scheduling opportunities and was slower.

The v16F control corresponds to **2.228 µs per equivalent 64-key tile/SM** versus **0.671 µs** for its incumbent control. At 0.970 TB/s and the assumed 1-µs latency, average outstanding traffic would be only ~7.35 KB/SM. The consumer requests the next tile too slowly; larger buffers fill and then sit ready. A 4.8-TB/s service period would be ~0.451 µs/tile/SM. Increasing nominal staging allocation alone cannot bridge this issue-rate gap.

Exact host-compiled layout checks: v2 T64 verify 61,568 B/CTA; v2 T256 verify 143,488 B/CTA; incumbent `vj` full verify 113,664 B/CTA (two fit); actual v16F WST dynamic allocation 205,008 B/CTA (one fits). V3 T64 verify is 77,952 B, and T128 verify 110,720 B: both can fit two CTAs by shared-memory capacity. Actual register-limited occupancy remains to be measured.

### 3. A concrete, independently proven decode-tail bank defect

V1/v2's local stripe exchange used `row*64 + 16*t + d`. For one scalar access, eight query rows and the two same-parity `t` lanes land on each bank: **16 different words/bank**. Even favorable vectorization retains four-way conflicts per 128-byte wave. This is an *epilogue* defect, **not** a false claim that the entire shared K/V staging is conflicted. The raw K load pattern and mandatory 128-byte waves of the V uint2 loads are conflict-free in the static bank model.

V3 replaces the exchange with a padded, lane-major 18-float peer state. Only the second stripe writes; the first retains its own state in registers. Scalar bank degree falls **16→2** (paired 64-bit accesses can do better), and exchanged state drops from 8 KiB of data plus statistics to **4,608 bytes**. This is the strongest evidence-backed fix for d1S/d16S.

### 4. Merge is a separate small-batch problem

For batched full verification, changing FP16/FP32 partials barely affects the 330–660 µs loss: v16F split32 parallel is 337.58/333.70 µs; v32F split16 parallel is 635.47/635.86 µs. Thus partial format is not the dominant batched full-case bottleneck.

At v1F, however, immutable split rows total 8.65 MB for 264 FP32 splits; half means reduce that to ~4.33 MB. A single last-arriver CTA serially folding those rows is not competitive. The old parallel merge also launched only **eight CTAs** for B1,L8, with a dependent load/use per split and eight dimensions/lane. FP16 conversions/merge scheduling may contribute to the especially poor half last-arriver result; without ptxas/stage timings, its exact cause must not be invented.

V3 keeps overflow-safe FP16 means but uses a warp/row and two dimensions/lane for verification, **eight independent partial-row prefetches**, and **32 merge CTAs** for B1,L8. Decode retains eight lanes/row so one CTA handles all sixteen output rows together, with the same batch-eight prefetch. This attacks reduction concurrency and load latency rather than assuming smaller payload automatically means faster code.

### 5. Parameter-address lowering is a real remaining compiler question

The old restart passed a large mutable `Params` kernel argument by value, then passed it by reference through helpers containing memory-clobbering inline assembly. Unlike the incumbent `vj` kernel, it omitted `const __grid_constant__`. Taking an ordinary kernel parameter's address can require per-thread local storage; whether the optimizer removed that copy here is **unknown**, not a measured spill claim. V3 explicitly uses the grid-constant contract for main and merge, and includes a preparation-only **plain-parameter control kernel with the exact same improved main body**. Matched attributes and cold main timings will isolate this compiler effect in the next GPU check. The full route can be reconsidered if that control overturns the current consumer-cost conclusion; it is not admitted speculatively.

## What v3 actually changes

1. Padded peer-only decode stripe exchange, eliminating the proven worst bank conflicts.
2. All independent page-table loads are issued before their dependent shuffles/copies; PAGE=1 full-tile interiors use checked 32-bit shift-add addresses, not the generic 64-bit page calculation. The old cache-layout correctness path remains for tails/page64.
3. Four-stage 64-key ring and a 128-key verification option; QK/PV still computes at most 64 keys at once. Copy operations are asynchronous and preissued ahead of consumption. The 96/128-KiB future-copy capacities exceed the Little's-law target, but are explicitly not claimed to fix the full consumer.
4. Batch-eight-prefetched merge with normalized FP16 or FP32 partials: warp/row for verification, eight lanes/row for decode; retain both last-arriver and parallel protocols where applicable.
5. Explicit grid-constant parameter storage for main/merge, with a matched plain-parameter diagnostic control rather than an unsupported claim about old spills.
6. Gate **only d1S, d16S, v1S**; all other cases keep the incumbent. Both cold-L2 ABBA rounds still require ≥2.5% wins and the public/tighter FP8-reference correctness bounds. No hardcoded winner.

## Register/compiler evidence gap and next-check diagnostics

`nvcc`, `ptxas`, and NVRTC were not found in PATH, Spotlight, or bounded common SDK/cache searches. A broader user-tree search reached its 120-second limit without a match. This ARM macOS host cannot run the required GPU code, so no local `ptxas -v` result is claimed.

V3 enables verbose `-Xptxas/-v`-equivalent output and **also prints persistent runtime resource rows** using `cudaFuncGetAttributes` and occupancy queries: `(L,tile,last,half,numRegs,localBytes,sharedBytes,activeCTA,stages,maxThreads,gridConst)`. Tile zero identifies a merge kernel, and gridConst zero identifies the matched plain-parameter control. `localBytes` may include stack/local arrays as well as spills; ptxas output is needed to distinguish them. Final resource rows are repeated at the end so the judge's log-tail limit cannot hide them.

Three bounded setup-only probes split the main and merge graph stages on independent synthetic full-attention inputs: v16F FP32, and v1F FP32/FP16. Each probe starts main after READ512 and measures merge immediately afterward, warm. V16F also times the matched plain-parameter main, separately READ512-flushed. Times include their separate graph replay costs. They are **diagnostic only**; full-context production dispatch never uses the restart, and no synthetic outputs/partials/graphs survive preparation. These probes can resolve the remaining main/merge and parameter-copy/spill questions in the parent's next check without redoing a full-context admission sweep.

## Validation and expected effect

* 52 CPU-emulated public-judge variants: pass; worst FP8-reference relative L2 **0.0028581**.
* 32 analytical constant/zero-V variants: exact BF16 outputs.
* 128 page/broadcast/32-bit-address/tail copy cases: exact bytes.
* Padded stripe-state storage/roundtrip and bank degrees: pass; independent raw K/V bank checks pass.
* 3,048 logical three-/four-stage ring iterations, batch-eight merge tails, and unique warp-row output ownership: pass.
* Python compilation/import and Clang CUDA host/device **syntax with API stubs**: pass; not real CUDA-header/PTX compilation.
* V1/v2 file hashes remain unchanged. Native v3 compilation, register/spill counts, GPU races, and speed: **unverified**.

Expected effect: **no full-case regression**, because every full-context call uses the incumbent. The only plausible gains are recovery of local exchange/prologue/merge latency on d1S/d16S/v1S. The old gaps were large, especially v1S, so **zero adoption is still possible**; no positive score estimate or measured v3 speedup is claimed. Next action is one parent-owned check of the exact v3 file and inspection of `RR3 resources`, `RR3 phases`, and the three admission summaries.

Evidence: `attn_restart_slowdown.json`, `attn_restart_v3_cpu.json`, `attn_restart_v3_cuda_syntax.json`; reproducible analysis/validation scripts under `scripts/`. Local branch-only ledger rows record the parent check results and v3 work; no main commit/push.
