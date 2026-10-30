# decode-moe-plan: implementation results (2026-10-30, branch `argon/decode-moe-20261030`)

Parent for every Decode candidate: `board_best/decoder_finreg_fp8pv_scored_2.0071.py` (scored 2.0071x,
job `646774bad23e4c7f`). The parent's own Decode checks span 1.9968x to 2.0064x, so single runs cannot
resolve changes below about ±1.5%. Every number below was measured by the official service on an H200.
The raw logs are in `experiments/decode_moe/profiles/`.

## Phase 0: prepare-time A/B harness

**Status: built per candidate. A shared module was not built.**

- The R1 harness is in `build_r1.py` and the WQ profiler is in `build_wq.py`. Each one:
  - compiles nothing extra (it reuses the extensions the candidate already builds);
  - checks every variant's output against the incumbent;
  - times the variants as interleaved cold-L2 CUDA-graph replays;
  - prints a table to the judge log.
- **Constraint found:** Decode `prepare` already takes 254–263 s against the 300 s limit. The first R1 run
  skipped its harness at 256.3 s (job `de55689ac40c483a`). The Decode file also has only about 3–5 KB
  of headroom under 1 MB.
- **Consequence:** MoE-kernel experiments run as profilers on the MoE track. It shares the identical
  kernel and has ample prepare time and file size. Decode then gets only the chosen setting.

## Phase 1: MoE at 16 and 32 tokens (WQ)

Job `6834342880f5474e` (MoE track, 5/5 tests passed, 2.3387x on the unchanged timed path).

- **Change:** a runtime stream-K range quantum `q` per phase. It rounds each CTA's stage range to `q`
  stages, so `q = 16` gives whole units and no fixups.
- **Also profiled:** reduced grids of 120, 100 and 66 CTAs, and MAIN versus MID at 16 tokens.
- **Workloads:**
  - Decode-like routing from the `muse_decode_layer` reference model: T = 16 and 32, three draws
    each, plus one draw at T = 128.
  - MoE-track routing: T = 16 and 32, two draws each, plus one draw at T = 128.
- **Offline (NVRTC sm_90a):** MAIN compiles at 96 registers, 8 B stack and 16/16 B spills (parent:
  52/84 B spills). TINY and MID are unchanged. A CPU model of range partition, owner lookup and fixup-slot
  uniqueness passed on 300 random job sets.
- **GPU result:** every quantum was slower than the incumbent fine split. The table shows the median
  time relative to the incumbent:

| Workload | q = 1/8 | q = 4/4 | q = 8/8 | q = 16/16 | grid 120 | grid 66 |
|---|---:|---:|---:|---:|---:|---:|
| Decode-like T = 16 | 1.058 | 1.102 | 1.138 | 1.079 | 1.019 | 1.299 |
| Decode-like T = 32 | 1.082 | 1.081 | 1.085 | 1.115 | 0.995 | 1.391 |
| MoE T = 32 | 1.036 | 1.054 | 1.087 | 1.106 | 1.019 | 1.513 |
| Decode-like T = 128 | 1.061 | 1.044 | 1.097 | 1.116 | 1.053 | 1.591 |

- **Decision: closed.**
  - Coarser splits cost more in load imbalance than they save in fixups.
  - Fewer CTAs lose HBM parallelism. Grid 120 is within noise.

**16-token skinny tile (`skinny_ring_ws` holding 9–16 tokens per expert): closed, not implemented.**
- The WQ run measured the MAIN (WGS) build against MID at 16 tokens on the same draws:
  - 0.966x on draw 3 (busiest expert has 14 tokens);
  - 1.12x on draw 2 (busiest expert has 8 tokens);
  - 1.06–1.10x on the MoE-track draws.
- So re-streaming the weights of split experts costs at most a few percent, and only when one expert
  is heavily loaded. That is well below the +18% traffic that the earlier analysis predicted as an
  upper bound.
- The change would mean rewriting the tuned mma.sync skinny ring for two n8 token tiles. That is
  multi-day work, testable only through 8-minute judge checks, for a gain bounded at about 3% of
  `moe_fused` on decode 16. **Not worth it under the current constraints.**

## Phase 2: routing + pre-expert norm fused for more than 8 tokens (P2)

| Job | Gate | Tests | Speedup | Note |
|---|---|---|---:|---|
| `ddc5d0263caa4435` | bitwise: FAIL in all 4 modes | 13/13 | 2.0000x (parent path) | ids, normalized rows and local ids exact; SGLang's weights differ by 1–2 ULP (≤ 3e-8) |
| `3b53f3cb23054eb9` | ids/rows exact, weights ≤ 1e-6 relative: PASS (mode 0) | 13/13 | 2.0001x | `_p2_kernel` 4.4–5.4 µs, versus routing_kernel + `_pn` 4.5–5.6 µs: **no gain** |
| `b9c4f2c1a7634ce2` | 1 or 2 warps: FAIL (normalized rows not bitwise equal at T = 128); 4 warps: PASS | 13/13 | 2.0025x | `_p2_kernel` 4.5–5.4 µs: unchanged, **no gain** |

**Decision: P2 closed.** One fused program per token is as slow as the two kernels it replaces. The
kernel time is bound by 24 serial 256-wide reductions, and fewer warps change the norm's FP32 rounding.

**Folding `_pn` into the `moe_fused` prologue: not implemented.**
- The pre-expert norm needs whole-row sums before any expert GEMM reads `x`.
- Inside the persistent kernel this would need one of:
  - a grid-wide barrier, costing about the same as the separate launch;
  - redundant per-CTA row reductions: at T = 32, 132 CTAs × 256 KB = 34 MB of L2 reads;
  - edits to every x-staging path (TMA skinny loads and the WGS gather).
- P2 already removes the routing+norm launch pair outside the kernel and showed that launch removal
  alone is worth at most about 1 µs.

## Phase 3: small-token router/pooler (R1) and `_sfn`

| Job | Result |
|---|---|
| `de55689ac40c483a` | 13/13 passed, 1.9990x. The harness skipped at 256.3 s (cutoff 240 s), so this measures the parent path. |
| `209ff320073b48f6` | Harness ran. Router launched first on the side stream: 1 token 32.38 → 25.38 µs, 8 tokens 40.80 → 34.72 µs. **The judge benchmark recheck FAILED at verify 1 (8 tokens): "routing: 2 of this GPU's experts touched where the reference touches 5".** |
| `522bb253f6144f71` | 13/13 passed, 1.9968x. Same-stream orders (router then pooler, or pooler then router) were 31.4–31.5 µs at 1 token and 40.8–41.7 µs at 8 tokens versus parent 32.2 / 39.9: below the 3% gate, so the parent was kept. |

- **Decision: R1 closed.** Only the concurrent arrangement helps, and it produced wrong routing under the
  judge's repeated-replay recheck at 8 tokens. Inspection did not find the root cause; it was not a
  numeric difference (the eager outputs were bitwise equal in the harness). It must not ship.
- **`_sfn` fold into the sliced unpooler: not implemented.**
  - `_sfn` needs two full-row RMS reductions: over the 1,024-column branch, then the 8,192-column
    updated stream.
  - The sliced unpooler GEMV (`smtrace_gemv`) spreads a row's 1,024 outputs across CTAs, so the fold
    requires a last-CTA reduction inside the Dense kernel family.
  - The bound is 3.5–3.7 µs on the four shapes with 8 or fewer tokens: about +1% score if fully
    removed.
  - Deferred behind the larger levers. It is still the best remaining small-token item.

## R2: exact FP32 router fused into the T = 1 / T = 8 pooler GEMV (Phase 3), the first measured gain

- **Design.** At 1 and 8 tokens the pooler GEMV (v18s_gemv / smallm2_gemv, 128 CTAs × 1024 threads) and
  `route_fp32` (128 CTAs) run as one launch:
  - warps 0–15 of each pooler CTA first compute that CTA's 16 router rows of one 1,024-wide K slice,
    using route_fp32's per-row FMA order, shuffle reduction and parts layout;
  - a named barrier (512 threads) plus one atomic per CTA elects the last CTA. Its warps run
    route_fp32's exact top-8 tail;
  - all warps then run the unchanged GEMV (a separate template instance, OPT bit 64).
- **Build.** The modified CUDA is derived at import from the parent's own templates by exact-anchor
  edits and built on a background thread.
- **File size.** The file was over 1 MB, so the MoE CUDA string is now stored zlib-compressed. It
  decompresses to identical bytes (checked against the parent).
- **Offline (NVRTC sm_90a).** Both fused instances: 64 registers, 0 stack, 0 spills.

| Job | Result |
|---|---|
| `454f650818fb417a` | 13/13 passed, 2.0097x. The R2 extension failed to build (missing `pybind11` include with `no_implicit_headers`), so the parent path ran. |
| `8cdc3919fe5c4f48` | Gate **PASS**, bitwise on pooler output, weights and ids: T = 1 in-process 30.40 → 29.47 µs; T = 8 39.01 → 32.86 µs. Both selected. **13/13 passed, 2.0307x, CI [2.0002, 2.0611]**. Verify 1 measured 123.11 / 123.85 µs (sliding / full) against 126.1–133.9 / 133.7–141.4 µs in the eight parent-path runs; decode 1 was unchanged at 85.06 / 92.63 µs. Prepare took 214 s. |
| `2027d3dda5664795` | **Scored run: 13/13 passed, 2.0339x, CI [2.0034, 2.0644]**, score 11121.5 us against 11270.1 for the published 2.0071x run. Gate passed again (T = 1: 30.94 -> 29.60 us; T = 8: 39.42 -> 32.96 us). Per-shape: decode 1 83.30 / 88.96 us; verify 1 122.91 / 131.41 us. No flags or rejections. At 02:57 UTC the Decode board still listed 2.0071 (job 646774ba), apparently not yet refreshed; overall rank 217 / 1641. |

The verify-1 gain (−5 to −13 µs per shape) is outside the spread of every parent-path run, so it is a
measured improvement. At 1 token the fused kernel (23.8 µs) is only about 1 µs faster than the
separate pooler plus router. The pooler at T = 1 is register-bound at one CTA per SM, so the router's
loads add to its critical path rather than overlapping it.

## After R2: R2b, S1 and the leaderboard

- **Board (02:58 UTC):** Decode **2.0339x, rank 217 / 1003** (was 2.0071, rank 234). Overall rank 217 / 1641.

| Job | Candidate | Result |
|---|---|---|
| `d8fbc31bb5bb43e6` | R2b: router prologue stages x in shared memory and preloads all 32 weights; same FMA order | Gate PASS (T = 1: 30.88 -> 29.12 us; T = 8: 39.26 -> 32.48 us). 13/13 passed, **2.0334x**, a tie with R2's 2.0339x. The fused kernel stayed at 24.0 / 22.4 us, so the load pattern was not the cost. **Not submitted.** |
| `69b63d4def8542ac` | S1: scored R2 plus a prepare A/B of `_sfn_kernel` num_warps (4/8/16/32), bitwise-gated | Every variant was bitwise equal to 8 warps. Graph times were T = 1: 8.10 / 7.87 / 7.65 / 7.62 us and T = 8: 8.35 / 7.74 / 7.26 / 7.81 us (each including about 4 us of graph-launch overhead). No variant was > 3% faster at both token counts, so 8 warps were kept. 13/13 passed, 2.0129x (in this run the R2 gate kept the parent path at T = 1: 30.24 vs 30.53 us). **Measured closure for the `_sfn` launch configuration: at most about 0.5 us.** |

**`_pn` fold into the moe_fused prologue: closed on measurement.**
- P2 (`3b53f3cb`, `b9c4f2c1`) measured the routing + `_pn` pair at 4.5-5.6 us. One fused launch did the same work in 4.5-5.4 us, so removing a launch boundary gains about 0.
- The `_pn_inputs_kernel` itself is 1.5-2.5 us in every profile. That is the ceiling for folding it into `moe_fused`, about 1% of the score.
- The fold would need either a grid barrier or redundant row reductions inside a kernel already at 96 registers. Not pursued.

**`_sfn` fold into the sliced unpooler: not implemented.**
- The ceiling is the kernel's 3.3-3.9 us at the four shapes with T <= 8.
- The unpooler is a cluster-reduction GEMV (`smtrace_gemv`, RED = 2). The fold needs a grid-wide two-stage reduction over 8,192 columns, and FP32 reduction order cannot match the Triton kernel bitwise.
- Lower value than removing the fused router's measured 10 us overhead (next item).

## Phase 0: shared harness (done)

- `experiments/decode_moe/harness.py`: `HARNESS_SRC` with these helpers, appended once by builders:
  - `_okh_same`: bitwise equality, or per-tensor relative tolerance;
  - `_okh_time`: interleaved cold-L2 CUDA-graph timing;
  - `_okh_pick`: margin selection with an optional allow-list;
  - `_okh_budget`: prepare-time budget;
  - `_okh_bg`: background build thread.
- **CPU checks:** equality, tolerance, dtype mismatch, pick margin, allow-list, budget and thread all behave as expected.
- **GPU check:** S1 rebuilt on it (`build_s1h.py`) as job `1f20126e3f994646`. 13/13 passed. Output `S1_SFN T=1 nw8=8.03 nw4=8.06 nw16=7.87 nw32=8.10`, `T=8 nw8=7.52 nw4=8.64 nw16=7.52 nw32=8.29`, `sfn_nw=8`: the same decision as the copied-code S1. Prepare took 253.8 s.
- That run scored 2.0430x, but its timed path is identical to scored R2, so this is R2 re-measured within noise. **Not submitted:** resubmitting identical code would only sample noise.

## R2d diagnostic (`04f8fa914e9742fb`: 13/13 passed, 2.0258x)

Timing-only modes in the prepare harness (graph times; each includes about 5 us of graph-launch overhead):

| T | pooler alone | x staging + barriers + atomic | + router math only | + top-k only | full fused |
|---|---:|---:|---:|---:|---:|
| 1 | 18.98 | 20.03 | 23.68 | 25.82 | 28.99 |
| 8 | 19.30 | 23.33 | 26.85 | 29.41 | 33.09 |

- The last-CTA top-k tail is the largest cost: about 6 us. It reads its 64 partials per lane with volatile loads, which serialize.
- R2e switches those reads to `__ldcg` (L2, pipelined; still after the `__threadfence`). The summation order is unchanged, so results stay bitwise identical.

## R2e and R2f: removing the top-k tail

| Job | Change | Result |
|---|---|---|
| `a805991775114e52` | R2e: `__ldcg` loads in the top-k tail | 13/13 passed, 2.0308x. Top-k-only graph time unchanged (26.05 vs 25.82 us at T = 1). The load type was not the cost; the cost was the last-arriving GEMV CTA running the tail before its own GEMV. **Not submitted.** |
| `d778271fc15348ed` | **R2f: a dedicated 129th block (on one of the 4 SMs the 128 pooler CTAs leave free) waits for the 128 arrivals, runs the identical top-k tail and resets its own counter; GEMV CTAs only publish partials** | Gate PASS (bitwise). Fused kernel T = 1: 29.31 -> **24.93 us**; T = 8: 33.18 -> **27.94 us** (pooler alone 19.1 / 19.0). **13/13 passed, 2.0633x, CI [2.0323, 2.0942]**. decode 1 77.48 / 85.33 us (scored R2: 83.30 / 88.96); verify 1 113.93 / 122.56 us (R2: 122.91 / 131.41). |
| `94572fe644414691` | **R2f scored submission** | **13/13 passed, 2.0510x, CI [2.0203, 2.0818]**; gate PASS (T = 1: 30.75 -> 24.70 us; T = 8: 39.49 -> 27.97 us). **Board: Decode 2.0510x, rank 214 / 1005; overall rank 214 / 1647** (was 2.0339 / rank 217). |
## A3 attention tuning on R2f (`9171012a24754fb5`)

- R2f plus the Attention board's A3 entry `(16, 1, True, 1): dict(n_splits=8, nc=1)`. 13/13 passed, 2.0590x.
- Decode-16 sliding took 135.03 us, with attention at 8.4 us. That is the same as the R2f runs (130.9-140.0 us; attention 8.2-8.5 us), so there was no measurable change. **Not submitted.**

## Scoring fit and the joeyhlu comparison

- **Scoring fit.** Each of the six loads is weighted 46 sliding : 16 full layers by time. The score is the geometric mean of the six per-load reference/ours ratios. This reproduces the reported speedup of R2b, scored R2f and A3 to within 0.0007 (2.0327/2.0334, 2.0505/2.0510, 2.0584/2.0590). It is a fit, not the published formula.
- **Value of 1 us saved, as % of Decode** (computed from the R2f per-shape times):

  | Load | sliding layer | full layer |
  |---|---:|---:|
  | decode 1 | 0.153 | 0.053 |
  | verify 1 | 0.103 | 0.036 |
  | decode 16 | 0.081 | 0.028 |
  | decode 32 | 0.061 | 0.021 |
  | verify 16 | 0.045 | 0.016 |
  | verify 32 | 0.034 | 0.012 |

- **joeyhlu** (Decode 2.2091x, rank 54). Their per-shape table is shared by them, and that run computes to about 2.17 under the fit. Compared with the average of our three R2f-based runs (gap = ours - theirs, us):

  | Load | sliding | full |
  |---|---:|---:|
  | decode 1 | +9.3 | +8.9 |
  | decode 16 | +13.9 | +12.0 |
  | decode 32 | +8.5 | +2.2 |
  | verify 1 | +8.6 | +2.7 |
  | verify 16 | -3.9 | -8.7 |
  | verify 32 | -8.0 | -9.4 |

  - Their lead is a roughly fixed 9-14 us per layer at 1-32 tokens, and we lead by 4-9 us at 128/256 tokens. The lead does not track MoE size or attention size.
  - Their notes describe starting each projection's weight fetch while the previous kernel finishes. On their separate boards they trail us on MoE (2.27 vs 2.38), Attention (2.32 vs 2.57) and Dense (1.12 vs 1.53).
  - Matching their 1-32-token times while keeping our 128/256 lead would be worth about +5% (to about 2.16).

## L2P: weight prefetch into L2 during latency-bound kernels (closed, measured)

- After the qkv projection at T = 1 / 8, a side-stream kernel pulls the next weights into L2 while attention and the norm run. Prefetches change no values.
- Prepare timed qkv -> 5 us window -> o -> 4.5 us window -> pooler+router for each variant against no prefetch (graph us). Outputs were bitwise equal in every variant; a variant is kept only if it is more than 3% faster.

| Job | Variant | T = 1 | T = 8 |
|---|---|---:|---:|
| `94c689655d17463b` | none | 52.99 | 56.22 |
| | bulk prefetch, 8 blocks: o (12.5 MB) | 52.67 | 56.26 |
| | + FP32 router weight (20.5 MB) | 58.05 | 57.76 |
| | + 12 / 24 MB pooler prefix | 77.79 / 77.86 | 77.28 / 77.22 |
| `bad2bc21ae5b420b` | none | 52.54 | 55.55 |
| | bulk, 132 blocks: o / o+router+12 MB pooler | 53.02 / 77.86 | 55.49 / 78.08 |
| | plain-load warm-up, all SMs: o | **51.74** | **54.75** |
| | load warm-up: o+router / +12 MB / +24 MB pooler | 51.97 / 52.90 / 55.42 | 54.94 / 55.97 / 58.91 |

- Both checks passed 13/13 (2.0509x and 2.0673x). Nothing was selected, so the timed path is R2f and both scores are noise. **Not submitted.**
- **Conclusion:**
  - Having the o weights already in L2 saves at most about 0.8 us. So the o GEMV is not limited by DRAM bandwidth at T = 1 / 8; its 5.2 us is mostly launch, ramp and tail latency.
  - Bulk L2 prefetch of the pooler region delays the pooler by about 20-25 us.
  - Load warm-ups larger than the o weights only add contention.
- **Also implied:** the whole 5-kernel chain is about 52.5 us of graph time, against about 44 us of kernel time plus about 5 us of graph overhead. The idle gaps between kernels are therefore only about 0.5-1 us per handoff.
- joeyhlu's 9-14 us per-layer lead is thus not bandwidth left on the table during latency-bound kernels. Remaining explanations:
  - a kernel's launch and prologue overlapping the previous kernel's execution (PDL), since only that hides ramp latency;
  - fewer or shorter kernels.

## Remaining work (needs a decision; not started)

Both remaining Phase 3 items require editing the generated Dense kernel family. The small-token
pooler `v18s_gemv`/`smallm2_gemv` and the sliced unpooler `smtrace_gemv` are built by `run_v18` from 41
zlib-compressed `_zsrc` templates.

1. **Router-into-pooler fusion at T = 1 and T = 8.**
   - Add 128 FP32 router CTAs (the `route_fp32` arithmetic, so outputs stay bitwise identical) at the
     head of the pooler GEMV grid. They then co-run inside one launch, which is the 6–7 µs that
     concurrent R1 showed, without the cross-stream hazard.
   - Expected value: about +2% Decode score.
2. **`_sfn` fold into the sliced unpooler.** Adds a last-CTA two-stage RMS reduction. Bound about +1%.

Each item is multi-hour, judge-only work at about 8 minutes per Decode check.

- No candidate has shown a gain outside measurement noise. **Nothing has been submitted for a score from
  this branch.**
- The measured opportunity that remains is the one R1 exposed: about 6–7 µs at 1 and 8 tokens from
  overlapping the FP32 router with the pooler. A correct version would need the router fused into the
  pooler kernel, as `sk2f` already does for 9–32 tokens, rather than two concurrently launched kernels.

## PDL early-start diagnostic: Step 1 (2026-10-30, current)

This section supersedes the older "Remaining work" wording above. The human's new brief
authorizes one timing-only PDL check, followed by production work only if its ceiling gate passes.
Worktree `decode-moe-plan`, branch `argon/decode-moe-20261030`; parent commit
`58db7a68d84c71e0cc0d4d8d7f3b6492610a7b1d`. No other worktree or `best/` is edited.

Baseline: `experiments/decode_moe/decoder_r2f.py`, full SHA256
`75fa2a73b4de60410c10dc3609141f3677c9ff79acff0512e45fdd6b5084f747`.
Its historical scored result remains 2.0510x (`94572fe644414691`). No new performance
claim is inferred from its other runs, and no live rank has been requested here.

### CPU evidence and bounded implementation

- A single-inline-assembly weight-load-plus-wait prototype compiled, but SASS placed
  `ACQBULK` at `0x60` before every global load. Rejected on CPU; zero GPU jobs.
- A `__noinline__` device wait helper produces real independent weight/escape `LDG`s
  before the call to the helper containing `ACQBULK`. This is a different compiler
  mechanism from the previously rejected direct-wait placement.
- Exact source edits are invertible and leave the math/store bodies intact. Dependent
  activation loads use coherent `ld.global.cg`; immutable packed weights/escapes retain
  the original load instructions. No rounding boundary is removed.
- For R2f, all ordinary CTAs wait before the FP32 router staging reads. CTA 128 takes
  the separate top-k path, waits at its entry, and never executes the weight preamble.
  Its SASS branches at `0x70` to `0x4710`, then waits at `0x4730`. The normal path calls
  the wait helper at `0x1c0`, before the first coherent router load at `0x300`.
- The launch attribute array has two entries and appends programmatic serialization
  after any cluster attribute. The four selected instances are noncluster RED0/S1.
- The one-CTA spin triggers dependents before its 5,000 ns busy loop. CPU SASS emits
  `PREEXIT` at `0x20`, before the first timer read at `0x50`.

| Twin | Original registers | PDL registers | Stack / spills | Independent LDGs before wait call |
|---|---:|---:|---|---:|
| O1 smallm2 | 50 | 50 | 0 / 0 | 6 packed vectors + 1 escape |
| O8 v18s | 50 | 56 | 0 / 0 | 6 packed vectors + 1 escape |
| R2f1 v18s | 64 | 64 | 0 / 0 | 3 packed vectors + 1 escape |
| R2f8 smallm2 | 64 | 64 | 0 / 0 | 3 packed vectors + 1 escape |

All numbers in this table are CPU NVRTC sm_90a compiler resources, not GPU timing.
Raw source, compiler logs, CUBINs, SASS and hashes are under `pdl-cpu/`; explicit
instruction addresses are in `pdl-sass-ordering.json`. Actual judge resources must
also be printed and validated by the diagnostic before measurements.

Candidate `experiments/decode_moe/decoder_pdl_diag.py`, SHA256
`044f8606805c8c79fcaff621a40edbb2f7a703217346bf19e78cc735fce019ad`,
962,982 bytes (37,018 bytes of headroom). Its original R2f prefix is retained exactly;
only a prepare wrapper and diagnostic helpers are appended. Production `kernel_fn`
and dispatch remain unchanged, irrespective of diagnostic results.

The diagnostic uses shared `_okh_time` for both arm orders, 30 replays per arm per
order, cold L2, and a separate scratch/counter for the PDL R2f twin. Each pair is also
captured and replayed on three changing input sets. `_okh_same` checks shape/dtype and
values; byte views additionally distinguish signed zero, with finite-output checks.
The signed-zero, dtype, nonfinite and missing-output guards passed CPU tests. Timing
and CUDA graph correctness have not yet been validated on H200.

Measurements planned: spin-to-O, spin-to-R2f, and QKV-to-spin-to-O-to-spin-to-R2f,
for M1 and M8. The last chain has real QKV and projection kernels but independent
probe inputs and spins substituting for attention/norm; it is a synthetic ceiling,
not an end-to-end Decode gain. The conservative combined gate requires at least
1.5 us savings for each handoff and 3 us for the two-handoff chain in both arm orders.
Every result prints `PDL_DIAG`; no production path is enabled in this source.

### Identity, quota and launch record

Official `okc show muse_decode_layer --json` was read before launch. TASK, AGENTS,
baseline, checker, reference and NVFP4 helper match the local pinned contents after
newline normalization (`pdl-official-identity.json`). The official TASK now explicitly
documents load-time weighting and geometric-mean aggregation; it agrees with the
earlier empirical fit. The official template is absent locally and is not imported.

At 2026-10-30 05:57 UTC, Decode quota was 3 checks and 8 scored runs remaining.
The routine four-check reserve was already crossed before this request. This new
direct brief specifically requests one diagnostic check; use that one check and
leave the remaining two. No public submission is authorized by an unchanged timed path.
Latest check was `bad2bc21ae5b420b` at 05:01:30.947 UTC; observe the 60-minute
spacing before upload. The 50-job snapshot contains no active queued/running job;
an old `duplicate` MoE record is not a new evaluation.

GPU launch: pending. Binding and preflight are saved in `pdl-diag-binding.json` and
`pdl-preflight.json`. No publication, board change, new production enablement or
GPU-derived speedup is claimed at this point.

### Step 1 check launched

At 2026-10-30T06:01:43.412891+00:00, source `044f8606` was uploaded once. The CLI acknowledged
job **`d5f8835a8a85480f`**, problem `muse_decode_layer`, check mode. No TLS failure or retry.
Production dispatch is unchanged. Result and prepare-time PDL measurements are pending.

### Step 1 result: ceiling passed (job d5f8835a8a85480f)

H200 check passed **13/13**, benchmark validation and integrity, no flags. Prepare
completed in **262.064 s**. Both R2f parent gates selected M1/M8. All four PDL
extensions built, all six diagnostic pairs passed bitwise checks across three
changing-input graph replays. Actual GPU resource queries reported O1/O8 56 registers
and 0 local bytes, with 1/4 blocks per SM respectively; R2f1/R2f8 64 registers,
0 local bytes, 1 block per SM. O1's judge count differs from the CPU NVRTC count;
use the actual GPU count when reasoning about integration.

Times below include the same graph overhead on each arm; units are microseconds.

| Tokens | Diagnostic | Order normal,PDL: normal / PDL / saving | Order PDL,normal: normal / PDL / saving |
|---|---|---:|---:|
| 1 | spin -> O | 15.680 / 11.872 / 3.808 | 15.840 / 11.840 / 4.000 |
| 1 | spin -> R2f | 31.296 / 28.000 / 3.296 | 31.328 / 28.224 / 3.104 |
| 1 | synthetic chain | 53.408 / 47.712 / 5.696 | 53.792 / 47.904 / 5.888 |
| 8 | spin -> O | 15.872 / 12.480 / 3.392 | 15.872 / 12.512 / 3.360 |
| 8 | spin -> R2f | 33.856 / 31.424 / 2.432 | 33.824 / 31.552 / 2.272 |
| 8 | synthetic chain | 56.480 / 51.872 / 4.608 | 56.512 / 51.648 / 4.864 |

Both shapes pass the conservative >=1.5 us-per-handoff and >=3 us combined-chain
gates in both orders. These are synthetic-ceiling measurements, not whole-layer
savings. The unchanged production path scored **2.046116180x**, CI
[2.015424438, 2.076807923], and will not be submitted publicly. Full evaluator identity,
raw result and logs: `pdl-status-final.json`, `pdl-diag-outcome.json`,
`pdl-diag-prepare.log`. One check consumed, no scored run. The direct brief now
conditionally authorizes Step 2 production integration; its own bitwise equality,
>3% cold-L2 graph gates and whole-layer validation still apply. Earliest next check
under routine spacing is 2026-10-30 07:01:46.883 UTC, not an immediate retry.

## Step 2 production integration packet (2026-10-30)

Step 1 evidence is committed and pushed as `0db477d` on
`argon/decode-moe-20261030`. The main chat performed all work; no subagent was launched.
The latest user AGENTS instructions were applied without changing the selected model.

Candidate: `experiments/decode_moe/decoder_pdli.py`, SHA256
`c15df9d3d83c55f4ba64be810e56502563e6cfe22bb5ef22172a1feb287654e5`,
940,518 bytes (59,482 bytes headroom). It retains the exact R2f source prefix, then
appends pure source derivations, a separate layer function and gated public wrappers.
The eight private CUDA instances are O1/O8, R2f1/R2f8, sliced unpooler1/unpooler8,
and the original QKV1/QKV8 producers with an entry trigger. No packing or arithmetic
is replaced. No `best/`, main, other worktree or other session branch is edited.

### Implementation and guards

- New QKV producers announce readiness at entry. The paired attention path uses its
  existing incoming PDL support. It waits before QKV reads; metadata/history work is
  independent. Every primary CTA must signal or exit before dependent launch, so
  unlaunched primary CTAs cannot be stranded by the secondary.
- O and R2f derive from the compiler-verified Step 1 twins, with an outgoing trigger.
  The stock attention-boundary norm remains the real predecessor for the R2f gate.
  No synthetic spin is used in this packet.
- Sliced unpoolers preserve their original cluster geometry, mbarrier initialization,
  expected bytes, peer stores, waits, FP32 reduction and final BF16 stores. Their two
  initial weight stages cover the entire split (KU=ST=2). Activation loads are moved
  after the noninline wait helper and use coherent CG loads. Static assertions exclude
  any instance for which later stages would require the original mixed loader.
- MoE's existing outgoing trigger supplies the unpooler handoff. The incoming MoE flag
  stays off: explicit cached dependent route/X reads were found, and this packet does
  not establish their incoming-PDL coherency contract. Its math and scheduler are
  unchanged. `_fpr_PDL` is only relevant at 9–32 tokens and is left for that separate
  regime; there is no global flag toggle affecting untouched shapes.
- One extension packages the eight private instances in distinct namespaces, with
  common CUDA headers and one pybind translation unit. This avoids eight separate
  header/pybind builds. All 16 primitive host exports have matching signatures.
  Occupancy queries use cluster-only configs; the PDL attribute is added on launches.
- Packed extension and original attention extension build on background threads.
  A background stock norm warm-up starts after the original core state is constructed,
  while Dense preparation proceeds. It only prepares immutable weight caches and
  synthetic tensors; no runtime input or output is cached.
- Incoming consumers wait before dependent activation reads, including R2f router
  staging. R2f's extra top-k CTA bypasses ordinary weight reads and waits at entry.
  Private routing scratch is 64 KiB plus one counter, reused sequentially and reset
  by the existing top-k completion. No expanded/repacked expert layout is added.

### Numerical/performance gates

Each M1/M8 handoff uses real producers: QKV->attention, attention->O, stock norm->R2f,
and retained MoE->unpooler. Gates require byte equality on three changing graph
inputs (with whole cache byte comparisons for attention) and >3% cold-L2 interleaved
CUDA-graph improvement in BOTH arm orders. Unpooler correctness includes one local,
all nonlocal and eight local routes. Timing uses one local expert/token as a bounded
representative pattern; this is recorded, not claimed as actual judge routing.

A single union of individually passing paths is then tested against the original
complete `kernel_fn`, including output stream, routing IDs and full cache bytes.
The union must independently pass bitwise equality and >3% in both orders. No
Cartesian combination search occurs. Regimes that fail or run out of preparation
budget retain the parent. Dispatch is shape/config-only; QA additionally requires
B1 and L1/L8. Other token counts and q_len fallbacks retain the original code.
No new work starts after 278 s; final prepare remains subject to the 300 s cap.

CPU NVRTC compiled the packed device module successfully. All eight kernels have
zero stack/spills. Registers O1/O8/R1/R8/U1/U8/Q1/Q8: 50/56/64/64/63/62/78/93.
All eight have an entry `PREEXIT`; six consumers issue the expected 7/7/4/4/7/7
weight/escape LDGs before the helper wait, then coherent dependent loads. Raw packed
SASS and proof: `pdli-cpu/combined.sass`, `pdli-cpu/combined-ordering.json`.
The final host-only occupancy guard leaves the compiled device proxy byte-identical;
this binding is recorded explicitly in `pdli-preflight.json`, not represented as a
new GPU measurement. Final packed CUDA SHA256:
`785ba7a5b64d17632d15ae7918cba5684972f3f7517f0ef992076b057746af15`.

Official validation, actual resource queries, actual selection and performance are
still pending. A score increase is not claimed from CPU compilation or Step 1.

### Step 2 admission

At 07:00 UTC the account has **2 Decode checks and 8 scored runs left**, with no
active job. The latest check was Step 1 at 06:01:46.882509 UTC. Earliest new upload
is 07:01:46.882509 UTC. The human brief conditionally authorizes one production check
after the ceiling passes; that condition is met. Use one check, leaving one, and
record its actual handle. No scored submission unless 13/13 and material targeted
and aggregate improvement satisfy the human's publication guard.

### Step 2 check launched

At 2026-10-30T07:03:22.358488+00:00, one upload of source `c15df9d3` was acknowledged as
**`59cfbd2b12fa43f2`**, check mode, `muse_decode_layer`. No TLS retry. A byte-identical
frozen payload is under `payloads/`. Actual selection, full correctness and timings
are pending; no public submission or board update has occurred.

### Step 2 check result and scored-confirmation decision

`59cfbd2b12fa43f2` passed **13/13 tests, 12/12 benchmarks and integrity**, no flags.
Prepare **264.746 s**. API **2.081797387x**, CI [2.050570426, 2.113024348]. Source
`c15df9d3` selected only **M1 sliding mask 10 = O + sliced unpooler**. Both private
kernel markers appear in that case's profile, nowhere else. Evaluator identity
matches the Step 1 parent check exactly. Actual resources: O1/O8 56/56 regs,
R1/R8 64/64, U1/U8 63/62, Q1/Q8 78/93, all zero local bytes. Occupancy queries:
1/4 blocks per SM for O, 1 for R2f; Q1/Q8 186/154 active clusters and U1/U8
124/132 active clusters.

The accepted whole-layer graph gate saved 2.944 / 3.200 us in the two arm orders
(new/old .956149 / .952584), with byte-identical outputs and caches. The official
1-token sliding case improved **79.027 -> 75.518 us** against Step 1's unchanged
R2f path. Its fixed-other-loads weighted contribution is about **+0.5475%**;
do not call the entire +1.50% API point difference causal. Cross-run control drift
and overlapping intervals remain. This is not a significance claim.

Other regimes stayed on the parent. R2f's real stock-norm predecessor yielded
0.672–1.088 us (M1) and 0.768–1.280 us (M8); neither passed >3% in BOTH orders.
QA M1 sliding saved only .224–.256 us. Several other comparisons failed byte
checks (attention a few BF16 bytes; M8 MoE outputs on dense-local routes; complete
full-context outputs). Those failures were recorded and rejected, not tolerated.
There is no 16/32 transfer of these failed paths.

Live board before confirmation: **2.0510x, rank 228**. After inspecting actual path
markers, the accepted paired full-layer gate and targeted benchmark reduction,
authorize exactly one scored confirmation of the identical checked bytes. The
new aggregate point exceeds 2.0510, with a visible source-bound targeted gain;
its uncertainty and diluted contribution are preserved above. This consumes a
scored run, leaving the final check untouched. No unchanged-code retries or
public rerun if it selects no new path. Decision details: `pdli-score-decision.json`.

Scored upload acknowledged job **`452fbd9df6af4ff4`**, source `c15df9d3`, once,
no TLS retry. The full source is byte-identical to the successful production check.
Result, actual selected paths and board placement are pending.

### Scored result: new board best, 2026-10-30 07:31 UTC

**`452fbd9df6af4ff4` passed 13/13, 12/12, integrity, zero flags.** Same source
`c15df9d3` / same API content hash as the production check. Prepare **260.968 s**.
Scored **2.075466170x**, CI [2.044334177, 2.106598163]. The board now reports
**2.0755x, Decode rank 210/1015 and overall rank 210/1663**; the immediately prior
snapshot was 2.0510x / rank 228. This is one scored confirmation, with no rerun.

Scored prepare selected **M1 sliding mask 14 = O + R2f + sliced unpooler**. R2f
passed >3% in both orders on this GPU (29.632->28.608 us, 29.792->28.320 us).
Whole-layer pairing passed byte equality and saved **3.008 / 3.040 us**
(new/old .962505 / .962331). The official 1-token sliding time was **75.819 us**.
Profiles confirm `pdli_o1`, `pdli_r1`, `pdli_u1` only in that case. The production
check had selected mask 10; selection differences are a consequence of the recorded
prepare-time gates, not a source change. Full-context and all M8 variants fell back.
QA and M8 MoE comparisons with byte differences remained rejected; the M8 full
union passed byte checks in this run but did not meet the performance margin.

The published point is **1.19% above** the previous board point. That entire
aggregate difference is not attributed to one changed case: intervals overlap and
unchanged loads vary. The supported evidence is the repeated paired one-token layer
win, explicit selected-path markers and the official targeted-case latency. Both
checks and the scored run use the original numerical contracts and fallback paths.

This task used **two checks and one scored run**. Fresh Decode quota: **1 check,
7 scored runs left**. No GPU job is pending; preserve the final check. The broad
top-50 goal is not achieved. M16/32 was not opened by transferring failed T8 paths.
Current retrieval summary: `PDLI_STATE.json`. Main publication of byte-identical
scored source and its Decode row is in progress in an isolated checkout.

### Publication complete

The exact scored source is published on `main` at commit
`2e504820c55c627019661d81a86f74ff2b2266bd`, artifact
`board_best/decoder_pdli_scored_2.0755.py` (940,518 bytes; full SHA256
`c15df9d3d83c55f4ba64be810e56502563e6cfe22bb5ef22172a1feb287654e5`).
The working file, staged blob and committed published blob were byte-checked.
A normal fast-forward push advanced `main` from `b8e92d6`; `ls-remote` confirmed
the exact publication commit. Its Decode board row records job `452fbd9df6af4ff4`,
2.0755x and the timestamped API ranks. Other track rows and all `best/` files
are unchanged. The main table now identifies its separate row snapshot dates.
A scoped `-text` attribute preserves the new artifact's exact submission bytes.

The implementation is committed as `c47fbf4` on
`argon/decode-moe-20261030`; the final evidence is committed separately on the
same branch. This packet completed the timing ceiling and bounded T1/T8
integration. Only one-token sliding passed its public-run gates. M16/32 was
not implemented or GPU-tested; there is no claim that the complete extension
backlog or top-50 objective is finished. No further GPU jobs were launched.
The final recorded quota remains **1 check / 7 scored runs**.

### TR8 MoE candidate: source frozen, GPU evidence pending

After publishing the Decode PDL gain, return to the requested MoE-first order.
The previous turn made verified progress. No active GPU job was observed in
50 recent jobs (all terminal, including one duplicate). Fresh quotas show
MoE 19 checks / 10 scored runs and Decode 1 / 7. The remaining Decode check
is preserved. The last check predates 07:31 UTC; this admission is after
12:18 UTC, satisfying the shared 60-minute spacing policy. One MoE check is
planned, leaving 18 checks and 10 scores. No score is preauthorized by this
candidate's CPU evidence; a material measured gain is still required.

TR8 replaces only small-route table construction with one warp and one CTA
barrier, retaining exact stable expert/route order, all arithmetic and every
completion dependency. Source SHA256
`958335479991016b3cebae2ad9517ef8f75c036fc1d8ac63a0b8459395219498`,
190,786 bytes. Baseline is retained `33114871`; board CUDA equality and
current official task identity were verified. Eight CPU compiler screens and
3,648 ownership/fallback models passed, with no spills but 2-5 extra registers.
This is not GPU validation or a claimed speedup. The exact frozen payload,
resource logs, review, source inverse and stopping criteria are in `tinyroute/`.
The candidate must pass bytewise changing-route graph probes and >3% in both
arm orders on two synthetic routing populations before selecting each M<=8.
All larger/failed paths remain parent. Synthetic routes are not official
benchmark measurements. No repeat of the rejected stream-K or skinny16 work.
