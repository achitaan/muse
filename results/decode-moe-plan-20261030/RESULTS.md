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
