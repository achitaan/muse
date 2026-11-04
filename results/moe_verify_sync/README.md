# Verify-only narrow-N tensor instruction experiment

Status: **offline prototype; no GPU check submitted or speedup claimed**.

Branch: `loop/moe-verify-sync`, based on worktree HEAD `63e253f`.

Self-contained check file:

`/Users/achita.anantachina/.local/share/opencode/worktree/e72615/loop-moe-verify/candidates/moe_verify_sync.py`

## Roofline correction and evidence

The public task (`.tools/okc show muse_moe_w4a16`) specifies EP8: this rank
computes only local routes. Gate/up has `2I` output rows, then down has `H`,
so dense matrix work is **6 H I = 100,663,296 FLOPs per local route**.
Using `4 M top_k H I` both omits the up projection and incorrectly charges
this GPU for other ranks' routes.

Exact achieved TFLOPS needs the measured check's local-route counts, which
were not supplied. The model below uses the public distribution already in
the incumbent `_moe_routes`, 32 draws with Python RNG seed 20261104, not the
judge's CUDA RNG stream. Reported times are supplied measurements; counts
and TFLOPS are modeled, **not matched per-check measurements**.

| Tokens | Time (us) | Local routes median | Useful TFLOPS | Issued TFLOPS incl. N padding | Dense-peak useful / issued | Active experts / 64-route jobs | N=8 jobs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 183.3 | 184.5 | 101.32 | 155.96 | 10.24% / 15.77% | 23 / 23 | 17 |
| 256 | 216.2 | 357.5 | 166.45 | 212.31 | 16.83% / 21.47% | 25 / 26.5 | 13.5 |

Useful TFLOPS ranges over those draws: 82.38–166.40 and 135.02–227.21.
Even padded issued work is far below 70% of 989 TFLOPS. This is **not a
dense-tensor-peak floor determination**. A narrow-N instruction/consumer
floor can coexist with low dense-peak utilization. Modeled weight traffic
alone is 3.55 / 3.47 TB/s, consistent with a streaming + consumer-overhead
problem rather than a conventional large dense GEMM.

The actual per-expert M is not 128/256. Most jobs have <=8 routes; the
incumbent uses `m64n8k16` WGMMA, dequantizing A (weights) **in registers**.
There is no dequantized-B SMEM path here. Packed weight `uint4` reads are
bank-conflict-free; scale reads are broadcasts. The new X reads also map
32 lanes to 32 distinct banks.

## One change

For jobs with <=8 routes, replace WGMMA with four independent warp-local
`mma.sync.m16n8k16` operations, each warp retaining exactly its existing
16-row A and accumulator fragment. This avoids eight WGMMA
fence/commit/wait groups per K256 stage. Keep every packed NVFP4 bit,
scale operation, expert global-scale boundary, routing/job table, TMA
ring, stream-K partial, SwiGLU, norm and final combine. N>=16 jobs retain
the original WGMMA body byte-for-byte. Only M=128/256 can adopt the arm.

Non-repeat evidence (TRIED ledger rows identified by job/name):

- `08c52957` P2-RS64 changed task size, CTA geometry and GEMM2 core. This
  does none of those: same 576-thread CTA, four consumer WGs and K256 ring.
- `79687699`/`adbd3ba4` VMS/VML was a three-launch small-batch engine with
  different tile ownership and split-K. Here the same one fused kernel and
  scheduler are retained, and only the N=8 instruction is changed.
- `79036669`/L1 register-budget redesign and `8d55ab94`/L2 wide-N jobs are
  not repeated. This adds no wide job, new stage layout or occupancy scheme.
- `60173f98` detached epilogues are unchanged; this is not a finalizer arm.

Correctness prerequisite: both arms include the already-kept empty-job
guard (`2cba38a5`, ledger row 73). The original board file otherwise hangs
on all-remote replay. The incumbent's existing preparation/selector logic
and non-verify dispatch are preserved. If it picks a non-WGS fallback for a
verify case, the new arm is not tested or installed for that case.

## Admission/fallback

- Actual kernel resource attributes must not grow (registers, local memory,
  shared memory). No offline resource values are claimed.
- Strict finite numerical comparison: relative L2 <=0.002 plus
  `allclose(rtol=.02, atol=.002)`; no loosening based on observed error.
- Captured graph replay is exercised across random, hot, 65/64/9/8/1-route
  and all-remote inputs, updating buffers in-place between replays.
- Two independent passes of eight fresh routing draws, with 12 ABBA and
  12 BAAB quads/draw. Every timed replay follows a **512 MiB READ** flush.
- Each pass needs median aggregate and paired candidate/base ratios <=.985
  in both orders; no draw's ratio may exceed 1.005.
- Compile errors, resource growth, numeric mismatch, timeout or failed
  speed gate keep the incumbent. Tuning uses private scratch and cannot
  supply a runtime output. Existing default outputs are never cached.

Expected effect, if the hypothesis survives H200: a small consumer-side
gain, plausibly 1–4% on verify, not a dense-TFLOPS breakthrough. No timing
evidence exists locally. Four warp instructions and explicit B loads may
instead be slower than one WGMMA; the admission gate intentionally makes
rejection easy. No second architecture attempt was built.

## Local verification

```
python3 scripts/build_moe_verify_sync.py
python3 scripts/analyze_moe_verify.py
python3 scripts/test_moe_verify_sync.py
```

Tests cover packed NVFP4 dequant/A/B fragment placement and 64x8 dense
products (21 full/partial stage cases), all stage/slot addresses, output
register coverage, X bank maps, 100 randomized 16-warp ring schedules,
original wide-WGMMA/epilogue identity, mock timing order/read flushing,
negative gate cases, exception fallback, timed dispatch and fresh outputs,
plus host C++ template syntax instantiation of both phases.

**Not available locally:** Torch, CUDA device, NVCC, NVRTC, ptxas. The host
syntax check uses mock declarations; it is explicitly not a CUDA/PTX
compile, a liveness proof or a numerical-rounding/tolerance test on H200.
Full GPU build, barrier behavior, rounding, resource usage, prepare limit
and speed remain unverified. The caller must run any OKC check.

The shared ledger helper would push main, which this lane is forbidden to
do. Its claim/result are recorded only in this branch's `TRIED.md`; caller
should publish the row to the shared ledger when appropriate.
