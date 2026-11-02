# Serving track, change S1: fused post-expert norm for small token counts (Marlin path)

Branch `argon/serving-20261102`. Pinned SGLang `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`
(checkout at WSL `~/muse-serving/sglang`, branch `my-patch`, created by `okc serve init`).
Serving opens 2026-11-05 12:00 PT (3 checks/day at C=16, 1 scored run/day).

## Candidate (patch `s1_postnorm.patch.diff`, 2,466 bytes, 2 files, no line > 400 chars)
- Call site: `MuseSpark1xFp4Experts._forward_marlin` -> `fused_post_expert_norm_weighted_sum(...)`
  (`python/sglang/srt/layers/muse_spark_fp4_moe.py`). Stock sends `tokens < 32` to a ~10-op PyTorch path
  (where-mask, float, mul, pow, mean, rsqrt, mul, mul, sum, cast); >= 32 uses the existing Triton kernel.
- Change: new optional kwarg `min_tokens_for_fused` on the helper (default None = stock 32; ignored off CUDA),
  passed only by the Marlin call as `_MARLIN_POSTNORM_MIN_TOKENS` (default 1; env
  `SGLANG_MUSE_MARLIN_POSTNORM_MIN_TOKENS=32` restores stock for local A/B). The other consumer
  (`muse_spark_bf16_moe.py`) is untouched. No change to routes, precision, norm order/form, weights,
  scheduler, cache policy or flags.
- Semantics: the Triton kernel skips slots with id < 0 (stock zeroes them before the norm: same result,
  and stock's zeroing is what keeps uninitialized remote rows out); FP32 weight-first RMSNorm per expert
  before the weighted sum; FP32 accumulation; BF16 output.

## CPU evidence (PASS, formula level only)
`serving/test_s1_postnorm_cpu.py`: 208 cases, T in {1,4,8,16,31,32,33,64}, all-local / mixed / all-remote /
one live slot in each of the 8 positions, scales 1e-3/1/300, weight-first and weight-last, remote slots
poisoned with NaN/Inf/1e30. Stock fallback vs a line-by-line torch emulation of the Triton kernel vs FP64:
both finite, all-remote exactly 0, worst rel-L2 vs FP64 1.7e-3 for both, max |stock - emulation| = 0 on CPU.
This does NOT execute Triton or a GPU (local GPU is a GTX 1660, no torch/triton installed).

## Where it applies
Marlin decode/verify steps with < 32 tokens per forward: DFlash verifies 8 draft tokens per request
(the kernel board's q_len 8), so C=1 (~8 tokens) always hits it, C=4 (~32) only when fewer than 4 requests are
in the batch, and C=16/32 do not. The single C=16 check rung cannot show a gain; only the scored run's C=1
(and partly C=4) rungs can. Expected effect is therefore small on the overall serving score.

## A/B protocol (needs 8xH200 in the pinned image; we have none)
1. Op level (any Hopper GPU): stock vs fused at T = 1,4,8,9,16,31,32,33 with real shapes (H=4096, topk=8),
   poisoned remote rows, CUDA-graph timing, read-flush, alternating order; outputs vs FP64 reference.
2. Server: stock-vs-stock repeatability first, then stock vs S1 (env switch) with the same image and
   request mix, natural cache; record tokens/s/GPU, tokens/s/user, TTFT, output equality, and the fraction
   of MoE forwards with T < 32. Keep only if the C=1/C=4 throughput gain exceeds the stock band (~2% floor).
3. Official: one `okc serve check` only to prove correctness/no regression at C=16 (S1 is inactive there);
   a scored run is the only place the C=1/C=4 effect is visible.

## Verdict: UNMEASURED
Missing evidence: GPU parity of the Triton kernel at T < 32 (incl. remote-row skip), op-level timing on
H200, and whole-server throughput at C=1/C=4. Low-occupancy risk: one program per token at T=1 (8 slots x
4096) may lose to the graph-captured PyTorch ops; only a GPU run can tell.

## S2 (later, isolated): align only the local expert IDs
`_marlin_expert_outputs` maps global ids to -1/0..31 then calls `moe_align_block_size(..., 256)`; sizing for
32 local ids (+ sentinel) shrinks the sorted-id buffers (e.g. T=16: 1024 -> 360 int32 slots). Not drafted;
must be tested separately from S1.
