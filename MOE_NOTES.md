# MoE optimization notes — October 28, 2026

Current order: MoE W4A16, Decode Layer, Attention Decode, Prefill Layer last.
Main chat only. Conserve checks and scored runs; every trial needs a concrete hypothesis.

## Completed phase 1

- Baseline 03449dfa701d4a95: 0.999760595x, all 5 tests and 6 benchmark cases passed.
- Small-M fused finalizer 74ba53564cfd42e4: 1.149649851x, full official validation passed.
- Smaller EP8 M tiles bd37e0d81b0542fa: 1.271471562x, CI [1.252399489,1.290543636], full validation passed.
- Direct single-token CUDA NVFP4 GEMV 67f2f03d27b44ede: correct but slower, 1.070874526x aggregate; single-token path about 148us versus 54us retained. Reverted. No follow-up parameter sweep.
- Evidence, frozen source hashes and controller history: results/resumed-moe-phase1-20261028/.
- Four MoE checks used, no MoE scored submissions.

## Precision review and new baseline

Dot supplied a review (no performance measurements) identifying the same small-M
fallback and highlighting the difference between the public starter and pinned
SGLang. Independently verified pinned helper:
https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py

SGLang's effective norm weight is BF16(1+w), but y * effective_weight stays FP32
through RMSNorm and route-weight reduction, with only final output cast to BF16.
The public starter adds an intermediate BF16 rounding. Phase 1 followed that
starter and passed all official checks; the new baseline removes the extra
rounding to match pinned SGLang's FP32 computation more closely. Do not roll back
to the older rounded variant just for a small timing point change. Establish a
fresh measured baseline for the stricter precision contract first.

Mask remote expert routes before loads/arithmetic, output zero for all-remote
rows, and overwrite outputs on every graph replay. Preserve FP32 accumulation,
original NVFP4 codes, FP8 block scales and FP32 global scales. In-kernel
dequantization is permitted; persistent BF16 predecoded weights are not.

## Prepared hypotheses, unmeasured

The original-layout single-token CUDA attempt is retained as negative evidence.
A grouped Triton W4A16 GEMM addon is being prepared under .tools. Its first CPU
check caught multiplication-order differences from the task dequantizer; fix
value * (scale * global_scale) before any H200 evaluation. It is not a speedup.
Marlin launch/scheduling remains another hypothesis; no library/harness patches.

## Other tracks

See TRACK_NOTES.md for earlier attention/decoder work. Attention scored job
f6b899f674914ef2 completed at 1.116457695x and passed. Decoder baseline
65ab5fffad0c4995 completed at 1.002013444x. The next focus after MoE is decoder
layer, then attention and prefill. No Dot advice authorizes external spending.
