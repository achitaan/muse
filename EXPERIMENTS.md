# Experiment log

Deadline: 2026-10-23 13:00 America/Toronto (17:00 UTC).
Only the primary native goal controls this sprint. Installed Ralph was inspected; no second controller was launched and no global orchestration settings changed.

## E00 — Starter baseline

Official PyTorch starter preserved at `best/starter.py`. Official H200 check `05a52f7e67084663` submitted. Result pending. No speedup claim.

## Planned controlled comparisons

- E01: small-M contiguous-K GEMV and split-K BF16 Tensor Core GEMM. FP32 partials, one final BF16 cast.
- E02: larger-M conventional tiled BF16 GEMM, modest split-K for insufficient CTA counts.
- E03: router fused BF16 hi/lo products, independent FP32 accumulation and split reduction.

Each candidate is a standalone file. Record its SHA256 at submission and retain the official result. Compare paired cold-L2 CUDA graph timings, reported intervals, and per-load score; accept only official correctness-passing candidates. Static checks are not GPU verification.
