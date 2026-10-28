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

## E00 result

H200 job 05a52f7e67084663 passed 9/9 correctness tests and 30/30 timed shapes. Score 0.908671x SGLang, 95% interval [0.895041,0.922301]. This starter is slower than SGLang. Larger-M FP32 router costs 19.7–46.0 microseconds versus 14.8–17.7 for SGLang, while M8 router costs26.4 microseconds for both. Router fusion is the first measured priority.

## E04–E06 queue

Combined small-batch + fused-router with cuBLAS larger projections; adaptive small-batch partitioning; original-FP32 M1 router GEMV. These are independent hypotheses, not measured wins. Each new launch uses an immutable SHA256-named payload via scripts/official.py. Initial local file hashes are preserved in results/initial-hashes.json; server content_hash is a distinct server identifier and is not assumed to be the file SHA256.

## E07 and final local checkpoint

Packed hi/lo cuBLAS router submitted as 4f2f2262cdc14b0e. Small-batch check 9b29fd36a8d646dd passed 9/9 tests and 30/30 shapes: 0.912009x SGLang, 95% interval [0.898329,0.925689]. Six other optimization jobs remain queued. No overall speedup over SGLang established; no scored submission. User requested GitHub handoff and local cleanup; all workers finished and the local goal is paused.


## October 28 checkpoint

The current scored result is 1.521764418x (job 1183fb1c393c4422), rank 106 at 05:13 UTC. The best check is 1.522874509x (60a349bee6944644). All official tests and benchmark cases passed. The target top-50 placement is not achieved. Complete raw evidence and attributed sources are in results/sprint-20261028-evidence; phase1 and phase2 directories retain the controller history. See HANDOFF.md for the current deadline and evaluation quota.
