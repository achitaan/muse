# Small-batch candidate

Scope: BF16 projection at M=1 through M=32. Router uses original FP32 PyTorch
matmul. Larger M, non-contiguous inputs, and unsupported K use PyTorch matmul.
`prepare` stores original contiguous BF16 weights and a lossless BF16 transpose.
The unpooler honors the supplied N, including its first-1024-row slice.

## Evidence and hypothesis

The published task measures cold-L2 CUDA graph replays. Its H200 lower bounds
show memory traffic dominates these small batches. The baseline uses cuBLAS.
These are confirmed facts from TASK.md and baseline.py; there is no local H200
profile or before/after measurement yet.

The candidate hypothesis is that explicit streaming improves M=1 weight
bandwidth, while split-K provides enough independent blocks for M=8/16/32.
Neither performance claim is confirmed. Integration should keep a branch only
if official paired replay timings improve beyond the reported confidence band.

## Implementation and tuning choices

* M=1: four output rows per CTA, 1024 contiguous K elements, four warps. Convert
  both BF16 operands to FP32; multiply and accumulate in FP32; reduce K in FP32.
* M=2..32: M tile 16 or 32, N tile 64, K tile 64, four warps, three stages.
  Contiguous-K partitions give eight independent blocks per N tile, or sixteen
  for the sliced unpooler. Each writes FP32 scratch, then one deterministic
  reduction kernel sums partitions and converts to BF16 exactly once.
* No atomics, lower-precision weight copies, activation compression, data-based
  branches, output caching, timing hooks, harness access, or runtime network.
* Scratch is allocated per call and each valid element is written before read.
  The prepare-time transpose doubles BF16 weight storage for this standalone
  candidate; the parent can retain only required layouts when integrating.

## Validation completed and required

Python compilation passed on the source without importing dependencies. An
exhaustive static address model passed for 18 public/odd-M projection cases:
every valid output and scratch element has exactly one writer, and every K
index belongs to exactly one split. Windows Python has torch but no Triton.

CPU numerical-order models (seed 20261023, sampled N=65) passed against
`(x.double() @ w.double().T).to(torch.bfloat16)`: M=1 at K=1024/4096/8192,
and split reduction at (M,K,split)=(5,4096,16), (7,8192,8), (8,1024,8),
(16,8192,8), (32,4096,8). The split cases include a 30x outlier channel and a
single-entry power-of-two probe. Rounded BF16 outputs matched that independent
reference in these sampled models. These models do not execute the GPU code.

Local hardware cannot establish H200 BF16 tensor-core correctness or
performance. The parent
must run official `okc check` against this candidate or the integrated file.
Normal path: scored M=1/8/16/32 projections. Failure/stress path: outliers at
M=512/1000 go through the ordinary BF16 PyTorch fallback. Integration edges:
M=5 sliced unpooler, M=7 pooler, M=33 o, and M=130 unpooler; also router M=8/9.
Nearby M values in 2..32 retain masks; M=33 switches to the fallback.

Priority is correctness first, then measured speed. If split-K's second launch
outweighs occupancy gains (particularly K=1024), fall back to torch for that
shape family or reduce SPLIT_K. If GEMV trails cuBLAS, keep the fallback for M=1.
No GPU runtime pass or benchmark improvement is claimed by this note.
