# Adaptive small-batch candidate

Scope: standalone variant of frozen `small_batch.py`. Only M=2..32 BF16
launch partitioning and its direct-output path change. M=1 GEMV, router FP32
fallback, large-M/unsupported fallbacks, and lossless weight layouts retain
the original implementation.

## Evidence and intervention

Confirmed static evidence: the 64-column output tiling already provides 128
independent CTAs for N=8192. Fixed split8 expands this to 1024 CTAs, requires
an FP32 scratch tensor, and always launches a reduction. H200 has 132 SMs.
Whether that additional partitioning helps bandwidth is unmeasured.

Hypothesis: avoiding scratch and the reduction launch for already-wide output
grids reduces latency, especially at K=1024. Smaller grids still benefit from
some split-K parallelism. This variant makes only that bounded change; no GPU
speedup is claimed.

The split count depends only on public output shape. With 64-column tiles,
use split1 for >=128 N tiles, split2 for >=64, split4 for >=32, otherwise
split8. This aims for about 128 CTAs without targeting each layer by name.

| Public family, M=2..32 | N tiles | Old split / CTAs | New split / CTAs |
|---|---:|---:|---:|
| o: N8192 K1024 | 128 | 8 / 1024 | 1 / 128 |
| full unpooler: N8192 K4096 | 128 | 8 / 1024 | 1 / 128 |
| QKV: N2304 K8192 | 36 | 8 / 288 | 4 / 144 |
| pooler: N4096 K8192 | 64 | 8 / 512 | 2 / 128 |
| sliced unpooler: N1024 K4096 | 16 | 16 / 256 | 8 / 128 |

For split1 the existing GEMM writes directly to BF16 output and no scratch or
reduction is created. Products accumulate in FP32, then the output store does
one final BF16 conversion. For split>1, each K partition still writes FP32
partials, and the deterministic FP32 reduction converts to BF16 once. No
atomics, lower-precision weights, output caching, or input-derived state.

## Validation and remaining uncertainty

No official checks/submissions are run by this worker. Native Python cannot
execute the Triton kernels; H200 compilation, numerical correctness, and
paired cold-L2 graph performance remain the parent's responsibility.

Python AST parsing and compilation pass. Static output, scratch, and K-address
coverage checks pass for 56 cases: M=2/5/7/8/16/17/32 across all five public
families plus N=65/1025/2305 tails. Each output coordinate is written once,
split scratch intervals do not overlap, and every K index is covered once.

CPU FP32 arithmetic models pass against the rounded FP64 reference at seed
20261023, sampling 65 output columns for (M,N,K,split): (8,8192,1024,1),
(16,8192,4096,1), (32,2304,8192,4), (8,4096,8192,2), (5,1024,4096,8), and
(7,4096,8192,2). Each model includes a 30x outlier channel and one power-of-two
single-entry probe row. Rounded BF16 outputs match the reference in these
models (global, row, and probe errors zero). These do not execute GPU code.

Correctness reference: `muse_bf16_proj/reference.py` computes an FP64 product,
then rounds to BF16; BF16 outputs must meet 1% relative L2 with per-row guards,
and 0.1% on power-of-two single-entry probe rows. Fewer K partitions change
FP32 summation order, so runtime checks must include probe and outlier rows.

First normal path: M8/M16/M32 o and full unpooler direct-output variants.
Failure surfaces: odd M and nearby N/K tails, QKV/pooler split coverage and
probe/outlier numerical behavior. Integration edge: M5 sliced unpooler must
use only the first N=1024 columns of the lossless full weight transpose;
M33 and M130 still use the existing PyTorch fallback.

Follow-up: compare direct versus fixed split8 per shape, keeping changes only
when paired graph timings improve beyond uncertainty. If 128 CTAs cannot
saturate bandwidth for full unpooler, split2 is the smallest next experiment.
Reverting dispatch to frozen `small_batch.py` is the fallback strategy.
