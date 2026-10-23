# Large-batch BF16 candidate

Scope: `x @ W[:n].T` for the four BF16 projection layers at M >= 32.
Router and smaller M use standalone PyTorch fallbacks for parent integration.
Input BF16 and weight BF16 precision are preserved. Every dot product uses
FP32 accumulation and converts to BF16 only at the final output store.

## Evidence and hypothesis

Confirmed from task: H200 target, cold L2 before each graph replay, native
BF16 weights, M in 32/128/256 for the larger scored loads, odd-M checks 33/130,
and outlier checks at M 512/1000. The full pooler/unpooler weights are 64 MiB.

Hypothesis (unmeasured): at M=32, 32x64 output tiles produce 36 QKV or 64 pooler
CTAs, leaving much of H200's 132-SM device without an output tile. Bounded
split K increases QKV to 144 CTAs and pooler to 128 CTAs. The cost is one FP32
workspace and one reduction launch. Full unpooler has 128 CTAs already and
uses a single kernel. These counts are static, not measured occupancy.

All larger scored loads use ordinary grouped output tiling, one launch, and
no split K. Grouping four M tiles can reuse weight blocks through L2 during
one call; this does not rely on warm cache across graph replays.

## Initial configurations

| M | BM | BN | BK | split K | warps | stages |
|---|---:|---:|---:|---|---:|---:|
| 32 | 32 | 64 | 64 | QKV 4, pooler 2, o/unpooler 1 | 4 | 3 |
| 128 | 32 | 64 | 64 | 1 | 4 | 3 |
| 256 | 64 | 64 | 64 | 1 | 4 | 3 |
| 33 (o) | 32 | 64 | 64 | 1 | 4 | 3 |
| 130 (unpooler) | 64 | 64 | 64 | 1 | 4 | 3 |
| 512 (QKV) | 64 | 64 | 64 | 1 | 4 | 3 |
| 1000 (unpooler) | 64 | 64 | 64 | 1 | 4 | 3 |

For other shapes, split K is enabled only for M <= 64, K >= 4096, and fewer
than 96 output tiles: split 4 below 48 tiles, otherwise split 2. Each split
owns disjoint contiguous BK-aligned K blocks; masked blocks cover K tails.
Aligned shapes omit provably unnecessary masks; irregular shapes use row,
column and K masks. PyTorch small-M fallback respects
the sliced unpooler's N=1024.

## Validation and remaining risks

No official check or submission is run by this worker. Parent owns those.
Local WSL command was denied (`Wsl/E_ACCESSDENIED`); native Windows Python has
PyTorch but no Triton. Therefore AST compilation is the available initial
check; neither an H200 speedup nor numerical pass is claimed. Static launch
mapping and K-coverage checks pass for all target shapes and extra irregular
N/K examples: grouped mapping is a bijection over output tiles, while each K
index belongs to exactly one split.

Parent's first runtime check must cover:

1. Normal: M=32 QKV and pooler (both split-K choices), plus M=128/256.
2. Failure surfaces: M=33 o and M=130 unpooler tails; M=512 QKV and M=1000
   unpooler with outliers; sparse probe rows and no NaN/Inf.
3. Integration edge: unpooler at M=5 uses exactly its first 1024 rows; router
   uses its full FP32 fallback until the parent integrates a router candidate.

Correctness source: `muse_bf16_proj/reference.py` computes FP64 products then
rounds to output dtype, applies 1% BF16 relative L2 / per-row safeguards and
0.1% relative error on one-nonzero power-of-two probe rows. FP32 dot and
FP32 split reduction should fit these bounds, but must be checked on H200.
Reduction changes summation order; no atomics or BF16 partial rounding occur.

Before/after plan: parent runs one official correctness check, then compares
paired cold-L2 graph timings with the baseline. Keep split K only when its
total GEMM + reduction latency wins outside the reported uncertainty. Try
split=1 and split=2/4 before broader tile sweeps. This is the smallest
reversible configuration change if split overhead dominates.

Official API source consulted via Context7:
https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
and https://triton-lang.org/main/python-api/generated/triton.language.dot.html.
