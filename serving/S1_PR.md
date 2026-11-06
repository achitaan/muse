# Use fused post-expert norm for small NVFP4 Marlin batches

## November 5 preparation update

The v2 patch has now been tested through normal imports on the official H200
image and pinned SGLang commit: **115 focused tests passed**, and the combined
S1/dormant-MoE integration suite passed 159 tests. Fresh paired measurements of the
complete synthetic expert operation saved approximately 23–40 microseconds at
T=1/8/16, with three captures, stock controls, and separate cold/steady timings.
The routing/shape sweep covered 336 cases. Generated-answer agreement and full-server
throughput remain unmeasured; some observed stock-repeat-equal outputs differ under
S1, so numerical operator tests do not establish serving acceptance.

See [the completed handoff](PRECOMP_HANDOFF.md), [raw results](results/precomp-20261105/),
and [the grouped measurements](PRECOMP_RESULTS.json). The earlier cleanup validation
and “no local GPU” statements below describe the historical preparation stage.

## Motivation and change

`MuseSpark1xFp4Experts._forward_marlin` already uses the Triton post-expert
RMSNorm + routing-weighted sum for batches of 32 or more tokens. Smaller
batches fall back to a sequence of PyTorch operations. Allow the Marlin caller
to select the existing kernel for every nonempty CUDA batch instead. No new
kernel or environment variable is needed. The shared helper's default cutoff
remains 32; the BF16 MoE consumer is unchanged, as are weights, routes, norm
ordering, epsilon, and BF16 activation/output precision.

The checkout is shallow at `a55ab7d5`. Blame attributes the cutoff to the initial
"Support Muse Spark Open" commit; no comment explains why 32 was chosen. There
is insufficient history to claim a historical performance rationale.

## Existing measurements (operator level only)

Prior H200 diagnostic `cd2e95e7`, source `serving/s1s2_opdiag.py` on
`origin/argon/serving-20261102` in the Muse repository: hidden size 4096, top-k 8,
CUDA-graph replay, read-based L2 flush outside timed events, alternating A/B
order, median timings. The image helper matched the pinned stock implementation.
These measurements are inherited evidence, not rerun during this cleanup.

| Tokens | Stock us | Fused us | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 33.8 | 6.8 | 4.9x |
| 4 | 37.9 | 9.8 | 3.9x |
| 8 | 39.1 | 8.5 | 4.6x |
| 9 | 39.7 | 9.1 | 4.4x |
| 16 | 42.3 | 9.2 | 4.6x |
| 24 | 45.9 | 9.2 | 5.0x |
| 31 | 48.3 | 9.2 | 5.3x |

At 32/33 tokens both implementations already use Triton and timings were
essentially unchanged. Server throughput, TTFT, and generated-token agreement
are **unmeasured**. No challenge checks or submissions were run for this PR.

## Correctness and tests

Both implementations apply each expert's RMSNorm before its routing weight,
with FP32 norm arithmetic/accumulation and BF16 output. Invalid local expert
IDs are excluded: the fallback zeros those slots before normalizing, while
Triton skips them without reading potentially uninitialized expert outputs.
The kernel and zero-token early return are unchanged.

The prior H200 diagnostic covered local, mixed, all-remote, and single-live-slot
patterns, poisoning remote rows with NaN/Inf. Outputs stayed finite; all-remote
outputs were exactly zero. Both paths had relative L2 error 1.66–1.70e-3 against
an independent FP64 reference; maximum stock/fused difference was 0.0078
(BF16-ulp scale).

Focused tests live under `python/sglang/test/kernels/`: CPU fallback/override
equivalence, mocked CUDA dispatch without GPU allocation, and CUDA numerical
tests against FP64 around the original cutoff. CUDA tests explicitly skip when
CUDA is unavailable. CPU tests require the project's PyTorch/Triton dependencies
but do not require a GPU.

Local cleanup validation: 32 CPU numerical cases and three mocked CUDA-dispatch
cases passed with PyTorch 2.14.1 using an isolated source-loading harness (the
production helper and test functions, with no Triton execution). Full pytest
collection/import was not run because pytest/Triton are absent; CUDA numerical
tests were not run. `py_compile`, `git diff --check`, allowed-path/source checks,
and the 400-character/400-KB limits passed.

## Risks and remaining validation

Changing reduction order can move a BF16 output by an ulp and could flip a
greedy token versus stock. Operator accuracy is not proof of server answer
agreement or end-to-end benefit. Run stock-vs-stock and stock-vs-patched serving
comparisons, especially low-concurrency/small-token batches, when GPU access and
the serving window are available (Nov 10, 09:00 PT). No local GPU was available.
The BF16 consumer's cutoff is intentionally not changed without separate evidence.
