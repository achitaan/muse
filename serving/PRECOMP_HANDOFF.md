# Serving preparation: completed bounded run

Prepared November 5 Toronto time; GPU logs use November 6 UTC.

**S1 is the first candidate. The direct-Marlin MoE port is experimental and disabled.**
No official check or scored submission was used. Generated-answer agreement and
whole-server throughput remain unmeasured for both patches.

Branch: `codex/serving-precomp-20261105`, based on main
`19e09dd694d748d6ef77bd0e09308f56c3b210ca`. Earlier experimental branches and retained
kernel sources were preserved. The original four local baseline files were brought
across and extended here. Work was performed in the continuing chat without subagents.

## Artifacts and pins

| Artifact | Status | SHA256 |
|---|---|---|
| `s1_postnorm_v2.patch.diff` | First candidate; 7,375 bytes | `ea7f03abf143c671694d4caa421530f8d76bcc531a7c4b6063a358a1aabf9a7f` |
| `moe_direct_marlin.experimental.patch.diff` | Dormant integration; 168,310 bytes | `e85da7cab68b2536dc6fe259834c062a7b4378a93abd423ec9c3246b53a9db8e` |
| `native_marlin_wgs.cu` | Frozen third prototype | `3eb38b8d658e3becd0d314c2ce4ad717786b9859c7570cee4392458411340736` |

All source hashes are in [PRECOMP_SOURCE_HASHES.json](PRECOMP_SOURCE_HASHES.json).
The S1 patch's Windows line endings were normalized to LF; its code change is the
same v2 patch retained on main. Both patches apply to the exact SGLang base and also
compose successfully. The experimental patch has an **empty accepted-shape set**;
applying it does not enable the native kernel or allocate its workspace.

- SGLang: `a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`.
- Official image: `lmsysorg/sglang@sha256:8cfc5a72cb296771f5f8a4a4688cb6d05685610a830cbb616fa4ab0eb7bb2b0a`.
- Runtime: `/opt/sglang/bin/python`, Python 3.12.3, PyTorch 2.14.1+cu130,
  Triton 3.8.0, nvcc 13.0.88, driver 595.91.07.
- Hardware: one H200, 143,771 MiB reported VRAM, 132 SMs, US-CO-1.
- The official image's default Python does not provide this Torch environment.
  Its ordinary `/sgl-workspace/sglang` checkout was not the pinned revision; clean
  pinned stock/S1/MoE checkouts were created explicitly.

## Fresh measurements

These are **synthetic single-rank expert operations**, after pre-expert norm:
routing/layout preparation in the call, GEMM1, SwiGLU, GEMM2, and the stock
post-expert norm/combine. They do not include the router, projection, communication,
DFlash generation, or the complete server. Synthetic weights are not checkpoint
weights. Performance was measured at EP offset 32 with uniform global top-8,
mixed two-local/six-remote, and hot-expert routes.

Both arms use identical inputs/routes within each comparison. There are three
separate graph captures, balanced randomized orders, two stock arms, and raw samples.
Cold-L2 and steady replay results are separate. CUDA events are converted from
milliseconds to microseconds by multiplying by 1,000. Compilation, preparation,
and the read-based 256-MiB L2 flush are outside timing.

S1 saved **23.26–40.42 microseconds** across the measured T=1/8/16 cases. T>=32
retains the existing fused path and stayed within the stock-control variation.
See `results/precomp-20261105/results/s1-paired-v2.jsonl` and the grouped values in
[PRECOMP_RESULTS.json](PRECOMP_RESULTS.json). At T=1 some uniform draws have no local
experts, so their large ratios must not be generalized to a server workload.

Final MoE latency reductions below span all three blocks and both cache conditions.
Positive means faster; negative means slower.

| Tokens | Uniform global top-8 | Mixed | Hot experts |
|---:|---:|---:|---:|
| 32 | -13.00% to -10.31% | -10.44% to -9.92% | -11.76% to -11.23% |
| 128 | 32.17% to 32.90% | 27.97% to 29.90% | -0.78% to 0.32% |
| 256 | 39.37% to 41.37% | 33.62% to 35.13% | -1.02% to 1.07% |

The native implementation therefore **does not pass the retention gate across
routing regimes**. T=32 remains stock. No input-dependent routing-count selection,
benchmark-case detection, or new shape regime was added to rescue the result.

## Implementation and numerical decision

The port adapts the retained persistent/WGS kernel to use 2-D TMA loads from resident
Marlin packed codes and encoded scale bytes. It adds no persistent weight bank.
It reconstructs routes every call, initializes every per-route output, returns BF16
expert outputs, and delegates norm/combine to stock. A zero-job range guard prevents
reading a nonexistent first job.

The integration passes a keyword-only `allow_decode_kernel=False` through the actual
Muse call chain. It admits only decode/target-verify with the required TP/EP layout,
without SP or fused AR, after pre-expert norm. Other callers default to stock.
Preparation is attached after stock repacking/global-scale collapse, outside capture.
One model-owned workspace is shared across layers; all calls and retained graphs must
execute sequentially. It must live as long as any graph references it.

The bounded numerical corrections were:

1. v1 used a 2^57 operand rebasing inherited from the board design, undoing it at
   the BF16 accumulator boundary. Its complete operation was fast, including hot
   routing, but tiny activations exposed a serving-equivalence issue.
2. v2 kept stock's encoded operand/accumulator scale. This exposed subnormal loss
   in the inherited global FP32 atomic fixups: one tiny-input GEMM1 comparison had
   35.8% relative L2 difference.
3. v3 uses CAS-based FP32 addition for those fixups to preserve subnormal bits.
   This reduced that comparison to approximately 0.785% GEMM1 / 1.149% SwiGLU
   relative L2 difference, at tiny absolute magnitudes. Both final combined outputs
   were zero in that probe. Ordinary-input differences remain as well.

The second correction costs the hot-expert performance advantage. The architecture
is frozen after the two correction cycles. No old kernel-track tolerance was used
to accept it. Stock-versus-stock variation exists in some cases; in other observed
pairs stock matched exactly while the candidate differed. Those differences remain
**unresolved serving-equivalence issues**, not numerical passes or statistical wins.
S1 likewise has small output differences in some observed stock-repeat-equal cases.

The tile oracle was also corrected to include stock's BF16 dequant multiplication
before undoing exponent rebasing. Treating encoded bytes as ordinary FP8 values or
using an ideal FP32 product was incorrect for subnormal scale inputs. CUDA comparisons
then passed, including zero/subnormal encodings and unequal pre-collapse gate/up
global-scale fixtures. The fixture explicitly models the loader's gate-column collapse;
it is not a checkpoint-load test.

The FP32 global-atomic behavior is documented in the
[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/archive/12.2.2/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red).
The timing consequence above is our measurement, not a prediction from documentation.

## Validation and limits

- CPU coordinate/scale checks and CUDA byte comparison against the stock repacker passed.
- S1: 115 tests through normal imports, then 336 routing/shape cases across four EP
  offsets (312 nonempty, 24 empty), including fresh-input graph replay and exact-zero
  all-remote/padding results.
- Final native program: 1,392 cases across every EP8 offset and all planned boundary
  and captured shapes; 6,720 fresh-route/input replays; finite outputs, local-ID
  agreement, and initialized invalid slots. These checks record numerical differences
  rather than treating mere finiteness as serving acceptance.
- 36 retained-graph replays alternate T=1/32/128/256/257 on the same scratch,
  including all-remote, padding, hot, uniform, and zero inputs.
- Compute Sanitizer memcheck reported zero errors for the final numerical program.
  This is finite evidence, not a general race-freedom proof.
- 44 actual-import integration tests passed; both patches together passed 159 tests.
- Both patches satisfy the fetched path/suffix restrictions, 409,600-byte cap,
  400-character added-line limit, and forbidden-word checks. Clean patch application
  and combined `git diff --check` passed.

The large replay sweep and sanitizer used CUDA source `694e2b76...`. The distributed
source removes spaces from one blank line; the exact distributed hash was rebuilt
and used for `native-release-paired.jsonl`. Earlier numerical revisions, failed
builds, the diagnosed oracle failure, and the superseded harness assertion are retained
in the raw evidence rather than relabeled as successes.

Workspace is **175,163,680 bytes** for one bound layer including the shared output
and local-ID buffers; each additional layer adds 768 bytes of descriptors/global
scales. Sixty layers total approximately 167.1 MiB, below 256 MiB/rank. Each block
uses 231,168 bytes dynamic shared memory with 576 threads; runtime occupancy permits
one resident block/SM and the launch uses 132 blocks. The final cold extension build
and fixture preparation took about 67 seconds in one recorded build. Full-model
startup, eight-rank execution, and concurrent same-state calls were not validated.

## Cost and cleanup

Task pod `ndo64yfaozb0yf` was created at 02:37:07.969 UTC and deletion was verified
by 03:35:31 UTC on November 6. It used one H200 at $5.29/hour and an 80-GB ephemeral
container disk; no network volume or eight-GPU resource was created.

Compute estimate: $5.1475. Storage estimate: $0.0108. **Total approximately US$5.16**,
well below the $50 cap. These include provisioning time and use the deletion-readback
time as an upper bound. Runpod's billing API had no posted records yet; this is not
an invoice. Storage uses its published $0.10/GB/month rate, prorated with a 30-day
month; see [Runpod pricing](https://docs.runpod.io/pods/pricing).

All 55 raw evidence files were exported and checked against remote SHA256 sums
before deletion. Pod deletion returned 204; the subsequent list was empty. The
task's account SSH public key was removed, preserving any other keys, and its local
private key was deleted. There is no ongoing task-created compute/storage charge.
See [PRECOMP_SESSION.json](PRECOMP_SESSION.json).

## Reproduction and opening-day gate

Use [PRECOMP_REPRODUCE.md](PRECOMP_REPRODUCE.md) for commands and
[PRECOMP_RESULTS.json](PRECOMP_RESULTS.json) for machine-readable summaries.
`results/precomp-20261105/SHA256SUMS` binds the raw files. The experimental v1/v2
snapshots are under `experiments/`; do not substitute their better timings for v3.

The fetched [official serving manifest](https://openkernelchallenge.dev/api/v1/serving)
confirmed opening November 10 at 09:00 Pacific / noon Toronto, the pinned image and
commit, three C=16 checks/day and one scored run/day. The preparation cutoff remains
November 9 at 20:00 Toronto. Re-read the evaluator manifest before deployment.

The authorized opening sequence remains **one C=16 check for S1**, reserving other
checks for diagnosed repairs. A passing answer/TTFT check precedes a scored run at
C=1/4/16/32. No rank, server speedup, or generated-answer agreement is claimed here.

## Self-review

Reviewed the actual phase callers, pre-capture binding, shared workspace ownership,
original source hash, tile coverage, zero-job path, barrier/resource limits, BF16
boundaries, patch composition, raw controls, failed experiments, and cleanup evidence.
The accepted-shape set remains empty because neither small numerical differences nor
positive isolated timings establish serving readiness. No independent review is claimed.
