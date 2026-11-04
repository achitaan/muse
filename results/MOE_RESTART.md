# MoE restart: RSTREAM

## Check file

`experiments/moe_restart/moe_restart.py` — self-contained; native CUDA only
through `torch.utils.cpp_extension.load_inline`. Branch `loop/moe-restart`,
based on `63e253f`. No GPU check or submission has been made by this lane.

## Structural restart

Compressed weights never visit shared memory in the new core. Consumer warps
load their disjoint losslessly permuted NVFP4 codes into registers. Eight
lanes load each warp's unique scale bytes, then broadcast them at consumption.
Register lookahead is six K64 groups for N<=32, three otherwise. Broadcasting
is deliberately delayed: a shuffle in the prefetch function would wait for
the scale load and serialize lookahead.

A CTA owns 128 weight-output rows with **two** consumer warpgroups and one
activation producer warp (288 threads). M<=8 uses 64 rows/one WG/160 threads,
avoiding the very low decode1 task count of 128-row tiles. WGMMA consumes
register A fragments directly. Only activations use a three-stage K64 shared
ring. Native token axes are 8/16/32/64/96/128/192/256: an expert has **one
entire token list**, including hot experts with >64 tokens.

Whole-output-tile dynamic ownership assigns each `(expert, tile)` exactly
once, traversing full K. No stream-K partials or float atomics. Gate/up are
interleaved in eight-row groups. A writes BF16 SwiGLU directly into B's
packed activation layout. One persistent launch performs routing, gather,
A, B, and norm/weighted sum. Three ordinary resident-grid barriers order
gather/A/B. No readiness flags or PDL added. The 128-row weight loads retain
evict-first L2 policy to protect activation packs; tiny uses readonly loads.

Differences from closed structures:

- **MAIN/WGS:** no weight shared ring/producer, <=64-token jobs that duplicate
  weights, cost-table stream-K scheduling, or fixups.
- **VMS/VML:** not three launches or swapped-operand mma.sync on staged weights.
- **P2-RS64:** not a single-WG shared-weight consumer repeated at two CTAs/SM.
- **SS64/32:** no dequantized-weight shared writes/reads.
- **Producerless Humming:** tiles own weight rows, not eight activation rows;
  weights are not reread for different token tiles.

Fallback is the attributed incumbent, not an approximation. Its known
empty-local-job WGS range bug is repaired by an empty-range early return.
Otherwise arithmetic, builds, setup and the existing incumbent build selector
are retained. RSTREAM admission is separate and stricter.

## Corrected byte/latency accounting and score

For **each actually touched local expert**, H=I=4096:

| Item | Bytes |
|---|---:|
| gate/up E2M1 codes | 16,777,216 |
| gate/up E4M3 block scales | 2,097,152 |
| down E2M1 codes | 8,388,608 |
| down E4M3 block scales | 1,048,576 |
| **Total** | **28,311,552** |

Also: 8 global-scale bytes/expert and 8,192 norm-weight bytes/call. Block
scales are **1/9 of compressed payload**, not free. Codes remain exactly
four bits/value; scales remain eight-bit E4M3. No five-bit deltas, weight
dequantization cache, or requantization.

At 4.4 TB/s, unique compressed payload costs 6.434 us/touched expert. Count
actual touched experts, not all 32 or the task's 0.75 average-routing-share
floor heuristic. Routing/top-k are inputs: router logits/top-k selection
need not be computed. Every local route, BF16 activation and supplied FP32
route weight is still processed.

For 128-row tiles, activation shared traffic/expert is
`(64 A tiles + 32 B tiles) * 4096 * 2 * Npad * (1 write + 2 WG reads)`
`= 2,359,296 * Npad bytes`, **Npad/12 times** the weight payload. There is
no compressed/dequantized-weight shared traffic. Applying this to the
actual histograms in `d813e0ad059d4c23`:

| Case | Median padded tokens/touched expert | Shared TB/s at 4.4 TB/s weights |
|---|---:|---:|
| decode16 | 8.00 | 2.93 |
| decode32 | 8.68 | 3.18 |
| verify16 | 14.19 | 5.20 |
| verify32 | 21.45 | 7.86 |

The 64-row tiny core has ratio Npad/9, or 3.91 TB/s at Npad=8. These are
logical counts, not measured throughput. All-hot patterns can still become
shared/math bound. Activation packs are reread from global addresses for
different output tiles but should remain L2-resident **within a call**;
failure to retain them is a material risk, not excluded from accounting.

The supplied paired raw-graph diagnostic measured unique-stream gaps of
15.2 us (decode16), 25.7 us (decode32), 10.5 us (verify1), 32.9 us (verify16),
and 64.0 us (verify32). It did not profile decode1. Verify32 really has
repeated heavy-expert weight jobs; its gap is not a 4% roofline entitlement.
RSTREAM targets consumer/ring losses as well as duplicate jobs. A launch is
still required. The public spec's ~4 us fixed graph overhead is subtracted
for scoring; distinguish it from raw graph events and the once-read stream
floor. Decode1 has fewer tasks and larger relative launch/ramp/finish costs;
the 64-row path is an attempt, not a proven small-batch win.

The six loads have equal weight (60). Each load's ratio enters the geometric
mean equally: this is **not the sum of six microsecond numbers**. Going from
5651.4 to 4834.6 us displayed time requires ratio 0.85547, or 14.45% less
time. The measured gaps leave that goal physically plausible; this structure
removes the rejected shared-traffic obstacle and duplicate weight passes.
**Reaching 2.78x has not been measured or established.** The load/decode
pipeline and tails must actually deliver; the gate may admit no cases.

No public leader design is known or assumed here. Their aggregate scores do
not imply that they beat every per-case streaming floor. The local motivation
is independent: duplicate jobs cost about 7.7% at M256, and phase B has
consumer back-pressure (shared ledger rows 48, 77, 82 in the supplied copy).
Producer-dequantized SS-WGMMA/native-N128 remains a separate follow-up, not
part of this candidate; it should first win a GEMM2-only comparison. Scale
dictionary/entropy coding is likewise not implemented, and this lane does
not touch the other session's five-bit scale-delta work.

## Per-case admission and numerical boundaries

Prepare compiles RSTREAM concurrently with fallback setup. Only SM90 and
positive-normal FP8 bytes 0x08..0x7e are eligible; other weight domains stay
on fallback. New arithmetic preserves BF16 GEMM output, BF16 global-scale
multiply, BF16 SwiGLU, and BF16 effective norm weights. Norm/reduction are
FP32, epsilon 1e-8.

Per case, the gate:

1. Mutates inputs **at the same captured graph addresses** through hot,
   all-remote, all-local, zero, and random transitions.
2. Requires repeatable BF16 output and L2 error <=0.005 globally / 0.02 per
   row versus fallback, with an average-row floor for near-zero rows.
3. Checks an independently written full FP32 NVFP4 reference on an M-token
   hot expert in every case, and a multi-expert M256 draw, with <=0.012
   global / 0.05 row error. Public bounds are 0.015 / 0.075. This is exact
   computation within the numerical contract, not bitwise agreement with
   occasionally nonrepeatable fallback output.
4. Times eight independent realistic skews with read-flushed L2, four ABBA
   and four BAAB quads each. Both order-specific geometric means must be
   <0.97; six of eight draws must be >=1% faster, none >1% slower. Raw graph
   overhead is not subtracted, making admission conservative.

Only admitted counts enter `state['restart_cases']`. Other counts, including
nearby correctness-test shapes, use the incumbent. Exceptions or a prepare
budget limit leave unadmitted cases on fallback. Test inputs/references are
discarded. Timed calls read fresh inputs, allocate fresh output, and never
consult saved activations/reference outputs. Shared scratch needs sequential
use, as in the incumbent.

## Local verification

`results/moe_restart_cpu.json` records passing CPU tests:

- full-K, 512-row code/scale roundtrips for tiles 64/128; direct packing
  equals conversion from retained WGS bytes;
- all 16 E2M1 codes times all 119 admitted FP8 bytes decode exactly;
- 60 routing patterns, including 65/97/129/193-token hot boundaries,
  all-remote/all-local, activation bijections, descriptor offsets, capacity;
- 660 ownership plans and ring/register-lifetime models;
- analytical one-nonzero-column MLP, BF16 boundaries, RMSNorm;
- admission-policy rejection and zero/nonfinite guards;
- Python compile/import/AST and generated MMA operand checks.

Reproduce:

```sh
uv run --offline --no-project --with torch --with numpy python experiments/moe_restart/validate_cpu.py
python3 -m py_compile experiments/moe_restart/moe_restart.py experiments/moe_restart/validate_cpu.py
for side in host device; do
  /usr/bin/clang++ -x cuda --cuda-${side}-only --cuda-gpu-arch=sm_90 \
    -nocudainc -nocudalib -fsyntax-only -std=c++17 -DRSTREAM_DEVICE_ONLY=1 \
    -Iexperiments/moe_restart/cuda_syntax_stubs results/moe_restart_instantiated.cu
done
```

Both instantiated CUDA families passed Clang host/device **frontend semantic
checks with declaration-only stubs**. This is NOT NVCC/NVRTC/PTXAS validation:
there is no CUDA SDK, NVRTC, ptxas, GPU, or NVPTX code-generation backend here.
Stubs/generated `.cu` files are validation-only, not used by the check file.
Harness files were initialized to read the spec, never edited or imported
by the candidate.

## Risks / next check

- NVCC/load_inline linking, PTX assembly (including hinted vector loads),
  actual register counts/spills and WGMMA scheduling are unverified.
- N256 needs 128 accumulator registers under the 288-thread/224-register
  ceiling. Ptxas could spill/serialize. Runtime occupancy is checked.
- Async visibility/parity, capture/replay and correctness have not run on
  GPU. A severe device fault/deadlock is not recoverable by Python fallback.
  Residency/grid barriers assume sequential scratch use, not concurrency.
- Register loads may lose to bulk TMA despite lookahead; sparse experts and
  whole-tile tail quantization may lose.
- The gate has a 275-second deadline, but compiler/setup overrun itself
  cannot be interrupted by it.
- No measured speedup exists. Next action: one authorized `okc check` of
  the candidate; inspect `RSTREAM_REF`, `RSTREAM_GATE`, `RSTREAM_SELECTED`
  and ptxas diagnostics.

The ledger was searched read-only with `scripts/tried.py`; configured
`.tools/tried.py` does not exist. This branch records a row. Publishing it
to shared main is left to the orchestrator: this lane was instructed never
to commit to main or push.
