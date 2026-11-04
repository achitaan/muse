# MoE restart: SS64/32 output-owned decompression engine

**Decision: design only; do not build yet.** The HBM-only ceiling has room,
but a credible >=3%-above-best end-to-end gain is not established. No candidate
or OKC check was created. Branch `loop/moe-redesign` starts at `63e253f`.

## A fundamentally different structure

Two launches for M16/M32, falling back to the existing kernel elsewhere:

1. **GEMM1:** full-K, output-owned 64-row gate/up tiles. A CTA has one producer
   warp, one dedicated 128-thread dequantization WG, and a different 128-thread
   **SS-WGMMA** WG (288 threads). Compressed four-slot lookahead feeds two
   decoded BF16 shared-memory tiles; raw scales remain full E4M3 bytes. Compact
   active-expert metadata is rebuilt from fresh routes in each CTA. Every
   expert's NVFP4 codes/scales are read once, with no K splitting/fixup atomics.
2. **GEMM2 + finish:** full-K, 32 useful rows padded to the hardware's 64 rows,
   so both phases have 128 tasks per touched expert and avoid a coarse B tail.
   B writes weighted FP32 y and route sums of squares. One resident-grid
   completion barrier precedes the final norm/route sum in the same launch.
   Use cooperative launch with verified residency; this is not per-expert
   readiness signaling or PDL. A clears all per-launch counters/sums.

Target four CTAs/SM: about 33/40 KiB shared per M16/M32 CTA and <=56 registers
per thread would fit SM90 resource limits. These are estimates, not compiler
results. The intended advantage is overlapping dequantization and tensor-core
issue in **different** WGs at higher residency—not tweaking MAIN/WGS. Compared
with closed P2-RS64 (ledger row 83, `08c52957`), the MMA WG does no dequantization
and uses SS operands. Compared with VMS (rows 85–93), this is not swapped-operand
mma.sync or a three-launch split-K engine. FS1's K16 outer products (row 62)
are also absent. The price is substantial extra shared-memory traffic.
The closed dense-c12 SS-WGMMA experiments (rows 132–136) are a warning, not
evidence that the more aggressively expanded NVFP4 path will work.

## Correct byte floor: do not use the published SOL directly

For one touched expert, H=I=4096:

| matrix | E2M1 codes | E4M3 scales | total |
|---|---:|---:|---:|
| gate/up | 16 MiB | 2 MiB | 18 MiB |
| down | 8 MiB | 1 MiB | 9 MiB |
| **one read** | **24 MiB** | **3 MiB** | **27 MiB = 28,311,552 B** |

For actual routes, bytes are **27 MiB x `len(unique(local topk_ids))`**, not
32 experts, not the route-pair count, and not `0.75 x distinct`. Global scales
add 8 B/expert; norm weights add 8 KiB. The supplied profile does **not** contain
actual route histograms, so their job-specific byte counts cannot be recovered
from timing. `actual_weight_bytes()` provides an exact counter if routes are
supplied; no synthetic routes are presented as judge routes.

The public reference's SOL uses **0.75 x nominal mean distinct**. Undoing that
discount gives the following **nominal workload means**, not measured draws:

| M | nominal mean distinct | codes MB | scales MB | once-read MB | HBM floor incl. x/out | ours |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 11.86 | 298.467 | 37.308 | 335.775 | 70.008 us | 93.409 us |
| 32 | 15.97 | 401.898 | 50.237 | 452.135 | 94.304 us | 126.908 us |

HBM=4.8 TB/s. Even all 8M routes' unpadded GEMMs require only 13.0/26.1 us at
989 TFLOP/s, below the byte bounds. Optimizing just these two of six loads
gives the idealized score `2.347287 * ((93.409/70.008)*(126.908/94.304))^(1/6)`
= **2.58782**, +8.75% over the 2.3797 best. Thus **MoE is not proven at its HBM
floor**. This is an upper bound, not an expected speedup.

## Why the implementation gate does not pass

Targets of 84/109 us would score **2.45054**, only **+2.977%** over the best,
against the requested >=3% threshold of **2.451091**. The insignificant decimal
difference is not the real concern: those targets already assume very high
bandwidth and little launch/ramp/barrier/tile overhead, with no measured margin.

More importantly, each expert now causes 27 MiB raw TMA writes, 27 MiB raw
decoder reads, **128 MiB decoded writes**, and **128 MiB tensor-core weight
reads** (B's 32/64 padding included). At 84/109 us this requires generic shared
load/store traffic of **22.95/23.81 TB/s**, TC operand reads of **18.95/19.66
TB/s**, and **45.90/47.63 TB/s summed** weight traffic, before X traffic.
Whether these bandwidths share ports or can run concurrently is decisive.
Adding them as one bottleneck without hardware evidence would also be wrong.

**Bounded question for the parent's Oracle/researcher:** can SM90 sustain these
three traffic classes concurrently at four 288-thread CTAs/SM, or does shared
bank/operand bandwidth invalidate the target? Also obtain actual M16/M32 D/R
histograms before making an actual-byte headroom claim. No CUDA toolchain/GPU
is available locally to resolve either question. Per the implementation role,
stop rather than guess this architectural throughput assumption.

## Gate if that question is resolved positively

Prepare compiles the new engine, verifies >=4 cooperative-resident CTAs/SM and
no spills, and otherwise retains the incumbent. On at least 32 same-input
route draws per M, log exact D/R/weight bytes, check finite outputs, relative
L2 <=0.005 and row <=0.02 against the incumbent, and test hot/empty repeated
graphs. Use read-flush paired ABBA, not separate prepare build-selector scores.
Select only shapes whose confirmed gains support >=3% projected board gain;
fallback on any failure. No timed input caching, host sync, precision change,
expert dropping, scale deltas or PDL. **This gate is planned, not implemented.**

Reproduce the analysis: `python3 results/moe_redesign_roofline.py` and
`python3 results/test_moe_redesign_roofline.py`. Local evidence is analytical
and CPU-only; JOINFULL is closed, and pending EXACTDIV was not queried.
