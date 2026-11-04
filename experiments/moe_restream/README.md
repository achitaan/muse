# TQ64 — one raw-weight stream per heavy expert

**Offline candidate, not a measured improvement. No CUDA compiler/GPU locally.**

Check file: **`experiments/moe_restream/moe_restream.py`** (standalone).
Readable CUDA: `restream.cu`. Source assembler: `build.py`; prepare/dispatch
append: `host.py`. The candidate contains all of these, without file imports.
Parent: `loop/moe-headroom` / `e578f78`, as requested.

## Exact root cause / terminology

In the incumbent, `OKC_FT_T=64`, hence `FT_T=64`, `WGS_NT=FT_T/8=8`.
`route()` assigns an expert with C>8 routes `ceil(C/64)` jobs, each with its
own `jE` (same expert), `jS` (different token start), `jC<=64`. Small experts
have one job. MID is different: `OKC_SPLIT=1` makes <=8-token jobs.

For each MAIN job, phase A runs 32 output units of 256 raw gate/up rows;
phase B runs 16 units of 256 down rows. A unit has 16 K256 stages. Each stage
copies 36 KiB: four WG pieces, each 64 rows (8 KiB codes + 1 KiB scales).
Each WG owns different rows but the same token slab. Native WGMMA N=8*NTT,
NTT<=8, with at most 32 FP32 accumulator registers per thread.

The weight producer addresses by **expert + output unit + K stage**, NOT
`jS`. Therefore sibling token jobs reread the identical 18 MiB A + 9 MiB B
weights. Unique weights are 24 MiB E2M1 codes + 3 MiB E4M3 block scales per
expert. `dup_jobs=sum(ceil(C/64))-D`, each worth another logical 27 MiB.
L2 may suppress some transactions; this is not a hardware-transaction counter.

**`quads=8` in MOE_HEADROOM is eight ABBA timing quads (32 replays), not a
kernel job/quad count.** Packing's `q=0..3` denotes the four WG row pieces;
`NTT=8` denotes 64 token columns. Neither is `dup_jobs`.

## Actual histogram model — before implementation

Source: supplied log `moe_headroom.d813e0ad059d4c23.log`. `MODEL.json` preserves
the actual eight verify histograms and recomputes all D/R/duplicate counts.
Use each draw's measured stream rate, not 4.8 TB/s or its entire residual:

| M256 draw | heavy counts | duplicate jobs | raw bytes saved | bandwidth-equivalent us | byte-only cut |
|---|---|---:|---:|---:|---:|
| 0 | 74, 225 | 4 | 108 MiB | 25.55 | 10.54% |
| 1 | 93 | 1 | 27 MiB | 6.42 | 2.99% |
| 2 | 168, 88 | 3 | 81 MiB | 19.34 | 8.98% |
| 3 | none | 0 | 0 | 0 | 0% |

Median opportunity at measured effective bandwidth: **5.98% of verify32 time**,
~1.03% of the six-load board if fully realized. The independent 4.8-TB/s byte
cost is about **5.48%**, avoiding attribution of fixed graph/ramp cost to every
duplicate. A 2% normal-stage overhead assumption leaves **3.5–4.0% per
verify32**, roughly 0.6–0.7% board. In the measured-rate model, 3% overhead leaves 2.98%; 5% leaves
0.98%. This passes the requested >=3% case-level modeling threshold only under
the explicit <~3% exposed-overhead hypothesis. It is NOT a measured prediction.

**All four M128 draws have zero duplicates** (max C=54): zero byte-saving
opportunity from this mechanism. M128 stays on the incumbent, without a TQ gate.

## Different reuse mechanism, not a closed repeat

One job now owns all C<=256 tokens of an expert. Exchange WG row parallelism
for token parallelism **inside the same CTA**:

| C | row groups R | token groups | output rows/unit | K128 copy/stage |
|---|---:|---:|---:|---:|
| 1..64 | 4 | 1 | 256 | 18 KiB |
| 65..128 | 2 | 2 | 128 | 9 KiB |
| 129..256 | 1 | 4 | 64 | 4.5 KiB |

WG low bits select a row group; high bits select a <=64-token slab. All token
WGs read the same CTA-local compressed tile, so one TMA stream serves them.
They still dequantize independently. Empty fourth slabs participate in ring
barriers but own no output/fixup/readiness contribution. K128 permits **three
weight slots plus two 64 KiB x slots**. Native N remains <=64; no larger
accumulator, SS-dequant surface, external partial spill or precision change.

- Not cluster multicast (row 27, `6627d540`): no clusters/DSM/remote barriers.
- Not L2 hint/co-location/interleave (rows 50/51, `c384c9f1`/`b9ccafb6`): the
  repeated global copies are removed, not merely hoped to hit L2. L2 is shared
  across SMs; putting siblings on one SM is not itself an L2 retention policy.
- Not native wide-N (rows 49/65/69, `8f3b54a3`/`8d55ab94`): every WG still
  uses the original N<=64 accumulator and ping-pong register buffer. Three raw
  slots remain, unlike the closed two-slot cap128 layout.
- Not SS64/32: no decoded weight matrix stored in shared memory.

Existing stream-K fixups still combine K shares, but each weight K128 piece
belongs to exactly one CTA share and is fetched once. Existing **K256** global
dependency groups are retained (`st>>1`); only their counts reflect the merged
job's active token WGs. No finer readiness policy, new global flags, or PDL.

## Important counter-cost, not hidden by the byte model

Fewer row groups mean 2x/4x more x/inter **L2-to-shared activation copies** for
heavy experts. Additional activation bytes in draws 0/1/2/3 are **291/36/222/0
MiB**. The positive model assumes these overlap weight/compute work. Total
activation rates required at the byte-model time are 2.38/0.97/2.12/0.78 TB/s
across the GPU; local burst/latency and two-slot behavior remain unverified.
If extra X time were fully serial, even a hypothetical 40 TB/s X path plus
2% stage overhead reduces the modeled median cut to 2.41%; at 20 TB/s it is
0.84%. `MODEL.json` reports these sensitivities. **The GPU gate must decide.**

## Prepared fallback gate

The original packing/build core and original `kernel_fn` remain fallback. The
TQ64 stage now runs **before** any optional legacy selection (see
`SKIP_REPAIR.md`); neither the old headroom measurement nor the exhaustive
M128/M256 selector runs first. A bounded, cached-only M16/M32 selector can run
afterward without changing the M256 gate comparator. Both incumbent and TQ64
use one explicit **C++20 host + NVCC** policy. TQ64 builds a separate extension
and private losslessly packed weights/scratch.
Only M256 can select TQ64. Two independent passes of **16 fresh judge-like
draws** each use read-flush balanced ABBA (six quads). Each pass requires a
median TQ/base ratio <=0.97 and a no-duplicate-draw median <=1.01. Numerical
gates require finite values, relative L2<=0.005 and row<=0.02 vs incumbent.
Hot 256-token experts and repeated empty graphs are also checked. Failure,
deadline or compilation error leaves the incumbent selected. No actual input
is cached; timed calls always produce a fresh output and branch only on shape.
Budget: TQ64 has the first claim on remaining setup time, with a 285 s total
gate deadline and >=60 s needed before starting its compiler. Optional cached
small-shape selection stops at 290 s. Every admission/skip prints physical
layout, scale decoder, elapsed time, total remaining time and gate remaining
time; non-PRMT and budget skips are distinct. Compiler invocation is not
interruptible, so exceptional compile duration remains a risk.

## Local evidence / gaps

Run `python3 -B experiments/moe_restream/build.py`, then `test_restream.py`.
Native compilation of actual extracted metadata helpers verifies 269,484,032
output coordinates (C=1..256), >1.2M inverse-owner checks, and the original
K256 contribution counts. Native shared-layout estimate: **209,024 B**, with
accumulator/fixup widths unchanged. Additional CPU tests cover actual packer
permutations via lazy tensors, 10k FP4 word roundtrips, read-once raw pieces,
activation transposes, 500 asynchronous active/empty-WG ring traces, fallback
dispatch and exception handling. CUDA/NVRTC/ptxas, numeric GPU tolerances,
actual register/stack resources, graph replay, gate time and speed are **not
verified**. Source attributes and per-draw gate results are printed on H200.

## Decode and nonduplicate verify residuals

M32 has no duplicate jobs in all eight draws, yet a 25.7 us median gap; M16's
nonduplicate draws retain ~13–16 us. M128 retains ~32.9 us with no duplicates;
M256's zero-duplicate draw retains 50.1 us. Restreaming cannot explain these.
The floor omits routing/gather, activation traffic, dequant/MMA, dependencies,
stream-K fixups, imbalance and norm/finalization. It does not localize a cause.
Prior finalizer optimizations/ablations (rows 37/59/98) recovered only a small
part, so attributing all of these gaps to the finalizer/router would be wrong.
Another decode attempt needs a **resource-matched** phase/stall trace on the
same D/R/histograms to separate gather/startup, A/B consumers, dependencies,
tail and finalizer. That is outside this one bounded restream attempt.

No OKC submission/check, main commit/push or other-worktree modification was
performed. Parent must merge the local TRIED row into the shared ledger.
