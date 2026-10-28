# MoE second skinny tile: CPU prototype

This is a prepared alternate CUDA source, not a selected or GPU-tested kernel.
Prototype SHA256 `921dc83ec4bdf76b0c33e777c028adfdfec84265439ca390a7369e1b6f410f96`.
The complete checked MoE33114871 source is an exact prefix. Its CUDA, packing,
preparation and dispatch remain unchanged. `ALT_SRCS['sk16']` and one define
tuple are appended; no current call launches this alternate.

## Measured reason and distinct algorithm

The retained 16-token MoE uses eight-token skinny jobs. A popular expert with
9..16 local tokens needs two jobs, each streaming and dequantizing the same
weights. One 16-token job can feed both token tiles from a single weight read
and dequantization. Counts1..8 still need one pass, and counts9..16 would need
one instead of two. This is a logical traffic reduction for hot experts,
not a measured total HBM reduction or a speed claim.

The pinned public code's dormant SKT16 branch is incomplete: it computes two
token tiles, but stores only the first float4 into the shared partials array.
The second tile's reduction reads the unwritten float4. The current SKT8
incumbent cannot enter that path, so this is not a current-submission defect.
The helper-warp path also explicitly requires SKT8; the prototype uses the
existing static non-helper ring. It is different from the rejected mixed
skinny/fat selector and from the rejected two-warpgroup WGS architecture.

## Repair and numerical contract

Only the alternate changes SKT8 to16, its static skinny_ring, and the kernel
name for profile evidence. A reversible ledger recovers the original CUDA
bytes exactly. Dynamic skinny and helper-WGS paths are not selected. Every
original weight/scale format, FP4 decode, BF16 Marlin GEMM boundary, SwiGLU,
FP32 norm accumulation, routing and final BF16 output remains.

Each token tile has four FP32 accumulators, one for each K16 subgroup, just
as the checked first tile does. Both use the same dequantized weight registers.
The second tile has its own quartet, its own original-order merge, and now
stores its float4 into shared red[...,4..7] before the existing CTA barrier.
This avoids the serial single-accumulator arithmetic in the dormant branch.
It preserves the intended original arithmetic grouping; CPU BLAS is not a
hardware MMA emulator and cannot establish bitwise CUDA identity.

The existing upper-eight input loader already stages fresh activations in
warp-private [8,520] BF16 storage. Its __syncwarp precedes the reads. Shared
partial reductions wait at the existing CTA barrier. The parity-two partial
buffers and each next-iteration CTA barrier protect reuse. No new barrier,
global counter, weight copy, persistent activation or output cache is added.

## CPU evidence

`audit_moe_skinny16_cpu.py` checks all counts1..16, eight warps and32lanes:
524288 second-tile BF16 operand words exactly match the checked eight-token
loader applied to a job starting eight entries later. Payloads include signed
zero, subnormals and nonfinite bit patterns as representation probes. Padding
has unique writers and zeros; alignment padding is never read. All4096 K
elements occur exactly once. Four modeled uint4 shared-memory transactions
have no bank conflicts.

The reduction/store ownership model covers every cell of a [count,16] output
tile exactly once for16 counts, and detects the missing-second-store mutant.
The fixed non-helper Smem ABI model has sizeof36368 and dynamic shared214656
bytes, below H200's232448-byte limit. This is an ABI estimate; actual CUDA
resource queries and occupancy remain mandatory before attribution.

Independent split-job and combined-row FP32 numerical models compare against
FP64 for M9,15,16. Their maximum FP64 relative L2 error is9.25e-8. M9's two
CPU BLAS arrangements differ by2.38e-8; M15/16 agree in this sample. These are
finite CPU screens, not full MoE numerical verification or a hardware proof.
At M<=16 there is at most one new job per local expert: <=32 jobs versus
JMAX88, with at most128 routed entries versus capacity3072.

## Work needed before a GPU experiment

Do not submit this prototype as a performance trial: its original dispatch is
unchanged. First add a bounded prepare comparator and metadata-only selector
for the whole M9..16 bucket. Keep M<=8 and M>=17 on their exact incumbents.
Use self-generated diffuse, hot and mixed routing with unique IDs and positive
normalized weights; exercise nearby M9/15/16 and an analytic all-remote case.
Compare actual GPU outputs with the checked incumbent and reject nonfinite or
excess per-row errors. Preserve all samples from paired cold-L2 CUDA graphs
on M16, alternating order. Enable only a consistent material winner.

Freeze a source-bound full5-tests/6-benchmarks/integrity guard that requires
nonempty selection, actual moe_fused_sk16 in the16-token profile, complete
GPU probe/paired evidence and affected-versus-unchanged-control improvement.
Original preparation took163.434 seconds; one extra compilation and probes
must fit300 seconds. No geometry sweep, fresh GPU baseline or score for noise.
Dense4d443b13 is already fully prepared for its next check; this MoE prototype
is subsequent work. MoE/Dense checks were exhausted at the last API snapshot
and reset at00UTC. The broader top50 objective remains active.
