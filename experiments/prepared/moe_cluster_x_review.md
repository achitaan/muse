# MoE cluster activation multicast: feasibility packet, not GPU-ready

Baseline is retained33114871, CUDA1e60cf46. No full CUDA candidate, host wrapper,
GPU evaluation or speedup exists yet. The fixed design under investigation is
two CTAs per cluster, activated initially for M>=64 with exact PRMT eligibility.
All smaller and ineligible paths must retain their existing implementation.
The existing chunking contract must cover a final short chunk as well.

## Concrete mechanism and scope

`wgs_xproducer` bulk-copies the same job/stage activation tile for different
output units. The proposed pair handles adjacent output units with the same
K-stage range and job. One leader multicasts X to both CTAs; packed weights
remain private and unchanged. It preserves four consumer warpgroups, both
producer warps, NTT geometry, weight layouts, scale bytes and arithmetic.
This is distinct from the rejected occupancy, full-producer-WG, four-tile and
unconditional unit-alignment designs.

Keep stream-K. Schedule over paired-unit groups and G/2 clusters; do not round
all ranges to whole units. Phase A has32 units/job, B16, so their paired domains
have16 and8. The source-table per-stage cost stays unchanged. A group-stage s
and rank r maps to actual stage `32*(s//16)+16*r+s%16`. Its inverse is unique.
Per-job unit counts are even, so a pair cannot cross an expert/job boundary.
The unit-start owner is computed in the paired domain and mapped to `2*c+r`.
Fixup slots remain within the existing160-CTA capacity. Whole-grid completion,
gather partitions and generation counters still count the original G CTAs.

`results/moe-cluster-x-feasibility-20261029/` contains1,658 CPU models. They check
stage coverage, inverse owners and total work across the saved draws, all
single-job counts, nearby token counts, empty, one-hot and uniform routing.
Median maximum estimated CTA cost is unchanged. Worst skewed draw increases
reach4.33%; this risk is not hidden or treated as measured latency.

For normal M32/128/256 draws the old median logical source-copy bytes across
both phases are51/108/172.5MiB. The model halves source requests and source
bytes; destination shared-memory bytes stay unchanged. These are logical TMA
requests, not measured L2/HBM traffic or an exposed critical-path fraction.
Original requests may hit cache. Added cluster synchronization and scheduling
must earn their cost in complete-kernel paired measurements.

## Required ring and lifetime protocol

Use the existing three X slots. Both CTAs must initialize their local mbarriers
before any remote operation. Their existing full-CTA WS_JOIN already occurs
after routing, range construction and gather. Add a full cluster rendezvous
immediately after that join in each of the three thread paths: weight producer,
X producer, and consumers. Do not put a full-cluster barrier in just one branch.
No need to move initialization before the producer/consumer split.
Place the startup rendezvous before producer helpers' lane!=0/rank1 returns.
The512/32/32 roles are warp-aligned. PTX's aligned cluster barrier requires
warp-uniform execution, not identical PCs across every warp; explicitly
reconverge producer warps after their lane0 helpers before the exit rendezvous.

Both producer warps still participate in that join. Only rank0's X producer
then issues copies. It arms both local and peer full barriers for exactly
`npad*512` bytes and issues one two-recipient bulk multicast. Both receivers
wait for their own completion barrier before WGMMA reads. The X producer's
job/stage order must match both consumers, including phase-A trailing-unit
reordering and phase-B readiness waits.

Every consumer warp in both CTAs releases the leader's empty barrier after
its WGMMA reads complete. That barrier expects32 arrivals, not16. The leader
cannot reuse a slot until both CTAs release it. Waiting only for the leader's
local consumers would permit overwriting the peer's operand. The peer's private
empty counters must not be mistaken for completion of the shared protocol.

Keep all threads alive through a final full-cluster rendezvous. Producer paths
must rendezvous before returning, and consumers must rendezvous before their
final return. Prove all remote shared-memory accesses complete before either
CTA can exit. Audit every early-return path, empty work, final slot and parity
wrap; the original whole-grid barrier does not replace that lifetime audit.

Drain the leader's last occupied X slots before it reaches the final rendezvous.
For N copies issued, slot x has last use `x+3*floor((N-1-x)/3)` when N>x;
wait for its combined empty phase `floor((N-1-x)/3)&1` to complete. This proves
both receivers completed their reads and their remote acknowledgements landed.
Handle N=0 without division. A cluster barrier by itself does not complete
asynchronous operations. The final barrier is additional shared-memory lifetime
protection after the explicit drain, not a replacement for it.

Release/acquire scopes and async-proxy ordering need an explicit justification.
The initial primitive uses conservative cluster-scope release/acquire, and its
assembly contains additional GPU fences. Do not assume their cost is free.
NVIDIA's pinned CUDA13.0 CCCL headers also expose relaxed cluster operations;
their existence is not permission to remove required ordering. Choose and prove
one final protocol before a GPU trial, rather than sweep scope flags afterwards.

An isolated sm_90a probe compiles a4096-byte multicast, remote expect_tx,
remote empty arrivals, and entry/exit cluster barriers. Source, compiler log,
complete SASS and hashes are in the packet. It has never been loaded/executed.
It proves instruction encoding support only, not three-slot correctness,
performance, occupancy or CUDA-graph support of the complete design.
Its illustrative empty count8 assumes128-thread CTAs with four consumer warps;
the full native design has16 consumer warps per CTA and needs the count32 above.

## Integration and numerical obligations

Only eligible MAIN uses this source. Preserve TINY/TINY2/MID, non-PRMT fallback,
original scale eligibility and every BF16/FP32 boundary. No persistent expanded
expert weights or new packed layout. Keep original scratch dimensions and
offsets; barrier counts need no larger arrays. A prepare-only comparison may
use one separate157MiB scratch bank, as before. Preserve sequential-state use.
Changed partitions can change FP32 partial-reduction order; no bitwise output
equivalence is claimed. Require both the tighter incumbent-comparison probes
and the official mathematical-reference checks.

Update weight-producer actual-unit addressing, X-producer common addressing,
consumer unit indices, both cost-prefix shifts, range endpoints and inverse
fixup owners together. Add explicit empty-job guards before division/table
reads in the proposed source. Never change only a unit constant or producer.
Keep all K contributions, arrivals, deferred fixups and source BF16 stores.
Recheck phase-A doneAs: the paired adjacent units still supply the same eight
arrivals for each phase-B activation slice.

Concretely, prefix/range shifts change9/8 to8/7, range grid/index become G/2
and blockIdx.x/2, and each producer/consumer actual unit becomes
`2*(group_stage>>4)+(blockIdx.x&1)`. X uses the common group/job, while weight
addresses use that actual unit. In the inverse owner, first convert an actual
unit-start stage to `(actual_stage>>5)<<4`, then return twice the cluster owner
plus rank. The physical scratch slot and global completion grid stay in CTA
coordinates. Stage offsets within each16-stage unit do not change.

For progress, audit the dependency graph as well as the ring. Original phase-A
fixup completion has no designated-finisher spin (DF bit0 is clear). Phase B
keeps increasing paired-unit order: its designated stage0 finisher can depend
only on contributors in later cluster ranges. Both ranks share that order.
Check this argument against the exact deferred-flush placement and producer
backpressure; the CPU bijection alone is not a deadlock proof.

Full-grid progress requires G/2 clusters resident at once. A block occupancy
query alone is insufficient. Require even G, actual cluster-launch support,
required cluster dimension2 and active-cluster capacity >=G/2. Verify a legal
cooperative cluster launch and CUDA-graph capture before relying on global
spin synchronization. Unsupported combinations must fall back; no unresident
CTA or cluster may be needed to release an operand or completion counter.

The pinned CUDA13.0 headers declare both Cooperative and ClusterDimension
launch attributes valid for graph nodes/launches. That is API evidence, not
proof of the combined launch on this evaluator. Prefer an explicitly cooperative
cluster launch for the whole-grid spin contract, with actual capacity and error
checks. Do not silently use a plain launch after a cooperative rejection.

Query both BIG variants. Require zero actual local/static shared bytes,
unchanged231168dynamic bytes and576threads, suitable registers, and the full
cluster residency guarantee. CPU NVRTC is a filter, not this runtime proof.
Validate namespace/export/query binding before upload; do not repeat the old
paired-store resource-query wiring defect.

Keep the official evaluator identity, full suite and actual-path markers.
Use fresh changing-input/route graph replays, nearby M64/65/127/129/255/257,
chunked384/385, skewed and all-nonlocal cases, and PRMT-ineligible fallback.
Freeze exact source, scope, resource protocol and material-gain gates before
any new check. Controls include the unchanged M1/8/16/32 loads; only M128/256
should show the new namespace. No score or Decode transfer unless complete
source-bound correctness and material affected/control improvement pass.

The old phase diagnostic remains inadmissible for timing fractions. This
proposal rests on source-proven duplicate requests, not those phase spans.
Even a standalone win needs its own full Decode integration evidence. D9's
native MoE equals this baseline, so a precise MAIN-only transfer is possible
if the result warrants it; packaging draft307dd791 provides preparation room.

Next work is the event/interleaving model and complete device-source patch,
then both BIG NVRTC/SASS preflights and host/query audits. No GPU job is queued.
The existing pacing window/reserves still apply; quota is not a spending target.

Primary mechanism sources: [NVIDIA PTX mbarrier/TMA semantics](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html),
the pinned CUDA13.0 CCCL generated mbarrier headers, and
[NVIDIA CUTLASS barrier implementation](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h).
These support mechanisms, not profitability for these shapes.

CPU reproduction: `python experiments/prepared/screen_moe_cluster_x_schedule.py`.
With the previously pinned workspace compiler available, the isolated primitive
can be recompiled with `python experiments/prepared/local_nvrtc_compile.py --source results/moe-cluster-x-feasibility-20261029/cluster-x-primitives.cu --minimal-std-tags --name "&cluster_x_probe" --out .tools/cluster-x-probe-reproduction`.
This command compiles on the CPU and launches no kernel. The full native-source
patch, graph host wrapper and ring proof remain separate unfinished work.
