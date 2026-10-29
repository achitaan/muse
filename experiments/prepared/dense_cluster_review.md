# Dense large-token cluster experiment

Prepared source: `169dbce77be377eea3c49b05428d67d3141c0a6f506f8dd23e57ad4cd7e269e3`.
This supersedes unsubmitted `4d443b13` only by correcting event ms-to-ns
conversion; the earlier source and its evidence remain archived. Current
authorization permits local implementation/CPU checks only, with no GPU runs.
D1 remains conditional on preparation, nearby performance and whole-layer value.
GPU status: unmeasured. CPU evidence cannot establish CUDA compilation,
synchronization correctness, or performance. The retained publication is
`6131bc96427e4486`, source `edc61a1e`, at 1.527771327x.

## Why these kernels

The exact published Dense profile spends 13.7 us in QKV and 19.8 us in pooler
at M128. The corresponding reported paired times are 13.416 and 19.754 us.
The five-projection load totals 65.731 us, and each of six loads contributes
equally to the aggregate. These two projections are about half this load.
Improving both toward their listed 8.4/14.6 us floors would improve the whole
problem by roughly 2.8%. That is an optimistic bound using listed floors, not
a performance prediction. One proposed fixed configuration per projection
is appropriate; a broad geometry sweep is not.

The public source supplies BF16 WGMMA shared/shared GEMMs with cluster K
splitting and original weights. This is different from the rejected small-M
compressed SK2 projections. The large M128 pooler currently uses cuBLAS.
No new compression format, router precision change, external library or GPU
provider is involved.

Public implementation, Apache-2.0:
https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_bf16_proj.py

The pinned source is tracked at
`results/dense-pinned-codec-screen-20261028/upstream_dense.py`, SHA256
`c1dd241c0c08b42fddfa7bc9656d0f596b59bb13005c50c0db4418d9f2f1dc7e`.
The builder preserves the native CUDA pieces except dispatch narrowing and
generating only the needed m64n128k16 BF16 helper. It includes attribution
and the complete upstream Apache license in the submission.

## Fixed scope and algebra

QKV: original W[2304,8192], M33..128, GQX BN128/NWG1/stages8/cluster3.
Pooler: original W[4096,8192], M65..128, GQ BN128/NWG1/stages8/cluster2.
The public defaults enable GQ and GQX. Its pooler M33..64 chooses a different
BN64 configuration, so this experiment leaves those rows on the incumbent.
All other layers, token counts, unsupported layouts and device cases use the
original dispatch. Each timed call computes from its fresh activation pointer.

For 128 K64 tiles, rank r owns [floor(r*128/CL),floor((r+1)*128/CL)).
Cluster2 divides K into 4096/4096; cluster3 uses 2688/2752/2752.
All K values occur exactly once. Partial sums accumulate in FP32 and are
reduced in FP32 before the final BF16 store. Each output has a single owner:
GQ splits the 64 weight rows into halves; GQX partitions sixteen 8-token blocks
into 5/5/6 blocks. The CPU enumeration checks every M1..128, not just M128.

GQ uses 215184 bytes dynamic shared memory; GQX uses 230544, including alignment
slack, TMA rings, barriers and peer fragments. Both fit 232448 bytes, with the
larger configuration close to the limit. Launches have 160 threads and
128/108 CTAs respectively. Actual resources remain to be checked on H200.
The CUDA cluster launch contract and shared-memory limits were checked against
NVIDIA's guide; actual occupancy and launch acceptance remain GPU questions:
https://docs.nvidia.com/cuda/archive/12.8.1/pdf/CUDA_C_Programming_Guide.pdf

TMA loads min(M,128) activation rows and expects exactly that byte count.
Unused columns in the BN128 tile only feed dropped outputs; valid output
columns do not mix with them. GQ waits for its peer barrier before summing.
GQX writes fragments after WGMMA completion, uses a release/acquire cluster
barrier before peer reads, and keeps every CTA alive through another barrier.
Staging reuses the retired TMA ring and stays disjoint from live fragments.
These are source arguments; the CPU audit does not prove CUDA race freedom.

## Preparation and correctness

The full EDC source is an exact prefix. Its existing base aliases bypass only
the final WG tuning wrapper. The exact published log has three WG_KEEP lines
and no WG_SELECT, so those timed kernels were unchanged by that wrapper in
the baseline run. This does not assert identical selection on every possible
unseen weight distribution. The preceding v18 preparation remains intact.
Do not substitute `best/lean.py`: its final unpooler wrapper also differs.
Baseline preparation took 90.439 seconds, against a 300-second limit.

Preparation compiles just the two fixed new instantiations. It compares with
FP64 on deterministic synthetic BF16 activations for QKV M33,64,65,127,128
and pooler M65,96,127,128, two patterns each. Patterns include one-hot power-of-two
rows, scaled outlier columns, and a near-zero row. Global L2 must be <=0.004,
impulse L2 <=0.001, worst normalized row <=0.02 and worst impulse <=0.005,
with finite outputs. The row denominator includes the task's 0.1-average floor.

Only a numerically passing family gets two synthetic M128 comparisons against
the incumbent, seven paired cold-L2 graph samples for each pattern, alternating
arm order. Both median new/old ratios must be below 0.97 to enable that family.
Selection caches only static weights and metadata, never timed inputs/outputs.
The compact evidence marker retains all oracle cases and paired raw timings.
Budget gates at 210/255/270 seconds leave headroom for return; a compilation
or launch failure remains visible rather than being hidden as a speed result.

CPU verification: 5134 metadata dispatch cases, 256 exact ownership grids,
both complete K partitions and memory bounds, 20 FP32 split/reduction versus
FP64 numerical cases, global name binding and pinned native source checks.
Those numerical models test the algorithm, not the actual GPU instructions.

## Measurement and keep policy fixed before any GPU launch

Use `scripts/measure_dense_cluster.py` and
`scripts/verify_dense_cluster_evidence.py`. Scope is `candidates/active.py`.
The exact EDC scored baseline is reused locally, without API calls or reruns.
New evaluation needs positive Dense check quota from `okc problems --json`.
There was no Dense quota left in the last official snapshot; reset is 00UTC.
Launch once, persist the handle, and resume it after observation timeouts.
Do not re-upload after an unknown launch outcome without reconciling jobs.

Require all nine named tests, all thirty named benchmark cases, integrity,
H200 and preparation <=300 seconds. Require nonempty actual new choices,
complete numerical and paired evidence, and matching GQ/GQX profile symbols
on the selected M128 projections. Empty choices mean discard even if unrelated
benchmark noise raises the aggregate.

The affected projections must improve at least 3% geometrically, also at least
3% after dividing by the geometric change in unchanged controls. No selected
projection may regress by more than 2%. Whole-problem speedup must improve
at least 1.5% over 1.527771327x. These are practical retain thresholds, not a
statistical significance test. Report overlapping intervals honestly.
Only after this guard passes may one scored confirmation be used. An interval
overlap alone is not permission for repeated scoring. No nearby geometry
sweeps, router changes, speculative Decode transfer or fallback-only rescores.
The broader top50 goal remains active whatever this one trial's outcome.
