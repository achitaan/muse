# Dense D1: fixed kernels with nearby performance protection

GPU checks are now authorized sparingly until October29 at20:00Toronto,
2026-10-30T00:00Z. Follow EVALUATION_POLICY.json: one GPU job at a time,
at least60minutes between new checks and four checks per track in reserve.
Scores require convincing correctness and material performance evidence.

Candidate13224d30d3cf78a89bc731b3a6d036f4853c30cfb30529daa581050dd1f66c9b
is687638bytes. Frozen source and CPU evidence are in
results/dense-cluster-nearby-preflight-20261029/.

Its native CUDA and all nonselection AST match the earlier unlaunched169dbce7
packet. QKV remains cluster3 for M33..128; pooler remains cluster2 for M65..128.
Both use original BF16 weights. This adds no new kernel configuration.

The preparation comparison now covers QKV M33/64/65/96/127/128 and pooler
M65/96/127/128, two deterministic patterns each and seven alternating-order
cold-L2 graph pairs per pattern. A family needs both M128 median new/old ratios
below0.97 and every nearby ratio at most1.02. Existing FP64 numerical probes
and preparation budget gates remain intact. Selection retains only static
weight/metadata decisions. Raw paired times are saved in nanoseconds.

The guard reconstructs ratios from those raw samples. Fourteen malformed
evidence mutations are rejected. A conservative compact-diagnostic bound is
13534bytes, below the service's16000-character tail. Fresh official CLI task,
reference, checker, baseline and AGENTS files match the pinned copies exactly.

This candidate still has no CUDA compilation, residency, race-freedom or GPU
performance evidence. The CPU ownership and partition audits are filters only.

One full Dense check is justified as a bounded test of the two unchanged
families: their listed-floor screen allows about2.84% aggregate improvement,
while the existing1.5% retain gate requires about5.62us saved on the M128 load.
Decode source capacity is now available through the separate CPU-only package;
preparation feasibility and useful whole-layer transfer still require evidence.
No overall placement improvement is predicted.

Reuse the exact cached scored EDC baseline. Require full9tests/30benchmarks,
integrity, H200, preparation<=300s, nonempty choices, all numerical/nearby
evidence, and actual selected profile symbols. Retain only if affected and
control-normalized improvement both exceed3%, no affected projection regresses
by over2%, and aggregate speedup exceeds1.527771327x by1.5%. Empty selection
is a rejection even if the aggregate point rises. No baseline rerun or sweep.

Use scripts/measure_dense_cluster.py and verify_dense_cluster_evidence.py.
The experiment controller owns candidates/active.py; verified best files stay
unchanged. A failed candidate is recorded and reverted before further work.
