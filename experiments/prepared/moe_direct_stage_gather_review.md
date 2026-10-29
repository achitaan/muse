# MoE phase-A activation staging: direct gather hypothesis

Parent `331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`,
native CUDA `1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c`.
CPU model only; no native prototype, GPU experiment, or measured speedup yet.

The retained WGS path globally gathers routed inputs into xA before compute;
its producer later bulk-copies each stage to shared memory. A distinct candidate
could let all32 lanes of that existing producer gather phase-A stages directly,
removing the initial global gather and its readiness wait. Keep phase-B xB and
slice-ready overlap unchanged. Retain all original scratch allocations initially.
This proposal adds no producer warpgroup, cluster or separately launched kernel.

The source-bound CPU model checks1024count/stage layouts,1,179,648vector
destinations and256arrival schedules. It includes duplicate input tokens and
padding without a routed-entry load; valid uint4 addresses match the retained
gather exactly. These are model properties, not a memory-order or GPU proof.

For a job with n valid rows and p=8*ceil(n/8), the modeled phase-A activation
traffic changes from8192*(2*n+32*p) to8192*32*n bytes. This excludes weights and
phaseB. These logical requests do not quantify HBM traffic or elapsed time:
xA and original inputs may hit L2. The direct producer adds8..64copy rounds per
stage instead of one bulk instruction, plus route lookups and barrier registration.
That overhead may dominate any benefit, especially for hot64-row jobs.

A possible completion scheme uses default cp.async.mbarrier.arrive per lane,
then a warp synchronization before lane0 releases the barrier's initial count1.
The default form adds a pending arrival before asynchronous completion, per
[NVIDIA's PTX specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive).
This supports the count model; it does not prove proxy visibility or performance.

Next implement one native draft with exact inverse/source binding. Audit stage
generations across both phases, async-proxy visibility, producer convergence,
fresh routes, padding masks, chunk384+1 and all-nonlocal calls. Compare compiler
resources and code with the incumbent before deciding whether one paced GPU
check has a credible benefit. Preserve scale bytes and every numerical boundary.
No geometry, register or cache-policy sweep is authorized by this model result.
