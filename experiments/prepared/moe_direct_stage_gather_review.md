# MoE phase-A activation staging: direct gather hypothesis

Parent `331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`,
native CUDA `1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c`.
Current decision: park without a GPU check. Three native drafts compile, but
all show static48-byte stack/316-byte spill stores/668-byte spill loads per
BIG entry versus16/56/124 for a matched control. All use96base registers and
have no C751 warnings. These offline fields are not native H200 measurements.
No Python submission, candidate installation, GPU experiment or speedup claim.

The second draft hoists a routed-row lookup across eight copies. The third
moves completion/proxy publication to the producer and restores the exact
consumer loop, at the cost of a producer wait. Neither removes the spill
increase. A control with the same namespace and empty-job/range guards keeps
the incumbent spill counts, so those shared changes do not explain the increase.
All revisions and exact inverse ledgers, matched control, compiler inputs/logs
and disassembly are preserved under
`results/moe-direct-stage-gather-preflight-20261029/`.
The original hypothesis/model discussion below remains historical preparation.

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

The independent strip-mined layout audit checks another1,179,648vector
destinations. Source inverses restore the complete retained CUDA exactly.
No measured recoverable complete-load time offsets the higher static spill
burden and extra producer work. Do not spend a check on this design or search
geometry/register/cache flags to force admission. Preserve it as rejected
preparation; any materially different proposal needs its own opportunity screen.
