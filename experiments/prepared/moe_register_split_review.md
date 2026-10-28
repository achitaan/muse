# Next investigation: complete producer warpgroup and register redistribution

Status: design only, not implemented or measured. Retain33114871. The scale-six,
WGS32, scheduler-quotient and grid-parameter trials did not produce a retained
gain. Do not combine any of those losing edits into this investigation.

The measured original main WGS kernels use96registers/thread, zero reported
local bytes and231168bytes dynamic shared memory. WGS32 cut shared memory but
remained at95registers and acquired8local bytes, without improving32-token
speedup. Source comments identify the96register budget as limiting the WGMMA
pipeline. That motivates investigating register ownership directly, rather
than another tile-size, encoding or dispatch-parameter sweep. It does not prove
register stalls are the dominant runtime cost.

The current main block has512consumer threads and64producer threads. Hopper
setmaxnreg requires every warp in a warpgroup to execute the same instruction.
Do not apply a dec instruction to only the two producer warps and assume the
partial warpgroup is valid. One possible safe design uses a complete128-thread
producer warpgroup:640total threads, keeping512compute and64useful producers.
The remaining64threads participate in the producer-group register decrease
and the existing full-block WS_JOIN once, then return. They must not enter
the x-producer routing barrier or duplicate weight/x copies.

Candidate accounting to review: a96register initial budget for640threads is
61440registers, within Hopper's65536. Releasing producer-group registers to32
and increasing consumers to112 keeps the same total:
128*32+512*112=61440. Both counts satisfy PTX granularity. Each of the four
consumer warpgroups executes inc112 uniformly; all four producer warps execute
dec32 uniformly before any producer-role branch. If the actual initial budget
is smaller, these requests can deadlock: verify compiler launch/resource
metadata or set a documented compile-time allocation policy before evaluation.
Do not assume a source-level launch_bounds directive guarantees the needed
initial allocation or that a runtime hint necessarily takes effect.

Keep original GS2, all eight token-tile variants, original scratch geometry,
byte scales, DQ8, work ranges, flags, BF16 rounding and FP32 norms. No pipeline
group-size tuning in the same trial. Limit the new allocation/block shape to
the main OKC_WGS build; tiny/mid/non-PRMT variants retain original code and
launch sizes. The ordinary host launcher derives block size fromNTL, so both
normal and PDL launchers must use640for this exact specialization.

Before any GPU check: enumerate every thread's role, warpgroup and barrier
participation; model arbitrary legal arrival order and register-pool admission;
check WS_JOIN count640, routing barrier7count544, all consumer barriers512,
all sixteen consumer warp arrivals, and no second setmaxnreg after a partial
group exits. Audit the entire Params/control/math diff and reverse it to the
baseline. Check padding threads never touch scratch/output/counters and that
grid-wide completion still has one consumer contribution per block. Prove
all132blocks remain resident with the unchanged near-maximum shared memory.
Record actual compiler attributes, but distinguish static attributes from
dynamic register ownership.32register producer code could spill; measure it.

If this cannot be made reviewable without speculative resource assumptions,
do not spend a check merely to try it. Look for a different structural change.
If it can, use one full5tests/6benchmarks/integrity check and compare WGS loads
against unchanged small-token controls. Transfer to Decode only after a clear
measured gain; published sources remain unchanged meanwhile.

Primary references inspected October28:
https://docs.nvidia.com/cuda/parallel-thread-execution/
https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/primitives.html
https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
The checked attention kernel provides a local example of complete producer
warpgroups and register redistribution; it is not proof of this MoE design.
