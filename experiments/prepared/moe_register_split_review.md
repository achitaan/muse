# Next investigation: complete producer warpgroup and register redistribution

Status: implemented and CPU-audited; GPU unmeasured. Retain33114871. The scale-six,
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

Before any new kernel launch: enumerate every thread's role, warpgroup and barrier
participation; model arbitrary legal arrival order and register-pool admission;
check WS_JOIN count640, routing barrier7count544, all consumer barriers512,
all sixteen consumer warp arrivals, and no second setmaxnreg after a partial
group exits. Audit the entire Params/control/math diff and reverse it to the
baseline. Check padding threads never touch scratch/output/counters and that
grid-wide completion still has one consumer contribution per block. Prove
all132blocks remain resident with the unchanged near-maximum shared memory.
The ordinary prepare-time resource query must confirm residency before launch.
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

## Implemented preflight and CPU evidence

Frozen draft17adb3e692d36fbb92af4a38b1570bfefd028696426a29ec9293517e581f6824,
180781bytes. Builder and CPU audit are build_moe_regsplit.py and
audit_moe_regsplit_cpu.py. The exact ledger reverses two CUDA edits (NTL and
the WGS-only role block), the appended host resource query/export and the
prepare-only admission code to recover the complete baseline module AST.
All numerical CUDA helper bytes, GS2, jobs, scratch layouts and host dispatch
are unchanged. The tiny/mid/fallback builds retain288/288/288/256threads;
only main WGS changes576to640. No extra compiled variants or buffers.

The independent register-conservation graph explored45reachable states and
68transitions, allowing each producer warp to release and each consumer warp
to acquire in any legal order. Only the final state has no next action; no
blocked consumer can prevent a remaining producer from releasing before its
barriers. Every warpgroup has one uniform action and no later setmaxnreg.
Barrier membership is640(full),544(routing),512(compute). Padding enters
only the full join and returns before any input/output/scratch operations.

Actual cudaFuncGetAttributes and cudaOccupancyMaxActiveBlocksPerMultiprocessor
queries run in prepare after the existing synchronization and before any
kernel launch. They opt in to dynamic shared memory first. Require both
kernel instantiations to report exactly96initial registers and at least one
resident640-thread block per SM, and the whole persistent grid to fit. A
different allocation fails preparation explicitly rather than risking a
register-pool wait. CPU mocks exercised admitted/non-PRMT and seven rejection
cases (allocation, residency, register-file size, grid size, architecture).
The resource/enable diagnostics are also required by the full-evidence guard.

Use cached baseline2.365437416x and one full official check. A passing result
must include all5tests/6benchmarks/integrity and an enabled split; only then
compare the timing. Do not score a flat result or transfer unmeasured code.


## Phase10: full producer warpgroup, no retained speedup (19:08 UTC)

Checkfc4ff22b0b184caf/source17adb3e6 passed5tests/6benchmarks/integrity at
2.340478636x, CI[2.305371456,2.375585815]. Resource preflight was
(640,96,96,0,0,231168,1,1,65536,132): both initial allocations96, no static
local bytes, one resident block/SM. Split enabledTrue. No changed WGS load
improved over primary33114871. Reverted, archived, no score. All math helpers
and GS2 were unchanged. Static attributes do not directly measure dynamic
ownership or establish a compiler-stall cause. Full evidence and audits:
results/resumed-moe-phase10-20261028/. No official job pending at this boundary.

An inactive Decode transfer draft21bf949f is991255bytes with all non-MoE AST
unchanged, but it was never evaluated or promoted because standalone timing
did not improve. Retain09badc9b in Decode and33114871 in MoE.

One source-backed follow-up: four-step WGMMA groups on this admitted register
split. Original source specifically says GS4 serialized16..32-token variants
at96registers; GS4 needs16more live A-fragment registers, matching the reviewed
96to112compute budget change. Test that paired mechanism once, retaining all
geometry, original layouts and GSFCI3; capture ordinary compiler diagnostics.
Do not retry tile/encoding/grid parameters or score a flat result. Expected
MoE quota4checks/7scores after four checks since18:24 (not a fresh API query).

## Specific follow-up: GS4 on the admitted register split

build_moe_regsplit_gs4.py reads the immutable phase10 source, not a mutable
working candidate. Draft844c01650709fc470493f186ce1e303ea1fbef45301602bcf908713360e9bfba
is180849bytes. Its sole CUDA algorithm change from17adb3e6 is
OKC_WGSGS2to4. It leaves GSFCI3, geometry, all role/barrier/register budgets,
preflight and every layout/math helper intact. The module AST recovers exactly
after reversing this macro and the ordinary build-diagnostic flags.

The CPU schedule oracle enumerates the same16K16steps in the same order in
both settings. GS2commits8groups, GS4commits4groups; deferred flush3 exists
exactly once in either first stage. Ping-pong fragment banks are reused only
after the earlier group using that bank retires at wait_group1. The16additional
A-fragment registers match the96to112compute budget change. Final wait_group0
and old input-slot releases remain unchanged. This is static/control evidence,
not GPU timing or a proof of changed compiler scheduling.

The main build alone enables load_inline verbose output and ptxas-v, preserving
optimization flags. Inspect ordinary compiler diagnostics for ignored
setmaxnreg or WGMMA serialization; static resource attributes cannot establish
those details. NVIDIA notes the need for a complete launch-bound specification;
this kernel already uses launch_bounds(NTL,1), unlike the single-argument case
discussed at https://forums.developer.nvidia.com/t/setmaxnreg-ignored-unable-to-determine-register-count-at-entry/361145 .
Do not introduce system calls or further device functions for diagnostics.

Compare against retained33114871 and the diagnostic GS2trial17adb3e6, using one
full check. No scored submission or Decode transfer for an unchanged/noisy point.
No subsequent group-size or allocation sweep if this paired mechanism loses.
