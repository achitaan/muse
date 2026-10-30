# O16/O32 PDL draft: CPU implementation, no GPU launch

The scored Decode parent remains `c15df9d3` (2.075466170x, job
`452fbd9df6af4ff4`). The draft preserves its entire 940,518-byte prefix.
All newly selected paths default off. This is incomplete Step 3 work; it is
not an H200-validated improvement or the next admitted evaluation.

## Exact source and dispatch

O16 is the `mt_M16TRACE_SOURCES['o']` / `next_early` instance
`(1,2,1,1,0,0,256,0,8192,1024,2,4,4,1,7,1)`. Its weight-only first stage
and escape word precede a non-inlined checksum-dependent wait helper. Every
activation read uses the coherent L2 path after that wait. The private twin
supports the C12 plan; the parent's BF16 fallback remains original.

The selected O32 source is **b2o / r3_o32_o_direct_a**, under
`state['achita_dense']` -> `bundle2` -> `o`, configuration
`(32,128,8,1,1,1,1)`. Its direct BF16 register-store epilogue and 20-ms barrier
watchdog remain intact. The initial source inspection also extracted the old
generic `OLAYER_CUDA`; `olayer32-parent.cu` is that inactive historical template,
not the baseline in the final binding. The authoritative O32 parent is
`o32-parent.cu`, generated from `b2o_olayer_source`. No GPU job used the generic
draft. Final sources and all compiler records were rebuilt and rebound after
this dispatch correction. Private helper/export names are normalized only
inside the new extension, with an exact inverse proof.

O32 issues all eight independent weight bulk copies before a non-inlined
dependency wait in thread zero, then issues the eight activation TMA copies.
Other threads meet thread zero at the original CTA barrier. The complete
transaction barrier still requires both copies before WGMMA reads the slot.
Each slot expects 24,576 bytes: 16,384 weights and 8,192 activations. ST=IT=8,
so the complete K domain is in the initial ring; no slot is overwritten by a
refill during that initial pass. Dynamic shared memory stays 196,736 bytes.
No new traffic, scratch or weight representation is introduced.

Both kernels signal outgoing PDL at entry and attach incoming serialization
on real launches only. There is no dependency on concurrent execution for
progress. The O16 query configuration retains its original occupancy query;
O32 has a private occupancy export. Launch errors and nonzero local memory
must reject a plan rather than silently select it.

## Verification obtained

Four individual NVRTC SM90 screens and one combined-module screen passed.
O16 has 72 registers and O32 42, unchanged from their actual parents; all have
zero stack and spills. Machine-code auditing identifies the wait helper call
by its acquire target. O16 has exactly four independent LDGs before it and
coherent activation LDGs afterwards. O32 has eight UBLKCP weight copies before
it and no activation UTMALDG before it. Matrix/FP-add/BF16 conversion instruction
counts match each parent. The full source derivations invert exactly.

The transaction model proves disjoint, complete byte coverage of the full-K
ring. CPU construction of B16/B32,L1 metadata verified distinct in-range cache
locations, mixed positive context lengths and matching page-table locations.
Those metadata checks do not execute attention. Device proxies use the driver
CUtensorMap ABI (128 bytes, alignment 64). They exclude host launch code;
host extension build, H200 resource queries, preparation time and numerics are
not verified. CPU SASS is not proof of GPU compiler scheduling or a speedup.

Reproduction:

```text
python experiments/decode_moe/build_pdl16.py
python experiments/decode_moe/check_pdl16_cpu.py
python experiments/decode_moe/analyze_pdl16_opportunity.py
```

The wrapper uses the shared cold-L2 graph harness and the existing strict
bytewise checks (including entire cache bytes). It first compares projection
arithmetic, then the whole layer on three changing inputs and both arm orders,
requiring >3% whole-layer gain. Every other regime calls the exact scored
parent. New builds and 16/32 norm warmups run in background during prepare;
all compiler threads finish before graph capture. New probes stop at 278 s.
The 300-s preparation constraint is still unverified.

## Why no check was spent

The cached full-context layer times are 213.838 us at B16 and 312.057 us at B32.
A 3% gate requires 6.415 / 9.362 us respectively. The entire profiled O kernel
was only about 5.8 us in each case. Profile durations are not formal
critical-path bounds, and launch/overlap can affect the result, but this is
strong evidence against spending the final check on the O-only package.
Even a hypothetical 3-us improvement in all four affected layers would move
the six-load point only to about 2.0875x at fixed references. That is a
sensitivity calculation, not a forecast or measured gain.

The remaining 16/32 pooler and unpooler handoffs are not enabled. Inspection
found that the existing fused pooler already supports issuing P weight stages
before its producer wait and has a second consumer wait. `_fpr_PDL` is still
zero; simply flipping it without execution/order, arithmetic and performance
evidence is not this draft's change. `fpr-parent.cu` is inspection evidence only.
The failed TR8 MoE routing candidate is not transferred.

Primary mechanism references: NVIDIA
[PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
and [PTX griddepcontrol](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol).
These establish synchronization requirements; concurrency and profitability
remain opportunistic and require measurement.
