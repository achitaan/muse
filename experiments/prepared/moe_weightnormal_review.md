# WGS weight-load priority

One fixed memory-reuse hypothesis from retained33114871. Do not inherit the
rejected register/GS4, tile-size, six-bit or quotient changes. Main remains
576threads, GS2, original four compute warpgroups and all old source layouts.

A fat expert splits into64-token jobs. Its WGS weight source address depends
on the expert, unit and K-stage, with no token-job offset. Every additional
job on that expert therefore reads the same weight addresses again within
the same launch. The original global comment says weights stream once and
marks these copies evict_first. Keep evict_first for the other paths and
evict_last for intermediate stores; change only WGS weight loads to normal.

NVIDIA CUTLASS's enum confirms evict_first=0x12F0000000000000 and
evict_normal=0x1000000000000000:
https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_desc.hpp
These are performance hints, not memory-order or numerical changes:
https://docs.nvidia.com/cuda/parallel-thread-execution/
No CUDA code from CUTLASS is copied; existing Apache provenance remains.

Draft273c1f1a6d2aae2fd602cb20e5a7de965ac12686c86fdca614d8031c17b4c6a0,
178345bytes. build_moe_weightnormal.py defines a WGS-only policy operand and
changes the one corresponding assembly input. All four copy operands, size,
barrier, instructions, source addresses, scratch and math stay identical.
Its inverse ledger recovers the complete baseline module AST after reversing
this policy, prepare-only enabled diagnostic and main-only verbose/ptxas-v
build diagnostics. No additional module, buffer, synchronization or selector.
The original global evict_first and evict_last constants are unchanged.

Address audits cover counts0..3072, including duplicate physical routes.
Each expert's768phase/unit/stage addresses repeat for every split job; no
route is omitted. A separate public-formula model uses64independent draws
per shape and reports logical reuse opportunities, not private inputs or
cache hits. M256repeated expert weights in50/64draws, mean jobs/active ratio
1.07969; M12817/64, ratio1.01287. No repeats below64tokens in that model.
Nearby shape tests are required because normal priority can evict useful
activation data; this is a possible cost, not a guaranteed cache improvement.

One full official check against cached2.365437416x, all5tests/6benchmarks,
integrity, precision and explicit WGS-enable evidence. Inspect large loads
and unchanged small controls; record compiler diagnostics for original GS2.
Do not score a flat result, sweep cache-policy values or introduce persistent
input/output caches. A material gain alone justifies a whole-Decode transfer.
