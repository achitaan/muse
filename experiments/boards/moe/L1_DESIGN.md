# L1 / STEP1: MAIN-only asymmetric registers and early stage capture

Parent: moe_b13 (HINT+ERL+FINREG). No N96 jobs or other STEP2 work.
`OKC_L1` defaults1, gated by `OKC_WGS`;0 preserves the parent MAIN path.
Other builds retain their launch and compute paths.

Accepted plan:640 threads, four consumer WGs and one full producer WG. Initial
96 registers/thread accounts for61,440 registers. Producer128 threads execute
warpgroup-uniform setmaxnreg.dec32 before specializing two producer warps;
consumer512 threads execute warpgroup-uniform setmaxnreg.inc112. Final accounting
512*112+128*32=61,440, leaving4096 physical registers unused.

Threads512..543 retain weight production,544..575 retain x production and the
NT+32 routing/range barrier7. Added threads576..639 do not join barrier7, but do
remain alive through WS_JOIN (barrier2, now640 arrivals) before returning.
The existing WGS producers have only this initial routing/gather WS_JOIN; the
consumer-only CTA_SYNC stays512. Empty weight/x mbarriers still receive16
arrivals, one per consumer warp. Neither padding warp issues copies or credits.

Each consumer captures all16 code words and8 scale words of a packed stage
before its x wait. An all-word checksum drives the warp-sync mask through the
parent's opaque, host-initialized p.zero=0, forcing a real dependency on every
load without changing the full mask. After the warp rendezvous, lane0 returns
the weight-slot credit. Then wait on x and run the original lg/h/jj decode,
fence/MMA/commit/wait sequence verbatim. Same units, K order, split-K shares,
epilogues, buffer layouts, hints, generation barrier and finalizer.

Required evidence: ptxas role/register behavior, stage-loop local loads/stores,
SASS load->weight release->x wait order, WGMMA instruction sequence, non-MAIN
SASS identity, CPU packed-stage mapping and participant counts. Actual output
bit equality and performance require the judge; no local GPU execution.

Risks: dynamic-register allocation may alter compiler outlining/scheduling;
producer32-register budget may spill; whole-stage capture may still exceed112
in wide-N stage paths. Dependency checksum adds integer instructions. A weight
slot can be reused earlier, but gains depend on whether x waits or producer
refill latency are actually exposed. STEP2 remains deferred until measurement.

## Final delivered files

- `moe_l1.py`:190,023 bytes, SHA256
  `5608124583138c16d6f3739023c0ccd71a705118016ef68e0c51f3ddf9d69457`.
- `../diag/moediag_l1.py`:175,826 bytes, SHA256
  `faef2acf6855547df5ea5210589e0e65b8eb172777edb148afa2905140c0c156`.
- Parent SHA256 `b6390e9993caebd178645d2f33e780724fd84242d15b4a17639d98bc17a3c7ce`.

The candidate preserves the parent Python prepare/selector/packing and public
host interface. Only CUDA is modified, plus the extension cache-name prefix to
avoid sharing a module name with the parent. A forward declaration fixes an
existing strict-NVRTC declaration-order issue; macro-off SASS proves it does
not change generated arithmetic.

## Offline compilation / SASS evidence

CUDA13 NVRTC/ptxas sm_90a, C++20; CUDA cache disabled for complete spill logs.
All13 checked configurations passed: BASE/L1/OFF MAIN; BASE/L1 for each of
TINY,TINY2,MID,MAIN_FALLBACK; and both diagnostic TUs. Both BIG instantiations
were compiled for every configuration. This is device-only, not a Torch host
extension build, and does not execute GPU code.

| MAIN build | Launch regs | Consumer/producer limits | Stack | Spill stores/loads |
|---|---:|---:|---:|---:|
| BASE / L1 disabled |96|96 /96|16B|56 /124B|
| L1 enabled |96|112 /32|0B|0 /0B|

The zero-spill result holds for the entire MAIN kernel, not only the steady
stage loop, and is reproduced by the diagnostic TU. SASS contains no LDL/STL
instructions. A conservative branch/call traversal from the role-allocation
sites finds producer operands at mostR19 and consumer operands at mostR109
(including relevant vector operand extents), within32/112. The attribute96
reported by ptxas/cudaFuncAttributes is the **launch allocation**, not the
post-setmaxnreg consumer limit.

Both BIG instantiations contain:
- `USETMAXREG.TRY_ALLOC.CTAPOOL ... 0x70` at0x70:112 consumer registers;
- `USETMAXREG.DEALLOC.CTAPOOL 0x20` at0x5d7c0:32 producer registers;
- barrier2 count0x280=640, including padding paths;
- barrier7 count0x220=544, x producer+consumers only;
- consumer barrier1 count0x200=512.

All32 static stage-capture sites per instantiation (A/B x NTT1..8 x first/
continuing stage) were checked: four LDS.128 code loads and four LDS.64 scale
loads, then live checksum/mask dependency, WARPSYNC, empty-slot arrival, then
x-full wait, with no intervening weight load or WGMMA. Example in both kernels:
loads0x6900..0x6a30; WARPSYNC0x6b20; weight release0x6b70; x wait0x6b80.
All1,024 static HGMMA instructions across both instantiations retain BASE's
shape/count inventory; source tests verify the entire decode/MMA/fence/commit/
wait/epilogue tail is verbatim. OFF MAIN and **all non-MAIN variants have
instruction-for-instruction identical SASS** to their parent counterparts.

Smem fields and allocation function are unchanged: dynamic231,168B per the
parent layout. Both BASE and L1 offline cubins also report1,024B static shared,
so total232,192B remains below Hopper's232,448B block opt-in limit. Ring sizes,
packed tensors, scratch and arithmetic interfaces do not grow.

## Diagnostic and CPU validation

The diagnostic follows moediag7's `CUDA_B85/CUDA_SHA/SOURCE_SHA` packaging,
using exact moe_b13 BASE and L1 as the replacement comparison slot. HINT is
omitted (already in both kernels). Two thin-host CUDA TUs, one binding; no
explicit C++17 override. Pinned MAIN only, bypassing the parent's prepare-time
selector while retaining its WGS packing. Cases:
R32, R128, R128_NODUP, R256, R256_DUP3, R256_NODUP.

40 cold-L2 graph replays per measurement,256-MiB flush outside events,
forward BASE/L1 then reverse case order and L1/BASE. Reports24 timing/error
lines, two resource lines (including576/640 threads), and six `L1_BASE` ratio
summaries. Compares output bytes and max absolute/relative and relative L2
against saved BASE. Zero routing stays explicitly disabled.235-second
cooperative budget and bounded error diagnostics retained.

CPU checks passed for exact parent payloads/source hashes and binding linkage;
full packed-stage byte addresses (codes once, scales four-way replicated as in
parent); per-step code/scale mapping; complete WG register redistribution and
join participants; seven adversarial early-release ring models;40-sample event
protocol and all six cases/24 mocked measurements in both orders. Mock log
4,396B +1,000B reserve fits9,400B. Mock values are not accuracy/speed evidence.

Reproduce under `.l1_offline/`:
```
python -B .l1_offline/make_l1.py
python -B .l1_offline/make_diag.py
python -B .l1_offline/compile_l1.py
python -B .l1_offline/test_l1.py --sass
python -B .l1_offline/sass_l1.py
python -B .l1_offline/test_diag.py --compile
```
Resource logs, SASS, source snapshots and JSON manifests are retained there.

## Remaining runtime risks

Spills are eliminated offline, but speedup and output bit equality are not
measured. Extra checksum integer instructions, increased instruction footprint,
warpgroup register-pool handoff, or earlier refill contention can outweigh the
removed spill/x-wait exposure. The SASS-derived path/resource checks do not
replace runtime residency, synchronization, PDL or numerical validation. Torch
host-build duration and the diagnostic's235-second budget fit remain unknown.
No STEP2/native N96 jobs or local GPU execution was attempted.
