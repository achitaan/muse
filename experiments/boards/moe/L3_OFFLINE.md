# L3 feasibility gate: paused, not a submission

No `moe_l3.py` or GPU timing harness is promoted from this attempt. The requested
load-lifetime change reaches **zero spills at96 registers**, but ptxas achieves
that result by serializing **every WGMMA**, including the narrow paths. Thus
it would be incorrect to claim either that zero spills is mathematically
impossible, or that the intended asynchronous/narrow-fast-path target passed.

## Exact source change tested

The tested source starts from GPU-tested L2 on moe_b13, without L1. One detail
in the premise needed correction: final L2 already has `OKC_L2_GS=1` for NTT>8.
It also drains previous-segment deferred work before wide accumulators become
live. Those settings were retained, not added a second time.

For NTT9/10, the prototype loads only current `cwg/scg` at the top of each lg.
There are no next-lg cwn/scn variables in this compile-time branch. The last lg
returns the weight slot after a real all-load-dependent warp rendezvous. This
is later than L2's lookahead-based release; arithmetic/K order is unchanged.
The complete current L2 NTT<=8 load/decode/MMA source is kept verbatim in the
other compile-time branch. Runtime3x32KiB versus2x48KiB x selection is retained.

Source snapshot: `.l3_offline/runtime.cu`, SHA256
`8251d0793e32dc6af141f15d99f13f0d3a457e9467de09e7f2f35a170ac543b5`.
The launcher and host ABI are unchanged. Existing candidates/harnesses are not
edited by this attempt. Source generator: `.l3_offline/source.py`.

## NVRTC/ptxas evidence

CUDA13, sm_90a, C++20, cache disabled; both BIG instantiations agree.

| Build | Used registers | Stack B | Spill stores/loads B | C7512 |
|---|---:|---:|---:|---|
| Exact moe_b13 BASE |96|16|56 /124|no|
| L3 runtime x layout |96|0|0 /0|**yes**|
| L3 fixed2-slot wide-containing specialization |96|16|104 /184|**yes**|
|112-register negative control |112|0|0 /0|**yes**|
| Uncapped-to255 negative control |217|0|0 /0|no|

The controls above96 registers are not launchable at576 threads: Hopper's
hardware admission rounds18 warps to20, and per-warp registers round to256.
112 registers needs71,680 registers/block;217 rounds to224 and needs143,360,
both above65,536.217 is an observed allocation, NOT a proved minimum required
for asynchronous execution. Neither control is a candidate.

SASS confirms why the zero-spill result is not sufficient. Counts below cover
both BIG instantiations:

| Build | HGMMA instructions | DEPBar.LE1 (overlap retained) | DEPBar.LE0 (full drain) |
|---|---:|---:|---:|
| BASE |1024|512|32|
| L3 runtime96 |1280|**0**|**1280**|
| L3 fixed2-slot96 |1280|**0**|**1280**|
|112-register control |1280|**0**|**1280**|
|217-register control |1280|768|40|

This is not just a ptxas warning with no code effect: the96-register runtime
build has a full drain for every MMA instead of the parent's overlapped groups.
It has zero LDL/STL in the whole kernel and zero local instructions in every
identified steady-stage loop. The fixed2-slot build still has six local
instructions in one N72 steady-stage loop. Loop audit matches complete backward
branch intervals containing16 same-N HGMMA instructions and deduplicates compiler
backedges around the same MMA sites; BASE is its calibration/control.

The compiler log names C7512 as insufficient register resources. The tested
bounded changes do NOT establish that every possible96-register implementation
must serialize. They show that the requested no-lookahead/GS1 transformation
does not preserve the intended pipeline in these instantiations. No GPU speed
claim is made from the instruction counts.

## Separate exact-BASE dispatch question

Keeping narrow source text unchanged is not sufficient: ptxas's whole-function
serialization also changes the narrow SASS. A distinct exact-BASE kernel is
straightforward to compile (the BASE control above), but selecting it for every
no-wide **launch** is not a host-only shape decision:

- For valid distinct top-k routing, M<=64 guarantees no native wide expert.
- At M128/M256, R128/R256 and their NODUP cases have identical host-visible
  shapes/configs; the deciding counts live in CUDA `topk_ids`.
- Copying counts with `.item()`/`.cpu()` introduces timed host synchronization
  and is not CUDA-graph-capture-safe. Caching routing by input pointer or using
  the diagnostic's precomputed counts would move input-dependent work outside
  the timed implementation and is not an acceptable substitute.

An architectural decision is needed before implementing that part: e.g. an
explicit device routing predicate plus CUDA conditional-graph dispatch, or
host metadata genuinely supplied by the caller. Either changes the launch
contract/cost and needs separate validation. No such mechanism was guessed or
silently added here. This dispatch question remains open independently of the
register/pipeline problem.

## Measurement and validation

The read-flush correction is accepted. `read_flush_check.py` checks the actual
moediag13 bench function with CPU mocks:40 `torch.amax(flush)` reductions are
retained and ordered before start events; no zero_/add_ write flush. No new GPU
timing harness was produced, and historical write-flush times were not treated
as read-flush baselines or extrapolated into an L3 speed estimate.

CPU checks passed for the narrow source identity, current-group byte mapping,
single early deferred-work drain, unchanged host ABI, and register admission
arithmetic. All five device-only builds compiled. No real Torch host-extension
build, GPU execution, runtime correctness check or implementation commit.

Reproduce the evidence (all work stays under this worktree):
```
python -B .l3_offline/source.py
python -B .l3_offline/check.py
python -B .l3_offline/compile.py BASE RUNTIME96 TWO96 OVER112 OVER255
python -B .l3_offline/loops.py BASE RUNTIME96 TWO96 OVER112 OVER255
python -B .l3_offline/read_flush_check.py
```
Compiler flags/logs, cubins, SASS, loop JSON and input hashes are retained under
`.l3_offline/`. Paused here rather than presenting a serialized prototype as a
completed spill-free optimization or changing the dispatch architecture without
an agreed plan.
