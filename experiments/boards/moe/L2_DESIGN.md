# L2: native wider jobs, N80 fallback, on moe_b13 WITHOUT L1

N96 was implemented and compiled, but serialized WGMMA under the96-register
cap. The delivered default uses the user's permitted **N80 fallback**:
65..80 token experts become one job, padded to72/80 for native WGMMA.
Experts above80 (including81..96) retain the parent's64-token splitting and
final short tail. The measured68- and75-token experts are therefore merged;
the144-token expert remains64+64+16. Experts<=64 keep the same job/arithmetic.
`OKC_L2=1` enables this only in MAIN/WGS;0 disables it. Native72/80/88/96
definitions exist, with experimental `OKC_L2_CAP=96` gated off by default.
Three36-KiB weight slots and the original576-thread/96-register launch remain.
No setmaxnreg, producer padding, or whole-stage capture from L1.

## Narrow-output grouping invariant

Simply collapsing jobs and retuning costs changes other experts' CTA/K splits,
which could violate the requested bit identity for<=64-token experts. Instead,
native job stage cost is cost(64)+cost(tail), separately for A/B, where tail is
cnt-64. The baseline's total cost and every narrow job's absolute cost prefix
are then unchanged. Cost-based CTA boundaries and narrow K shares are preserved;
only native wide-job K grouping changes. Wide cost can exceed255; do not pack
the sum back into an8-bit field. The sum is evaluated from the two original
8-bit lookup entries, with uint32 prefixes and the unchanged safe total bound.
This is conservative for performance: it preserves numerical grouping rather
than claiming an accurately tuned native-N96 cost model.

## X ring and fixups

After CTA-local routing, S.preissuedB (unused by WGS otherwise) records whether
the launch has a native wide job. No new shared header fields. Every CTA chooses
3x32KiB x slots if false,2x48KiB if true; both consume96KiB. Weight ring stays
3x36KiB, and the host shared-memory allocation function remains unchanged.
The x producer and consumer use the same slot count, stride and parity; spare
barriers are initialized but unused in two-slot mode. Input/SiLU xA/xB rows are
padded to8 with actual ntt, so existing gather and epilogue mapping extends to12
n-tiles. Readiness remains npad rows / eight A warpgroup epilogues per B stage.

Fixup stride grows from8 to10 n-tiles in the delivered cap80 build (4096->5120
FP32 values per CTA-boundary/warpgroup slot). Python allocates the12-tile upper
bound6144, so cap96 also has enough space if explicitly built. Every CUDA fixup
and counter pointer uses its own compiled stride consistently; the extra Python
space is unused capacity, not a mismatched live layout. Allocation grows10MiB
over BASE (cap80 actually uses5MiB more). The diagnostic allocates this larger
scratch too, and clears it between BASE/L2 variants. Total padded x rows are
unchanged: rounding64+tail separately equals rounding the combined count to8.

## Compilation/validation plan

Native WGSMMA9..12 were added; existing1..8 definitions and narrow GS2 retained.
Final wide GS=1. Wide segments drain prior-segment deferred work BEFORE creating
their larger accumulators; narrow segments retain the parent's deferred flush
after eight k16 steps. This changes scheduling/liveness, not epilogue arithmetic.
Keep ring ownership, WGMMA register-source lifetimes and commit/wait correctness.
Test job/entry/unit/x-row/fixup coverage, x-slot bounds/parity, narrow cost/K
partition identity, both runtime x layouts, and no L1 inheritance. Diagnostic:
moediag7 packaging, BASE versus L2 on the requested six routes,40 cold-L2 reps,
both orders, byte equality/relativeL2, zero-routing disabled. No local GPU/git.

Unlike the closed cap128 WIDE experiment, this uses native wider WGMMA and
retains36KiB weight stages and three weight slots; no two64-token compute passes.

## Resource investigation and final choice

All entries below are offline CUDA13 NVRTC/ptxas sm_90a, C++20. Each build has
two BIG instantiations; their resource results agree. No local GPU was run.
The initial N96 GS2 build emitted C7512 (compiler serialization of WGMMA) and
80B stack,116/204B spill stores/loads. GS1 still emitted C7512 with72B stack,
112/200B spills. N80 without the deferred-work change also serialized.

Bounded live-range trials drained deferred epilogues before wide accumulators,
removed wide packed-group lookahead, and explicitly drained wide commit groups.
N96 still serialized. These extra no-lookahead/forced-wait trials are **not in
the delivered code**. N80 with GS1 and early deferred-work drain eliminated
C7512 and was selected. This is a resource fallback, not a measured speedup.
Prototype sources/logs remain in `.l2_offline/` for audit.

| Final compiler view | Launch regs | Stack B | Spill stores/loads B | C7512 |
|---|---:|---:|---:|---|
| BASE / L2 disabled |96|16|56 /124|no|
| Delivered L2, runtime3/2-slot selection |96|32|236 /456|no|
| Narrow-only compile, cap64 /3 slots |96|16|96 /160|no|
| Fixed-two-slot offline specialization |96|64|460 /624|no|

The last two are **separately recompiled diagnostic specializations**, not a
claim that the runtime binary changes its allocation between layouts. Fixed
two-slot mode is valid even for narrow jobs, and uses the same48KiB pitch.
The actual candidate always uses the delivered runtime3/2-slot build.

SASS confirms native HGMMA64x72 and64x80, no64x88/96 in the default binary, and
no L1 setmaxnreg instructions. Between the first and sixteenth HGMMA of each
native-wide stage, there are **zero LDL/STL instructions** in the delivered
binary. This limited inner-MMA interval check does not imply a spill-free whole
stage: whole-kernel totals above remain significant. Narrow first-stage intervals
can contain up to six local instructions (including deferred epilogue work).
That regression and added code footprint may erase duplicate-stream savings.

Dynamic shared allocation remains231,168B; both BASE/L2 offline cubins also
report1,024B static shared. Three36KiB weight slots +96KiB x ring are unchanged.
The cap80 wide x payload is at most40KiB inside each48KiB slot.

L2-disabled MAIN has the same instruction count and arithmetic as BASE; the
only observed SASS differences are exchanging R12/R16 for the x producer's
stage-count/parity variables (eight instruction differences across two kernels).
TINY/TINY2/MID/MAIN_FALLBACK SASS is instruction-for-instruction identical to
the corresponding parent builds.

## Delivered files and validation

- `moe_l2.py`:199,829 bytes; SHA256
  `1c70fc070da19e1a3508f462e040d530172ff76fc454d41c1ccabe46ac793ac4`.
- `../diag/moediag_l2.py`:178,050 bytes; SHA256
  `9ecf4f050713fefb62cb87dacd7f98a4ef4606be94727382d720aa958fcb3ea3`.
- Parent `moe_b13.py` SHA256 remains
  `b6390e9993caebd178645d2f33e780724fd84242d15b4a17639d98bc17a3c7ce`.

The diagnostic uses moediag7's CUDA_B85/CUDA_SHA/SOURCE_SHA dictionaries and
two thin-host CUDA TUs. Exact BASE is moe_b13; L2 is the delivered N80 fallback.
All six requested cases,40 cold-L2 samples, both case/variant orders, byte
equality/max-abs/max-rel/relative-L2 comparisons, zero-routing disabled.
`MD_CASE` reports native counts/job savings and x-slot selection. In addition
to24 ordinary timing/error lines,12 `MD_NARROW` lines compare the per-route
FP32 B outputs for experts<=64 against saved BASE. This isolates narrow outputs
even when final token outputs mix wide and narrow expert contributions.
Both diagnostic device TUs compile and reproduce the BASE/L2 resources above.

CPU tests passed:
-66 job/count models (including65/68/75/80/81/96/97/144 boundaries), both phases,
  grids1/7/132: every entry/unit/stage once, unchanged total cost and exact narrow
  cost prefixes/CTA K partitions, unchanged narrow x-row origins.
- x gather and A-epilogue mapping bijections through96-row capacity; both x-ring
  bounds/parities,12 delayed-release pipeline models, readiness8 per B stage,
 10-tile live/12-tile allocated fixup coverage.
- no L1 inheritance, unchanged narrow MMA/epilogue/gather code, exact payload
  hashes and2-TU linkage,24 mocked measurements with all40 repetitions/both
  orders, narrow-output selection and larger scratch allocation. Mock log5,110B
  plus1,000B reserve fits9,400B. Mocks are not numerical/performance evidence.

Reproduce:
```
python -B .l2_offline/make_l2.py
python -B .l2_offline/make_diag.py
python -B .l2_offline/test_l2.py --sass
python -B .l2_offline/test_diag.py --compile
python -B .l2_offline/compile_l2.py L2 NARROW TWO_SLOT OFF TINY TINY2 MID MAIN_FALLBACK
python -B .l2_offline/sass_report.py L2 NARROW TWO_SLOT
```

## Remaining risks / acceptance

Runtime correctness, narrow bit equality, wide rounding differences, performance
and Torch host-build duration are unverified. Narrow K shares/arithmetic were
preserved deliberately, but timing-dependent cross-CTA atomic ordering still
needs the `MD_NARROW` GPU check. Wide K-share grouping changes and is reported
through global byte/error comparisons. The preserved virtual costs overestimate
native work and can load-imbalance the launch; they were chosen to protect the
narrow numerical contract. Two x slots, earlier deferred-work drain, instruction
footprint and remaining spills are substantial performance risks. Do not treat
compilation success as evidence that L2 beats BASE. No local GPU or direct git
commands; no L1 implementation changes were included.
