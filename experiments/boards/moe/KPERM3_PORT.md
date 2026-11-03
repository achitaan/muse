# moe_b16: KPERM3 port into our b15 lineage

**This is our moe_b15 plus a reimplementation of the KPERM3 technique, not
alygg77's submission file.** The technique is credited to alygg77 under
Apache-2.0 beside the CUDA and Python integration. Our existing license and
attribution notices remain intact.

## Sources inspected, read-only

- `alygg/kernels/muse_moe_w4a16.py`: SHA256
  `2f7a20c15c4c5173bba176122aaa065dd437a8d647ea5809b284554fc2738252`.
  Relevant Python functions: `_kperm_src`, `_kperm_codes`, `_kperm_rows`,
  `_wgs_copy`, and the WGS portion of prepare. CUDA: `wgs_gather`, `wgs_seg`.
- `alygg/README.md:46-63`, `RESULTS.md:24-35`,
  `docs/H200_MEASURED_FACTS.md:111-117`; negative-results context was inspected.
- Our parent `moe_b15.py`: SHA256
  `eeeced020b79edec2e578d56cb116e4aacddf4152d0baf710f5004f1d182cc06`.

The reference has no OKC_KPERM switch: its WGS packing/gather/scale accesses
are hardwired to KPERM3. Other reference differences were NOT imported: its
older dq4 bit placement, numerical epilogues/norm treatment, context-binding
wrapper, diagnostics and dispatch remain outside this port. In particular,
our Marlin BF16 accumulator/global-scale boundaries and effective BF16 norm
weight remain exactly as in b15.

## Isolated technique and numerical consequence

Inside one256-column K stage, let physical WGMMA position
`p=16*(4*lg+j)+8*q+2*t+u`, with lg,j,t in0..3 and q,u in0..1. KPERM3 reads
original column:

```
k = 16*(4*t+j) + 4*lg + 2*q + u
```

Thus lane t's step j uses the SAME original per16 scale block `4*t+j` for
all four lg groups. A thread loads four original gate/up scale pairs in one
uint2 per stage; the compiler can build each pair once, instead of reloading
and decoding four pairs for each lg.

**The logical summation order necessarily changes.** For example, the first
physical k16 step reads original columns
`0,1,64,65,128,129,192,193,2,3,66,67,130,131,194,195`, not0..15.
It is not possible to get this sharing while retaining b15's original k16
operand grouping. The number/order of physical MMA instructions, stage order,
commit groups and waits remain unchanged; their logical K operands change.
No quantized value changes. Differences can originate from FP32 addition
association, and can propagate through the already-existing BF16/SiLU/norm
rounding boundaries. GPU output bit equality is therefore NOT promised.

Three transformations must agree:
1. Both W13 and W2 WGS copies permute K within each256-column stage.
2. A's input gather reads x in that order. Each16-byte destination vector
   gathers four intact32-bit BF16 pairs; the token/layout destination is unchanged.
3. Gate AND up rows are permuted together so physical intermediate position P
   holds original intermediate `perm(P)`. The unmodified A epilogue directly
   produces B's desired xB order. B output rows remain natural.

The intermediate permutation stays inside256-index blocks. The same two A
units/four WGs per unit still produce everything required by a B K256 stage,
so doneAs readiness accounting does not change.

## Local implementation differences

Our raw-code reimplementation uses a packed **byte transpose**, rather than
the reference's nibble-unpack/index/repack. Adjacent low/high nibbles remain
together under the permutation: old byte axes `[t4,j4,lg4,pair2]` become
`[lg4,j4,pair2,t4]`. This is a bijection of original bytes. Our existing
`_pack_codes_nat`/dq4 bit encoding is then used unchanged.

Scale bytes remain original values, with WGS storage axes changed from
`[lg,warp,g,j,h]` to `[warp,g,t,j,h]`. The scale address changes to
`8192 + (warp_in_WG*32+lane)*8`, once per stage. That uint2 is reused across lg.

We retain **our ERL release at lg2**, whereas the reference releases at lg3.
The KPERM3 ERL dependency checks the last four code words; all scale words have
already been consumed by lg0. Extending raw scale-register lifetimes merely to
repeat that dependency would be unnecessary. The non-KPERM branch keeps the
original six-word code/scale dependency unchanged.

HINT, FINREG, J==0 empty-range handling, routing, job construction, cost/range
partitioning, x producer, numerical epilogues, finalizer, PDL and host ABI are
preserved. The Python selector is unchanged: KPERM applies whenever MAIN/WGS
is selected (default threshold17+, with the existing selector behavior retained).
TINY/TINY2/MID/mma.sync fallback keep their original packed weights and code.
No additional WGS weight copy or scratch allocation is required.

## Mode control / integration contract

`OKC_KPERM` defaults to3 in CUDA;0 keeps the legacy WGS path. Python `KPERM=3`
selects the matching prepare layout and automatically supplies the matching
define for WGS builds. Set `KPERM=0` **before prepare** to build legacy layouts.
Contradictory explicit compile flags are rejected. Do not change KPERM after
preparing state, and do not run b15 CUDA on b16's permuted WGS tensors (or vice
versa). A future A/B diagnostic must prepare matching BASE and b16 layouts;
the old shared-weight-state diagnostic pattern is not valid for this port.

## Offline evidence

Submission: `moe_b16.py`,191,008 bytes, SHA256
`745ff92fc457c70aab40fbbc9e9706b30ca80e50e90fbde73c8a47e209ad53cd`.

NVRTC/ptxas sm_90a, C++20, CUDA cache disabled; both BIG instantiations agree:

| MAIN | Registers | Stack B | Spill stores/loads B | C7512 |
|---|---:|---:|---:|---|
| b15 |96|16|56 /124|no|
| b16 KPERM3 |96|16|60 /128|no|
| b16 mode0 |96|16|56 /124|no|

**The published zero-spill result did not reproduce in this retained b15
configuration/toolchain.** The port is not a demonstrated removal of the
56/124B whole-kernel spills. An equivalent explicit four-scale-pair predecode
trial produced the same60/128B result and was not adopted. No HINT/ERL/FINREG/
J0 or numerical changes were removed to force a nicer register report.

The scale-sharing mechanism is visible in SASS. Ordinary16-MMA stage spans
reduce scale LDS.64 instructions4->1 and PRMT16->4. Across both kernel
instantiations, total LDS.64 drops268->76 and PRMT1788->1034. A larger N48
backedge region also includes deferred/epilogue work, so its counts are not a
pure per-stage cost. Whole-kernel local instructions remain90->94. These are
static instruction counts, not measured execution times.

Both builds retain1,024 HGMMA instructions,512 overlapped DEPBar.LE1 waits and
32 terminal DEPBar.LE0 waits: no C7512 serialization. Mode0 MAIN and every
TINY/TINY2/MID/MAIN_FALLBACK build have instruction-for-instruction identical
SASS to b15. Dynamic/shared layouts and launch dimensions are unchanged.

CPU seed20261102 tests passed:
- Full one-expert W13+W2 repack/inverse: **25,165,824 code bytes and3,145,728
  scale bytes recovered exactly**, including all possible code/scale bytes.
- The inverse uses independent CUDA dq4 algebra and scalar fragment-coordinate
  mapping, not our packer's `_POS` inverse. Inputs are verified unmodified.
-65,536 sampled actual consumer fragments match original code and scale
  addresses; valid-scale encoded dequantized values match exactly.
- All4,096 activation gather coordinates and all16 A->B readiness blocks match.
- Independent FP64 exact-product composed MLP test verifies paired gate/up
  permutation, BF16 SwiGLU inter permutation, and B column cancellation.
- The three read-only reference permutation helpers agree with our independently
  expressed byte/index maps. Their extension/context wrapper is never imported.
- Legacy packing is byte-identical to b15; compile/packing mode guards pass.
- Source audit confirms our numerical boundaries, producer/epilogue/finalizer,
  HINT/ERL/FINREG/J0 and host interface remain intact.

## Reproduce / remaining limits

```
python -B .kperm_offline/make_b16.py
python -B .kperm_offline/test_repack.py
python -B .kperm_offline/compile.py
python -B .kperm_offline/audit.py --sass
```

Readable CUDA, focused reference/our diffs, logs, flags, SASS and JSON evidence
are under `.kperm_offline/`. The foreign reference tree was read only. No local
GPU execution or implementation commits. Device-only compilation is not a
Torch host-extension build, numerical tolerance check or speed measurement.
The reference's published speedups are motivation, not results for b16. Actual
tolerance and performance remain for the judge; any new timing harness must
use the judge-like **read flush**, not zero_/add_ write flushing.
