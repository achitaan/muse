# WGS phase-A adjacent-column BF16 stores

Prepared October 29, 2026. One fixed candidate; no CUDA compilation, GPU
correctness, resource admission or speedup has been established yet.

## Source and scope

Baseline `best/moe_verified.py` has SHA256
`331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.
Its primary check d600231c8a8244cb is 2.365437416088433x; its scored result
b6c70c9b83724d2e is 2.348299935218348x. These are distinct historical jobs.

Frozen source `results/moe-epair-store-preflight-20261029/submission.py`:
`e8e3086d34a30de0283e37f2f40d3b1d9742313203fc624c1720bc6a6badced8`,
336352 bytes. Alternate CUDA SHA256:
`9d6c435420eb3f43d9fa2cbbea343d1414a557ee883140e67d97718b2b0e3e5d`.
The original module is an exact prefix, with a reversible CUDA edit ledger.

Only the eligible M>16 MAIN/WGS family can select the new module. Original
TINY, TINY2, MID and non-PRMT fallbacks remain original. The new device namespace
is `okc_epair`; profiles must show actual execution. The WGMMA mainloop,
group sizes, token geometry, weight/scales bytes, routing, work assignment,
fixups, phase-B epilogue, finalizer and release/reset protocols are unchanged.
An explicit empty-job range return avoids reading an uninitialized job row
on all-nonlocal inputs, preserving every nonempty range.

## Observed work and mechanism

The retained phase-A epilogue computes two token values per lane and writes
them as separate 16-bit stores into phase B's activation layout. Adjacent
column lanes have the complementary values needed for one aligned 32-bit
store. This is different from the separate, parked activation-gather proposal.

Keep the original scalar arithmetic and BF16 conversion for each valid value.
Exchange one opaque 16-bit payload with lane XOR4, pack neighboring columns
into low/high halves, then write one 32-bit word. Every lane executes the
shuffle, including tail lanes whose temporary payload is initialized zero;
the final store predicate still excludes invalid token rows. No padding is
written. No new CTA/grid barrier or shared buffer is introduced.

For a full eight-token epilogue tile, two warp store instructions become one;
the abstract 32-byte-sector count falls from eight to four. A one-token tail
has no such sector reduction. For a job of count c, the scalar-store count
is 4096*c and the packed-store count is 2048*c; requested sector counts in
this layout are 512*c and 512*ceil(c/2). Payload bytes are unchanged. These
are source counts under the stated instruction/coalescing model, not measured
HBM traffic, latency or a promise of half the epilogue time. Added shuffles,
selections and register liveness may cancel the savings.

NVIDIA documents the [BF16 bit reinterpretation](https://docs.nvidia.com/cuda/archive/13.1.2/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__MISC.html)
used here and the distinction between sectors and serialized
[L1TEX wavefronts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#quantities).
Generated code and actual resources remain unverified.

## Numerical and ownership proof

The Marlin-equivalent BF16 accumulator/global-scale/product materializations,
FP32 SiLU division/exp/multiply expression, `probe_q` and final BF16 conversion
are literal exact. Only the converted payload bits move between lanes.
There is no additional rounding, FP8 activation or changed FP32 norm tree.
Phase B retains its BF16 effective gain and FP32 normalization/route reduction.

An independent CPU reference transposes complete dense token/column tiles
into the documented WGMMA layout. All counts 1..64 and all 4096 columns match:
8,519,680 unique halfword writes, untouched padding and aligned word stores.
All 65,536 BF16 payload patterns round-trip through low/high packing, including
signed zero, subnormals and NaNs as raw bits. Wrong-lane, swapped-column and
padding-writer mutants fail. These are CPU layout/bit proofs, not GPU tests.

Every `wgs_epi_tile` call is warp-uniform: the direct path's NTT loop is uniform,
and split-unit completion is broadcast at the existing warpgroup barrier before
its uniform ntt loop. The shuffle occurs after the per-token arithmetic branch
reconverges and before the new per-token store predicate. Existing release
arrivals remain after all contributing stores and their original barriers.
Other CTAs cannot observe a newly published incomplete activation tile.

## Resources and replay

No expert copy or extra layout is added. Original weights and scratch offsets
are exact. The comparison uses a distinct approximately157MiB scratch bank;
original inputs/layouts, original scratch, compile allocations, output graphs
and the128MiB cold-L2 flush buffer remain additional live allocations.

Before any new launch, require both BIG instantiations to report1..112registers,
zero local and static shared bytes,231168dynamic shared bytes,576threads and
at least one active block/SM. Set the dynamic-memory attribute before querying
occupancy. The original source sets grid to the device's SM count, and the
fixed gate additionally requires grid1..160. Do not relax admission after a
failure. Extra temporary register pressure is a principal risk.

One extra MAIN extension is prepared. Skip if original prepare exceeds185s,
compilation exceeds270s, or complete diagnostics cannot fit within300s. No
reachable extension compiles during graph capture. The wrapper returns fresh
outputs, preserves aligned int32 IDs, chunks at384tokens and never host-reads
timed inputs. Sequential scratch reuse remains the contract.

## Fixed evaluation and stopping rules

One future paced full check, cached baseline, no scored run at this checkpoint.
Do not upload before05:22:50.375730UTC; maintain one active job, the user window
through October29 at20:00Toronto, and the policy's per-track reserves.

Within prepare, compare the exact original and new modules on normal synthetic
M17,31,32,33,63,64,65,127,128,129,191,192,255,256,257,384,385 and skewed M128.
Ten cold-L2 graph pairs/case, balanced randomized AB/BA order, preserved raw
samples; CUDA-event milliseconds convert to nanoseconds with1e6. Replays are
not independent jobs. M32 gets two fresh-input/route/weight graph replays.
All-remote17/33/384/385 must produce exact zero after poisoning; duplicate-route
M33 must match the incumbent within0.5% global and2% worst-row error. Other
comparisons use the same tight bounds, independently of the official checker.

Select the single new family only if all correctness/resource/replay gates
pass, every paired case loses at most2%, and normal M32/128/256 each improves
by more than3% with their GM ratio below1/1.03. Otherwise select the incumbent
and treat the result as a rejected candidate, even if the official aggregate
point happens to rise. No selector based on favorable isolated cases.

Retention additionally needs full5tests/6benchmarks/integrity, the expected
new namespace in cases2/4/5 and original small paths in0/1/3, and actual evaluator
baseline/reference/checker/nvfp4 hashes matching the freshly verified packet.
Require affected-load GM>1.03, control-normalized GM>1.03, each normalized
affected load>=.99, unchanged-control GM in[.95,1.05], and track speedup above
2.4009189773297592x (1.5% over the cached primary check). These are engineering
gates, not a joint statistical confidence statement.

Host CPU mocks cover100 dispatch cases,25 chunk cases, six resource rejection
types and11 selection rejections. The full-result guard's latest CPU audit
is `guard-cpu-audit-v2.json`:12 bad fixtures rejected, including changed evaluator
identity. The earlier11-fixture audit is preserved as development history.
Six fresh official starter files match the pinned copies byte for byte.

A failed resource, correctness, new-path or material-gain gate closes this
specific candidate; no neighbor sweep or retry for a better noise point.
A retry requires a diagnosed bounded defect. Publish only after source-bound
material improvement. Preserve all verified sources meanwhile.

## Decode contribution

Three of six MoE loads are affected; equal improvement s dilutes to sqrt(s)
at track level. A3% per-load gain therefore means about1.49% MoE track gain,
not3%. The paired threshold asks for roughly3.65/5.22/6.27us recovery from
the saved M32/128/256 primary times; these are required savings, not forecasts.
The exposed epilogue fraction and exact overall-board aggregation are unknown.

The exact native MoE CUDA is also embedded in retained DecodeD9. A retained
win can replace its corresponding MAIN build while preserving small families,
layer rounding and overlap. The inactive source-preserving307dd791 Decode
package provides source/preparation headroom; construct the small CUDA patch
from its original source rather than duplicate a full literal/build. Whole-layer
correctness and46S+16F weighted-load evidence remain necessary before retention.
No Decode source or check is changed by this packet.

## Workflow exception

The user requires main-chat-only work and a fixed composer model. Automatic
approval review rejected the attempted Lane command because orchestration might
dispatch another model/agent. It was not retried or bypassed; implementation
and review continue in this chat. No independent-agent review is claimed.
