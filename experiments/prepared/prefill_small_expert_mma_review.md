# Next bounded Prefill inspection: small expert tiles

No candidate or evaluation yet. Keep93e57ab4, fully checked5/8 and scored
1.663469245x. Zero-row compaction is deferred by its routing proxy; no GPU
run was used. Source also already overlaps pooler/router. Preserve the one
remaining MoE check and one remaining Decode check for stronger evidence.

Read the _pg__CUDA_SRC literal safely through AST. Its persistent expert
kernel is different from the standalone33114871 WGS kernel. Do not transfer
its flags to MoE or repeat the rejected two-group geometry experiment.

Current flags include MOE_MMA32=0, MOE_GROUP4_MAX128, MOE_NACC4=1 and
MOE_FINE_MASK6. The alternative small-N code uses mma.sync.m16n8k16 instead
of WGMMA and skips8-token groups past nreal. It might reduce padded work for
very small expert tails, but packed weight decoding/bandwidth still costs
the same. MMA instruction issue and loss of asynchronous overlap can make
it slower. Existing upstream comments mention compiler serialization when
too many variants/call boundaries are present. Read those comments and
upstream experiment notes before treating the disabled path as new.

Resolve actual work-item N/nreal dispatch and count its share of the work.
Prove both GEMMs retain all real rows/K elements, original packed scales,
BF16 accumulator/global-scale boundaries, BF16 SwiGLU and FP32 norm stages.
The unused branch must be audited explicitly; a passing parent that never
executes it does not validate that branch. Check its finalization path uses
the corrected paired Marlin roundings before any GPU evaluation.

Only implement a bounded candidate if source/evidence supports a material
gain. Do not sweep flags or blindly enable a disabled experiment. Require
actual new-path profile/selection evidence, full5/8/integrity and a material
whole-layer gain before one score. If the path was already measured slower
or only affects negligible tails, close the inspection without a run and
research a different algorithmic improvement.

Also closed by inspection: standalone331's WGS gather already avoids
global stores for padding rows. wgs_gather sets d[k] only for r<cnt and
stores only nonnull d[k]; padding is counted in xready but not written.
wgs_xproducer copies only ceil(cnt/8)*8 rows per job/stage, not FT_T rows.
There is no missing easy zero-fill removal to spend the last MoE check on.

Inspection completed: real dispatch passes ti.w as nreal; only N32 expert tails use the optional synchronous MMA path. Weight traffic/decode and padded epilogue remain unchanged. No upstream positive timing evidence; leave disabled. The stronger exact-decode/paired-epilogue transfer was tested separately and discarded by its fixed profile guard; see results/resumed-prefill-pair-dq8-20261028/.
