# kb: building on keell0renz's public kernels (Apache-2.0)

Source: https://github.com/keell0renz/muse-open-kernel-challenge (commit 8e30e1b), files copied unchanged into this folder
with their LICENSE and NOTICE; every change below is ours and marked `achitaan` in the source.

Changes:
1. wgs_stage_at: `if (nj == 0) return 0;` in every copy (decode layer x2, MoE x2). Fixes the stream-K MoE hang when a launch
   has no local route (all experts remote): ranges become empty instead of reading the unset S.jC[0]. Bit-identical otherwise.
2. v1 (`kb/v1/`, also the current top-level files): three bit-identical MoE changes from our own lineage (measured on our kernel:
   R128 -2.1%, R256 -1.5%, R32 -0.6% with a read flush), applied to every stream-K copy:
   - ERL: free the weight ring slot once the stage's last codes are in registers (lg == 2), not after its math;
   - HINT: weights of an expert split into several 64-token jobs load with the normal L2 policy, not evict-first;
   - FINREG: the finalizer keeps pass 1's y rows in registers for pass 2 instead of re-reading them.
