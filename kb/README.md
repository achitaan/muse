# kb: building on keell0renz's public kernels (Apache-2.0)

Source: https://github.com/keell0renz/muse-open-kernel-challenge (commit 8e30e1b), files copied unchanged into this folder
with their LICENSE and NOTICE; every change below is ours and marked `achitaan` in the source.

Changes:
1. wgs_stage_at: `if (nj == 0) return 0;` in every copy (decode layer x2, MoE x2). Fixes the stream-K MoE hang when a launch
   has no local route (all experts remote): ranges become empty instead of reading the unset S.jC[0]. Bit-identical otherwise.
