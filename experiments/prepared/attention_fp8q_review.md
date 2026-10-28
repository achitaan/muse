# One-query FP8 QK inspection and trial contract

Parent: corrected Attention24782695, check1a16e2d658a8400d2.522139036x,
scored d0cdadc0960446ca2.541743138x. Keep this source unless the new path
passes full16tests/12benchmarks/integrity and the predeclared performance guard.
No repeat baseline evaluation. Main chat only, existing free official H200.

The L1 path currently expands FP8 K to FP16 and uses four m16n8k16 QK
operations per eight-key block. Use two m16n8k32 E4M3 QK operations instead.
The task explicitly permits query rounding to E4M3. Preserve BF16 Q/K norm,
BF16 RoPE output, BF16 attention output before gated RMSNorm, FP32 softmax
and accumulation, all keys, causal/window bounds, cache writes and scales.
PV remains FP16 tensor math. The L8 path and all Python AST stay unchanged.

Primary documentation:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
NVIDIA specifies FP8 MMA support from sm89; H200 is sm90a. The register
mapping was derived from the documented A/B/C coordinates, then checked by
reconstructing the entire integer-tagged operands, independent of QK products.

For each lane group g and lane within group t, A0 is query head g at
dimensions4t..4t+3; A2 adds16. A1/A3 are zero for the unused eight output
rows. B0/B1 contain the same dimensions of key g. Two K32 steps cover64.
Output C0/C1 are query g, keys2t/2t+1. Four eight-key blocks and two warp
subgroups cover every64-key tile for both KV heads.

New query stores occupy the first8bytes of each thread's original16byte
slot. Packing an entire row into64contiguousbytes without a barrier would
race another thread's raw BF16 query read in WS mode. Retaining128byte row
stride avoids that race and requires no additional synchronization. K needs
four32bit shared loads instead of one128bit load, with32distinct banks per
warp. This extra load instruction cost is the principal performance risk.
Q stays unit-RMS until FP8 conversion; multiply completed FP32 QK scores
by scale_log2*k_scale, preventing subnormal loss for k_scale1/64.

CPU audits:384 complete integer operand matrices,1536 store ownership
checks,768 conflict-free warp loads,32 scaled/rotated/partial-context online
softmax cases. CPU max relative L2 vs the allowed FP8-query FP64 oracle
0.000959605. These do not prove GPU accumulation behavior or speed.
Reversible source ledger restores original CUDA exactly; other Python AST
unchanged; existing BF16 boundary guard passes. No submission imports tests.

Guard fixed before GPU evaluation: six actual L1 profile symbols, full16/12,
no affected speedup ratio below0.98, affected six-case geometric ratio>1.025,
whole metric ratio>1.015 vs cached corrected check. Intervals may overlap;
these empirical thresholds do not establish formal statistical significance.
Only a retained material winner gets one scored confirmation and possible
Decode transfer. An aggregate timing fluctuation without new-path gain loses.


## One-query FP8 QK trial closed at21:25UTC

Job60aa54a9feeb49e9/source8463eaf3 passed16tests/12benchmarks/integrity
at2.542655363x, CI[2.504515532,2.580795193]. The six changed L1 cases
had geometric ratio0.999962266 vs corrected247; unchanged L8 controls
rose2.3628%. Batch-one sliding ratio0.96775 violated the fixed0.98 floor.
The controller discarded it (trial91092e8, revert04741e8). No scored run
or Decode transfer. Retain corrected247, scored2.541743138x. One check
used; expected Attention12checks/7scores left. Full payload, profiles,
source and CPU audits, fixed guard failure and immutable controller history
are in results/resumed-attention-fp8q-20261028/. Do not rerun for noise
or retain a selector based on these small individual fluctuations.

Next inspect L1 PV: its helper loads FP8 V and expands32packed FP16
registers per32-key block, with16FP16 MMA operations. A register-only
4x4 byte transpose could feed8FP8 K32 MMAs without extra shared memory.
Keep the retained FP16 QK path. Before a candidate, audit P scaling,
PTX A/B/C mapping, every BF16 boundary and actual FA3 internal precision.
