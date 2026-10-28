# One-query FP8 probability/value trial

Parent is retained corrected24782695, not the discarded FP8Q trial. Keep its
FP16 QK, BF16 Q/K norm, BF16 RoPE, cache writes, every key/mask and BF16
attention output before RMSNorm/gate. No Python dispatch, scratch, allocation,
barrier or shared-memory size changes. L8 arithmetic is untouched.

The existing PV helper loads FP8 values at2*tig+{0,1,8,9}, across two K16
halves. Its byte permutations expand into FP16 B fragments, then16 MMA
operations compute32keys and64dimensions. The new helper retains those exact
eight8byte V loads, packs four FP8 values per register using a two-stage4x4
byte transpose, and issues eight E4M3 m16n8k32 operations with FP32 accumulators.

Crucial K permutation: each logical quartet maps to physical keys
2*tig+{0,1,8,9}, with+16 in the second half. P already holds these values
as s[0/1] and s[2/3], so it packs directly into A0/A2. Both operands use
the same permutation; no key is discarded. A1/A3 are zero unused rows.
B column g represents output dimension8*g+t, retaining the old strided
output mapping. No additional shuffle, transpose buffer or synchronization
is required. Retain the old V loads' bank layout and existing QK implementation.

Probability scale is exactly256. The existing online max and rescaling
remain; both FP32 denominator and FP8 numerator use the same power-of-two
scale. P is bounded by256 before conversion, below E4M3's finite maximum448.
This reduces underflow compared with unscaled FP8 probability conversion.
No attention-weight thresholding or explicit skipping is added. The cache's
V scale is applied by the unchanged output finalizers.

Primary precision source: the supplied baseline calls FP8 FA3; FA3's internal
probability tensor uses FP8. Its implementation uses Max_offset8 for FP8:
https://github.com/Dao-AILab/flash-attention/blob/main/hopper/softmax.h
https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h
Instruction register mapping:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16832
This is an internal tensor precision already used by the baseline; required
BF16 materializations are retained. Do not generalize FP8 permission to Q/K
norm, RoPE or output activations, which must remain BF16.

CPU evidence:256 complete PV operand matrices,2048 exact byte transposes,
32 output cases with two scales, partial windows, coherent V directions,
80K context, large score jumps and stable online merge. Max output relative
L2 vs a FP64 softmax/PV oracle0.011853506, worst per-row0.025342845.
The CPU model does not prove GPU correctness. Reversible CUDA ledger restores
the baseline exactly, all other Python AST is unchanged and BF16 guard passes.

Predeclared guard: full16tests/12benchmarks/integrity, all six actual new
L1 symbols, affected GM ratio>1.025, no affected ratio below0.98, aggregate
ratio>1.015 versus cached247 check. One full official H200 check, no repeated
baseline. The previous FP8Q changed GM0.999962 shows why aggregate noise
alone cannot justify a score or Decode transfer.
