# Attention L1 QK operand swap

Parent9e922c35, exact scored8b767ea7b2b74de6 at2.537000514x, full16tests,
12benchmarks/integrity. Reuse that exact scored result as cached baseline.
One candidate469d8b04, one check, no sweep. Do not inherit rejected3CTA/FP8Q.

The old one-query QK maps eight query heads to the16-row MMA A operand;
half of each m16n8k16 result is used. Swapping operands computes16keys x8heads
instead of8heads x8keys, with the same FP16 query/cache values. Each32-key
warp tile needs8MMAinstructions instead of16. FP32 reduction order changes;
all64K dimensions still contribute once. No FP8 query conversion is introduced.

Use existing K shared-memory swizzle. ldmatrix.x2.m8n8.b16 loads16rows of
packed FP8 pairs; low/high pair conversion produces the four FP16 A fragments.
A common K permutation (0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15) within each
16dimensions lets Q supply B fragments without per-tile register shuffles.
Only the query prologue reverses inherited word order0,2,1,3.

Scores are16keys x8heads. Transpose them to the existing softmax/PV lane order
through warp-private128float buffers in dead query stagingS.kvq (4warps=2048B).
One consumer barrier after query reads prevents overwrite races. Each16-key
tile uses syncwarp after stores and after reads before reuse. XOR index
(head*16+key)^((head>>1)*8) gives unique scalar-store banks and ideal two
128Btransactions for float2warp reads in the32bank model. No extra shared
allocation/geometry/register-budget changes; real compiler behavior is unmeasured.

All BF16 Q/K norm/RoPE/output boundaries, FP16 query scaling, FP8 PV, softmax,
cache writes, segment/index bounds, scratch/cache workspace and L8 code remain.
The source inverse ledger recovers every original CUDA byte and other PythonAST.
This is a different arithmetic arrangement, not another occupancy/register trial.

CPU audit reconstructs actual address-provider/LDMATRIX and PTX MMA fragments,
checks every Kentry once, complete integer products against direct K@Q.T for
64cases, and exact score transpose. It proves2048Bwarp-private bounds and
the modeled bank schedule. Sixteen FP16/FP8numerical cases compare FP32 modeled
sums with independent FP64. CPU sum order is not a hardware MMA emulator.
Full official16/12/integrity remains mandatory.

Primary PTX layouts/collective requirements:
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816
https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-ldmatrix

Fixed guard: exact hash/full coverage/BF16 precision; all sixL1new-path profiles,
changed GM>1.025 and changed/control GM>1.025, noL1ratio<.98, aggregate>1.015.
No causal statistical-significance claim from cross-job controls. A score is
allowed only if the passing check's lowerCI clears parent upper2.575055522.
If it loses, preserve the negative result; no neighbor tile/bank sweep. Phase
2.70x is only an experiment checkpoint, not the overall top50Goal.
