# Prefill normalization/gather opportunity screen

CPU source/profile inspection only, exact93e57ab4 and primary job9ca79c7f676f4b63.
No Prefill code was changed and no new GPU job was used. The current Attention
foreground experiment remains the only queued evaluation.

The active Prefill kernel calls pre_expert_norm(expert_input) before its one
reachable _experts call. Final _experts invokes the persistent grouped path;
moe_gather_kernel loads one4096-element BF16 row per INPUT TOKEN, then scatters
that same row into each local route's tile slot. It does not reread the input
row per route. Padding CTAs take a separate zero-fill path. The gather already
contains an unused eight-float reduction workspace and a metadata barrier.

A later fusion could calculate the norm once in that per-token gather, retain
the BF16 normalized row bits while scattering, and remove the upstream norm
call. It must use the existing precomputed BF16 effective(1+w) gain, retain
epsilon and BF16 output materialization, and leave all route positions,
zero-fill, tile ownership and replay publication unchanged. Passing raw weights
and multiplying by1+w in FP32 would not prove equivalent rounding. Keeping the
upstream norm as well would normalize twice.

The removable logical materialization is exactly one BF16 write and read per
token:4*T*4096bytes. This is12,337,152bytes atT753,49,348,608atT3012 and
268,435,456atT16384. These are requested bytes, not measured HBM traffic.
The intermediate may be served from L2, especially at small T. Arithmetic at
4.8TB/s gives2.57/10.28/55.92us; these numbers are neither a forecast nor an
exposed-time bound. Normalization arithmetic, gain loads and any new reduction
or register costs remain charged to the fused kernel.

Retained gather profile times are9.1..9.2us atT753,24.8..27.4us atT3012 and
126.3..132.9us atT16384. The top-kernel summaries do not uniquely attribute
the separate expert-input norm, and the entire gather is not removed by this
proposal. Kernel sums do not establish critical-path latency. No net gain is
established from these figures.

The extension uses --use_fast_math. NVIDIA documents that this enables FTZ
for single-precision denormals:
https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math
Adding ordinary floating-point normalization into the bit-copy gather therefore
requires an explicit subnormal/rounding review or instructions that preserve
the required behavior. This is a prospective fusion requirement, not evidence
of a numerical defect in the incumbent's bit-copy gather.

Prefill does not count toward the overall board. Decode's gather differs: it
distributes expert rows across persistent CTAs and intentionally batches input
loads to hide latency. Transferring this fusion would need its own complete-row
norm ownership and a cost model for duplicate routes, barriers, register use
and lost input-load overlap. The Prefill per-token layout alone does not prove
a profitable Decode integration.

Decision: park GPU evaluation pending a positive net-cost bound or justified
overall transfer. Preserve the source-bound screen and original sources. Do
not revive the failed DQ8/paired-epilogue transfer, MMA32 or CIT flags from this
traffic observation. Reproducer: screen_prefill_norm_gather_opportunity.py.
