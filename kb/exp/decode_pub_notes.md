# Decode publication port

Base: `origin/argon/keell-base-20261102` (`f50a51d`), `kb/v1/decode_v1.py`.
Kernel hunk copied exactly from `loop/attn-pub` `42a3d35`,
`candidates/attention_pub.py`'s `_PUB_OLD_EXCHANGE` / `_PUB_NEW_EXCHANGE`.

Matching decode exchange: base lines 2648–2682, `segment`, called by
`attn_kernel`; L=1 non-CONT cross-warp handoff of two key halves.
`OKC_NOXCH1=0`. Shared K starts at 1024-byte alignment; its stages exceed
the 5120-byte requirement for 64 aligned 80-byte records (static assertion).

`ATT7_CUDA_SRC` and `ATT7_CPP_SRC` remain byte-identical. The separate
candidate TU has `OKC_DEC_PUB`, default 0, compiled with 1. Its only kernel
change: four float4 accumulator transfers and float2 m/l, 80-byte stride.
FP32 expressions/order, barriers, global reductions, split counters,
launch plans and PDL are unchanged. Production defaults to incumbent.
Inherited inline builds retain math flags but pin host/CUDA C++20 and
SM90a gencode through `_dp_load`. The obsolete non-selecting MFAB
write-flush diagnostic is disabled/omitted for prepare/file budget.

Gate: d1S/d16S/d32S; three independent seeded synthetic task-distribution
inputs with scored context capacities and private arm scratch. Each pass
has 16 balanced ABBA/BAAB whole-layer graph blocks for base/candidate and
independently captured base/base null. Restore new slots, then READ512
before every timed replay, both outside CUDA-event timing. Observe QKV,
attention, layer outputs, ids and KV stores. Deterministic attention is
bit-identical; otherwise relative-L2 and absolute error must fit observed
incumbent repeatability (no tolerance floor). Require exact new KV/ids,
all untouched cache bytes and scratch guards. Require both null statistics
in [0.995,1.005], every ratio/paired <1, median three paired <=0.995.
Incomplete/failed gates retain incumbent. Final `DEC_PUB_PICK` is authoritative.

Validation: seven CPU tests, AST, and forbidden-API scan pass; file <1 MB.
GPU build, correctness, performance and <285 s prepare remain UNTESTED:
no local torch/CUDA/nvcc. This lane submitted no check. Soft deadline 275 s
leaves cleanup margin; inherited prepare/in-flight GPU work is not forcibly
interruptible. Build uses one daemon thread, no process-starting APIs.

Check file: `kb/exp/decode_pub.py`.
