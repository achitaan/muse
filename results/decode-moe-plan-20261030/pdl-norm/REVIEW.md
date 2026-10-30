# Private stock norm with an earlier outgoing trigger

CPU-prepared hypothesis; **no H200 validation or speedup** belongs to this
draft. The scored Decode source remains `c15df9d3`, 2.075466170x.

Draft `experiments/decode_moe/decoder_pdln_draft.py`, SHA256
`7f5a03dd52e9c888fe7d9854c4acbe96b0b79bca90806283d0c49a58f77ca671`,
991,647 bytes (8,353 headroom). It retains the complete scored parent prefix.
The preceding O/FPR/unpooler CPU draft `985abda3` is preserved independently.

## Mechanism and exact scope

Inspection of the pinned stock scalar norm shows its PDL trigger **after all
output stores**. Enabling the next pooler's PDL flag alone therefore has only
the remaining grid tail to overlap. This draft moves that trigger before the
norm's incoming dependency wait. It does not move the wait or an activation
read; the pooler can begin only its independent weight preamble until its own
wait confirms complete norm output visibility.

The private norm is derived from SGLang commit
`a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f`. Its body, 256 logical threads,
eight-warp XOR trees, vector layout, FMA sequence, BF16 post-norm round trip,
FP32 residual and output stores remain source-identical. Exact inverse edits
change only the scalar trigger position and private scalar-kernel/host-class
names. The unused cluster path remains unchanged. This is distinct from the
rejected cooperative/cluster reduction and warp-count experiments.

At import, a background build reads the installed public SDK header and six
helper headers, requiring their exact canonical source hashes. It derives an
owned temporary header and uses the SDK's documented `load_jit`/wrapper API,
with its normal target/compiler flags. Installed files, loaded modules and
global environments are not patched. The new C++ builds join while weight
preparation is still on the core path, before the parent's selection captures.
Actual host/CUDA/TVM linking and prepare <300 seconds remain unverified.

Only the M16/M32 FPR pair uses this norm. Bit 2 now denotes **private norm plus
FPR**, not an independently timed norm choice. The paired unit probe compares
normalized input, FP32 residual, pooler output, logits and routing bytes.
All candidates retain the whole-layer output/cache byte guard and >3% in both
interleaved cold-L2 orders. No mask or norm-geometry sweep is added.

The adapter preserves `pre_norm_dtype=None`: **FP32 input to the pre-norm**,
with the original final BF16 output. It does not invent an extra BF16 round.
It reuses the parent's resolved BF16 pre-weight and FP32 residual coefficients.
Unsupported dtype, shape, stride or alignment calls the original norm. The
small-token exact FP32 router, original O/U paths and q_len fallbacks remain
unchanged. TR8 is not integrated.

## Evidence and limits

Public source/header and JIT loader/spec copies are in `upstream/`. The SDK
supports absolute owned CUDA source paths and generates the same typed wrapper
signature. The CPU adapter audit executes the pinned public Python adapter
with recorded C++ calls and real CPU tensor metadata: M16/M32 select
8192/256/PDL/prefetch/cluster1, with identical 12 arguments and false
pre_norm_bf16. Stride fallback and preparation-join order are checked.

The scalar device proxy preserves the full arithmetic/vector/CTA body and
uses selected CUDA intrinsic adapters for host-heavy SDK types. Both parent
and early versions compile at 175 registers, 64 static shared bytes, zero
stack/spills, with identical arithmetic/shuffle instruction counts. Parent
trigger address 23792 follows final store 23776; the early trigger at 208
precedes wait 224 and first global load 288. These are proxy compiler screens,
not actual SDK/H200 resources; separate runtime queries must prove zero local
memory and positive residency.

The FPR prologue logically requests 15 MiB / 9 MiB of packed weights at M16/32
with its existing depths 5/3, versus 3 MiB for the parent's initial stage.
Those are requests, not measured HBM traffic or guaranteed hidden work. The
register model does **not** fit norm and FPR together on one illustrative
65,536-register SM budget. Any overlap depends on other available SMs and
tails; no concurrency or speed is assumed.

The layer and selector edits round-trip exactly to the prior CPU draft;
public dispatch, O, U and FPR function ASTs are unchanged. Full-context >3%
still requires about 6.4/9.4 microseconds saved. New source identity,
`PDLN_RESOURCE`/P16 selection markers, private-kernel execution, all official
tests, prepare headroom and measured affected/control performance are required
before any publication. The final Decode check remains reserved.

Upstream sources: [norm implementation](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/csrc/muse_spark_v12/norm_residual.cuh),
[JIT loader](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/utils/compile/loader.py),
[wrapper specification](https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/jit/utils/compile/spec.py).
