# Prefill preparation - October 28, 2026

Prefill is last in the requested track order. Its official TASK states it does
not contribute to overall decode-step placement. No GPU result exists yet for
the candidate described here; CPU/static evidence does not establish speed.

The first candidate retains the official SGLang layer's attention, router,
normalization modules and residual outputs. Only the expert implementation
changes to a persistent grouped NVFP4 kernel, derived from Vjeux's Apache-2.0
source at revision f35a41ee1c13e316dabbf772830d47b23b453393.

Original layer code normalizes expert inputs before calling the expert path.
The native gather therefore copies BF16 bits directly into its tile layout,
with the metadata publication barrier retained; it does not normalize twice.
Both expert GEMMs preserve Marlin's BF16 accumulator boundary, BF16 global
scale, and BF16 product before SwiGLU/post norm. Effective post-norm weights
come from SGLang in BF16; scaled norm values, reductions and weighted sums stay
FP32 until final output. All local routes are included; remote routes are
masked before output reads. Fresh route counts, positions and activations are
rebuilt on the GPU every call. Weights stay packed NVFP4 with original scales.

CPU audit (seed 20261028) inverted 1,048,576 codes and 65,536 scale bytes and
verified the complete gate/up row permutation. Static checks resolved all 108
global bindings. Source SHA256:
`93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7`.
Prepared builders/audits are under experiments/prepared/.

The initial scoped verification will measure this prepared candidate directly
against SGLang in the official paired evaluation, establishing this track's
first measured starting point. It does not repeat an untouched starter run.
The fixed verifier requires all 5 correctness tests, 8 benchmarks and integrity
checks; a separate guard checks the cached evidence without using another run.
Retain the source only as verified if that complete coverage passes. Do not
publish a speedup based on CPU checks or a public author's board result.

Conserve quota: one concrete hypothesis, no parameter sweep. Native launch
choices remain the public source's defaults. Its full-layer tuning, residual
recomputation and attention variants were not adopted.
