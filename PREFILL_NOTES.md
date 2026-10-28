> Latest checkpoint: October28,22:53UTC. No pending job/controller. Goal active; top50unmet.
> Two checks this continuation, no scored runs. Prefill28d04382 passed5/8/integrity
> at1.680162079x but fixed normalized-profile guard failed1.004422(<1.02); reverted.
> MoEf5737777 passed5/6/integrity at2.348851666x; native65280gate audit passed,
> but aggregate declined vs2.365437416; controller discarded before guard. No Decode transfer.
> Retain MoE33114871(scored2.348299935x),Prefill93e57ab4(scored1.663469245x),
> Attention9e922c35(scored2.537000514x),DecodeD9checked1.990224880x/unscoredrepair.
> API limits: Dense0/0,MoE0/7,Decode1/4,Attention9/5,Prefill18/9(checks/scores).
> Total28checks/25scores; reset00UTC. Latest overall snapshot remains150/1447at22:34UTC.
> Next source inspection: dense_pinned_source_review.md; Dense cannot evaluate before reset.
> Conserve other tracks: no new candidate justified yet. Main chat only.

# Latest verified Prefill result

First full check passed at **1.639290758x**, with all 5 tests, 8 benchmarks
and integrity checks passing. Exact source is best/prefill_verified.py;
metadata and confidence interval are in best/prefill_metadata.json.
Scored publication 23e7c8b4c3e54869 passed at **1.663469245x**. Do not duplicate it.
The controller's local 1.5x target was reached; that is not completion of the
native overall Goal. This track is excluded from overall placement.
Full evidence and immutable phase journal: results/resumed-prefill-20261028/.

The preparation record below is historical; its unmeasured status has been
superseded by the full GPU result above.

---

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

Latest: Prefill scored job 23e7c8b4c3e54869 passed at 1.663469245x.
Current decoder fusion draft has passed bitwise codec/layout and Python
binding audits; GPU correctness and speed remain unmeasured. One check will
test 9..32-token fusion, preserving the complete FP32 small-token router.
Publish only if its checked interval excludes the retained interval.


## Read-only follow-up inspection at21:47UTC

Retain93e57ab4, scored1.663469245x. No additional Prefill evaluation used.
Cached primary profiles show persistent experts278.5us/44.4% atT753,
811.2us/40.3% atT3012,3807.6us/37.2% atT16384. Large dense projections
account for much of the remaining time. Norm/residual stages are each
about21us,78us and400us at these sizes, roughly6..8% combined. They are
likely bandwidth bound; audit the exact stream/router output byte traffic
before assuming a launch fusion removes their work. Prioritize an
algorithmic expert/dense improvement for a material Prefill gain. This
track stays last and does not improve overall placement.


Prefill zero-row unpooler proposal deferred without a GPU run. Public reference
has256experts and selects the busiest32-expert range; the earlier384/50%
assumption was wrong. Eight CPU routing proxies gave3.19..20.19%zero rows,
mean11.59%; these omit attention/gain/router calibration and are not actual
challenge measurements. Custom matmul benefit is unproven. Pooler/router
overlap already exists in this source. Next read prefill_small_expert_mma_review.md.


## Prefill follow-up, October28 at22:46UTC

Exact-decode/BF16pair transfer28d04382 passed full5/8/integrity as221b6741d74c4aa2 at1.680162079x, CI[1.654959648,1.705364511]. Fixed guard rejected normalized expert-profile GM1.004422088 (<1.02). Controller reverted; retain93e57ab4/scored1.663469245x. One check, no score. Source/audits/full result/controller: results/resumed-prefill-pair-dq8-20261028/. MMA32 and CIT remain disabled; no blind flag experiment. Next guarded MoE SiLU screen is CPU-audited, GPU unmeasured.
