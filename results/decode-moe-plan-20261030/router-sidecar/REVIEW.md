# Joint pooler/router scheduling — one fixed diagnostic

The current Decode source is `decoder_pdli.py`, SHA256 `c15df9d3d83c55f4ba64be810e56502563e6cfe22bb5ef22172a1feb287654e5`. Its literal `_PDLI_EXPECTED['r1']` binds the retained R1 native source `fa82410b1833f245d3a036e442b951f5405a50cef84822c1094b0dc5c8d38571`. The source-verified prototype is `5baeea7f74b411a4998713053de3c2c75d9e6bd7d49e1095eec609a5cf8b8fd0`; packaged hashes and the exact runnable payload are in `payload-binding.json`.

## Hypothesis

R2f already overlaps router and pooler work within each CTA. Its lower16 warps perform routing before their pooler work; upper16 start pooler earlier, and the pooler reduction joins the K partials. This prototype gives all128 pooler CTAs their original pooler work immediately after the retained PDL preamble/wait. It moves router work to64 router-only CTAs and retains one top-k CTA. There is no extra kernel launch or duplicated router-weight matrix traffic. New scheduling waves, CTA setup, bandwidth contention and later outgoing-PDL eligibility can negate the benefit. The combined profile duration does not quantify a removable router stage.

Fixed geometry: pooler IDs0–127, top-k128, router IDs129–192,1024 threads per CTA. Router CTA b computes ks=b>>3 and32 consecutive expert rows `(b&7)*32+warp`. Its two16-warp halves exactly reproduce old CTAs `2*b` and `2*b+1`. One shared input slice serves both halves. Every FP32 FMA and shuffle expression, partial address and top-k expression is retained. All32 writer warps fence before a1024-thread barrier, then one leader publishes two arrivals. The unchanged top-k threshold remains128 and resets the counter after all units arrive. No scheduling-order assumption is used for correctness; only top-k waits on this grid.

The helper uses the separate FP32 normalized router input. Pooler input/output remain BF16. There is no BF16 approximation to the router and no use of the MoE variance exception.

## Completed CPU evidence

Two paired device compilations cover the prototype and its packaged kernel-name variant. Parent and candidate use64 registers,32768 static shared bytes, zero stack/spills/local instructions. At1024 threads,64 registers consume the Hopper SM register capacity; actual occupancy must still be queried on H200. [NVIDIA Hopper occupancy limits](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/index.html#occupancy).

The integer model covers2048 unique K-slice/expert partial outputs and131072 sampled address identities; the unchanged loop expressions establish the remaining addresses. A200-invocation model alternates M1,8,2,4 and performs12800 counter observations, preserving stale unused partial rows. These are ownership/protocol models, not GPU race or floating-point tests.

The source-bound SASS audit identifies dispatch, each role's dependency wait, coherent router-input loads after its wait, writer fences, the1024-thread publication barrier, and atomic arrival. The packaged candidate's instruction sequence matches the screened prototype exactly after its private kernel rename. Forty mock public-call cases delegate directly to the retained Dense function. The22-argument native ABI and nonzero-error handling pass CPU contract checks. Host NVCC/linking is not reproduced by NVRTC.

Incoming and outgoing PDL behavior still needs actual graph validation; concurrency is not a guarantee. [CUDA programmatic dependent launch](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization). Additional CTAs can delay the point where every producer CTA has announced readiness.

## Evaluation design frozen before collection

Use one unscored Dense check as a joint-projection diagnostic, with the actual Dense pooler/router weight bank. The public Dense API requests one projection at a time; its public calls continue to use the exact retained Dense source `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b`. No Dense board gain or selected public fast path will be claimed. Private unit evidence alone cannot establish a complete Decode improvement. This preserves Decode checks until a component gain is credible.

Preparation builds both extensions synchronously before capture, queries actual resources, and requires64 registers, zero local bytes,32768 static/16384 dynamic shared bytes, one active block per SM and at least two SMs. Missing exact plan/FP32 weights or failed resources stop the private test. Original weight bytes are unchanged. All synthetic inputs, outputs, counters and partial buffers are private; no actual benchmark activation is cached or precomputed.

Require twelve changing-input graph probes across M1,8,2,4, with an unchanged-parent repeat control. Compare raw bytes for pooler output, router weights, router IDs, the entire FP32 partial buffer and counter; counter must be zero after each replay. Include zero inputs with tied router scores and a last-column spike. The private `sidecar_gemv` marker must appear in a real CUDA profile.

Measure M1 only, normal and spike populations, both arm orders,25 cold-L2 graph observations per arm through the shared preparation harness. All four candidate comparisons must cost less than0.97 times parent. If they pass, both unchanged-parent timing controls must be within2%. Preserve unsorted event observations. The shared harness only adds an optional collector; its default medians/order/event-unit conversion remain unchanged and CPU-tested. Replays within one job are not independent jobs and these thresholds are not significance claims.

Preparation cutoffs are268 seconds before probes,274 before profiling,278 before timing groups. No other parameters or shapes are tuned. A numerical/control/resource failure or failed timing gate retains the baseline. There is no timing-noise retry. Any setup repair needs a concrete diagnosis and a separately recorded bounded decision. No automatic score or Decode transfer follows even from a passing unit test; the whole-layer critical-path gain and conformance must justify its own check.

## Self-review

Source inverse, retained arithmetic, publication ownership, role-specific waits, resource limits and unchanged public delegation checked. The exact current official Dense TASK/AGENTS match local pins. GPU numerical equality, actual host build/resources, runtime markers and performance remain unproved until the diagnostic finishes. The source and raw evidence are preserved without changing verified incumbents.
