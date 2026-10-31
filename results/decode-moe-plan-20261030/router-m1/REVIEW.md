# Retained R1 one-token specialization — CPU screen

Baseline Decode: `decoder_pdli.py`, SHA256 `c15df9d3d83c55f4ba64be810e56502563e6cfe22bb5ef22172a1feb287654e5`. Its pinned R1 native source is `fa82410b1833f245d3a036e442b951f5405a50cef84822c1094b0dc5c8d38571`. Candidate native source: `8cba8791a059fd4f821d41ad8d581594fe60eee4303b5de767661d2e54fa920e`.

The separate-router-CTA experiment is closed by its30–34% regression. This candidate returns to the retained129-CTA layout, original16 router warps per pooler CTA, original top-k CTA, counter protocol, dynamic shared allocation and PDL preamble. It introduces no extra scheduling waves.

The retained M1 entry instantiates the pooler template with CM=0 and reads runtime ra.M in both router helpers, supporting up to eight tokens. This prototype sets CM=1, binds both helper M values to1 and sizes the router's shared input slice for one token. The existing host `(CM && m != CM)` guard rejects M!=1 before launch; non-M1 use requires the retained backend. The host constructs ra.M from the same m argument. No arithmetic expression, weight, K partition, FMA/shuffle order, selected-expert rule or precision boundary is changed in source. An exact textual inverse restores the parent.

The source index argument covers258 host M values, all1024 shared input positions and2048 unique router partials. With M=1, the copied slice and every j*32+lane read stay within0..1023. Partial addresses retain the original8*8*256 allocation; unused rows remain untouched. This argument does not establish GPU races or byte equality.

Paired CPU NVRTC compilation for sm90a passes. Registers fall64→57, static shared bytes32768→4096, total static instruction count3888→2984. There are no stack/spill/local instructions. All-width shared-load instruction counts are272→64; the original dot-qualified-only counter missed plain32-bit LDS instructions and was corrected from raw SASS. Global-load instruction count remains114. These are static counts: many removed multi-token paths were masked at M1 already, so neither instruction reduction nor shared allocation is a latency forecast or measured traffic reduction. Register capacity still bounds both1024-thread kernels to one resident CTA per SM, before other limits. [Hopper occupancy constraints](https://docs.nvidia.com/cuda/archive/13.0.2/hopper-tuning-guide/index.html#occupancy).

The current packet is CPU-only. Before one paced joint-unit diagnostic, it needs a new source-bound payload with a one-token adapter, matching actual-resource guards, byte probes, nearby-shape fallback tests, role-specific PDL/SASS audit and the same strict timing controls/raw observations. The closed193-CTA payload must not be reused unchanged: its resource expectations and M8 direct launch are incompatible with this candidate. No GPU job, score or Decode transfer is admitted here.

Self-review: verified parent binding, source inverse, host shape guard, bounds and compiler outputs. Fewer registers do not increase residency here. All verified incumbents remain unchanged. Next normal owned GPU-check eligibility remains2026-10-31 04:41:30.304784 UTC; a ready candidate and fresh admission are still required.

## Packaged qualification — before any GPU collection

Runnable source `dense_router_m1_diagnostic.py` is bound in `payload-binding.json`: SHA256 `9bd760ff0696ba59464e6d8b19a1f3a4920b261d981b0da6b49fde6ca4ea800e`,717903 bytes. Packaged candidate CUDA is `c2c6cba7e6fa6451891849bf6d254d0e2e8d34ea5fe1a6805ce77846366f107c`. Original public Dense prefix/delegation is retained; it is a joint-unit preparation diagnostic with actual Dense weights, not a Dense board improvement or a Decode integration.

The private `_M1Extension` forwards all22 native arguments unchanged. It chooses the specialized module only at M1 and the retained module for every other shape. All258 selector cases and40 public delegation cases pass CPU checks. The adapter owns its references and does not edit loaded extension modules. Packaged parent/candidate device instructions exactly match their screened versions after the candidate kernel rename. Source-bound SASS checks retain the outgoing trigger, both dependency waits, four independent constant-weight loads before the pooler wait, and coherent input loads after it. No GPU behavior is inferred solely from instruction addresses.

The new resource guard expects parent64 registers/32768 static shared bytes and candidate57/4096; both require zero local bytes,16384 dynamic shared bytes, one block/SM and at least two SMs. It does not reuse the closed193-CTA candidate's resource expectations or unsupported M8 launch.

Freeze20 changing-input graph probes across M1,8,2,4: zero, normal, last-column spike, negative zero and tiny values. Compare raw pooler output, router weights/IDs, full partial buffer and counter bytes, with unchanged-parent repeat controls. M1 must execute the actual `m1_gemv` marker; nearby shapes exercise the explicit retained-module fallback. No MoE tolerance exception applies.

Timing remains the fixed M1 normal/spike design: both arm orders,25 observations per arm, all four candidate costs below0.97 times parent, then both unchanged timing controls within2%. Preserve raw unsorted event observations. These are repeated observations in one job, not independent jobs or significance estimates. Preparation cutoffs remain268/274/278 seconds. Failed correctness/control/resources/timing closes the candidate without noise retries or a parameter sweep. A positive unit result still requires a credible whole-layer opportunity before a Decode check; no score or transfer is automatic.

Final CPU self-review: immutable old runtime is hash-bound before adaptation; generated Python parses below1MB; ABI, shape fallback, exact device binding and PDL/source checks pass. GPU host build, actual resources, byte equality, marker and latency are still pending. GPU jobs used by this packet:zero; fresh quota/queue/identity/spacing admission is mandatory.

## GPU outcome — closed

The human removed fixed60-minute spacing on2026-10-31: "Wait don't follow60m spacing just do when you think ur fine". Fresh queue, quota and five official files were checked before admission. Job `8efb0121db7848c2` completed on H200:9 tests,30 benchmarks and integrity passed; preparation99.248623382 seconds. Actual parent/candidate resources matched the frozen guards, all20 changing-input byte probes passed, and the `m1_gemv` execution marker appeared.

Four parent/candidate median costs in microseconds were25.184/25.376,25.408/25.024,25.280/25.120 and25.344/25.184. Candidate/parent ratios are1.007624,0.984887,0.993671 and0.993687. None passes the predeclared0.97 gate. Unchanged timing controls were conditional on all four primary passes and were therefore not run. Raw25-observation arms remain in the saved log and outcome JSON.

Close without a noise retry, score or Decode integration. The official public Dense path remains the incumbent; its1.502905751x point is not a candidate gain. One Dense check used; fresh quota is16 checks/10 scores, with20 Decode checks still available. Verified incumbents remain unchanged.
