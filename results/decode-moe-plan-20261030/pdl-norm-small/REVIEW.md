# Lean early-norm packet: T1/T8 exact FP32 router

CPU-only draft: `experiments/decode_moe/decoder_pdls_draft.py`, SHA256 `85a0ae65ee872d63fc9ca707883b9c9a5d2d1c9c8c23808ae4bab6578b7d818f`, 953,028 bytes, 46,972 bytes below the source ceiling. No H200 job, admission, score or publication.

The complete scored parent `c15df9d3` remains a byte-identical prefix. The earlier four-kernel chain and M16/32 private-norm drafts are preserved separately. This draft adds only the HasRouter=true SDK host instantiation and an owned norm module; it adds no projection build. Its CUDA body/query is `f24c9d33e6be7f433a8c9629019f66f250b569f4cd235fd80b098499c399795e`.

## Exact scope and mechanism

For B1/L1 and B1/L8 only, the private scalar norm sends its outgoing PDL signal before its incoming dependency wait. Every input read remains after the wait. The successor is the retained incoming-PDL R2f kernel, whose weight/escape prologue precedes its wait and whose router/activation reads follow that wait. The norm still performs every reduction, FP32 residual update, FP32 pre-norm input operation, BF16 output store and separate FP32 router-output store in the original source. It does not add a BF16 pre-norm input round.

The scalar source derivation has an exact inverse. The host call uses the original 12-argument interface with post_weight=None, the original cached BF16 pre-weight and FP32 alpha/beta, original epsilon values, pre_norm_bf16=False and an FP32 router output. The public pinned adapter independently selects H8192/256 threads/prefetch/cluster1 for both rows. Eight installed header identities, including host launch/runtime support, are required; mismatch closes the new path. The upstream source and Apache-2.0 license are preserved in `../pdl-norm/upstream/`.

## Gates and fallback

All original MoE, projection and router functions are unchanged. All strict byte/performance gates are unchanged. The norm/R2f unit compares both arms with the SAME incoming-PDL R2f, isolating the outgoing norm signal. It requires byte-identical BF16 norm output, FP32 residual/router, pooled output, route weights and IDs across three changing input replays, plus >3% in both cold-L2 arm orders. One combined whole-layer candidate then inherits the parent's already-selected paths and adds this norm/R2f pair. Complete outputs and both FP8 cache byte arrays must match, with >3% in both arm orders. There is no mask sweep.

Unsupported shapes/strides/dtypes/alignments use the original norm. Unsupported public shapes use the entire parent path; unadmitted private calls raise. The private compiler joins before the parent's first selection capture. New diagnostics start only before 274 s; shared timing refuses work after 278 s. Actual build contention and <300 s preparation remain unverified.

## CPU evidence and limits

- SDK argument and tensor metadata audit passes at M1/M8, with FP32 router and pre-norm input. Three unsupported-shape/stride fallbacks, an unadmitted call and seven public dispatch sentinels are covered.
- The complete layer clone reverses exactly to the parent's layer by undoing its name and one norm call. The parent prefix and MoE/projection/router/strict-guard ASTs match.
- Exact scalar-body/vector-storage/CTA-reduction device proxies compile for HasRouter=true. Both use **175 registers**, 64 bytes static shared memory and zero stack/spills; floating-point conversion/math and shuffle instruction counts match.
- Parent trigger is at SASS address 23920, after its last store at 23904. Early trigger is at 208, dependency wait at 224, first global load at 288. This establishes proxy ordering, not actual SDK/H200 code generation or runtime correctness.

The initial CPU ABI recorder omitted the public SDK's run_std_router method; its stub was corrected and the final audit passed. No GPU retry was spent.

## Opportunity screen

The source-bound R2f instances at M1 and M8 each have ST=1, RT=1, CWR=2, CWK=16 and three int4 weight loads per lane. Across 128 projection CTAs this is 6 MiB of independent logical packed-weight requests, plus 512 KiB of logical escape requests (with reuse). These counts are not measured HBM traffic or hidden time. The extra top-k CTA waits before dependent reads and contributes no weight prologue.

The user-supplied older R2f profile reports roughly 4.4–4.8 us for the stock norm. Earlier signaling exposes some of that interval, and possibly a predecessor tail, to the consumer; recoverable work is limited by the actual consumer prologue and scheduling. No same-SM co-residency, bandwidth-floor speedup or >3% whole-layer result is assumed. Standalone unit evidence would not by itself establish a material aggregate Decode gain.

Readiness is therefore CPU proof only. Actual SDK compilation, H200 resource queries, replay/cache bytes, causal norm/R2f timing, whole-layer latency and preparation headroom remain unresolved. The last Decode check remains reserved. Existing MoE nondeterminism is not an exemption from this draft's byte gates.
