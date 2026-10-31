# Router and empty-route ownership — read-only source review

Reviewed 2026-10-31 with the authorized researcher helper. No implementation or GPU job follows from these distinctions. These are source facts and unresolved hypotheses, not measured improvements.

## Retained Decode R2f

Binding: `experiments/decode_moe/decoder_pdli.py`, SHA256 `c15df9d3d83c55f4ba64be810e56502563e6cfe22bb5ef22172a1feb287654e5`, scored job `452fbd9df6af4ff4`. That job selected R2f at M1 and M8; M1 sliding additionally selected incoming-PDL mask14.

`experiments/decode_moe/r2f_v18s.cu:646` and instance764 launch 129 independent CTAs of 1024 threads, RED0/S1, without DSM clusters. CTAs0–127 each own32 pooler rows and router partials. CTA128 handles top-k only. In helper181–214, ks=blockIdx.x>>4 selects one1024-element K slice; nb=blockIdx.x&15 selects16 experts. All32 warps stage the router input and synchronize. Warps0–15 compute router partials and use a512-thread named barrier; warps16–31 can enter pooler work immediately. Partial same-CTA overlap already exists. Lower16 warps still execute routing before their pooler work.

The retained incoming-PDL R1 (`pdli-cpu/r1.cu:343`) preloads independent pooler weight/escape data before the dependency wait, then uses the same helper. Router input is the separately materialized FP32 normalized tensor; pooler consumes BF16 (`decoder_pdli.py:7460`). Preserve32 ordered FP32 FMA operations per lane, shuffle-down16/8/4/2/1, eight K-partial sums in ascending order, FP32 scale/sigmoid, biased score ordering with lower-ID ties and selected-raw-score normalization epsilon1e-15.

Router-only GEMV CTAs within this launch would change ownership; R2f's extra CTA only handles top-k. It is distinct from shared-input, Redux and PDL-preamble trials. However, simply adding128 CTAs creates257 CTAs with the same1024-thread resource allocation. Concentrating router units into fewer CTAs trades scheduling waves against serialization. Existing combined profile time does not establish a removable3.5-microsecond router tail. This direction is parked without a critical-path and resource argument.

## Retained MoE invocation-wide empty routing

Binding: `best/moe_verified.py` SHA256 `331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`, extracted CUDA `1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c`. CUDA line numbers below refer to that exact native source, also preserved in the existing `opus-fresh/results/moe-private-fixup-preflight-20261029/v4/baseline.cu` record.

TINY/TINY2 have no invocation-wide no-local early return. Route construction sets njF=njS=0 and pos=-1; FAST1 emits no units when its touched-expert mask is empty (2280–2319). Zero-job calls already omit all expert GEMM/weight work. They still perform helper joins, SS-bank maintenance, the grid barrier and masked finalization.

Helpers fork at2225, before routing at2352. A return inserted after routing would strand helper warps at WS_JOIN (1520,1594,2379,2418). Each CTA clears its share of the other SS bank, CTA0 clears the next generation counter (2365–2370), the grid barrier advances generation exactly once (2440–2447), and cleanup clears dynamic counters/doneA (2451–2457). Finalization masks invalid routes before SS/ybuf reads and writes positive BF16 zero (2520–2569).

A uniform decision before helper fork is not fundamentally forbidden, but skipping maintenance changes the retained scratch/generation protocol. A conservative alternative must preserve bank/counter clearing, grid arrivals/generation, cleanup, complete zero-output ownership and outgoing dependency behavior. It could avoid empty-phase bookkeeping; no exposed saving or actual benchmark incidence is established. One empty token is insufficient: every token must have no local route. Existing all-nonlocal correctness probes are evidence of behavior, not route-frequency or speed evidence. No GPU candidate is admitted.

Self-review: existing overlap and zero-job skips are explicit; precision and replay dependencies are retained; no probability estimate is presented as an observed frequency; no source changes or new execution claims were made.
