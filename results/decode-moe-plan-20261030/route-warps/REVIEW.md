# Tiny routing warp specialization — CPU screen, not GPU admitted

Baseline Python: `best/moe_verified.py`, SHA256 `331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`. Baseline CUDA: `1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c`. Candidate CUDA: `aec3724f6d3447d826b97616f5c64b8d8bda45071e72aa2ed6b8424ca0d017ed`.

## Mechanism and exact scope

The retained generic route builder uses eight counter rows, although E=8*M<=32 for TINY M1–4 and E<=64 for TINY2 M5–8. With the original CH formula, CH=32 in both regimes. This prototype initializes/scans/stores only one or two used counter rows. It keeps the eight-row allocation and every downstream layout unchanged. The C++ launch guard bounds M and nloc; unsupported capacities cannot enter an enabled private variant.

The first compute-group barrier orders only accesses within each warp: each warp writes/reads its own32 eid entries and its own wc row. It becomes a full-mask warp barrier. For TINY's single routing warp only, the next barrier also becomes a warp barrier. TINY2 retains that barrier because warp0 reads warp1 counts. The final two compute-group barriers remain, publishing job metadata before early_issue and entry/position metadata before expert work. Helpers, FAST1 preissue, global barriers, generation counters, scratch maintenance and arithmetic are unchanged.

CUDA13 documents warp-barrier memory ordering among participating lanes; matching/shuffle intrinsics alone do not provide that ordering. This prototype explicitly retains the needed warp barriers. [NVIDIA CUDA13 synchronization functions](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#synchronization-functions). This API contract supports the access argument, not a claim of race-free GPU execution or speed.

## Checks completed

`screen_route_warps_cpu.py` proves an exact textual inverse outside the routing edits and capacity guard, and checks that no other code reads wc. Its2880 CPU cases cover M1–8, three shard offsets, five local-expert counts, all-nonlocal, duplicate/skewed/alternating IDs and seeded random routes. Counts, offsets, stable sorted entries and inverse positions match the baseline; unused counter rows are poisoned. Exact metadata plus unchanged downstream job construction preserves its fat/skinny partition, rather than assuming duplicate routes cannot occur. This is a metadata proof, not validation of baseline support for every adversarial GPU input.

Eight paired NVRTC compilations (TINY/TINY2, PRMT/LUT, parent/candidate) succeeded. Registers remain146 for PRMT and141 for LUT; stack, spills, local instructions and static shared allocation remain zero. Compiled barrier instructions fall12→10 for TINY and12→11 for TINY2, matching the two/one removed compute-group barriers. Total instruction counts change5328→5312 and5352→5336 for TINY PRMT/LUT, and5376→5344 and5400→5368 for TINY2. These counts are static whole-kernel footprints, not dynamic cost or latency forecasts. No GPU job was launched.

## Historical comparison and admission decision

This differs from `tinyroute_source.py`'s TR8: it retains the existing match-any histogram and generic job construction, without the ballot-bitplane transpose or a new fat-job fallback branch. It therefore is not a literal retry of TR8. Nevertheless, the source-bound bounded TR8 job `7f34a396bd784aa1` passed its120 numerical fixtures but cleared zero of32 timing gates. Recorded M1 savings ranged0.7225–2.9957%; M4 ranged-0.5402–0.9818%; M8 regressed0.3817–0.5711%. These are the old TR8 measurements, not measurements of this prototype, and they are not independent replicated jobs.

Actual retained MoE benchmark cases in `norm-register/profile-9b9b978e.json` have M1,16,32,8,128,256. Only two of six scored loads can change here. Even3% lower cost in both yields only `(1/0.97)^(2/6) = 1.010205x` at track level. Decode transfer additionally depends on its own routing, overlap and whole-layer timing. Weight traffic, both GEMMs and norm work are unchanged. No current evidence quantifies routing time exposed on the load critical path.

Decision: preserve as a CPU-screened alternative; no runtime payload, GPU admission, score or Decode integration. Before reconsideration, require source-bound stage evidence that distinguishes its recoverable cost from the related failed TR8 design. A static resource pass alone is insufficient. If eventually measured, retain byte-identical inter/ybuf, the approved MoE norm/output limits, changing-route replays, PRMT/LUT coverage, both arm orders and the strict performance gate. Do not rerun it for timing noise.

## Other bounded public-source finding

Marlin's authors describe asynchronous global weight loads and double-buffered shared loads as key latency-hiding mechanisms. A speculative direct-global-load replacement would remove those mechanisms and needs its own dependency/resource model; this review does not adopt it. Their published results concern a different format/hardware setting and do not predict this H200 kernel. [Marlin primary implementation and technique notes](https://github.com/IST-DASLab/marlin#techniques).

Self-review: capacity and cross-warp publication boundaries are explicit; no shared-memory layout or required numerical boundary changed; exact source binding and all compiler logs are retained; GPU correctness and performance remain unproved; no evaluation quota was spent.
