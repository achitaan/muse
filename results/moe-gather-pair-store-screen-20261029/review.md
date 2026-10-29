# MoE gather store screen

CPU evidence only, October 29, 2026. No runtime source was edited, GPU job
launched, score published or incumbent replaced. Reproduce with
`python experiments/prepared/screen_moe_gather_pair_stores.py`.

Baseline `best/moe_verified.py` is SHA256
`331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.
The exact CUDA and `wgs_gather` body hashes are recorded in `screen.json`.
Its original eight-row load batching, valid-row masks, 512 gather threads,
CTA ranges, job padding and release counter protocol define this screen.

## Mechanism and ownership

Each retained warp loads a contiguous input-row segment, then each lane stores
16 bytes at a 128-byte stride in the destination WGMMA activation layout.
An even/odd row pair within one eight-token block occupies complementary
halves of the same 32-byte sectors. Currently those halves are written by
separate warp store instructions.

Exchange lane bit 4 with the row register bit using four 32-bit XOR-16 warp
shuffles, plus conditional register moves. The first resulting store covers
both rows for source lanes 0..15; the second covers both rows for 16..31.
For that pair, the abstract distinct-sector count falls from 32+32 to 16+16.
The byte payload, number of 16-byte stores and input loads do not change.

The CPU test verifies 16,384 layout cases across all counts 1..64 and all
16 consumer warps: 1,048,576 unique uint4 destination/payload checks. It
detects a wrong shuffle-distance mutant. Boundary fixtures include empty
work, odd CTA starts, padding, short tails and split fat jobs. Pairs never
cross an eight-token group, CTA range or the existing eight-row load batch.
Other rows retain the original stores. No FP32 or BF16 arithmetic changes
are proposed; this is an opaque bit rearrangement.

## Coverage and limits

The existing 192 synthetic stress draws supply job counts. They select the
busiest 32-expert shard and are not actual evaluator routes or an unbiased
distribution estimate. At fixed grid 132, their median store-sector-request
ratios are:

| M | Normal | Skewed |
| --- | ---: | ---: |
| 32 | 0.9900 | 1.0000 |
| 128 | 0.7186 | 0.6952 |
| 256 | 0.6476 | 0.5968 |

M32 has almost no useful pair coverage; do not change that path on this
evidence. M128/M256 have a distinct possible mechanism from the closed
occupancy, scale-compression, cache-priority and BF16 factor-cache trials.
The register-only transform adds no shared-memory buffer or global payload.
It does add shuffles, selections, address/predicate work and temporary
register lifetimes inside a resource-sensitive persistent kernel.

The sector model follows NVIDIA's documented warp-wide coalescing model:
[CUDA Best Practices 13.0](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory).
This model does not establish generated 128-bit stores or hardware counters.
Sectors, requests, serialized L1TEX wavefronts and HBM bytes are different
quantities; a sector reduction does not imply a matching wavefront reduction.
See [Nsight Compute's definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#quantities).
Later cache merging can reduce any downstream benefit. No timings, bandwidth
gain, resource admission, CUDA race proof or speedup follow from this screen.

## Decision

Keep this as a source-bound offline hypothesis. A separate CUDA implementation
would still need inverse-edit binding, uniform shuffle participation and masks,
compiler/resource evidence, unchanged readiness publication and replay tests.
Do not allocate a check until the exposed gather/store cost and plausible
whole-layer saving justify the extra instruction and register work. Existing
phase diagnostic timestamps missed resource admission and cannot supply that
cost fraction. The MoE and Decode verified sources remain unchanged.
