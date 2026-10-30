# TR8: exact single-warp small-route construction

Baseline: retained `best/moe_verified.py`, SHA256
`331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.
Its CUDA is byte-identical to the current MoE board artifact's CUDA. Decode's
FINREG source differs, so a successful standalone result still needs a separate
source derivation and whole-layer gate before transfer.

The generic routing prologue reconstructs counts, offsets, jobs, entries and
positions in every CTA. For M<=8 it uses at most two route-containing warps,
but initializes/reduces all eight warps' counters and uses four compute-CTA
barriers. The candidate transposes expert IDs into five ballot bitplanes in
warp zero. It computes the same histogram, scans offsets in expert order,
emits jobs in that order and uses match masks to retain stable route order.
Only one compute-CTA barrier is needed to publish the complete tables.

This changes control work and shared-memory traffic, not weight traffic or
expert arithmetic. There is no extra timed kernel, global scratch, allocation,
weight layout or persistent expanded representation. The full generic route
path remains for M>8, split-job builds, WGS, nloc>32 or counts above eight.
FAST1's independent preissue remains unchanged; its `early_issue` function is
already a no-op. The helper warp joins after the route tables and work ranges
are ready. Completion arrivals, generations and both GEMMs remain original.

For a valid route i in chunk b, its output position is the sum of counts of
smaller experts, plus that expert's count in earlier chunks, plus the number
of earlier equal-expert lanes in this chunk. This is stable counting sort.
Each local route writes a unique position in [0,sum(counts)); nonlocal
positions are -1. All warp intrinsics execute with all 32 named lanes.
All table publication precedes the compute-CTA barrier. There are no new
cross-CTA dependencies and no host routing reads in timed execution.

CPU evidence: 2,911 route tables exactly matched an independent stable-sort
oracle; 737 oversized histograms correctly requested the generic fallback.
Cases span M1..8, four shard starts, local counts 1/13/32, unique top-k rows,
all-remote, repeated slots, heavy skew and boundary experts. Eight NVRTC
SM90 screens cover parent/candidate x TINY/TINY2 x LUT/PRMT. All passed with
zero stack/spills. Candidate registers are 146 LUT / 148 PRMT versus parent
141 / 146. This does not prove GPU correctness, occupancy or profitability.
No CUDA arithmetic was changed; the original Python AST is identical except
the CUDA literal, before appended wrappers. Exact inverse derivation is checked.

Evaluation is one fixed candidate, not a sweep. Each M1..8 gets graph-replay
byte checks on five routing populations and three changing inputs. All-remote
tokens have NaN hidden rows and must still produce finite zero outputs. Only
bitwise-equal candidates faster by >3% in BOTH arm orders for BOTH one-local
and skewed route populations may be selected. Timing uses the existing shared
cold-L2 graph harness, 25 observations/order; those observations are correlated
within one GPU job, not independent experimental jobs or a significance test.
All compilation completes before capture, and probes stop at 278 s.

The optimistic removed work is three CTA barriers and generic table bookkeeping.
Its exposed fraction is not measured. For two changed standalone loads, the
track ratio is (G1*G8)^(1/6); an isolated score point is not proof of overall
Decode improvement. No publication follows empty selection, unchanged controls,
failed numerical gates or an inconclusive improvement. A Decode transfer must
also account for interaction with the accepted PDL tail overlap.

Official task/reference/checker contents were refreshed and matched the pinned
copies after newline normalization; see `official-identity.json`.
Warp collective semantics: NVIDIA CUDA 13.0 Programming Guide, sections
[warp vote](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-vote-functions)
and [warp match](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html#warp-match-functions).
These support the mechanism, not its predicted speed.
