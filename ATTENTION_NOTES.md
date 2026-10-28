# Attention Decode checkpoint - October 28, 2026

Best check: **2.589760797x**, job `ea693135be544501`, all 16 tests,
12 benchmarks and integrity checks passed. Interval: [2.550914385, 2.640640588].
Exact source: `best/attention_checked.py`; SHA256
`b56e07b1dbc930ef8fd4adaeba108e6d17b62a0f390d68c024b98d3cd608aee4`.
Scored publication `20524d06e3894595` passed at **2.562402309x**.
Do not repeat it for timing noise.

The candidate derives from Vjeux's Apache-2.0 implementation at
https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393.
Full license and local change notices are embedded in the source and preserved
in `THIRD_PARTY_NOTICES.md` and `licenses/vjeux-Apache-2.0.txt`.
The upstream leaderboard result is not a measurement of our candidate.

The native CUDA path combines Q/K preprocessing, cache writes, paged attention
and output normalization/gating. It traverses every visible key, retains causal
and sliding-window masks, and performs stable softmax reduction. Native CUDA
algorithm payloads are unchanged. Dispatch uses public shapes and device data.

Local cache changes retain scratch per device and launch signature and reuse
immutable RoPE constants. The first attempt used stream identity in these
keys. Eager warm-up and graph capture then allocated separate buffers, causing
initialization to become replayed graph operations. Removing that distinction
raised the checked result from 2.016854097x to 2.589760797x. Scratch is reset by
the native kernel. Same-signature calls require sequential replay; this source
does not provide independent scratch for concurrent calls sharing a signature.

Earlier work and negative evidence:

- Original fused preprocessing: check `89f9f7d227424d20`, 1.120441329x;
  scored `f6b899f674914ef2`, 1.116457695x.
- Custom partitioned online softmax: `cdc1670c333d4ceb`, 0.699554216x,
  correct but slower. Serial partition loops were expensive.
- Adaptive split count and shorter key chunks: `449fd95cf481493a`,
  0.825614005x, correct but still slower. Rejected without parameter sweeps.
- First native implementation with per-stream initialization:
  `2885135a42f441ca`, 2.016854097x, correct; the cache correction above retained.

Evidence, exact trial sources, profiles and archived controller history are in
`results/resumed-attention-phase2-20261028/`. The previous phase remains in
`results/resumed-attention-20261028/`. The phase is archived to allow publishing
the checkpoint; the overall native Goal remains active and top 50 is unmet.

Use fewer checks and scored runs, per the user's latest request. Each new run
must resolve a concrete question with a material expected gain. Reuse cached
baselines and job handles. A small timing fluctuation is not a reason to repeat
a check or spend another scored submission. Prefill remains last; its official
TASK states it does not contribute to the overall board.
