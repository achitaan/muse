# Attention Decode checkpoint - October 28, 2026

Current corrected check: **2.522139036x**, job `1a16e2d658a8400d`, all16tests,
12benchmarks and integrity passed. Interval: [2.484306950,2.559971122].
Exact source: `best/attention_checked.py`; SHA256
`24782695e507801dcf4e753349a97de723d99e3463b2aac0c67a645ff53a884a`.
Scored confirmation `d0cdadc0960446ca` passed at **2.541743138x**,
interval [2.503616991,2.579869285], full16/12/integrity.
The older2.562402309x publication20524d06 used omitted BF16 boundaries;
withdrawal approval is pending. Read ATTENTION_PRECISION_AUDIT.md.
Do not repeat it for timing noise.

The candidate derives from Vjeux's Apache-2.0 implementation at
https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393.
Full license and local change notices are embedded in the source and preserved
in `THIRD_PARTY_NOTICES.md` and `licenses/vjeux-Apache-2.0.txt`.
The upstream leaderboard result is not a measurement of our candidate.

The native CUDA path combines Q/K preprocessing, cache writes, paged attention
and output normalization/gating. It traverses every visible key, retains causal
and sliding-window masks, and performs stable softmax reduction. The CUDA
prologue/finalizers now restore BF16 norm, RoPE and output materializations.
Dispatch uses public shapes and device data.

Local cache changes retain scratch per device and launch signature and reuse
immutable RoPE constants. The first attempt used stream identity in these
keys. Eager warm-up and graph capture then allocated separate buffers, causing
initialization to become replayed graph operations. Removing that distinction
raised the historical unfixed check from2.016854097x to2.589760797x. Those
numbers predate the BF16 correction and do not measure the current source.
Scratch is reset by
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


## BF16 boundary correction: full check passed

Corrected Attention24782695 passed16tests/12benchmarks/integrity as
1a16e2d658a8400d at2.522139036x, CI[2.484306950,2.559971122]. This is now
best/attention_checked.py. Exact-source scored confirmationd0cdadc0960446ca
and Decode transfer check9a176eaead1a44ec (source1fb5d05d) are pending.
See ATTENTION_PRECISION_AUDIT.md and results/attention-bf16-boundaries-20261028/.
Withdrawal of five affected historical publications needs explicit approval;
automatic review rejected it and no withdrawal executed. Do not claim that
historical unfixed scores belong to these corrected source files.


Corrected Attention publication d0cdadc0960446ca passed16tests/12benchmarks/
integrity at2.541743138x, CI[2.503616991,2.579869285]. It measures exactly
24782695, the source of the primary2.522139036x check. Historical20524d06
still belongs to the omitted-boundary source and remains on the board pending
withdrawal approval; do not report its2.562402309x as this corrected source's
score. Source, publication result and profiles are saved under
results/attention-bf16-boundaries-20261028/publication/.


## One-query FP8 QK trial closed at21:25UTC

Job60aa54a9feeb49e9/source8463eaf3 passed16tests/12benchmarks/integrity
at2.542655363x, CI[2.504515532,2.580795193]. The six changed L1 cases
had geometric ratio0.999962266 vs corrected247; unchanged L8 controls
rose2.3628%. Batch-one sliding ratio0.96775 violated the fixed0.98 floor.
The controller discarded it (trial91092e8, revert04741e8). No scored run
or Decode transfer. Retain corrected247, scored2.541743138x. One check
used; expected Attention12checks/7scores left. Full payload, profiles,
source and CPU audits, fixed guard failure and immutable controller history
are in results/resumed-attention-fp8q-20261028/. Do not rerun for noise
or retain a selector based on these small individual fluctuations.

Next inspect L1 PV: its helper loads FP8 V and expands32packed FP16
registers per32-key block, with16FP16 MMA operations. A register-only
4x4 byte transpose could feed8FP8 K32 MMAs without extra shared memory.
Keep the retained FP16 QK path. Before a candidate, audit P scaling,
PTX A/B/C mapping, every BF16 boundary and actual FA3 internal precision.
