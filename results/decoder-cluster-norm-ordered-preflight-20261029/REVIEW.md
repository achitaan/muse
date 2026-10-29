# Decode reduction-order repair: preparation only

The eleven-hour audit found nine private checks and zero retained performance
wins. The subsequent cluster-boundary check 63acbd4d99324fea failed its setup
comparison before official tests or benchmarks. This packet establishes no
speedup, accuracy verdict on H200, or rank improvement.

Candidate 5acf30a5 preserves the packaged D9 source as an exact prefix and
adds boundary wrappers. Its native source 49bf73e6 distributes the original
256 scalar-kernel logical threads across four 64-thread CTAs. It preserves
vec8/four-pack accumulation order, the eight original warp partials, separate
statistic banks, and the final distributed-shared-memory lifetime join.
Required BF16 branch materializations and FP32 residual arithmetic remain.

Offline compilation reports 56 registers and zero stack/spills in each of
four semantic modes. Sixty finite CPU arithmetic cases match the modeled
scalar reference; 256 DSM schedules pass and both lifetime mutants fail.
The separate result guard accepts a synthetic positive fixture and rejects
27 faulty fixtures. These are CPU evidence, not GPU measurements. Native
resources, graph replay, hardware rsqrt/FMA behavior, and latency remain
unverified. The actual failed GPU input and its M/kind were not recorded.
The new comparison records that context and adds small/zero-branch probes.
The numerical and paired-timing gate functions are unchanged.

**Launch is withheld by an explicit driver admission gate.** The retained
tiny-token FFN uses `_sfn_kernel`, a Triton sum of separately rounded squares
with fusion disabled. It does not use the pinned scalar SGLang reduction
modeled here. Matching the scalar tree therefore does not justify claiming
agreement with that retained path. Resolve its reduction mapping or narrow
the candidate to leave this path untouched before a GPU retry. Triton is
not installed in the existing WSL Python; no packages were installed and no
global environment changed.

All six freshly downloaded official task files match the pinned copies.
No candidate was installed, no controller started, and no GPU job launched
for this repair. D9 and every other retained source remain unchanged.
The next permitted check time is 2026-10-29 17:03:03.179453 UTC; that time
permits consideration of a justified check, rather than requiring one.
The overall goal remains active and unachieved.
