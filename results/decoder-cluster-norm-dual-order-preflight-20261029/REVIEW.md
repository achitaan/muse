# Decode dual-order repair: one bounded check

Candidate1c23cdc6 repairs the setup-rejected f382e1af without relaxing numerical
or performance gates. All four modes and M1–32 boundaries remain in scope.
The exact packaged D9 module is its prefix; other component kernels and
unsupported fallbacks remain. Native source5d189e9a, combinedCUDA65b393c4.
This packet establishes no GPU speedup or better rank.

The rejected run failed an incumbent comparison before official tests or
benchmarks. Its actual input/M/kind were not recorded. Pinned scalar SGLang
source and finite reproductions identify changed accumulation/reduction order
as a hypothesis. Ordered5acf30a5 was withheld because it did not model the
retained tiny-token Triton FFN separately.

The exact retained tiny-FFN source was compiled offline with the official
[Triton3.9.0 wheel](https://pypi.org/project/triton/3.9.0/), explicit sm90,
eight warps and fusion disabled. Generated PTX has separately rounded squares,
32strided elements per logical thread, an adjacent balanced local tree, warp
XOR sums and an eight-partial final tree. Address/provenance analysis checks
six logical thread IDs. The evaluator's actual Triton version is unknown;
this is evidence for the named compiler, not evaluator binary identity.
Artifacts: ../decoder-sfn-offline-reduction-20261029/. WheelSHA256:
5e52ab4e2f1d64f0ac494c9aa723697b793f81a91c4cae4c36608ca0bc4267bf.
Extraction stays in .tools; no global packages changed.

CUDA distributes256original logical threads over four64-thread CTAs. Scalar
modes retain vec8/four-pack FMA chains; tiny FFN uses the generated strided,
non-fused tree. Both preserve full8192-element norms, BF16 post-gain and
normalized-branch materializations, FP32 residual/router values, separate
statistic banks and final DSM lifetime join. No moment shortcut bypasses
BF16 rounding. Implicit zero tail remains legal only at T<=8.

All four offline modes have56registers, zero stack/spills and16static shared
bytes. Sixty finite CPU cases match their respective models bitwise;
256DSM schedules pass, both lifetime mutants fail, and27guard mutants are
rejected. Host ABI/fallback and six fresh official identities pass. Hardware
rsqrt/div, scalar FMA contraction, native resources, replay and latency remain
GPU obligations. The old CPU guard artifact predates the portable kernel-source
binding; guard-CPU-admission-v2.json binds the final guard.

Unchanged gates: BF16output error<=1e-3, FP32residual/router<=2e-5;
36comparisons/eight changing-input replays; native<=64registers,0local,
16static shared,>=32clusters,>=132SMs,cluster4,64threads/CTA. Ten alternating
pairs per canonical boundary must show no individual>2% regression and
combined>=3% gain plus>=1000ns net saving per changed complete layer.
Full13tests/12benchmarks/integrity and new paths on indices0–7 are required;
8–11 remain controls. Reconstruct six loads with46sliding+16full weights.
Affected/control gains, per-case regression limits and track materiality
thresholds remain unchanged. Synthetic guard fixtures are not measurements.

Existing source-bound profiles show about8–9microseconds combined small-token
boundary time, not a measured critical-path fraction. A hypothetical net1us
saving per changed complete layer implies about0.514% track gain in the
recorded conditional model. Net means after DSM, launch and traffic costs.
No overall aggregation or rank forecast is inferred.

One paced check is admitted, earliest2026-10-29T17:03:03.179453UTC. Failure,
empty selection or insufficient materiality keeps D9. No noise retry, lowered
accuracy gate or partial-family harvesting. Publication requires complete
source-bound correctness and real material affected/control improvement.
The top50goal remains unachieved; quota is not a spending target.
