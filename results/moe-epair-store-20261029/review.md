# Paired-store check: resource-query compilation failure

Official check98e3538e85244c1d, sourcee8e3086d, is terminal. Setup failed while
compiling the new extension. The diagnostic resource query refers to namespace
`okc_epair_y`, but the actual new kernel uses `okc_epair`. Captured errors are
in query lines2721..2730. No new-kernel correctness test, benchmark, resource
attribute or performance result was produced. Integrity screening passed.
The controller's zero metric represents a failed check, not a measured0x speedup.

This was an avoidable source-wiring defect. The existing CPU layout tests did
not validate the query's namespace. The result does not test the performance
hypothesis or justify changing its numerical/resource/performance thresholds.

Controller run9997790027004ae09ded739524755f9a committed trialbadb91b, then
discarded/reverted it asfe81752. Original33114871 and all other protected sources
and metadata are unchanged. Controller history and raw result are preserved.
The earlier TLS upload failure created no job or quota charge; this accepted
check used one check and zero scores. Derived balances are MoE15checks and
93checks/50scores across all tracks, not a fresh post-result quota observation.

No job/controller remains active. The next new check cannot start before
06:27:37.228843UTC (02:27Toronto). No score or Decode transfer is justified.

## Bounded repair

`results/moe-epair-store-repair-20261029/` contains source9fc2993a,336334bytes.
Exactly nine resource-query namespace references change. CUDA algorithm bytes
remain9d6c4354; every Python runtime function AST, all diagnostic comparisons,
dispatch, thresholds and CUDA arithmetic are unchanged. The old failed source
and its audits remain intact, with no relabelling as the repair.

New query-binding regressions reject the original mismatch, a bad geometry
namespace, a missing Python export and an incorrect prepare-time method.
The corrected query is exact to the query actually compiled in the earlier
factor experiment after namespace/export-name binding. The full-result guard
also rejects its twelve malformed fixtures. These are stronger CPU checks,
not CUDA compilation or GPU validation of the repair.

One namespace-only retry is justified by this diagnosed defect and must keep
normal hourly spacing and the original full-suite, actual-path, resource and
material-gain gates. No parameter sweep or post-result threshold relaxation.
Preserve all incumbents unless the repaired source clears the complete gate.
