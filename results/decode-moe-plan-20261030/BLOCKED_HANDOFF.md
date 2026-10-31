> Current 2026-10-31: queue snapshot router-m1/jobs-20261031T041838Z.json contains 47 done and 3 duplicate jobs, with none active. Only own-check pacing remains until 04:41:30.304784 UTC; this is not a blocked goal. Historical record below remains preserved.

> Resolved by human2026-10-30: "Space only your own checks". MoE check0db9141b952a4efe launched. Historical blocker record follows.

# Pacing-scope blocker

Goal remains unmet. Three consecutive goal turns observed the same constraint: another client keeps launching Attention checks, moving the globally interpreted 60-minute interval. The pending human question asks whether spacing should apply only to this worker's checks, while still waiting for all other GPU jobs to finish. The current interpretation remains global until clarified.

Latest authoritative activity read at approximately 2026-10-30T23:20Z: Attention check `0b48e4f57db44c68` is running, submitted 23:18:18.499637Z. Global eligibility is therefore no earlier than 2026-10-31T00:18:18.499637Z. Earlier external job `eaebaa8ffe6f4c29` is terminal/pass. No external source binding or performance gain is asserted. The owned foreground timer62099 is terminal; no owned GPU job is running. Do not restart any observed job.

Ready first: `experiments/decode_moe/moe_norm_group.py`, SHA256 `e5c0e7ec4e28ccc2c9a91af85e41231a9ec356daaf0db13667c1560d6110b85b`. Actual-resource guard requires217600 dynamic bytes,288threads,zero local/static and at least one resident block/SM. Four offline compiles and the static ownership/numerical/loader audits pass; H200 evidence remains missing.

Separate alternative: `experiments/decode_moe/moe_norm_register.py`, SHA256 `21a652ccf6f5dd4bc72bea7b68d30654a639129f23fb9a110b732e67897bd537`. Actual-resource guard requires214784 dynamic bytes,288threads and the same residency/local/static conditions. Four offline compiles have146/141registers and zero spills;396ownership and144positive-addend cases pass. H200 evidence remains missing.

On resumption, refresh shared activity and quota, verify the applicable spacing scope and exact source hash, compare current official files with the pinned identity, then admit at most one focused MoE check. Preserve its acknowledgment and poll that same job. Require all official tests/benchmarks/integrity,120private fixtures, exact inter/ybuf bytes, approved MoE-only norm/output bounds, strict greater-than3% paired gains in both populations and arm orders, and actual selected new-path profile markers. Empty choices or fallback-only profiles are not gains. Do not automatically score, rerun, or integrate into Decode.

Latest quota snapshot at23:12Z:MoE13checks/10scores;Decode1check/7scores;Dense0checks;Attention16checks. Refresh before acting. Decode's final check remains reserved. The last owned GPU check was34a6e66a63f44a13 at22:07:03Z, so spacing only this worker's checks would already be satisfied once the shared GPU is idle, if the human selects that scope.

Bounded unaffected work is complete: both tiny payloads are packaged;the exact native1e60cf46 WGS follow-up parks repeated fixup/tail designs. Further speculative revisions without measurements are not a justified next step. No measured gain or rank improvement has been claimed for either unrun payload.
