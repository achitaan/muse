# Namespace-only repair: CPU verified, GPU unmeasured

Source9fc2993a9cc3b3723ad24c8016cca6bf9915d53958b821da222f5ebe240d944d
is336334bytes. It replaces the nine incorrect `okc_epair_y::` references in
failed sourcee8e3086d's resource-query constant with `okc_epair::`.

`python experiments/prepared/repair_moe_epair_query.py` proves the entire module
AST differs only in that constant. CUDA9d6c4354 is unchanged, as are every
runtime function, scratch layout, numerical expression and diagnostic/resource/
selection/performance gate. The earlier8,519,680-write layout proof,65536bit
pair proof and100host-dispatch tests remain bound to the unchanged code; their
old source hashes have not been relabelled.

The new query-binding audit checks all nine namespace references, the exported
function, C++ declaration and prepare-time method. Four deliberate bad variants,
including the exact failed original, are rejected. It also compares the query
to the version that compiled in factor check92577a6426f74605 after exact
identifier substitution. This does not establish new GPU compilation.

`scripts/moe_epair_store_repaired_evidence.py` validates that exact repair chain
and reuses the original scientific guard without changing its thresholds.
The twelve guard counterexamples pass their rejection tests. Measure with
`scripts/measure_moe_epair_store_repaired.py`; guard with
`scripts/verify_moe_epair_store_repaired_evidence.py`. Baseline33114871 remains
cached and fully source-bound; no new baseline job is needed.

Do not evaluate before06:27:37.228843UTC. There is no active controller/job
at this preparation checkpoint. Initialize the foreground controller from a
clean named branch and use only this frozen source. If the local pacing wait
plus expected job time could exceed the controller's default1800-second
measurement timeout, start closer to the slot or configure an appropriate
timeout explicitly; never restart a live job because observation expires.

Before promotion, require original full5tests/6benchmarks/integrity, valid
resources, actual new-path execution, fresh-input replay and material affected/
control gains. No GPU correctness, timing, speedup, rank improvement, scored
publication or Decode transfer is claimed for9fc2993a at this checkpoint.
