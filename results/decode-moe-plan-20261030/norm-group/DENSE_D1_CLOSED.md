# D1 is closed

The event conversion was corrected in source `169dbce7`. Source `13224d30` added nearby protection. Check `62f428608bbf41aa` selected neither fixed cluster family: M128 QKV new/old ratios were 1.0624 and 1.0401; pooler ratios were 1.0487 and 1.0724. Its nine tests, 30 benchmarks and integrity check passed using fallback paths.

The later resource-instrumented source `e48cba9a`, check `f026c3b14a864be0`, confirmed actual QKV/pooler register counts of 90/124, zero local and static shared memory, legal cluster capacities of 39/66, and 18 passing FP64 probes. QKV ratios of 1.0366/1.0414 and pooler ratios of 1.0703/1.0318 still lost. Selection was empty; its 1.501728512x point came from fallback paths. There was no score or retained gain.

Do not reopen the same timing correction or rerun these two fixed families for noise. The earlier review-plan description of D1 as awaiting its first GPU test is superseded.

Read-only sources inspected in sibling `opus-fresh`: the latest terminal header in `HANDOFF.md`, `results/dense-cluster-nearby-20261029/verdict.json`, and `experiments/prepared/dense_cluster_review.md`. No sibling files changed. This note records historical evidence, not a fresh GPU measurement.
