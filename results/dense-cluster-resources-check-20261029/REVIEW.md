# Fixed Dense M128 families rejected

Jobf026c3b14a864be0, sourcee48cba9a, completed on H200. Both fixed kernels
are legal under the resource gate: QKV90registers/0local/0static shared,
230544dynamic bytes,39cluster capacity; pooler124registers/0local/0static,
215184dynamic bytes,66cluster capacity. Both have132SMs,160threads/CTA,
one resident block/SM and enough native capacity for the requested grids.
Occupancy capacity is not an observed simultaneous-residency timeline.

All ten QKV and eight pooler FP64 oracle cases pass. Nearby synthetic shapes
also improve, but both scored M128 pattern comparisons lose: QKV new/old
1.03663/1.04137; pooler1.07031/1.03177. No family is selected. The earlier
listed-floor opportunity did not materialize in these actual kernels.

All9 official tests/30benchmarks/30profiles/integrity pass using original
fallback paths. The1.501728512x point and[1.479202584,1.524254439] interval
do not measure new-kernel scored integration. Preparation78.339seconds is
from a separate hosted run, not a causal preparation-speed claim. No score
was submitted. Nearby unscored gains are preserved as evidence, not retained
progress toward the overall placement goal.

The controller discards and restores EDCedc61a1e, whose retained scored result
is1.527771327x. Trial2840c0c9, revert9dc2f820, controllerd9b4b258,
archive20261029-181018. The controller skipped its guard because the point
was lower; an independent guard call also rejects the empty choice tuple.
All44 protected hashes match. No live job/controller remains.

Fresh18:10:21UTC quotas: checks Dense18,Attention16,MoE12,Decode17,Prefill20,
83total; scores10each,50total. Next new check>=19:03:28.127090UTC,
15:03Toronto. Close these two M128 families: no noise retry, rescore or
unscored-neighbor-only retention. Any different experiment needs its own
source, mechanism, whole-load bound and admission. Top50 remains unachieved.
