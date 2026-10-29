# Decode cluster boundary design rejected

Job545f0b524917423b, source1c23cdc6, completed on H200. All four native modes
have56registers,0local bytes,16static shared bytes,248active cluster capacity
and132SMs. All36 numerical comparisons and eight changing-input graph replays
pass. This fixes the earlier setup comparison failure in the covered probes.

Every paired native boundary loses: attention takes1.65–1.87times as long,
full FFN1.74–1.87times, and tiny FFN4.22–4.57times. Combined attention/FFN
ratios at M1/8/16/32 are3.066/2.833/1.723/1.866. Raw samples are preserved.
The entire family is unselected. Close this boundary-only cluster design;
the current evidence does not justify occupancy or parameter retries.

All13 official tests,12 benchmarks and integrity pass using retained
fallback paths. Their1.977485326x point and[1.947823046,2.007147606] interval
are not new-native whole-layer performance or integration evidence. No
native cluster symbols occur in those profiles. No score was submitted.

The controller discarded and restored D9. Trialddbb2653, revertcc881431,
controllere680f141, archive20261029-171355. All44 protected hashes match.
The controller short-circuited its guard because its metric was lower;
an independent offline guard invocation also rejects unselected family.

Fresh17:14:41UTC quota: checks Dense19,Attention16,MoE12,Decode17,Prefill20,
84total; scores10each,50total. The exact next check limit from the driver's
submitted-time ledger is18:03:10.328859UTC,14:03Toronto. Review-v2.json uses
that timestamp; initial review.json used server creation89ms earlier and is
preserved. No live GPU job/controller remains. Top50 remains unachieved.
