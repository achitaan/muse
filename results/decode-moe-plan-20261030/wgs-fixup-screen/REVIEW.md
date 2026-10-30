# Larger MoE stream-K fixups: park the avenue

Authorized read-only researcher follow-up; no new GPU measurement, payload or profiling. Retained Python source is 33114871; native CUDA is 1e60cf46. Exact inspected artifact hashes are in source-binding.json.

The retained stage ranges are disjoint. Producer reordering visits each assigned K stage once. A job requests 512 phase-A and 256 phase-B stages, each 36 KiB: 27 MiB per job. Splitting a unit introduces partial-result handling, not additional weight stages. Multiple jobs for one expert can request the same logical weights; this is separate from measured HBM transactions and does not revive rejected cache-policy trials.

With 132 CTAs, at most 131 internal boundaries split units in a phase. Each contiguous range has at most two partial endpoints. If B boundaries split S distinct units, contributors total S+B, at most 262. A contributor writes 8192*ceil(job_count/8) FP32 bytes across four warpgroups, at most 64 KiB. Atomic payload is therefore bounded by 16.375 MiB per phase; finisher read and zero requests have the same combined bound. These are logical operation counts, not exposed time.

Saved synthetic 32-draw normal cases contain 88–131 split units per phase at M32, 115–130 at M128 and 109–129 at M256. Their atomic payload ranges from 1.375–6.484 MiB per phase; skewed draws reach 15.172 MiB. These historical synthetic counts do not establish actual challenge traffic or critical-path fractions.

The earlier disjoint-cut screen checked 53,960 models and removed only median 0–2 split units per phase (about 0–1.64% of partial participants). The earlier private-endpoint design adds 20 MiB scratch and its best compiler proxy worsens stack/spill fields from 16/56/124 to 24/100/204. Phase A already processes its trailing partial first and defers completion behind the next segment's WGMMA work. Phase B has a designated finisher; per-slice doneAs readiness already overlaps phases.

Source anchors in sibling results/moe-private-fixup-preflight-20261029/v4/baseline.cu: traversal 1766–1794, completion/read-zero 1925–1983, partial publication 2074–2083, stage partition 2140–2150. Historical screens: results/moe-disjoint-cut-screen-20261029/review.md and experiments/prepared/moe_private_fixup_review.md. No sibling files changed.

Decision: no new cut, finisher or endpoint payload. Reopening requires specific exposed-time evidence or a distinct compiler result. Aggregate full-kernel profiles do not identify removable fixup latency. Self-review separates request counts, GPU timing and performance predictions; none of the counts above is a promised speedup.
