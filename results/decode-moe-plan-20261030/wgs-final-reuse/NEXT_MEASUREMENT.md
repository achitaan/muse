# Next investigation: retained MoE phase timing

No new GPU job or instrumentation candidate is admitted by this note.

The retained native source already contains disabled DBG hooks, a gtimer helper reading `%globaltimer`, an optional p.dbg pointer and a launcher accepting a fourth scratch tensor. These were inspected in the exact1e60cf46 source after the read-reuse result. The existing source offers eight per-CTA timestamps without redesigning its scheduler:

| Execution order | Tag | Boundary |
|---|---|---|
|1|0|Kernel entry before dependency wait/routing|
|2|1|Routing/setup boundary|
|3|7|WGS gather returned|
|4|6|Producer/consumer join returned|
|5|2|This CTA's phase A returned|
|6|3|This CTA's phase B returned|
|7|4|Grid barrier returned|
|8|5|Finalizer completed|

Tags6/7 are deliberately listed in execution order. Source anchors: helper/macro257–262, entry2219,routing2365,gather2399–2402,phases2406–2409,barrier2448,finalizer2596. Only one CTA thread writes each timestamp. The existing inline PTX helper is dead in the incumbent; its syntax and documented timer semantics must be validated before enabling it. Keep raw ticks until units and observation granularity are established.

A useful bounded follow-up would enable these hooks only in an unscored private diagnostic of the retained parent. First CPU-compile both BIG entries, bind unchanged arithmetic and source, check resources, then qualify output/live xB/ybuf and instrumented-versus-uninstrumented overhead. Fresh queue/quota/identity admission is still required. Large instrumentation overhead invalidates attribution. Do not interpret sums of per-CTA durations as a critical-path fraction: phases overlap across CTAs, producer work overlaps consumers, and early CTAs' barrier waits can merely reflect the last required worker. Retain absolute per-CTA boundaries to inspect actual joins and tail ownership.

Collect fresh probe routing/job statistics outside timed calls: local routes, active/per-expert counts, padded rows and job counts. Label their synthetic distribution; they are not actual challenge routing measurements. A normal uniform population and hot-expert population bracket different behavior, as the large timing differences in the closed experiment show.

Future diagnostic logs should emit critical summaries at the end and compress raw arrays into a bounded packet. The16000-character log cap clipped the earlier resource/probe lines here. Do not spend a check merely to restore those lost lines: this variant already failed its full-component timing gate.

Existing complete-kernel profiles establish substantial MoE time, but not which internal wait, data path or tail to optimize. The next change should follow that phase evidence, rather than repeat the closed occupancy, norm, cache, split-cut or scheduling flags.
