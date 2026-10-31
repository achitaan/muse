# Attention: specialize the existing nc1 merge controls

Parent Python SHA2569e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f;native241b67fe. Device prototype SHA256e29efa5e07587dfa5f61d1c0a9b51946bfc6eca6c4462370a86750359a0fd6bc. There is no runnable Python submission or GPU admission yet.

The retained code dispatches among nc1, nc8 and generic merge routines at runtime. For a future metadata-guarded nc==1 family, this prototype replaces26device reads of p.nc with the same constant1. Host code and launch plans remain unchanged;exact inverse replacement restores the device source. This is distinct from the rejected sweep that changed the number of copies to2/4/8. No split,cooperative flag,geometry,precision,mask,key coverage or FP32 arithmetic expression was intentionally changed.

One fixed assumption was screened on two prespecified L1 families. Flat MODE0 shrinks from3560to3048instructions (14.38%),with128registers unchanged. Nonflat SPEC-ONE MODE13 shrinks from2208to1560 (29.35%),with102to89registers. Both havezero stack/spills/local instructions and no compiler warnings. Both pass the preset5% footprint gate without relaxing it. CUDA13offline compilation does not establish actual H200resources, cache-miss latency, output bytes or speedup.

The completed other-client timeline in branch argon/decode-next-20261030 commitdaba560 attributes roughly0.93us of its4.8us decode1sliding timeline to entry-to-segment start, tentatively instruction-cache related. That is a hypothesis,not proof;the rest of the timeline has multiple dependent stages. Existing split/coop/nc sweeps,finisher outlining and the earlier page-metadata screen remain closed.

Before packaging:prove eligibility from the actual launch selector,guard nc==1,L==1 and only the two screened families,preserve all other native/Triton fallbacks and effective cooperative decisions,bind actual resources and preparation headroom,then inherit exact output/cache byte gates. A lower register count must not silently change cooperative launch selection. General page sizes,large index domains,mixed lengths and changing-input graph replay remain mandatory where supported. No GPU evaluation before the owned-check interval and fresh admission.

Self-review:the compiler removes alternatives to the same one-copy path;there is no new approximate math or input-derived timed dispatch. Static footprint is a filter,not a measured speedup. No gain or rank improvement is claimed. No Decode check spent.
