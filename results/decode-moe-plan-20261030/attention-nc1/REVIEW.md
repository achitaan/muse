# Attention: specialize the existing nc1 merge controls

Parent Python SHA2569e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f;native241b67fe. Device prototype SHA256e29efa5e07587dfa5f61d1c0a9b51946bfc6eca6c4462370a86750359a0fd6bc. There is no runnable Python submission or GPU admission yet.

The retained code dispatches among nc1, nc8 and generic merge routines at runtime. For a future metadata-guarded nc==1 family, this prototype replaces26device reads of p.nc with the same constant1. Host code and launch plans remain unchanged;exact inverse replacement restores the device source. This is distinct from the rejected sweep that changed the number of copies to2/4/8. No split,cooperative flag,geometry,precision,mask,key coverage or FP32 arithmetic expression was intentionally changed.

One fixed assumption was screened on two prespecified L1 families. Flat MODE0 shrinks from3560to3048instructions (14.38%),with128registers unchanged. Nonflat SPEC-ONE MODE13 shrinks from2208to1560 (29.35%),with102to89registers. Both havezero stack/spills/local instructions and no compiler warnings. Both pass the preset5% footprint gate without relaxing it. CUDA13offline compilation does not establish actual H200resources, cache-miss latency, output bytes or speedup.

The completed other-client timeline in branch argon/decode-next-20261030 commitdaba560 attributes roughly0.93us of its4.8us decode1sliding timeline to entry-to-segment start, tentatively instruction-cache related. That is a hypothesis,not proof;the rest of the timeline has multiple dependent stages. Existing split/coop/nc sweeps,finisher outlining and the earlier page-metadata screen remain closed.

Before packaging:prove eligibility from the actual launch selector,guard nc==1,L==1 and only the two screened families,preserve all other native/Triton fallbacks and effective cooperative decisions,bind actual resources and preparation headroom,then inherit exact output/cache byte gates. A lower register count must not silently change cooperative launch selection. General page sizes,large index domains,mixed lengths and changing-input graph replay remain mandatory where supported. No GPU evaluation before the owned-check interval and fresh admission.

Self-review:the compiler removes alternatives to the same one-copy path;there is no new approximate math or input-derived timed dispatch. Static footprint is a filter,not a measured speedup. No gain or rank improvement is claimed. No Decode check spent.

## Launch guard and native packaging

The guard was exercised through the actual retained Python launch function and plan helpers, with fake tensors only: 5,376 metadata/override/PDL combinations and 17 boundary counterexamples. Accepted calls map only to flat L1 MODE0 or nonflat SPEC-ONE L1 MODE13. Other copy counts, L8, cluster launches, cooperative flags and PDL remain outside the new path. Requested flags must be zero, preventing a lower register count from silently enabling a different cooperative plan.

The private launcher instantiates only those two entries. Its original Params construction and grid are preserved by an inverse source check. A native TORCH_CHECK repeats the metadata guard; original page-size/index bounds remain intact. The screen's device prefix is unchanged except for the profiler identifier attn_kernel_nc1. Packaged CUDA SHA256 is a7e423588c8f87bd580d71d3c0aae7764dcae30b7721f79c1adfa98b6043bdb0. An initial prefix assertion included a separator newline; correcting that assertion preserved the exact device prefix rather than relabeling the screened source.

Compiling both packaged entries with an exported constexpr report gives: flat L1 uses 67584 dynamic shared bytes and 256 threads, with 128 registers; SPEC-ONE L1 uses 58368 bytes and 128 threads, with 89 registers. Instruction counts remain 3048/1560. Both have zero stack/spills/local instructions and no warnings. These are device compiler/layout results. The native resource bridge will set attributes and query actual residency before any new-path launch; its host compilation and H200 behavior remain unverified.

The current official Attention task was saved and hashed. It defines six loads, each weighting sliding/full by 46/16, with 16 tests. Its official output tolerances do not waive this task's inherited exact-parent gate. A future wrapper still needs independent synthetic correctness/cache checks, unchanged-parent replay controls, cold-L2 comparisons, budget control and all fallbacks. No GPU-facing Python payload or admission exists yet. No measured speedup is claimed.

Self-review: no input-value reads were added to timed host dispatch; every family decision uses the actual native metadata after plan overrides. Both flat and SPEC guards exclude cluster/PDL/cooperative requests. The CUDA parameter setup, integer bounds, cache addressing and arithmetic prefix are preserved. Remaining work is the runtime wrapper and validation gates, not additional geometry exploration.
