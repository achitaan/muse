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

## Guarded standalone runtime payload

Python source `experiments/decode_moe/attention_nc1.py` SHA256 `066d89325f4adee4ea0156c050fcf35b98b4fbe09db4f28f09b9df85e0278837`, 328054 bytes. CUDA remains the packaged a7e42358 source. Parent prefix and all original function definitions are preserved. No verified source was edited.

Standalone Attention exposes kernel_fn(data), so calibration runs during the first eager setup call; it receives only the device, never caller activation or sequence values. Both compilation threads finish before graph capture. The timed dispatcher uses only native-call metadata and performs no new build, resource query or tensor-value read. Unsupported calls and first-capture calls use the original parent. No loaded extension attributes are patched.

The fixed synthetic fixtures cover both screened families, page1/page64, mixed lengths, deliberately mismatched speculative page widths, scaling and multiple head pairs. All twelve fixture plans were checked through the actual retained launch function. Each has three changing qkv/length/cache-location variants. Exact raw output and complete KV-cache bytes plus unchanged-parent repeats are mandatory. The MoE norm-variance exception does not apply. Earlier real-chain f6a94233 had parent_repeat=True; other changed-chain controls failed. That history does not establish universal parent instability, so the new setup checks it directly.

Each family needs eight comparisons exceeding3% (two cohorts, two synthetic distributions, both arm orders), plus both nearby controls within2% regression. The shared harness is retained unchanged. Replays are repeated measurements within one GPU job, not independent experiments. Setup cutoffs remain274/278seconds; full official tests/benchmarks/integrity and actual new-path profiles are still required. A fallback-only aggregate point is not a gain.

CPU boundary evidence:15resource guard cases,12actual-plan fixture cases,16dispatcher/rejection cases and4public-wrapper cases. Scratch reset is device-scoped and touches only internal workspaces; caller KV caches are never reset. An initial ctx129 full fixture selectednc3 and was corrected toctx63/nc1 before GPU use. The prior89a94912 packet and the pre-reset997ce9b0 packet remain archived, both unrun.

Self-review: native parameter construction and grid are inverse-bound to the parent; nc1 flags0 prevents cooperative/PDL/cluster mode drift. Resource queries precede all private probes. Scratch is zeroed after diagnostics, including gate failure, before actual inputs are processed. Synthetic calibration stores only selected family names/module handles. Capture fallbacks and original support/index checks remain. Actual host compilation, H200 resources, byte controls and speed remain pending. No GPU admission yet.
