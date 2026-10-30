# Real Attention to O32 diagnostic

This packet addresses a measured-evidence gap before using the final Decode check. Frozen source `dense_real_attention_handoff.py`,SHA25674f71a73387abce17a9ed58eb38b7adfa47becc9409964404447dd58f7ca83a7,764282 bytes,keeps the retained Dense prefix and public kernel unchanged. It enables nothing and has no scored/publication path.

## Why the one-CTA ceiling is insufficient

The retained M32 sliding Attention planner launches256 CTAs,128 threads each. A CPU-compiled sizeof tag gives58368 dynamic shared bytes per Attention block. The measured H200 has233472 shared bytes per SM. One such block plus ST8 O32 requires255104 bytes,so they cannot share a busy SM,even before accounting for other resources. ST8 can overlap only on unused or freed SMs. Summing its synthetic saving with U32's is not a layer forecast.

One fixed ST4 configuration uses98368 dynamic bytes. Two Attention blocks plus ST4 need215104 bytes. This removes the shared-memory obstruction,but does not prove concurrent residency or profitability. CPU compilation gives42 O32 registers,zero spills/stack. Its original ring-refill code is retained: WGMMA wait_group1 completes the old group before refilling done=i-1; a final wait_group0 precedes BF16 stores. The slot/phase model covers all eight K tiles once,in unchanged accumulation order. [NVIDIA PTX WGMMA waits](https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf)

## One signal change; four fixed comparisons

The retained Attention CUDA source is241b67fe. The private variant removes the two segment-level outgoing-trigger sites and calls that same helper at uniform kernel entry. An inverse edit reproduces the original source exactly. Arithmetic,masks,all keys,cache conversion,rounding,workspace generations and launch-selection logic are unchanged. Whole CUDA/host state is enclosed in a private namespace,with a28-argument typed forwarding alias. The native Python planner/launcher is extracted unchanged from c15df9d3 and executed in an isolated dictionary.

Both parent and private M32 device proxies use129 registers,zero stack/spills. SASS moves the first scheduling signal from offset30128 to176. The compiler hoists one independent seq_lens load ahead of it; the signal still precedes path exits and the mainloop. Projection activations remain behind their dependency wait. The signal allows scheduling; it does not grant access to incomplete output. [NVIDIA PDL semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)

Four predeclared arms use the actual Dense O weights and real Decode Attention code/data flow:

1. parent: retained Attention and original O32;
2. late8: retained Attention and incoming-PDL ST8;
3. early8: entry-signaled Attention and incoming-PDL ST8;
4. early4: entry-signaled Attention and incoming-PDL ST4.

This is a causal ladder,two fixed geometries,no parameter sweep. No O16,U16,FPR,MoE or other family is rerun. A duplicate parent graph is used only to test reference byte stability,not timed as an independent treatment.

## Required evidence and stop rules

All compiler threads must join before probes. Actual H200 projection and private Attention registers/local/static/dynamic memory and occupancy must meet the source-bound resource gates. If compilation or resources fail,stop without another retry; the Dense reserve must remain at least4.

Changing normal,zero and outlier QKV values must produce finite exact Attention intermediate,O output and all-cache bytes. Record independent per-arm results and mismatch fields. If the unchanged parent is not byte-stable,admit no performance treatment. Failed candidate byte gates remain excluded.

For valid arms,use the unchanged shared interleaved cold-L2 graph harness,25 repetitions in both arm orders. Real ST8 transfer must beat parent by >3% and >=1.5us in both orders. Early8 must also beat late8 by those margins. ST4 must additionally beat early8 by those margins to justify its new schedule. These are fixed acceptance gates,not confidence intervals. Preserve raw job output and exact source binding.

Start no new diagnostic work after274s; stop timing admission at278s; the official300s preparation limit remains. Public Dense timings are the unchanged parent and are not a new algorithm result. Any positive result supports only this real Attention/O chain; full Decode bytes,latency,preparation and selected paths remain unverified. No automatic Decode check,scored call,rank claim or publication follows.
