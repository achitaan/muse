# One output-projection trial

The initial hypothesis misattributed a truncated profile label: the roughly
14.3..14.7 microsecond stage is the unpooler. Source dispatch resolves the
output projection to m16trace_, about 5.6 microseconds. It reads an
8192-by-1024 BF16 matrix. The paired measurement below corrected this mapping;
do not repeat the mistaken bottleneck attribution in future work.

Use the single public o32h_r100 configuration from the pinned Vjeux source:
https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_decode_layer.py
Its geometry function at 16 tokens returns four stages. Instantiate
launch_sk2<1,16,2,2,2,2,4,true,false,1,1,bfloat16>: 32 output rows per CTA,
8192 output rows total and K=1024. Input masking handles tokens 9 through 15;
16 uses the full tile. Other token counts use the original incumbent.

Prepared source 9f415f258c8659c16067401355562adf954b4f7a260bbdf91bc26db3f2031396
is 994443 bytes. It replaces all three rejected new instantiations with this
one, sharing the existing fused pooler/router extension. Original CUDA bytes,
the FP32 router contract, attention, norms and MoE are unchanged. Three retired
router setup functions still return disabled; the actual wide-router bank stays.
The full Apache/MIT notices are preserved.

The codec audit independently reconstructs 131072 BF16 values and 26 escape
corrections exactly, checks stream bijections, both 16-row shared-memory and
store geometry, fallback conditions, dispatch across all token counts 9..16
and factory reuse. Structural inversion recovers the complete baseline AST by
reversing only the enumerated setup/factory/source append edits. The 1208
dispatch probes are control-flow checks, not numerical evidence.

Prepare compares just two arms at 16 tokens on two independent synthetic
activation patterns, seven paired cold-L2 graph replays each. Select only if
the combined ratio is below 0.97 and neither pattern exceeds 1.02. Only static
weight packs and the configuration choice persist. The same 260/265/280-second
comparison cutoffs apply; total preparation must remain below 300 seconds.

One full official check is planned with cached baseline 09badc9b. The guard
requires all 13 tests and 12 benchmarks, integrity and choice ('o',16), so a
passing incumbent-only run cannot be retained as the new optimization. GPU
correctness, nearby-shape behavior and speed remain unverified until that run.
No parameter sweep or scored intermediate is planned.

## Official result

Check 008ff9489dec46d2 passed 13/12/integrity at 2.007000100x, interval
[1.976895099,2.037105102]. The paired output comparison was slower: ratio
1.0493 combined, worst pattern 1.0629. Relative L2 stayed below 0.0000553.
SMALL_PROJ_CHOICES was empty, so the guard discarded the trial and retained
09badc9b. No new output kernel was selected in the official timed calls.

The incumbent's paired times were 10.720 and 10.688 us; candidate 11.104 and
11.360 us. These graph measurements include fixed timing overhead; they do not
contradict the ~5.6 us individual-kernel profile. The profile mapping was the
mistake. No score was submitted for this rejected source. A separate scored
confirmation of the retained 09badc9b update is job 441eb580f3074669. Full trial
evidence/controller is in results/resumed-decoder-phase9-20261028/.
