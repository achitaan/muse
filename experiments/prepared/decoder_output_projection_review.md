# One output-projection trial

The 16-token decoder output projection takes about 14.3..14.7 microseconds in
the saved profiles. It reads an 8192-by-1024 BF16 matrix. This is the next
measured projection bottleneck after the three smaller alternatives were
rejected by actual paired comparisons in phase8.

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
