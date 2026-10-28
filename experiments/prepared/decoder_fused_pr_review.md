# Evidence-backed next decoder hypothesis (unimplemented)

Current decoder check 2f41872843fe4490 passes at 1.815511785x; scored job
95bbfd2c81224715 passes at 1.804641371x. Latest overall snapshot is 221.

The checked profile at T=16 shows a 15.7us pooler, two router GEMM kernels
totalling 9.3us, two split-K reductions totalling 3.7us, plus routing. This
supports studying a fused pooler/router backend before any parameter sweep.
These are profile spans, not a prediction that all of them can be removed.

Primary source already downloaded:
https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_decode_layer.py
Function bfp_fused_pooler_router implements one launch for M<=32: lossless
12-bit BF16 pooler with exact escape correction, BF16 pooler output, and
BF16 hi/lo router weights with FP32 accumulation/output. It shares the BF16
input and reduces the router partials in fixed order.

Only consider it for 8 < T <= 32. For T<=8 the original layer keeps a FP32
router input; preserve that complete path, including F.linear. For T>32 keep
the current checked path initially. Do not adopt the public whole-layer glue
as-is: it routes from BF16 input at small T and forms norm gains differently.

Relevant helpers: bfp_sk2_compress, bfp_sk2_tileunits, bfp_sk2_stream,
bfp_sk2_table, bfp_sk2_prepare, bfp__sk2_src, bfp__get_ext and
bfp_fused_pooler_router. Separate out only this backend, preserve Apache and
local change notices, audit bitwise BF16 reconstruction/escape algebra and
scratch reset. Avoid capturing first-use zero initialization: eagerly allocate
fused_part/fused_cnt in prepare. Shared scratch requires sequential calls.

No code has been integrated and no GPU run has been spent on this hypothesis.
Prefill's single existing check is the active phase; finish it before changing
controller scope. A future check must test the concrete fusion, not a sweep.
