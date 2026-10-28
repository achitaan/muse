# Read-only dense source inspection for quota reset

No dense candidate or evaluation in this continuation. Dense quota0/0.
Retain edc61a1e, sourcebest/verified.py, publication6131bc96427e4486 at1.527771327x.
No performance claim follows from public source comments or an author's score.

Pinned public source inspected:
https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_bf16_proj.py
RawUTF8SHA256c1dd241c0c08b42fddfa7bc9656d0f596b59bb13005c50c0db4418d9f2f1dc7e,
188559bytes, currently cached.tools/vjeux-bf16-proj-upstream.py. Apache2.0.

It offers 12-bit BF16 streams plus exact FP32 escape corrections, skinny sk2
MMA kernels and scalar GEMV specialists. Original BF16 bits must reconstruct;
no persistent dequantized BF16 copy is allowed. pack_compressed rejects >0.5%
escapes or corrections that do not recover weights inFP32. Audit signed zeros,
subnormals, uncommon exponent ranges and nonfinite/fallback behavior before
calling this bitwise lossless: its numerical equality test alone is insufficient
for signed zero. The separate sk2/cg compression tables also need independent
inversion. Original BF16/FP32 weight storage remains available for fallback.

Important defaults: USE_SK2=True; WIDE_BF16=False, so wide BF16 custom GEMM
is not the ordinary route. Router is BF16 hi+lo with FP32 accumulation; audit
the standalone TASK/router tolerance rather than transplanting it into Decode's
preserved full-FP32 small-token router. Shapes and sliced unpooler semantics
match the challenge only after checking every dispatch and nearby shape.

Do not rerun the Decode small-projection alternatives that selected nothing.
First compare their extracted families to this source and the current dense
incumbents. Any proposed standalone candidate needs a distinct measured reason.
The old Decode selection result is negative evidence, not proof a different
standalone workload will be faster or slower.

prepare() currently builds a CUTLASS pooler route on a Python background thread,
and runs24cold-L2 graph-replay tuning repetitions for six M values over many
backends. This conflicts with our bounded main-chat experiment discipline if
copied blindly. A future port should omit that optional build and narrow the
candidate family before full-source preparation; preserve the300second limit.
Do not install tools or libraries for it. CPU audit first, one meaningful full
check after quota is available, fixed guard, then score only a material gain.

Attention page-table follow-up was also screened from the exact retained source:
L1flat already bulk-loads64page indices via TMA, PD3ahead/PS4slots. The comments
record a0.7%loss for that strategy on L8. It is not a fresh global-load-removal
hypothesis; do not spend another run adding the existing optimization.
