# Next hypothesis: small-batch compressed dense projections, unmeasured

The latest full passing decoder check is 09badc9b at 1.994319167x, with implicit
FFN zero padding and early input loads. It remains unpublished. The published
a61735ee scores 1.965324622x. Profiled small pooler projection is still about
14 us; QKV about 10 us and sliced unpooler about 4 us. Target these remaining
memory/launch costs rather than another FFN warp sweep.

The pinned public Vjeux decoder source contains curated SK2 lossless BF16
projection configurations for these exact operation families. The current
decoder already extracts its fused pooler/router implementation for 9..32
tokens. Generic compression, escape correction, stream packing, CUDA templates
and CPU inversion techniques are therefore already available.

Potential curated configurations from bfp_SK2_CONFIGS at
f35a41ee1c13e316dabbf772830d47b23b453393:

```python
('pooler8h_r72', 'pooler', 4096, 16, 1, 2, 2, 4, 1, 1)
('qkv8_r176', 'qkv_gate', 2304, 24, 3, 1, 4, 2, 1, 1)
('slice8_r192', 'unpooler', 1024, 8, 1, 1, 8, 2, 2, 1)
```

Read bfp__sk2_geom, stream/table helpers and the actual upstream dispatch before
instantiating these. Do not infer parameters from labels or invent a geometry
sweep. This is a hypothesis, not a performance claim or a ready implementation.
Source: https://github.com/vjeux/okc-2026/blob/f35a41ee1c13e316dabbf772830d47b23b453393/muse_decode_layer.py
The original downloaded source is .tools/vjeux-decoder-upstream.py; the repository path is confirmed by the pinned README.

Preserve original BF16 weights losslessly, including exact escape corrections
and negative-zero/nonfinite fallback handling. QKV/input/output shapes and
slice semantics stay exact. Keep the full FP32 router up to eight tokens;
this proposal concerns the separate BF16 projections only. Existing 9..32
pooler/router and large-batch wide-router paths should remain unchanged.

Prefer a bounded two-way prepare comparison against each current incumbent,
using independent synthetic activations and paired cold-L2 graph timing. Cache
only configuration choices and static weights. Full 13/12 official coverage
is required. Keep byte-for-byte frozen sources and avoid a score until a real
algorithm choice plus sufficient performance evidence supports it.

Preparation fell from 279 s for the published source to 215 s in the latest
check after removing the unselected WG sweep. Preserve the 300 s budget.
Current source is 988314 bytes, leaving about 11.7 kB under the upload limit;
reuse existing generic helpers and retain full Apache/MIT notices.
