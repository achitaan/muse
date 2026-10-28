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

## Prepared implementation and CPU evidence, October 28 at 17:10 UTC

The builder now produces source 92fe19e2e94211b479baf0b8b5ab2bd2a705cd8c50ce09e5060479614d6a68aa,
995430 bytes. Every byte of baseline 09badc9b remains its prefix. Three fixed
instantiations reuse the existing generic CUDA template in a separate extension;
their pipeline depths come from the pinned upstream geometry function: pooler 3,
QKV 6, sliced unpooler 4. The original extension and router paths are untouched.

The CPU audit independently recovers 1,441,792 BF16 values exactly, including
233 escape corrections, checks every stream tile/unit once and reconstructs
correction tables. It exercises negative-zero, nonfinite, unsupported dtype,
wrong shape, excessive escapes and shared-memory fallback. It checks store
coverage for tokens 1 through 8, dispatch for nearby and larger shapes, module
reuse and global bindings. These are finite CPU checks, not GPU verification.

Prepare compares each fixed candidate only with its incumbent on two synthetic
activation patterns, for token counts 1 and 8, with seven paired cold-L2 graph
replays per pattern. Enable only a mean ratio below 0.97 with neither pattern
above 1.02. Tokens 2 through 7 use the selected eight-token geometry with explicit
masking. Configuration choices and static weights are the only cached results.
Packing/building is skipped if the initial prepare leaves insufficient budget;
selection stops at 275 seconds to preserve the 300-second prepare bound.

The guard requires full official 13/12 coverage, integrity, the existing FP32
router contract and an explicit nonempty SMALL_PROJ_CHOICES tuple. An empty
selection cannot be retained as a measured optimization. One full official
check is planned; no tile sweep or scored submission is authorized by evidence
yet. Reuse the cached baseline check without a new GPU job.

## Official outcome

Check afd35730c7d3476f passed 13/12/integrity at 1.997916935x with interval
[1.967948181,2.027885689]. However, preparation took 257.03 seconds and printed
SMALL_PROJ_SKIP_BUDGET. None of the new paths was packed, built or selected.
The nonempty-selection guard discarded the trial. Keep the original 09badc9b
source and treat this only as an independent observation of its timed compute.
No scored submission was made. GPU correctness and performance of the new
projection kernels remain unverified. Full records and archived controller are
in results/resumed-decoder-phase7-20261028/.

Before another check, reduce preparation overhead or integrate instantiations
into the existing extension build. Do not merely raise the cutoffs: observed
baseline setup varies from about 215 to 257 seconds. The pinned source also
offers o32h_r100 for output projection, a possible later target for the profiled
16-token output cost (~15 us). It has not been implemented or audited here.
