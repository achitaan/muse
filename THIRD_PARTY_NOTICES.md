# Third-party source

The optimized candidate derives from Nathen Fernandes, muse-bf16-proj, commit
7385e5d8f70980cfd5322ea254edc0b2b8ca90f7:
https://github.com/NathenFernandes/muse-bf16-proj

Local changes extend the upstream additive lossless decoder to four kernel
families. The original license notice is also embedded in the submission.

MIT License

Copyright (c) 2026 Nathen Fernandes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


## Native attention source

Attention derives from Vjeux, https://github.com/vjeux/okc-2026, pinned commit
f35a41ee1c13e316dabbf772830d47b23b453393, under Apache License 2.0.
The complete license is in licenses/vjeux-Apache-2.0.txt and embedded in the
submission. No NOTICE file was present at that revision.

Local changes replace scratch caching with a device/launch-signature cache and
reuse immutable RoPE constants across streams. This removes initialization
from captured graph replays and preserves buffers for captured shapes. Native
CUDA algorithms are unchanged. Calls sharing one signature require sequential
replay, consistent with this candidate's shared-workspace contract.

The prepared native MoE hypothesis also derives from the same Vjeux Apache-2.0
revision. Local changes remove diagnostics and restore Marlin BF16 GEMM outputs,
BF16 global scales and BF16 effective post-norm weights; norm arithmetic stays
FP32. The candidate passed full official coverage at 2.299797371x. The prepared source embeds the full
Apache license. Shared scratch requires sequential replay.

## Grouped Prefill expert component

The prepared Prefill expert component derives from Vjeux's same pinned Apache
revision. Local changes extract the component, preserve Marlin BF16 boundaries,
copy already-normalized BF16 inputs without arithmetic, rebuild route metadata
from fresh inputs, and integrate FP32 post-norm/weighted sum with SGLang. The
original SGLang full-layer pipeline remains. The candidate passed complete official coverage at 1.639290758x;
preserve embedded Apache license and this change notice.

## Prepared fused pooler/router component

The new decoder hypothesis extracts Vjeux's same Apache revision's sk2 fused
pooler/router path. Original native arithmetic is retained; only 9..32 tokens
use it, while all other shapes retain the previously checked path. Local
changes isolate compilation, check lossless fallback/workspace bounds, and
eagerly allocate shared control buffers. The component passed whole-layer checks and publication in phase4; see DECODE_NOTES.md. The additional full-FP32 small-token router/top-8 kernel is a local implementation.


## Exact NVFP4 bit placement

The newer checked MoE source adapts the eight-operation FP4 placement from
Lee Penkman, lee101/openkernels commit
c565e373d34575c5f7df862b05f8be7619078b0f, under Apache-2.0.
Copyright 2026 Lee Penkman. The complete Apache license already retained in
licenses/vjeux-Apache-2.0.txt and embedded in the submission also covers this
component. No NOTICE file was present in the pinned root. Local changes derive
the matching 32-bit packing bijection, integrate it with the native MoE, and
retain the corrected BF16 epilogues and FP32 norm arithmetic.
Source: https://github.com/lee101/openkernels/blob/c565e373d34575c5f7df862b05f8be7619078b0f/muse_moe_w4a16/submission.py


## Sliced FFN norm/residual boundary

The newer checked decoder's small-token implicit-padding boundary adapts the
SGLang fused_norm_residual semantics at a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f,
Apache-2.0. The single-CTA CUDA and Triton sources were verified for the BF16
round trips and FP32 pre-norm input. Local changes use an implicit zero tail,
keep the full 8192-column norm domain, and issue independent reads early.
The full Apache license is retained in the integrated submission and under
licenses/. The pinned SGLang root had no NOTICE file.
Source: https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_norm_residual.py


Attention a310b988 (October28): local FP8 probability/value MMA and register-only byte transpose preserve inherited V load ordering, FP32 accumulation, BF16 Q/K/RoPE/output boundaries and Apache-2.0 notices. Probability scale256 follows the FA3 numerical approach; implementation and CPU permutation audits are local. Full measured payload and source edit ledger are archived in results/resumed-attention-fp8pv-20261028/.


October28 segment-domain repair: local changes enlarge the L1 flat segment list to512, protect int32 prefix-space limits, and use the existing general fallback for L8B>64. Apache-2.0 attribution retained. Decode keeps its original FP16 QK/PV; no standalone FP8PV transfer was made. Exact source payloads and inverse ledgers are archived in the two segment-repair result folders.
