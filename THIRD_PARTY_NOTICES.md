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
