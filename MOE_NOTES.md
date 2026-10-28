# Latest MoE check - October 28, 2026, phase4

Checked source best/moe_verified.py: SHA256
`331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.
Check d600231c8a8244cb passed all 5 tests, 6 benchmarks and integrity at
**2.365437416x**, CI [2.329955855,2.400918977]. This is not the published score.
The interval overlaps the prior source's check upper bound 2.334294331,
so no scored run was spent. best/moe_published.py retains dd6049cb and the
2.312613084x publication b4a10b76347e4cb0. The published decoder still embeds
that older MoE implementation; no uncertain standalone gain was transferred.

Two measured changes: paired BF16 epilogues (375685be8be14e1b) passed at
2.318616664x, CI [2.283837414,2.353395914]. Pairing retains the BF16 accumulator,
BF16 global-scale and RN-even BF16 product boundaries, followed by FP32 norm
arithmetic. All 1,566,720 products in the independent CPU audit matched analytical
FP64 RN-even rounding, including signed zeros, subnormals and overflow.

The second change uses Lee Penkman's exact eight-operation FP4 placement and a
matching prepare-time bijection of all 32 packed bits. It changes representation
only: same NVFP4 bit count, no requantization, same BF16 fragments, scales,
routing, reductions and norm math. CPU audits checked 1,048,576 codes and 65,536
scale bytes across MMA, natural-K, fat, unit and WGS layouts. All 256 possible
inputs to the masked multiply were enumerated, proving no carry interference
in its selected output bits. Source inversions recover all original native
CUDA except the explicitly changed helpers. GPU speed is established by the
full check, not by source-level operation counting.

Provenance: native Vjeux source remains pinned f35a41ee; exact placement derives
from lee101/openkernels c565e373d34575c5f7df862b05f8be7619078b0f, Apache-2.0,
copyright 2026 Lee Penkman. The pinned root had no NOTICE file. Full Apache
license and additional attribution are retained in the source and notices.

Evidence: results/resumed-moe-phase4-20261028/. Builders and CPU audits are in
experiments/prepared/ and read immutable baseline sources. The controller is
archived, no job is pending, and the overall Goal remains active. This phase
used two checks, no repeated baseline and no scored submission. Further work
needs a material hypothesis; do not spend a run merely to chase interval noise.

---

# Next MoE hypothesis - paired epilogue, GPU unmeasured

Prepared source 6862c3a54429fb4149e7fd78f63b24db71d70648a1947172826ea7f134082409
pairs eleven independent GEMM-output sites through BF16x2 RN-even operations.
BF16 accumulator/global-scale/product boundaries and FP32 norm intermediates
remain. CPU audit matched 1,566,720 products to independent FP64 rounding,
including signed zeros, subnormals and overflow. Reversing the transformation
recovers all original native CUDA bytes. This is not a measured speedup.
One full MoE check will test it against cached baseline dd6049cb at 2.299797371x;
no geometry sweep or automatic scored publication. See
experiments/prepared/moe_paired_epilogue_review.md and saved CPU audits.

---

# Latest native MoE checkpoint - October 28, 2026

Check `6792a5c832ee4aff`: **2.299797371x**, all 5 tests, 6 benchmarks and
integrity checks passed. Exact source `best/moe_verified.py`, SHA256
`dd6049cbf7a2a41666f2d9e46010f81e0dd22c0840881bc5f187ac5f3ba1d3a3`.
Scored publication `b4a10b76347e4cb0` passed at **2.312613084x**. Do not duplicate.
This supersedes the previous 1.268945620x checked Marlin candidate below.

Derived from Vjeux's Apache-2.0 source at revision
f35a41ee1c13e316dabbf772830d47b23b453393. One persistent native grid performs
routing, both NVFP4 matrix multiplies, SwiGLU and final norm/weighted sum.
All local routes remain present; routing is rebuilt from fresh inputs each
call. Weight layouts are bit/byte repacks, without requantization or persistent
BF16 expansion. CPU inversion recovered 1,048,576 weight codes and 65,536 scale
bytes across MMA, natural-K, fat, unit and WGS layouts, with seed 20261028.

Local changes remove diagnostics and restore pinned SGLang Marlin's BF16 GEMM
outputs: BF16 accumulator boundary, BF16 global scale and BF16 product before
SwiGLU or post norm. Effective (1+w) norm weights are prepared in BF16; scaled
norm values and reductions stay FP32 until final BF16 output. This preserves
Dot's precision correction while using the native scheduler. The structural
precision guard now recognizes both this source and the prior Triton finalizer.
CPU/static checks alone do not establish GPU correctness or speed.

Scratch is prepared once, fully overwritten/reset by the kernel, and shared
between sequential calls. Concurrent calls sharing this state are unsupported.
Full source/license and exact check evidence are in
`results/resumed-moe-phase3-20261028/`. The native source includes Apache license
and local change notices; preserve them. Main-chat-only work, no sweeps.

The next single hypothesis integrates the separately checked native attention
and MoE into the whole decoder. The FP32 router path, expert ID rotation and
residual boundaries are preserved. The draft is unmeasured. Prefill stays last.

---

# MoE optimization notes — October 28, 2026

Current order: MoE W4A16, Decode Layer, Attention Decode, Prefill Layer last.
Main chat only. Conserve checks and scored runs; every trial needs a concrete hypothesis.

## Completed phase 1

- Baseline 03449dfa701d4a95: 0.999760595x, all 5 tests and 6 benchmark cases passed.
- Small-M fused finalizer 74ba53564cfd42e4: 1.149649851x, full official validation passed.
- Smaller EP8 M tiles bd37e0d81b0542fa: 1.271471562x, CI [1.252399489,1.290543636], full validation passed.
- Direct single-token CUDA NVFP4 GEMV 67f2f03d27b44ede: correct but slower, 1.070874526x aggregate; single-token path about 148us versus 54us retained. Reverted. No follow-up parameter sweep.
- Evidence, frozen source hashes and controller history: results/resumed-moe-phase1-20261028/.
- Four MoE checks used, no MoE scored submissions.

## Precision review and new baseline

Dot supplied a review (no performance measurements) identifying the same small-M
fallback and highlighting the difference between the public starter and pinned
SGLang. Independently verified pinned helper:
https://github.com/sgl-project/sglang/blob/a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f/python/sglang/kernels/ops/muse_spark_v12/fused_post_expert_norm_weighted_sum_triton.py

SGLang's effective norm weight is BF16(1+w), but y * effective_weight stays FP32
through RMSNorm and route-weight reduction, with only final output cast to BF16.
The public starter adds an intermediate BF16 rounding. Phase 1 followed that
starter and passed all official checks; the new baseline removes the extra
rounding to match pinned SGLang's FP32 computation more closely. Do not roll back
to the older rounded variant just for a small timing point change. Establish a
fresh measured baseline for the stricter precision contract first.

Mask remote expert routes before loads/arithmetic, output zero for all-remote
rows, and overwrite outputs on every graph replay. Preserve FP32 accumulation,
original NVFP4 codes, FP8 block scales and FP32 global scales. In-kernel
dequantization is permitted; persistent BF16 predecoded weights are not.

## Prepared hypotheses, unmeasured

The original-layout single-token CUDA attempt is retained as negative evidence.
A grouped Triton W4A16 GEMM addon is being prepared under .tools. Its first CPU
check caught multiplication-order differences from the task dequantizer; fix
value * (scale * global_scale) before any H200 evaluation. It is not a speedup.
Marlin launch/scheduling remains another hypothesis; no library/harness patches.

## Other tracks

See TRACK_NOTES.md for earlier attention/decoder work. Attention scored job
f6b899f674914ef2 completed at 1.116457695x and passed. Decoder baseline
65ab5fffad0c4995 completed at 1.002013444x. The next focus after MoE is decoder
layer, then attention and prefill. No Dot advice authorizes external spending.

## Phase 2 checkpoint and move to Decode Layer

FP32-intermediate baseline 0535f51d98d548ee passed all 5 tests, all 6 benchmarks
and integrity checks at **1.268945620x**, CI [1.249911436,1.287979804].
Exact source: best/moe_verified.py, SHA256 eab0e08d75f29658ec33a8f370d947ca023a9c10e3d09666684e6df1ef05f92d.

Grouped Triton check c8a6c6e4c46142e1 failed compilation because a masked FP8
load used an integer zero. One diagnosed retry 8f5c231df84b4147 compiled and
passed all correctness checks but was much slower: 0.448401955x aggregate, with
M128/M256 around 7.7/10.6ms. Reverted. No further geometry sweep.
Evidence and immutable controller history: results/resumed-moe-phase2-20261028/.

Seven MoE checks were used. One scored publication of the stricter FP32
candidate is now pending: 650aa66fd3334d8c. Observe that job rather than
resubmitting it. Remaining quota at launch: 13 checks and 9 scored runs.

Next focus is whole Decode Layer, carrying over the small-M FP32 finalizer and
M-tile choices. Whole-layer baseline 65ab5fffad0c4995 is already measured and
cached at 1.002013444x. Reuse it without another GPU baseline run.
