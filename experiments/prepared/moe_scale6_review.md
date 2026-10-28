# Next MoE hypothesis: lossless scale-byte compression

This began as a CPU-audited representation prototype. A GPU candidate is now
implemented and CPU-audited as described below; its GPU correctness and speed
remain unverified. Retain 33114871 as the verified/published MoE source.
The native CUDA is extracted for inspection in .tools/moe-current-inspect.cu.
Its original source/license and the DQ8 modifications must remain intact.

The large-batch WGS path reads per-warpgroup weight pieces of 9216 bytes:
8192 packed NVFP4 code bytes and 1024 FP8 E4M3 scale bytes. Packing scales in
six bits would reduce the latter to 768 bytes, pieces to 8960 and full stages
from 36864 to 35840 bytes: 2.7778% less weight traffic. Added decode instructions
could erase that benefit; only an official measurement can establish speed.

Eligibility must be checked on static local scale banks in prepare: every byte
must be in 0x40..0x7e, with FP8 E4M3 dtype. The public nvfp4.py random_weights
helper documents ordinary scales clamped to [8,448], which fit this interval.
It also documents a wider scale distribution. Those and every other ineligible
bank must use the unchanged original layout and CUDA path. Do not assume all
inputs fit, approximate values or quantize scales. The predicate implies the
existing PRMT scale eligibility but must not replace its original fallback.

Eight low-six-bit fields fit in six bytes. For packet i, read two aligned u32
words starting at 4*floor(3*i/2). If i is odd, shift the 64-bit pair right by
16 (a funnel shift for low u32, right shift for high). Otherwise keep it. The
four pairs of fields start at bit offsets 0,12,24,36. For each pair p, scatter
(p&63) and ((p&4032)<<10) to bits 0..5 and 16..21. Multiply by the same opaque
k16c=16 and add 0x60006000 rather than 0x5c005c00. Restoring bit 0x40 in each
original byte contributes exactly 0x400 per BF16 half. No carry crosses halves.
NVFP4 codes, BF16 fragments, GEMM outputs, norms and final sums stay unchanged.

audit_moe_scale6_prototype_cpu.py round-tripped 1,310,720 scale values, including
all 64-by-64 adjacent field pairs at every position. Every proposed scale pair
matched the old PRMT/IMAD bit pattern exactly. It proves the final two-u32 read
ends at byte 767 of a 768-byte block. Pieces stay 128-byte aligned; stages and
three-stage rings stay 1024-byte aligned. This is finite representation evidence,
not a proof of a not-yet-written complete kernel or its synchronization.

Suggested implementation boundary: add an OKC_S6 compile flag and compress only
the _wgs_copy scale tail, preserving its existing [group][warp][g][j][h] order.
Keep the original UNIT_BYTES for non-WGS rings. Add a WGS_UNIT_BYTES derived from
WGS_PIECE and use it only in the WGS weight producer, x-ring base, wgs_seg weight
slots, and host WGS shared-memory calculation. In wgs_seg, conditionally replace
the two old uint2 scale loads with an aligned two-word 48-bit load helper, and
the PRMT scale-pair construction with the exact six-bit equivalent. Every other
CUDA/math path must stay byte-identical or be recovered by structural inversion.

Prepare may add -DOKC_S6=1 only when eligible. Small non-WGS builds keep original
layouts and arithmetic even if the flag appears in their cache key. The existing
main WGS build can be replaced for eligible weights rather than compiling a
second main variant; preserve the 300-second prepare limit. No input/output
caching, held-out seed knowledge or dynamic data inspection is allowed.

Before a GPU check, audit actual full-layout inversion, fallback on all 256
possible scale bytes, code bytes unchanged, all load bounds, stage transaction
sizes, shared-memory bases, compile variants, binding/factory consistency and
the original BF16/FP32 precision guard. Require full MoE 5/6/integrity and clear
evidence the compressed WGS path was enabled; unchanged fallback-only timing
cannot be retained as an optimization. One fixed encoding, no geometry sweep.

## Implemented candidate, GPU unmeasured

build_moe_scale6.py produces source
4d91bbd6d6dc8043ee96c20b07b6afaf5a6156567c53a15b1a99a23ab3b305f0,
179916 bytes, from immutable 33114871. It follows the WGS-only boundary above.
Its ledger reverses every CUDA edit to recover the entire original string.
Reversing just the enumerated host setup, pack and factory-name edits recovers
the entire original Python AST. kernel_fn is unchanged. Full licenses remain.

audit_moe_scale6_cpu.py checks complete 8192-row gate/up and 4096-row output
banks, plus a permuted 512-row bank. It independently recovers 3,276,800 original
FP8 scale bytes, compares 26,214,400 unchanged NVFP4 code bytes, and verifies
byte-identical fallback. All 256 predicate inputs are exhausted. CPU spies of
the actual prepare function confirm agreement between chosen packs, PRMT/S6
flags and main/fallback variants, with four compiled variants in every case.
The real factory, using a fake loader, reuses each format/variant key without
crossing modules. The original BF16 GEMM/FP32 norm guard passes structurally.

NVIDIA documents the exact hi:lo right-funnel semantics used by the load helper:
https://docs.nvidia.com/cuda/archive/12.5.0/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__INT.html
Only shifts zero and sixteen normalize the loaded packet; the pair crossing
word boundaries uses shift24. Scalar shared loads retain four-byte alignment.
Compile-time assertions protect PRMT-domain and WGS layout alignment.

The evidence guard requires full official 5/6/integrity and a true
MOE_SCALE6_WGS diagnostic from prepare. One full H200 check is planned, using
the cached 2.365437416x baseline. Nine MoE checks and seven scored runs remained
at 18:08:22 UTC. No synthetic selector or tile sweep is added to this candidate.
No scored publication is planned without a material measured improvement.
