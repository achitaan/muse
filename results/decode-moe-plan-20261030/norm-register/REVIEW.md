# Helper-local norm accumulation: CPU prototype

CUDA source: `53f6c1560886441c4a2541a482ea8fa36b022c415ec30ad70c4bd8a2c04c85cb`.
Parent: retained `best/moe_verified.py`, SHA256 `331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50`.

The authorized read-only researcher identified two FP32 sums and one active job index per helper lane. This prototype changes only `ws_helper` and adds a compile-time static-tiny guard. Compute-tail updates, routing, GEMMs, SwiGLU, BF16 boundaries, ybuf stores, phase-A completion, finalizer and barriers remain in their parent source. An exact inverse restores the helper. No shared storage or extra scan/barrier is introduced.

On a phase-B job transition, lanes gg==0 flush the previous sums using the previous job's entry start and count, then reset. Each lane owns entries 2*tt and 2*tt+1. The last partial sum flushes before the existing threadfence/join. Group by routed job/entry, never original token ID. Tiny's existing no-fat-job prerequisite remains; this does not extend duplicate-route coverage. Each valid unit addend explicitly flushes subnormal inputs before register addition to preserve global atomic input semantics.

All four offline TINY/TINY2 PRMT/LUT variants compile: PRMT146 and LUT141 registers, zero stack/spills, no warnings. These are compiler results, not actual residency. The 396-case integer model verifies complete helper-plus-tail coverage and transition/final flush contributions across jobs0..32 and grids1/2/7/132/148/512. It does not prove CUDA race freedom. The 144-case positive-addend model, including invalid-slot NaNs and a missing-FTZ mutant, has maximum relative norm change4.976019108e-7 against a scalar FTZ baseline. It is not GEMM, output or device validation.

The timing-fixture metadata model uses local_count32 and the existing seed7 route generator. Logical norm atomic counts below are not bytes, exposed latency or predicted gains.

| M | Population | Jobs | Parent arrivals | Register-group arrivals |
|---|---|---|---|---|
| 1 | one | 1 | 256 | 256 |
| 1 | skew | 1 | 256 | 256 |
| 2 | one | 1 | 512 | 512 |
| 2 | skew | 2 | 768 | 594 |
| 3 | one | 3 | 768 | 396 |
| 3 | skew | 5 | 1536 | 480 |
| 4 | one | 4 | 1024 | 396 |
| 4 | skew | 5 | 1792 | 558 |
| 5 | one | 3 | 1280 | 440 |
| 5 | skew | 5 | 2304 | 483 |
| 6 | one | 5 | 1536 | 320 |
| 6 | skew | 7 | 3072 | 463 |
| 7 | one | 4 | 1792 | 462 |
| 7 | skew | 7 | 3328 | 498 |
| 8 | one | 5 | 2048 | 427 |
| 8 | skew | 6 | 3840 | 660 |

M1 and the M2 one-route fixture have no aggregation opportunity. Any future selector must preserve the same numerical and paired speed gates; no exception for these shapes. There is no runnable Python payload, GPU admission, GPU check, score or measured speedup for this prototype.

Self-review: ownership flushes use old-job metadata, empty helper intervals flush nothing, invalid entries remain masked, the original compute-tail and finalizer remain, and no timed host logic or persistent expanded weights are introduced. The scalar and integer models have narrower scope than the complete CUDA contract. A future package would still need actual resource guards, exact buffers, approved norm/output limits, replay checks and full official evidence. Keep frozen shared candidate e5c0e7ec first; this is a separate alternative, not another lowering revision to it.
