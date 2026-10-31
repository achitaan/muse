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

M1 and the M2 one-route fixture have no aggregation opportunity. Any future selector must preserve the same numerical and paired speed gates; no exception for these shapes. At the initial screen there was no runnable Python payload. The subsequent packaging below remains unrun and not GPU-admitted.

Self-review: ownership flushes use old-job metadata, empty helper intervals flush nothing, invalid entries remain masked, the original compute-tail and finalizer remain, and no timed host logic or persistent expanded weights are introduced. The scalar and integer models have narrower scope than the complete CUDA contract. A future package would still need actual resource guards, exact buffers, approved norm/output limits, replay checks and full official evidence. Keep frozen shared candidate e5c0e7ec first; this is a separate alternative, not another lowering revision to it.

## Separate executable payload, still unrun

Payload `experiments/decode_moe/moe_norm_register.py`, SHA256 `21a652ccf6f5dd4bc72bea7b68d30654a639129f23fb9a110b732e67897bd537`, is 338502 bytes. Its packaged CUDA SHA256 is `72611e8617e1d25f2f9af5b37ac5fd3c5b960f480c2b4ec17d2aae70109ad007`; the screened device prefix is byte-exact. The host resource bridge is the previously checked bridge under the distinct `norm_reg` namespace and cache name. Frozen grouped-tooling template hashes are recorded in tooling-template-binding.json.

The compiler's exported constexpr values give 36576 bytes for Smem, 36608 aligned, and 214784 total dynamic shared bytes in both tiny families, with 288 launch threads. The real loader will require those values, zero local/static shared bytes and at least one block/SM before any new-path probe. Seven simulated native-resource guard cases pass. No actual device resource query has occurred.

The parent prefix, all original definitions, shared harness, byte/variance gates, route mutations, prepare selection and timed dispatch match the retained gate by AST. Private compilation is absent from timed launch. There is no mutation of a loaded extension's attributes. Runtime validation still requires all 120 fixtures, both populations and arm orders exceeding 3%, full official tests/benchmarks/integrity and profile marker `norm_reg::moe_reg_fused` on selected scored cases. Empty selection is not a gain. M1 and M2 are not granted exceptions to the unchanged gates.

Self-review after packaging: the constexpr report agrees with the loader guard; source bytes have an explicit no-conversion Git attribute; template transformations do not alter the shared candidate or its builder. The existing shared payload e5c0e7ec remains first. No automatic launch, score or integration is authorized by these CPU artifacts.

## Terminal H200 result

Check9b9b978eed5b45d4 passes5tests/6benchmarks/integrity and120private numericalfixtures. Actual144registers,zero local/static,214784dynamic,288threads,oneblock/SM;prepare167.163s. No32pairedcomparison meets3%;choicesempty;all6officialprofilesparent. Fallbackpoint2.3421325305x is not a new-path gain. Reject;close norm aggregation;no score,rerun or Decode transfer. See outcome-9b9b978e.json.
