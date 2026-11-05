# Decode SK2 full-wave row tiling

Check file: `kb/exp/decode_qkvfill.py`, based on `c5251b3`.

| Configuration | T | R | CTAs | NT/NWR/WSK/UPW/NACC/CL | Ring stages | Stage bytes | Shared bytes (pmax=64) |
|---|---:|---:|---:|---|---:|---:|---:|
| qkv16fill_r144 | 16 | 16 | 144 | 2/1/4/2/1/1 | 5 | 28,672 | 149,488 |
| qkv32fill_r176 | 32 | 16 | 144 | 2/1/4/2/1/1 | 4 | 45,056 | 190,688 |
| unpooler16fill_r72 | 16 | 16 | 512 | 2/1/4/1/2/1 | 5 | 14,336 | 77,808 |
| unpooler32fill_r100 | 32 | 32 | 256 | 2/2/2/2/1/1 | 3 | 28,672 | 100,432 |

The CUDA kernel is byte-for-byte incumbent SK2, including FP32 accumulators,
K-warp reduction, escape correction, and the single BF16 epilogue store.
Only NT (rows handled by each row warp) decreases. NWR, WSK, UPW, NACC,
CL and ring-stage counts match the corresponding incumbent. Caps were
adjusted to keep stage counts, hence the escape-warp stage assignment.
Actual escape pmax contributes to the runtime shared-memory check.

Constraints: R=NWR*NT*8, N divisible by R, K divisible by WSK*UPW*64.
Thus R=17/18 are invalid; there is no masked row tail. QKV R16 gives 144
productive CTAs, with the q/k/v/gate boundaries 1024/1152/1280/2304 all
aligned. For N8192 no divisor yields 132–144 CTAs: the choices bracketing
that target are R64 (128) and R32 (256). Unpooler16 already has 256 CTAs;
the R16 experiment tests smaller tiles/more waves, not an idle-SM claim.
Unpooler32 increases 128 to 256 CTAs. These are valid >=132-CTA grids,
without introducing padding, partial stores, or a new arithmetic path.

R changes **both** the weight stream layout and escape CTA/local-row mapping.
Both are rebuilt from the original BF16 weights, never borrowed from the
old 24/32/64-row streams. Compression is shared across the two token shapes.

## Full-layer gate

- Separate QKV and unpooler decisions for d16S/F and d32S/F.
- Three fresh-input passes, 16 alternating ABBA/BAAB rounds per pass.
- Every timing replay follows a 512 MiB read reduction, outside timing events;
  current-token KV locations are restored before the flush.
- A base/base null in every pass, with alternating placement before/after
  the candidate timing. All null metrics must be in [0.995,1.005].
- Median of three paired ratios <=0.995; every pass's paired, median,
  ABBA and BAAB ratios strictly <1.000.
- QKV, unpooler, hidden, stream, expert IDs and changed KV bytes must be
  bit-identical, including repeated baseline runs. This is deliberately
  stricter than tolerance/repeatability-based admission, with false negatives
  possible on nondeterministic incumbent outputs.
- Unpooler is gated incrementally on top of any accepted QKV choice, so an
  accepted combined production path is actually tested in-layer.
- Default remains incumbent; failed/incomplete/unsupported gates stay off.
- `DEC_FILL shape layer cfg ctas pass old_us new_us paired null admitted`
  rows include the required fields and additional order/numerical diagnostics.
- Only existing in-process `load_inline`, same explicit base flag helpers;
  optional build overlaps base prepare on a thread. Gate deadline is 275 s
  from prepare entry (10 s margin to the 285 s limit). The inherited base
  prepare and CUDA compilation cannot be forcibly timed out in-process.

## Validation / pending caller

CPU: `python3 -m unittest kb/exp/test_decode_qkvfill_cpu.py -v` (9 tests),
plus incumbent `test_decode_qkv_cpu.py` (7 tests); AST/py_compile, forbidden
operation grep and diff whitespace checks. All pass. Source <1,000,000 bytes.
No local torch, nvcc or GPU: native compilation, numerical identity, speed,
and total prepare wall time are pending caller's H200 check. No submission,
push or main commit by this lane. Ledger search found no closed full-wave
row-tiling idea; local TRIED row records readiness and pending GPU validation.
