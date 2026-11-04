# Rotated-code extension: pooler ILP, SK2, scalar tails and router

**Check file:** `candidates/dense_rotcodes2.py`.
Branch `loop/bf16-rotcodes2` starts from the proven rotated-code branch
`df43f08`, not the closed raw-BF16 redesign. The parent's production prefix
is AST-identical except for exporting its own source/packing helpers.
Its successful M=1 selections remain the fallback.

## Evidence and priorities

The supplied `ec0ea8d1555443ce` log explains pooler1's rejection precisely:
17.120 -> 16.672 us, median ratio **0.97383**, paired ratio **0.97700**.
The latter misses the unchanged 0.975 gate. Its ~0.45 us saving resembles
qkv/o/unpooler's ~0.4–0.5 us absolute saving, but pooler's longer weight
stream dilutes the percentage. The event measurements include the graph
cost; the task subtracts ~4 us when scoring. This is consistent with a
bandwidth/launch-tail limit, not evidence of a wrong decoder. GPU stall
attribution remains unmeasured.

From the supplied per-shape profile, all shape weights are 62, so load shares
can be computed directly from microseconds:

- pooler1: 13.30 / 34.19 us = **38.9% of decode1**.
- pooler32 + unpooler32: 29.62 / 50.24 us = **59.0% of decode32**.
- unpooler16: 13.79 / 47.02 us = **29.3% of decode16**.
- Verify256 pooler/unpooler are larger, but run raw cuBLASLt: there is no
  12-bit decoder to improve. Verify128 RQ and the active Chen GV/RQ choices
  at qkv16/32 and pooler16 already implement the rotated `dec1` formula.
  They are not patched or retuned.

## Implemented changes

1. **Pooler1 ILP, same 2-row/512-thread geometry.** Decode both rows' eight
   weights, then alternate their independent FP32 FMA chains. Within each
   output, j/g/element order, K ownership, escape corrections and warp/CTA
   reduction order are unchanged. There is no new exponent window, K split,
   precision, or adoption threshold. A matched serial-rotated vs ILP-rotated
   ABBA diagnostic isolates ILP; registers/local bytes/occupancy are logged.

2. **Rotated SK2 streams.** Repack only the 256-byte code plane of every
   768-byte tile-unit. Low bytes, escapes, row permutations, geometry,
   barriers, MMA order and epilogue are unchanged. Twelve existing native
   BF16 configurations cover M2..8, 16 and 32. The most important active
   targets are pooler32 and unpooler16/32. Each exact M2..7 cell is gated
   independently; no unmeasured tail bucket is forced on.

3. **Rotated scalar M2..8 for o and sliced unpooler.** Keep the scalar M=1
   kernel's per-output FMA/reduction sequence. A one-K-unit activation
   register window limits live registers. Standard-code and rotated-code
   twins are compiled; GPU validation additionally compares the fused
   result to separate original M=1 calls for every token.

4. **Rotated routertail hi+lo c12.** Repack both allowed BF16 hi/lo planes
   losslessly; keep the FP32 accumulation and exact BF16 escape overwrite.
   Standard and rotated twins are built in an isolated optional module,
   without enabling the parent's disabled RT12 mode. M2..8/16/32 must beat
   the actual FP32/raw-hi+lo incumbent, not merely the old c12 version.
   Router FP64 tolerance is 5e-5, stricter than the task's 1e-4. Router1 and
   the larger raw router kernels have no eligible decoder and stay untouched.

This differs materially from closed **19b6b942**: its CG8 lacked the proven
sign permutation; this candidate removes decode work and has exact scalar
order/control twins. It is not a rerun of the unchanged old CG8 route.
The old RT12 path is documented in the parent as 0.8–1.9 us slower; rotated
RT12 may still lose, and is never promoted without the same strict gate.
There are no new GV/RQ pipeline/geometry sweeps, sub-12-bit codecs or raw
GEMM redesigns. The closed redesign is not included.

## Selection and expected effect

The combined candidate first runs the unchanged parent preparation. Every
new arm must pass full code-plane inversion, exact arithmetic agreement
with its standard/matching control, independent FP64 normal/outlier/probe
checks, and changed-input/repeated CUDA-graph checks. Each arm then needs
**both median and paired ratios <=0.975 on three fresh-input/fresh-graph
40-block READ512 ABBA draws** against the actual parent winner. A fresh
validation follows timing; only the best passing arm per cell is retained.
The failing pooler paired ratio is explicitly rejected in policy tests.

Expected target: a few percent on eligible cells, potentially a sub-percent
additional board gain. This is unmeasured. Pooler ILP may be redundant with
NVCC scheduling or increase registers; scalar/router c12 may remain slower
than tensor-core/raw alternatives. Any such cell stays on the parent.
The absolute 270-second budget includes parent preparation and new builds;
scored cells are prioritized over tails. Router build failure is isolated
from the main module. Logs are bounded and preserve the final summary.

## Local gates

Commands run with the repository's CPU Torch/NumPy environment:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/cpu-python scripts/verify_dense_cg_rotcodes.py
PYTHONDONTWRITEBYTECODE=1 /path/to/cpu-python scripts/verify_dense_rotcodes2.py
```

- Unchanged parent test suite passes, including its compiled 7,929,856
  word/window cases and 32 scalar GEMV models.
- 7,929,856 exhaustive BF16/window cases; 7,933,970 random/basis code-word
  window cases; actual Torch permutation/inversion checks pass.
- Actual pooler ILP and serial source bodies compiled as CPU C++20 and
  matched bit-for-bit on **100,000 two-row FMA-order cases**.
- SK2: **720,896 weights / 134 escapes** roundtrip exactly; **132** models
  cover all native geometries, M2..8 and 15/16/31/32. Worst FP64 relative L2
  **0.001958**; probes exact.
- Scalar tails: **42** M2..8 models, exact standard/rotated equality and
  K-order traces. Worst FP64 relative L2 **0.002040**; probes exact.
- Router: **8,388,608 BF16 hi/lo payloads / 2,932 escapes** roundtrip exactly
  across both used geometries. Ten FP64 models have relative L2 <=2.814e-6;
  probe error <=2.733e-6.
- Fourteen fresh BF16 wrapper calls, two router control/rotated launches,
  28 fallback checks and six timing/correctness policy cases pass.
- Both generated translation units pass C++20 **syntax-only** checks with
  CUDA/ATen stubbed and PTX removed. This is not Torch ABI or CUDA compilation.

Artifacts: `results/rotcodes2-offline.json`, readable core/router `.cu`,
marked syntax-only `.cpp`, and pooler CPU `.cpp`. The offline extraction
omits only the parent's eager extension-build expression; it does not
import/patch the real submission or any loaded Torch module.

No nvcc, ptxas, NVRTC or GPU is available. Native compilation, resource use,
hardware graph behavior, prepare time and performance remain unverified.
No OKC operations or pushes were performed. Shared ledger edits are local
to this scoped branch and must be merged by the caller.
