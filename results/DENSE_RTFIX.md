# Dense fixed router defaults

Candidate: `candidates/dense_rtfix.py`, from `a1125e5` consolidation.

- Router 8 uses rt8w8 and router 16 uses rt8w16 after correctness only;
  there is no timing gate, null estimate, capture or confirmation for them.
- One prepare-only call per normal/outlier/one-hot/zero batch, deterministic
  per-cell RNG. FP32 output must match the full-original-weight FP64 equation
  with global relative L2 <= 5e-5 and row-floor error <= 2.5e-4; both outputs
  and references must be finite. An exception retracts only that router cell.
- `DENSE_RTFIX cell=... arm=... err=... row=... bound=... row_bound=... used=...`
  reports each decision, including the fallback reason.
- The board's entire outer tuner, prepare and dispatcher are restored verbatim.
  Consolidation's bounded factory still exposes exactly qkv16 t16s5curP-k7,
  qkv32 t32s5curP-k7, pool32 t32s6curP-k4, qkv256 n256s5m1hzm-k7.
  These use the original board correctness checks and both 40-block ABBA passes,
  with median AND paired ratios <= 0.988. No stricter null-MAD re-audit remains.
- All factories/native math/packing/build flags are AST-identical to the supplied
  consolidated base, including its C++20 SM90a extension settings. The inherited
  allocation wrapper is always unbound; there are no controlled borrowing draws.

Validation: `scripts/verify_dense_rtfix.py`, result `dense-rtfix-offline.json`.
CPU tests cover exact board gate lineage and threshold/confirmation behavior,
8,388,608 exact hi/lo BF16 pack payloads across both geometries, the FP64 hi+lo
model (relative error 2.50e-6), synthetic-input safety failures, independent
router fallback, original CURP selection preservation, fresh production outputs
and dispatch guards. AST parse, forbidden process/network grep, and diff checks
pass. Candidate contains no process starts or network imports/calls.

No CUDA/nvcc locally: actual H200 compilation/correctness, speed, and prepare
duration remain unverified. The unchanged board tuner has a cooperative 270s
budget; the small router pack/validation pass follows it within the remaining
30s judge allowance, but that duration still needs the caller's check.
No check submitted, no push, and no commit to main.
