# Independent attempt 2: M=1 scalar GEMV rotated-sign codes

## Candidate and independence

**Check file:** `candidates/dense_cg_rotcodes.py`.

Branch `loop/bf16-cg-rotcodes` starts directly at **63e253f**, not at the first
candidate's commit. The first candidate, its branch and its pending check
`8ba14018820847fd` were not changed or queried.

This candidate copies `board_best/dense_kbv2_scored_1.5886.py` and changes only
the factory's export of its own source/helpers before appending a standalone
candidate/gate. Existing production logic, including the original v2 tuner
and nearby-shape routes, is AST-identical. There is no token-owned epilogue,
new route geometry, router/MoE/attention change, or change to SK2.

## Idea and why it is not a closed repeat

The scalar M=1 `cgemv_kernel` reconstructs every eight BF16 high bytes by
masking/sign-shifting/merging two words. Repositioning the code-word's eight
sign bits during untimed weight preparation makes the two reconstructions:

```c++
ha = (C & 0x87878787u) + base4;
hb = (__funnelshift_r(C, C, 4) & 0x87878787u) + base4;
```

All low bytes and exponent offsets are unchanged. High-half sign bits wrap
within each 32-bit word. The base is <=120, so the byte-wise additions cannot
carry across bytes. Each word permutation is independently inverted and
checked in prepare before any candidate kernel can run. Stored weights remain
lossless **12 bits/value**; escaped original BF16 payloads and corrections are
unchanged. All FP32 FMAs and reductions execute in the same order.

This representation is already described as `dec1` in the incumbent's
chenwainuo register-A kernel. The new work ports it into the **different,
scalar row-major M=1 cgemv**, whose original decoder still uses the standard
sign-in-each-nibble representation. This is not a claim that the representation
itself is novel.

Relevant closed ledger rows and material differences:

- `33c77700`: a new simple persistent-TMA GEMV family lost 1.7–10x. This keeps
  the existing tuned scalar kernel, geometry, loads and escape warp.
- `ee7d7b81` / `cbd91613`: sub-12-bit ZS codecs added expensive decode/escapes.
  This retains 12 bpw, the same byte count and the same escape path, and aims
  to **remove** integer decode instructions.
- `19b6b942`: M=8 cgemv for unpooler/o lost. This changes M=1 only, without
  reusing the rejected multi-token implementation.
- `83c46449` / ledger's closed register-A pipeline sweeps: ST/KS/geometry or
  pipeline tuning. This is a representation/decode change in the scalar
  FMA kernel, not another register-A or route search.
- First candidate: changed SK2 epilogue communication. This leaves every
  epilogue unchanged and targets only scalar decode arithmetic.

## Expected effect and bounded GPU gate

Targets: **pooler@1, qkv_gate@1, o@1, unpooler@1**, using each existing
`CG_CONFIGS` geometry. No new variants are swept.

Source-level high-byte reconstruction goes from approximately eight integer
operations to five per eight weights; PRMT, FP32 FMA and memory traffic do not
change. For pooler this could remove about 12.6 million scalar integer ops
(roughly 0.8 us at the aggregate INT32 issue ceiling, not a measured saving).
**2–5% per-cell improvement on pooler/qkv is a plausible target, not a speed
claim.** Overall board improvement would be much smaller because only the
M=1 load changes. The sliced unpooler's incumbent may remain faster.

Compiler fusion, bandwidth overlap, or instruction scheduling can make the
real saving negligible. There is no SASS/PTX instruction-count evidence here.
One check is appropriate; if it adopts nothing, close this exact approach
rather than weakening the gate or sweeping launch geometries.

Prepare checks exact agreement with the original scalar kernel on normal,
outlier and multiple one-hot columns, plus independent FP64 accuracy and
repeated graph replays. A cell is adopted only after **three independent
fresh-input/fresh-graph draws**, each passing both median-time and paired
ratios **<=0.975**, using the existing 40-block ABBA and 512 MiB vector READ
flush. Post-timing exactness/replay validation uses fresh inputs again.
All failures retain the incumbent. The 270-second total budget includes the
unchanged parent prepare; budget exhaustion can leave no cells adopted.

## Local gates

Run with CPU Torch and NumPy:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/cpu-torch-python scripts/verify_dense_cg_rotcodes.py
```

Evidence: `results/dense-cg-rotcodes-offline.json`.

- Incumbent AST preserved apart from source exports; scalar kernel body is
  literally identical after reversing the two decoder-line substitutions.
- Every one of 65,536 BF16 payloads tested at all 121 legal window bases:
  **7,929,856 payload/window cases**. Another 7,933,970 random/basis code-word
  window cases passed; the 32-bit sign permutation is bijective.
- The actual Torch packer matched the independent NumPy permutation on
  65,570 words and rejected malformed/noncontiguous code metadata.
- The actual new decoder expressions compiled with **clang++ C++20** and
  passed **7,929,856** CPU word/window cases, using a CPU funnel-shift model.
- 172,032 BF16 weight payloads, including 52 escapes, reconstructed exactly.
- 32 complete fixed-order scalar GEMV models across the four native
  specializations: normal/outlier worst relative L2 **0.003047** versus FP64;
  zero outputs and all tested probes exact. Old/new modeled outputs agree.
- Four fresh-output wrapper pairs, four control calls, 20 fallback cases,
  and six mocked timing/correctness admission cases passed.

**Not verified:** CUDA host/device compilation, actual GPU instruction count,
registers/spills, GPU FMA behavior, CUDA graphs/replay, prepare time, or H200
performance. nvcc/ptxas/NVRTC and a GPU are unavailable. CPU C++ compilation
does not establish CUDA compilation. No OKC calls or pushes were performed.

The generated readable `.cu` and CPU decoder `.cpp` are verification artifacts,
not submission files. `TRIED.md` was updated only in this scoped worktree;
the caller must merge the result row into the shared ledger.
