# Candidate implementation packet: FINREG and O128 (October 29, 2026)

Branch `opus/fresh-investigation-20261029`. This builds on the reviewed handoff
`HANDOFF.md` (commit `827b1e2`). All of this is local implementation, CPU tests
and CPU-only NVRTC compilation.

- **No** official check, scored run, GPU job, kernel load or launch, API call,
  subagent or merge was performed.
- All five `best/` files and `candidates/` are unchanged.
- **No performance gain is claimed for any candidate.**

## Feasibility decisions for the four reviewed ideas

| Idea | Decision | Reason (analysis, not measurement) |
|---|---|---|
| Keep the MoE WGS phase-B finisher's own partial in registers | **Implemented: FINREG** | Precise edit inside `#if OKC_WGS` only. Proof and checks below. |
| Lossless 12-bit dense weights at M=128/256 | **Implemented, gated: O128** | Reuses the existing exact packer and kernel. Enabled per shape only after prepare-time exactness and timing gates, because of a real L2 risk (below). |
| Earlier per-job Phase-A release (`skinny_ring_ws`) | **Not implemented; the earlier claim is withdrawn** | Static equal-cost ranges end Phase A at about the same time on every CTA. Any job spanning an interior CTA completes only at that CTA's range end, which is the phase end. Releasing the boundary CTA's credit earlier therefore cannot make `doneA[j]` complete earlier. Expected benefit is about zero. |
| Reorder WGS Phase-A finishers (`OKC_WGSDFIN=3`) | **Not implemented** | The retained `OKC_WGSREORD` deliberately runs each CTA's trailing partial unit first, so split units finish early for Phase-B overlap; the source records this as tuned. Reversing it delays those units. Keeping the Phase-A last-arriver partial in registers would need a new two-counter protocol: higher race risk for a small payload. |
| Shared-memory/`float4` staging of router input (`route_fp32`) | **Not implemented; the earlier claim is downgraded** | The redundant per-warp `x` reads hit the same 128 B lines and are likely served by L1. Staging adds a CTA barrier before the first FMA. It also does not address the measured problem: contention with the concurrent pooler. NVRTC compiles it (48 regs, 0 spills), but there is no credible mechanism for a gain. |

## Candidate files

All candidate files are in `experiments/opus_fresh/candidates/`. They are built
by `build_candidates.py` and recorded in `manifest.json`.

| Candidate | Parent (SHA256) | Candidate SHA256 | Bytes | Change |
|---|---|---|---:|---|
| `moe_finreg.py` | `best/moe_verified.py` `33114871…` | `fe72a4b2…63cad7` | 179,267 | FINREG CUDA edit, plus extension name `pair1dq8` → `finreg1` |
| `decoder_finreg.py` | `best/decoder_verified.py` `d9fe7558…` | `267fe3e1…7c5121` | 989,346 | Same CUDA edit and rename only |
| `dense_o128.py` | `best/verified.py` `edc61a1e…` | `b638ca2a…5b03ab7f` | 644,219 | Parent bytes as exact prefix, plus `o128_common.py` and a prepare/kernel_fn wrapper |
| `decoder_o128.py` | `best/decoder_verified.py` `d9fe7558…` | `e55b3821…051805` | 995,461 | Parent bytes as exact prefix, plus `o128_common.py` and a `prepare`/`_achita_dense_project` wrapper |

Notes:
- The finreg CUDA hash is `f2436030…eeba2` (parent CUDA `1e60cf46…`, the
  retained payload).
- All files are under the 1,000,000-byte service limit. `decoder_o128.py` has
  4,539 bytes of headroom.
- FINREG and O128 are kept as separate files. No combined Decode file was built.
  A combined file would be about 995.8 KB.

## FINREG: what changes and why it is correct

**Change.**
- In `wgs_seg`, the phase-B designated finisher (the split share with `s0==0`)
  no longer `red.global.add`s its own partial into the fixup buffer and then
  reads it back in `wgs_flush`.
- Instead, it waits for the other shares' stage counts (the same `fc`
  acquire-spin).
- It then does `bar.sync(3+wg)`, `fence.acq_rel.gpu`, and loads the other
  shares' summed partial (`__ldcg`). It zeroes the buffer, adds into its
  registers, and zeroes `fc`.
- Finally it falls through to the existing unrolled `wgs_epi_tile<PB>` loop.
- `wgs_flush`'s phase-B DF branch keeps only the non-finisher release.
- Phase A is untouched (DF bit0 = 0, atomic last-arriver path unchanged).

**Structural argument.**
- Phase B has no range reorder (`wgs_split` reorders only `pb == 0`).
- A split share with `s0==0` covers `[unit start, g1)` with `g1` inside the
  unit, so it is necessarily the last segment of its CTA's phase-B range.
- The 20,303 synthetic split units checked earlier agree with this.
- Non-finisher shares never wait on the finisher. The finisher waits on the
  same counts as before, at essentially the same point in time.
- Deadlock-freedom therefore rests on the same co-residency assumption as the
  retained kernel.
- Fixup-slot uniqueness is unchanged: one stage-0 partial unit per CTA per
  phase.

**Numerics.**
- The retained FP32 atomic sum order across shares is nondeterministic. FINREG
  computes `finisher + (sum of other shares in atomic order)`.
- A CPU model over 20,000 random 2–3-share units found **0** results outside the
  retained protocol's admissible set. The inputs included signed zeros and
  magnitudes from 1e-3 to 1e3.
- BF16 epilogue, global scale, norm gain and all other boundaries are untouched
  (same `wgs_epi_tile` call).

**Measured offline (CPU-only NVRTC 13.0.88 sm_90a, cache disabled).**
`candidate_verification.json`:

| Build | Registers | Stack | Spill store/load | Cubin |
|---|---:|---:|---:|---:|
| MAIN WGS PRMT, baseline | 96 | 16 | 56/124 | 3,859,728 |
| MAIN WGS PRMT, FINREG | 96 | **8** | **52/84** | **3,695,760** |
| TINY / MID PRMT | 146 / 150 | 0 | 0/0 | Identical cubin SHA256 to baseline when compiled without line info |

- No C751x serialization warnings.
- The inverse edit restores the parent CUDA exactly.
- The candidate AST is unchanged outside the CUDA constant and the extension-name
  literal.

**Expected effect (hypothesis).**
- The change only touches the end of phase B on the WGS path: M>16, that is,
  decode32, verify16 and verify32.
- It removes one `red.add` burst (about 16 KB per finisher warpgroup set) and
  the read-back round trip on the finisher's critical path.
- Plausible size: on the order of a microsecond per call. **Unmeasured.**

## O128: what changes, and the risks the gates address

**Change.**
- For exactly M=128 and M=256 on `qkv_gate`, `pooler` and `unpooler`, the
  candidate adds a `BN=128` instance of the existing lossless 12-bit
  RS-WGMMA `m32w_gemm`.
- This is one `if constexpr(BN==128) mma128(...)` line plus a generated
  `m64n128k16` helper.
- It covers three `(WK, split)` variants, using the existing
  `tt_mw_pack_weight`, `tt_mw_verify_onehot` and `tt_mw_launch_raw`.

**Gates.** In `prepare`, per layer and variant:
1. Resources: `localSizeBytes == 0`.
2. Bitwise one-hot reconstruction at M=128 and M=256.
3. A random-activation check against an FP64 reference: global relative L2
   < 2e-3, worst row < 1e-2, finite, including a zero row.

Then per (M, layer), a paired cold-L2 CUDA-graph timing must beat the incumbent
by more than 3% (the retained WG-selector policy). The incumbent is timed before
and after the candidates, and the minimum is used.

**Dispatch.**
- Only on the exact `(M, layer, K)` key, with matching `n`, a contiguous BF16
  input and a 16 B aligned pointer.
- Nearby shapes (for example M=129), sliced unpooler rows, the router and all
  other layers use the incumbent. The CPU control-flow test with mocks confirms
  this.

**Preparation budget.**
- The extension compiles on a background thread started before the base
  `prepare`, so `nvcc` overlaps the parent's builds.
- Selection is skipped (incumbent kept) once elapsed preparation exceeds 250 s
  (Dense) or 235 s (Decode).
- The Decode parent historically prepared in 215–257 s. **The Decode variant may
  frequently fall back entirely.** That is a fallback-only run, not evidence
  about the kernel.

**Measured offline.**
- NVRTC compiles both distinct template instances (`<128,64,1,2,2,true,2>` and
  `<128,64,2,2,2,true,1>`) at 141 registers, with 0 stack, 0 spills and 0
  warnings, for both the Dense and Decode sources.
- Dynamic shared memory per the `launch_t` formula is 34,816 B (WK1) and
  69,632 B (WK2).
- A CPU model of the wgmma `m64n128` D-fragment epilogue covers every
  (token, row) exactly once per warpgroup. The split/WK reduction loop covers
  every item exactly once.
- Static name binding finds no unbound globals.
- Synthetic weights from the task generator have about 0.014% escapes. That is
  about 9 per 16×4096 group, against the packer's limit of 64.

**Principal risk (analysis, unmeasured).**
- `m32w_gemm` assigns 64 weight rows per CTA. At M=128 each row tile re-reads
  the whole 2 MB activation from L2:
  - about 134 MB of L2→SM traffic for pooler/unpooler, against 50 MB of 12-bit
    weights from HBM;
  - about 268 MB at M=256.
- Each `mma` helper also waits synchronously (`wait_group 0`).
- The kernel may therefore be L2- or latency-bound and lose to cuBLAS. The
  timing gate exists for this reason.

Other unresolved risks:
- A CUDA fault during selection would be sticky and cannot be caught by the
  `try` block.
- Concurrent `load_inline` builds on a thread are untested on the judge.
- The one-hot check rejects any weight containing −0.0, because FP32 sums
  produce +0. That causes a fallback, not an error.

## Tests performed

1. `build_candidates.py`:
   - Parent hashes equal the retained hashes.
   - Inverse edits restore the parents byte for byte (FINREG).
   - The parent is an exact byte prefix (O128).
   - AST outside the edits is unchanged.
   - Sizes are under the limit.
2. `verify_candidates.py`, with results in `candidate_verification.json`:
   - NVRTC MAIN/TINY/MID baseline-versus-candidate builds.
   - TINY/MID cubin identity without line info.
   - O128 device compile from each candidate's own constants.
   - Host dispatch case count.
   - Static name binding.
   - Shared-memory budget.
   - Epilogue coverage model.
   - FP32 order model: 0 mismatches in 20,000 trials.
3. `test_o128_control.py`, with results in `o128_control_flow_test.json`, using
   CPU mocks:
   - Without CUDA, `prepare` keeps the incumbent.
   - Empty choices use the incumbent.
   - The new path is taken only for the exact key. M=129, other layers,
     non-contiguous input, FP32 input and sliced unpooler rows all use the
     incumbent.
4. `py_compile` of all four candidates. Each keeps its `# okc: problem=` first
   line.

**Not tested:** GPU compilation of the host extension, H200 correctness, races,
CUDA-graph replay, occupancy, preparation time, peak memory and any timing.
These require a separately authorized GPU evaluation.
