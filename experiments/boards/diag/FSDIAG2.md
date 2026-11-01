# FSdiag2: one diagnostic round, no candidate changes

Context: user-fixed host build (remove forced C++17; Torch2.14 ATen needs C++20).
GPU log `fsdiag.71a878e1.log` shows correct FS1 but R1=40.5 vs30.9us,
R8=129.5 vs59.4us; ZERO8=10us. The successful FS kernel reports148/168
registers,0/88B local memory on the judge. The MoE result70fa067e is slower too.
Neither `moe_fs1.py` nor `fsdiag.py` is edited by this round.

## What fsdiag2 measures

Copy of the user's current fsdiag with appended overrides, keeping its embedded
FS1 source and judge-like routing/weight generation. Eight thin-host CUDA TUs:
BASE TINY, BASE TINY2, exact FS1, four independent ablations, and FS timeline.
No C++ standard override on the host or CUDA build: inherit Torch's standard.
Build errors retain compiler lines containing `error`, first600 characters,
rather than the trailing ninja command.

R1/R8,40 cold-L2 repetitions each;256-MiB flush outside events; case and variant
order reversed in pass2. A235-second cooperative budget covers setup/build and
measurement dispatch. Diagnostic-owned output has a hard9000-byte normal limit
plus one short overflow marker, keeping it below10KB. All variants keep the same grid, four TMA slots, weight
volume, inactive scratch clearing, final grid barrier and finalizer. Each has
independent zero-initialized scratch with normal in-kernel replay clearing.
BASE/FS/TL correctness checks run, but ablations are explicitly **not correct
MoE outputs** and never trigger the production prepare guard.

Macros (default0; one at a time):

| Macro | Removed | Deliberately retained |
|---|---|---|
| `FS_NOB` | B shared weight loads/dequant/MMA, shared+upper output accumulation, global output flush | Both A stages/math/epilogue, B TMA+full/empty waits, finalizer |
| `FS_NOFLUSH` | Only global `red.add` operations | A/B math and accumulator/flush reads; replace global reductions with volatile shared sink stores |
| `FS_NOA` | A dequant/MMA and SiLU/Marlin epilogue; write zero inter | Original x/A-code/scale loads made explicitly volatile, A reduction buffers/barriers, B computation |
| `FS_NOACC` | Only shared FP32 y additions | B loads/dequant/MMA via one register checksum and one volatile shared sink per task; upper-register/shuffle accumulation; global flush (shared part zeros) |

Actual volatile shared sink stores prevent dead-code elimination of work whose
final output is intentionally discarded. Empty asm inputs were NOT sufficient:
an initial SASS check found that ptxas removed low-token B math in NOACC and
upper-register liveness in NOFLUSH. NOACC retains a register checksum (extra
FP32 additions); NOFLUSH replaces reductions with cheaper shared stores rather
than claiming a zero-cost sink. NOA uses actual volatile loads. These
are compiler-reoptimized ablations, not fixed-resource cycle subtraction;
resource reports and SASS MMA counts help interpret changed register pressure.

## Timeline definitions

Only `FS_TL=1` adds a trace pointer to its own Params. Lane0 of each warp writes
private timestamps; no additional CTA barriers. Trace `[grid,9,160]` stores up
to64 task A/B pairs and8 segment start/end pairs per CTA. These bounds exceed
the R1/R8 requirements, and the host rejects an overflow or missing task stamp.
The producer's early return is **not** counted as the CTA's consumer exit.

`FS2_TL` reports min/median/max across CTAs, microseconds from the first entry:
- entry before PDL wait;
- route: after routing/barrier initialization and the original initial CTA sync;
- Wfull: earliest consumer observation of first full A slot (after timed scratch
  clearing, accumulator initialization and x loads, NOT physical TMA arrival);
- flush0: first segment flush start, after the existing consumer CTA sync;
- flushN: last segment flush end, after the existing consumer CTA sync;
- barout: consumer release from final grid barrier;
- exit: last consumer lane0 observation just after finalization/stores issued.

`segments=s:start,end` also reports both timestamp triplets for each local
segment ordinal, restricted to CTAs with that segment.

`FS2_TASK` gives every local task ordinal's A-done and B-done min/median/max
over CTAs that own that ordinal. Each CTA's A-done is the maximum over consumer
warps, after both A epilogues/inter synchronization; B-done is maximum after
the W2 compute/accumulation and slot-release arrival. Task counts are summarized
as min/median/max. Exact a/b task-range metadata is checked on the host.

`Bsum` sums A-done->B-done per CTA: includes W2 full wait, B math/accumulator
work and slot release. `flushsum` sums all segment flush intervals; `flush_to_bar`
also includes the publish fence and the grid wait. `tail_active` excludes
nonfinalizer CTAs whose barout->exit observation is under0.5us. It is descriptive,
not a replacement for event timing. TL runs cold once per case in each pass,
separately from the40-repetition uninstrumented timings; tracing perturbs code.

## Likely culprit and concrete next fix (proposal only)

The all-shared four-token accumulator is read/modified/written after **every
K16 task**, even for one-token experts. Each task therefore moves128KiB through
shared memory for its four-token accumulator, in addition to108KiB weights.
The high specialization additionally carries64 upper accumulators across A,
spills under the168-register ceiling, and performs four shuffles per B MMA,
even when a particular expert has at most four tokens. These costs scale with
task count, consistent with the large R8 regression. ZERO8's10us also shows a
substantial fixed cost from routing/clearing/barrier/launch, not A/B work alone.
This is a hypothesis; the ablation/timeline results must establish dominance.

Interpretation:
- Large NOACC gain + Bsum dominance: shared read/modify/write loop is costly.
- Large NOFLUSH gain + flushsum dominance: segment reduction transactions/fence
  contention dominate, not task accumulation.
- Large NOB gain with smaller NOACC/NOFLUSH gains: W2 dequant/MMA address work,
  upper shuffles or the hybrid register spills are the main remaining suspects.
- Large NOA gain: A math/reductions or register lifetime across A dominates.

Concrete proposal if B/upper pressure dominates: **use eight warps total instead
of nine**, with consumer warp0/lane0 replenishing the four-slot ring after the
existing empty-slot arrivals (and the original A reduction sync points), rather
than reserving a ninth producer warp.256 threads admit224 registers/thread
(57,344 registers total); the current288-thread CTA is hardware-accounted as
twelve warps, capping it at168. Earlier device-only224-register compilation
eliminated spills but was unlaunchable at288 threads. At256 it is a plausible
spill-free option; preserve per-task arithmetic and ring parity, then measure
whether relinquishing the dedicated producer overlap is worth it. If NOACC is
the main gain, first predicate shared updates on actual token count (`2*t<cnt`,
`2*t+1<cnt`) rather than computing/storing four tokens unconditionally.

No implementation of those proposals is included in this diagnostic round.
Offline tools: `.fs_offline/build_diag2.py`, `check_diag2.py --compile`, and
`test_diag2_run.py`. No local GPU or implementation commits.

## Offline result

Final `fsdiag2.py`:102,680 bytes; SHA256
`3c0b2868abeacfb4f7c66ce0df1c5a01f70df55aa7041aeb21b96030812cb071`.
All8 device TUs/16 kernel instantiations compile with NVRTC/ptxas sm_90a,
C++20. This is not a full Torch host build or a local GPU run.

| Variant | Registers low/high | High stack B | High spill store/load B |
|---|---:|---:|---:|
| FS |151/168|80|112/128|
| NOB |140/140|0|0/0|
| NOFLUSH |143/168|48|88/96|
| NOA |110/167|0|0/0|
| NOACC |148/168|64|88/104|
| TL |159/168|120|192/244|

Low specializations have zero stack/spills. SASS verifies that NOACC and
NOFLUSH retain FS's MMA instruction count, while NOB and NOA remove only the
intended phase's MMA instructions. TL increases high-path spilling: treat its
absolute timings as perturbed, not substitutes for uninstrumented event times.
The BASE resource reports are146 registers,0spills in this offline toolchain.

CPU tests cover unique host linkage, original FS control source identity,
no added CTA barriers, exact timeline aggregate math (including producer
exclusion), compiler-error extraction,24 timings with40 repetitions and both
orders, and four timeline/task-summary lines. Mock log3965B leaves room for
real multi-task trace text below the prior9400B budget. Mock values are not
performance evidence. Reports/SASS are under `.fs_offline/diag2/`.

Frozen-file hashes verified unchanged after generation/testing:
- `moe_fs1.py`: `1e8b87610b13ac77dbc44769bfd76b892a690170eeb91aedb0c739a5539e7116`
- `fsdiag.py`: `5fa0d01ba81fb2e1f78994776cab7070716a0520e2a30f5ee67be3dda2568cbf`
