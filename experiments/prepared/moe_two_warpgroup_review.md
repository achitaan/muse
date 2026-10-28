# Fixed two-warpgroup WGS experiment

> Closed: check0fb1da41657d4346 passed full5/6/integrity with real two-CTA
> occupancy and no local storage, but lost on all three changed loads. Whole
> speedup2.045556897x; discarded, no score. Retain33114871. See phase13 evidence.

Main chat only. Parent is MoE33114871, checked2.365437416x/published2.348299935x.
No controller or GPU job is active when preparing this packet. The top50overall
Goal remains active. Latest API MoE rank54/top50cutoff2.3557; Decode/overall147,
with older omitted-boundary publications pending withdrawal approval. Do not
retry that rejected external action without the user's response.

This is a coupled architecture change: two128-thread consumer groups,64producer
threads,32-token jobs, two X-ring slots, half-sized weight units and264CTAs on
132SMs only if actual resource queries prove at least two active blocks per SM.
It replaces the original fast main module; tiny/MID and ineligible-scale fallback
modules remain byte-for-byte original. No additional old fast-main compilation.
Use an independent main weight/scratch bank, final prepare synchronization and
the original BF16 Marlin epilogues, FP32 norm storage, SwiGLU and route weights.

The public profile motivation is MoE60.5% of corrected Decode's M256 summed
GPU time. More blocks may hide producer/dequantization latency and improve
load balance. The cost is more32-token chunks and potentially repeated weight
traffic at larger expert counts. L2 reuse is not measured; this may lose. A
timing result must decide. This is not a repeat of the prior FT_T32-only trial.

The CPU ABI model predicts110848bytes of dynamic shared memory, including the
changed JMAX136 and NW8 metadata. The old declaration reproduces the previously
measured231168byte allocation. Two320-thread CTAs at96registers need61440
registers. These arithmetic checks do not establish CUDA occupancy. Prepare
reports actual register/local/shared/occupancy attributes for both BIG variants;
the host launch rejects a grid beyond that capacity or its272fixup slots.
[NVIDIA's launch-bounds documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
describes the compiler register bound. Actual occupancy remains a runtime check.

Wider ranges exposed a real uint32 overflow: valid384-token all-local counts
97on31experts and65onone yield127jobs, total19040256 and product5026627584
at264CTAs. Use full uint64 numerators and an FP32 starting quotient corrected
with exact integer comparisons. Owner floor((xe*G-1)/tot) follows directly
from floor(tot*c/G)<xe. Empty-job ranges return zero without division. This is
integer scheduling, not a change to activation or weight precision.

Pre-check CPU evidence:

- Full864MiB generated packed A/B byte permutation round-trip, plus the actual
  packaged Torch repack checked independently on56.6MB. All stage/group pieces
  remain whole, preserving gate/up pairing and every code/scale byte.
- Both epilogues and two-pass gather/final stores cover all4096columns. The
  gather's per-pass job cursor is reset; a stale-cursor mutant is detected.
- 5,797,128wide quotient/owner checks including overflow and an injected two-ULP
  estimate envelope; exact correction matches the independent integer oracle.
- Local ring BFS has no dead ends at two X slots; a one-slot mutant deadlocks.
  Global fair-interleaving protocol model executes3,742,896actions, including
 264CTAs, deferred flushes, designated B finishers and split-slot ownership.
  Every case completes with eight A arrivals per B stage and cleared counters.
  This does not prove CUDA memory ordering; original release/acquire operations
  remain unchanged, and a full GPU check is mandatory.
- FP32 finalizer model versus full-domain FP64 norm has maximum relative error
  1.11e-7; a missing-upper-half mutant is caught. This is not CUDA FMA identity.
- Five mock compiler-hook tests prove new/old source wiring;24host dispatch
  cases cover thresholds, fallback and chunk tails. These are not GPU tests.
- The original CUDA literal and unrelated base AST are unchanged. The alternate
  CUDA's95edit ledger reverses exactly to the parent. No hidden state, real-call
  input probing, private harness access, precision shortcuts or cached outputs.

Prepare also runs three owned analytic all-remote cases at M17/33/384, expecting
exact zero while exercising scratch reuse and both BIG variants. It retains no
probe tensors or outputs; they never supply an actual timed call's result.

Before checking, commit the packet, initialize one scope candidates/moe_active.py
with the cached331baseline and scripts/verify_moe_two_warpgroup_evidence.py,
then install the frozen draft and use controller finish. One primary check,
no duplicate baseline. Guard requires full5tests/6benchmarks/integrity, actual
okc_w2g kernels in all three affected profiles, analytic-zero success, at least
two active CTAs for both real variants and changed-load geometric speedup>1.02
with no affected ratio below0.99. Controller additionally requires whole metric
improvement. Report local/spill bytes honestly; no post-result threshold changes.

The result is archived in results/resumed-moe-phase13-20261028/. The achieved
resource change did not improve performance: changed-load ratios0.80046,
0.74681,0.72405; geometric ratio0.75644. No scored confirmation or Decode
transfer is justified. The overall Goal remains active. Do not repeat this
architecture or blame only FT_T32 weight repetition: M32 also lost sharply.
