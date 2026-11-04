# Large-M lane: one token-slab attempt

Branch: `loop/dense-m128256-decode`, based on `63e253f`.
Candidate/check file: `candidates/dense_large_slabs.py` (self-contained).
No okc invocation, submission, push, or main commit is performed by this lane.

## Evidence and roofline interpretation

Measurements: RESTART lines in the supplied
`dense_redesign.646c489f50b34d66.log`; actual incumbent kernel identities are
confirmed by profiles 20–28 in its adjacent `.txt`. Numbers below use decimal
TB/TF and **unique minimum traffic**, including one activation read/output write.
They exclude c12 escape metadata and partial-reduction traffic. Consequently
these are effective bandwidths, not measured HBM counters. Real traffic can be
higher; neither an ALU stall nor an HBM stall can be established from time alone.

The event measurements include roughly 4 us graph overhead, whereas the scored
profile times have replay cost subtracted. Both denominators are reproducible
with `python3 scripts/analyze_dense_large.py`. Approximate work-only numbers:

| M | layer | incumbent | work us (~event-4) | TFLOP/s | effective TB/s | raw/c12 floor us |
|---|---|---|---|---|---|---|
| 128 | qkv_gate | c12 rq CURP | 11.936 | 404.8 | 2.597 | 8.42 / 6.46 |
| 128 | o | raw ol_projection | 6.704 | 320.3 | 2.854 | 3.99 / 3.11 |
| 128 | pooler | c12 rq CURP | 16.880 | 508.9 | 3.168 | 14.64 / 11.14 |
| 128 | unpooler | c12 rq CURP | 17.248 | 498.0 | 3.100 | 14.64 / 11.14 |
| 256 | qkv_gate | pinned raw cuBLASLt | 17.296 | 558.7 | 2.493 | 9.77 / 9.77 |
| 256 | o | pinned raw cuBLASLt | 8.528 | 503.6 | 2.521 | 4.48 / 4.34 |
| 256 | pooler | pinned raw cuBLASLt | 24.288 | 707.3 | 3.022 | 17.37 / 17.37 |
| 256 | unpooler | pinned raw cuBLASLt | 24.096 | 713.0 | 3.046 | 17.37 / 17.37 |

The `.txt` independently reports ~11.7/6.5/17.3/17.4 us at M128, and
~17.9/7.8/24.4/24.1 us at M256, consistent with this overhead correction.
The board file can gate qkv256 onto rq, but this particular comparison did not:
profile 25 is `nvjet_sm90_tst_72x64_64x12_1x1_v_bz_TNN`, not c12.

* M128 qkv/pooler/unpooler are **bandwidth-favored by the roofline**; their
  ~54–66% effective bandwidth is a real gap, but includes startup, decode,
  activation rereads from L2, per-stage issue/waits, and split-K epilogue time.
  `o128` is bandwidth-favored in the ideal model, but its small ~19 MB footprint
  and 6.5–6.7 us work time make startup/tile scheduling especially relevant.
* M256 qkv/pooler/unpooler are **compute-favored by the unique-byte roofline**,
  albeit only marginally for raw BF16. M256 o is marginally bandwidth-favored
  (4.48 us bytes versus 4.34 us math), with substantial startup/tile overhead.
  Pooler/unpooler achieve ~72% of
  peak, not a hard floor; qkv/o achieve ~56%/~51%. The raw Lt winners have
  **no 12-bit decoder**. Calling their remaining gap decode ALU cost would be
  wrong. At these sizes the opposing HBM/math demands, skinny-GEMM issue,
  occupancy, startup and tails matter together.
* The old rq256 alternatives are slower (log lines 35–64): pooler best ~30.3 us
  versus 28.3 event, unpooler ~30.8 versus 28.0. Those use 128 weight rows x
  all 256 tokens, two math WGs, 128 FP32 accumulators/thread, one CTA/SM.
  Moving to c12 only saves bytes; it does not fix this consumer geometry.

## Why the incumbent works

rs128 swaps operands: compressed W becomes register A; activations are TMA-loaded
SW128 shared B. Two math WGs own disjoint 64-row weight slabs and reuse each
decoded fragment across all tokens. Large bulk W stages and producer/consumer
ring barriers overlap loads and math. At M128, per-thread escape cursors and
short high-byte patching remove serial warp-list work. KS=4/2/7 for
pooler/unpooler/qkv gives 128/128/126 CTAs, avoiding a short second wave. The
one-wave grid also permits the existing in-kernel zero handshake and TMA
reduce-add epilogue. The cost is rereading X per output tile from L2, stage
bookkeeping, 18/19-stage qkv splits, and split-partial rounding/epilogue traffic.
At M256 the tuned cuBLASLt raw kernels recover enough tensor-core throughput
to beat the extra c12 arithmetic and the high-register all-token rq tile.

## The one change

New optional extension only: **64 weight rows x 128 tokens per CTA**; 256
threads, one consumer WG + one producer WG, four 22 KiB stages. At M256,
adjacent CTAs cover token slabs 0:128 and 128:256 of the same W tile, allowing
ordinary intra-call L2 weight reuse, never persistence between cold replays.
The existing optional evict-first weight hint is disabled for this shared-W
geometry; otherwise adjacent slabs could unnecessarily discard their W lines.
Each CTA has 64 rather than 128 FP32 accumulator registers/thread. Producers
return registers down to 40; the math WG grows to 216:
`128*216 + 128*40 = 32768` registers/CTA, exactly two CTAs/SM. Shared allocation
is 91,248 B/CTA, or 182,496 B/two CTAs, below H200's SM capacity.

M256 KS=2/1/3/1 for pooler/unpooler/qkv/o gives 256/256/216/256 CTAs; M128
KS=4/2/7/1 gives 256/256/252/128. There is **no parameter sweep**. The candidate
queries compiled registers/local memory/dynamic shared memory/occupancy and
rejects spills, occupancy <2, or a grid that cannot be fully resident BEFORE
launching its global-zero handshake. The occupancy query also sets the actual
kernel's preferred carveout to 100% shared, so a smaller shared/L1 carveout
cannot silently invalidate the two-CTA capacity assumption. The existing decoder, WGMMA FP32
accumulation and BF16 split-partial reduction remain unchanged; all K terms
and all outputs are computed. No new epilogue, cluster, multicast, Stream-K,
codec, precision mode, or router change is introduced.

Two independent passes, each with three fresh normal/outlier/one-hot draws,
check against FP64 and the **actual final board dispatcher**, require exact
power-of-two one-hot outputs, poison/replay checks and reset counters, then
40-block cold-READ ABBA timing. Every draw's median AND paired ratio must be
<=0.985. Each failing cell uses the untouched incumbent. Plans retain only
weight-derived packs, escape lists, reset counters, and their extension module.

## Why this is not a closed repeat

The copied main ledger's DW v5 row (`027ee79b`, lines 132–136) closed the
producer-decode-to-shared SS-WGMMA family and cluster multicast. This attempt
keeps register-direct A decode and changes resident CTA/token ownership, not
that pipeline. The v4 CURP256 row (`19b6b942`, line 295) changed escape decode
on the original all-256-token tile; this uses the original warp escape decoder
and changes the tile/register budget. The closed route tuner (`88a9aad0`, line
288) and ST/KS sweep (`83c46449`, line 290) did not add token-slabbing or an
NT128 H2 two-CTA kernel at M256. It does not retry the explicitly prohibited
from-scratch clustered redesign (`646c489f50b34d66`, 1.8–3.7x slower).

The helper expected at `.tools/tried.py` is missing. Its tracked replacement
`scripts/tried.py` was searched with fetch disabled against the supplied ledger
copy (Dense / token slab / HALF / CURP 256). The new claim/result is local to
this worktree's TRIED.md: publishing to main would violate this lane's scope.

## Validation / expected effect / risks

CPU checks: `scripts/test_dense_large_slabs.py`: **16 tests passed** (about 2 s).
The output is retained in `results/dense_large_slabs_tests.log`. They cover all
eight production ownership/zero maps, resource arithmetic, SW128 epilogue
coverage, all 65,536 BF16 encodings, an independent register decoder and KS=1/2/3/4/7
escape traversal, 30 numerical cases (M128/M256 x five KS x three input types),
source drift rejection, unchanged incumbent AST (apart from five helper exports),
CPU/prepare-failure fallback, wrong-math rejection BEFORE timing, and mandatory
success in both passes/all six fresh timing draws. Python bytecode compilation
and `git diff --check` pass. No harness files were edited.
No local GPU, NVRTC or ptxas is available. The complete submission's unchanged
nb factory builds at import, so tests deliberately evaluate only actual packing,
source-generation and policy helpers from AST; they do not pretend a full
submission import or a GPU kernel was validated.

Expected effect is **unmeasured**, not a promised speedup. If extra occupancy
recovers 1–3 us per big pooler/unpooler cell, a ~0.5–2% aggregate gain is
plausible. It can also lose: M256 loads compressed W twice from L2, performs
twice as many decodes, reloads X per 64 rows, and uses only one math WG/CTA;
L2 bandwidth, ring startup or CTA scheduling could dominate. Compiled register
allocation and actual two-CTA residency are unknown until H200 execution. The
prepare-time resource/correctness/speed gates are mandatory. Existing BF16
partial reduction remains nondeterministic across split completion order,
as in the incumbent; this is not a new full-FP32-finalizer design.
