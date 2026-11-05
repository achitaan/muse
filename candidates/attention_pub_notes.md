# Attention publication: vector cross-warp handoff

Check file: `candidates/attention_pub.py` (self-contained).
Parent: `43a0f06`, `loop/attn-phase`.

## Source audit

`segment<1,false,0,MODE>` between mainloop and publication:

1. `cp_wait<0>()`, CTA barrier before reusing K staging storage.
2. Warps 2/3 write their per-lane `(m,l,16 FP32 acc)` to K shared memory.
   Original record stride 18 floats: 4,608 bytes written and read per CTA.
3. CTA barrier, then warps 0/1 read the corresponding record, compute the same
   `max`, two exponentials, weighted l and sixteen weighted accumulator sums.
   Warps 2/3 no longer publish their own reductions.
4. A two-shuffle quad reduction forms `lsum`; exponentiation rescales to 2^0.
5. With `ONE`, four 16-byte `red.global.add.v4.f32` per lane plus one scalar
   row-sum reduction per quad publish 4,096 accumulator bytes + 64 sum bytes.
   The 16-byte vectors cover the whole row in the existing permuted dim layout.
   Without `ONE`, the incumbent instead stages padded 72-float rows in V shared
   memory, fences the async proxy, runs another CTA barrier, and thread 0 issues
   TMA bulk-reduce/commit/wait plus the global async-proxy fence. This path is
   NOT changed and is not a repeat of the closed BULKRED experiment.
6. CTA barrier, then one thread-0 `atom.acq_rel.gpu.global.add.s32` on the split
   counter. Cooperative mode additionally computes the output row divisions.
   There is NO live per-thread `__threadfence()` on this path; its call site is
   guarded by the constant-false `!(mode1 || !mode1)`.
7. The phase's `partial` stamp is after that atomic returns. Counter contention,
   release completion and the atomic round trip are therefore included.

Minimum three CTA barriers before publication; four with padded bulk reduction.
The measured 1.2 us cannot yet be attributed to one of exchange, global reduction,
or ticket latency without finer GPU timings. The new one-replay instrumentation
prints `ATTN_PUB_PHASE` for drain/barrier, store/barrier, load/merge,
rescale/global-reduce, and barrier/ticket, plus the original `ATTN_PHASE` lines.

## Exactly one kernel change

The pair L=1 handoff uses 20-float/80-byte aligned shared records:
16 accumulator floats at offsets 0..15 (four explicit float4 stores/loads),
then m/l at offsets 16/17 (float2). Padding 18/19 is not accessed.
Each eight-lane 16-byte transaction covers every shared-memory bank once.
Source-level transfers drop from eighteen scalar accesses to five vector
accesses in each direction; actual incumbent SASS vectorization is unknown
without GPU/toolchain evidence. Allocated handoff footprint rises 4,608 -> 5,120
bytes within the existing 24,576-byte K-stage allocation; transferred bytes do
not rise. Static assertions enforce capacity. No new shared allocation.

All FP32 arithmetic expressions and pair-reduction order remain the same. No
global layout, FP32 atomic, barrier, fence, counter, plan, or KV-store change.
Host and nvcc flags are explicitly C++20 on all six JIT call sites, SM90a only.

## Gate and fallback

- Actual incumbent pair routes only; d1S, d16S, d32S first. If incumbent selected
  a one-head route for a shape, skip rather than silently change the launch plan.
- v16S is explicitly reported unchanged/skipped: the modified exchange is L=1
  only and the normal v16S route is VP2.
- Baseline/base null: 16 cold read-flush ABBA blocks and full output/cache check.
- Two independent candidate passes, each 16 ABBA blocks, 512 MiB HBM read flush
  before each timed replay. Both median and paired ratios must be <= 0.985.
- Bytewise BF16 output equality after each pass, with full K and V byte equality.
  Both arms start from identical caches with new-token rows poisoned. No tolerance
  or retry-to-pass. Nondeterministic atomic order can conservatively reject a win.
- One separately labeled candidate phase replay with fine publication stamps.
  These stamps are NOT present in the timed production candidate.
- Only completed passing shapes dispatch to the new raw module. All other calls
  keep the incumbent. Synthetic graphs/tensors do not survive prepare.

## Local validation

`python3 candidates/test_attention_pub_cpu.py -v`: seven tests passed.
AST/bytecode parse passed; forbidden process-start grep clean; diff check clean.
No local torch, GPU or nvcc: CUDA compilation, exact GPU results and performance
remain unverified. Caller submits the check and updates the shared ledger result.
The claim is recorded only on this branch because the lane prohibits main writes.
