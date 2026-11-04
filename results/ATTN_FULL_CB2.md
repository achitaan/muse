# FULL-CB2: v16F producer/consumer register balance

**Status: offline-ready, no measured gain.** One added arm, **v16F only**.
Branch `loop/attn-full-consumer-budget`, directly from `63e253f`.
Check payload: **`candidates/attention_full_cb2.py`**. No check/submission/push
was made; no main commit or other worktree was changed.

## Cheap screen first: full-context useful bandwidth

Source numbers: `results/attention_redesign_baseline.json`, scored board-best
job **8f53cd76d828427c** (2.6536245x). Actual aggregate sequence lengths are not
logged there. The table uses the advertised **80k nominal context**, not a
hardware traffic-counter measurement. FP8 K+V payload is
`B * 80000 * 2 allocations * 2 KV heads * 64 dims = 256 B * 80000` bytes.
Do **not** multiply these already-reused bytes by q_len or the GQA ratio.
New-token traffic, page indices and Q/gate/output bytes are not included.

| Full case | KV MB, nominal | Corrected us | Useful TB/s | / 4.8 TB/s |
|---|---:|---:|---:|---:|
| d16F | 327.680 | 86.489 | 3.789 | 78.93% |
| d32F | 655.360 | 165.734 | 3.954 | 82.38% |
| **v16F** | **327.680** | **98.178** | **3.338** | **69.53%** |
| v32F | 655.360 | 183.659 | 3.568 | 74.34% |

None crosses 85% at nominal 80k, so none is falsely declared at the HBM peak.
Decode-full is nevertheless already about **91–95% of the measured 4.15-TB/s
PAGE=1 gather stream** (ledger 199): skip it in this bounded attempt. v16F has
the largest fractional headroom. v32F also has headroom, but uses a **different
generic flat kernel**; porting unchanged WST to it already lost (187–188).
The same NPW change cannot be applied there without changing that backend.

## Source diagnosis and one proposed change

The board-best v16F dispatch is `_zh_wst`, specifically
`wst_ct14b<4,2,4,1,0,0>`, not the generic flat path described by the ledger's
104-register shorthand (row 197). Direct source inspection gives:

| | Incumbent NPW4 | Candidate NPW2 |
|---|---:|---:|
| Consumer WGs / threads | 2 / 256 | **unchanged 2 / 256** |
| Producer WGs / threads | 4 / 512 | **2 / 256** |
| Total threads | 768 | 512 |
| Producer register target | 40 | 40 |
| Consumer register target | 160 | **200** |
| Target register sum / CTA | 61,440 | **same 61,440** |
| Dynamic shared, B16 | 205,008 B | **same 205,008 B** |
| Resident CTAs / SM | 1 (shared limited) | **still 1** |
| Producer row passes | 2 | 4 |
| Warp cp.async issues / tile | 64 | **same 64** |

**Hypothesis:** give the unchanged eight consumer warps **25% more registers**,
with fewer producer polling/arrival warps competing for issue slots. This may
reduce consumer spill/rematerialization/serialization pressure without losing
consumer occupancy. It is not a generic launch-plan search: only NPW4 -> NPW2,
using existing source branches. There is no split-count or parameter sweep.

The generated CUDA changes **one entrypoint template argument**. Tile128,
four stages, PF2 lookahead, RR1 scheduling, G=SM cooperative launch, balanced
logical traversal, both KV heads, all query/cache/math/merge code and compiler
flags remain unchanged. In particular the incumbent's compiled QK precision,
FP32 partials, and `ZH_SCH=1` P path are preserved. This does not introduce a
new FP16-attention experiment or touch physical-order traversal.

**Not a closed repeat:** rows **187–188** tried the original NPW4 WST backend
at B32; row **194** changed one-head/split mapping; row **177** moved V transpose;
row **180** changed exp2. None changed WST's producer WGs/register donation.
Ledger searches for Attention/NPW, Attention/balance, and Attention/warpgroups
found no closed producer-balance trial. Row 197 motivates consumer throughput,
but its generic104 description must not overwrite the actual WST160 source.

## Admission / fallback

The **entire scored incumbent is a literal unchanged prefix** of the candidate,
including its original initialization. Additional experiments probe only v16F;
all other cells call that captured incumbent unchanged. Its existing initialization
is not a new sliding/one-head optimization attempt.

- Skip the added experiment unless incumbent v16F is ready WST4 (not a failed
  build's generic fallback or a newly selected one-head route). No backend or
  precision comparison is smuggled into the balance test.
- Query compiled NPW4/NPW2 function attributes, local-memory bytes and occupancy;
  print them before probing and refuse a nonresident or undersized-register-pool
  instance (setmaxnreg.inc must be funded before it is safe to probe). `numRegs` is not
  represented as measured dynamic consumer registers; local bytes may include
  stack, not just spills.
- Independent scratch/tickets for NPW2. One synchronized eager probe, graph
  replay with poisoned new-cache slots, unchanged judge/global/row bounds and
  exact whole-cache equality. Full-precision error/bit-equality diagnostics.
- Read-flush **512 MiB**, original **16-block ABBA**, then fresh independent
  confirmation. Both median and paired ratios must be **<=0.98** in both passes.
- All no-worse reference/row comparisons must also pass for adoption. A
  judge/cache-valid numerical rejection can be **timed diagnostically**, but
  can never select the candidate. No threshold slack was added.
- Atomic publication only after full confirmation. Exact B16/L8/full/page1/
  unit-scale/layout/device guard; other inputs retain incumbent. No actual-call
  timing, host synchronization, output caching or retained synthetic inputs.
- Additional import budget 90 s (build wait <=70 s), after incumbent's original
  <=85-s tuner. Both synthetic scratch sets are discarded after preparation.

## Local gate

```sh
PYTHONDONTWRITEBYTECODE=1 \
  /Users/achita.anantachina/Documents/Codex/2026-10-29/can/muse/.venv/bin/python \
  scripts/validate_attention_full_cb2.py
```

**17 tests passed**: fail-closed single-argument source change, unchanged full
incumbent prefix and compiler flags, exhaustive producer row/chunk coverage,
exact K/V staged-byte oracle at partial/full tile edges, unchanged new-KV write
ownership, register/shared/barrier budgets, 32 length-draw models of stream-K
scratch/tickets/folders, identical host wrapper arguments, scoped metadata and
runtime fallback, register-pool preflight, all performance/numerical/cache admission failures, and actual
resource-helper C++ syntax with host API shims. AST and CPU-only Torch import
pass; import reports incumbent fallback. Parent/candidate whitespace delta is
clean; whole-new-file staged checking preserves the parent's trailing spaces.

**Not verified:** CUDA translation-unit/host-extension build, ptxas, actual
register/local-memory use, producer/consumer progress, cache writes, numerical
equivalence, graph capture or H200 speed. nvcc/NVRTC/ptxas are unavailable locally.
CPU byte/protocol models do not replace the caller's H200 check.

## Expected effect and stop rule

Target hypothesis: **4–10% lower v16F time**, equivalent to roughly **+0.51–1.31%
board**, if consumer pressure is exposed. This is not a measured prediction or
a claimed +3% board gain. Even the 2% admission threshold is only +0.25% board
for this one cell. Real risks are producer underfill, higher per-thread copy/
page-index pressure (and producer spills under 40 registers), no consumer spill
to remove, or compile/import-budget failure. Reject and stop if the fixed NPW2
arm does not pass; do not launch a wider NPW/stage/split sweep.

Payload SHA256: `ea827f82f86acd490144414103794f82e19b4f29938afb025f0359c8034d7cd2`.
The branch-local TRIED row must be copied to the shared ledger by the caller;
its normal helper pushes to main, forbidden in this lane.
