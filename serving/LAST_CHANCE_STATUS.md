# MUSE_LAST_CHANCE.md: status of every item (Nov 2, 2026)

## 1. Banked kernel result: verified
- P36 source sha256 `449ef16c1e3af46254d6297497d658d6dd7f931d597aef4dd2ed2f62bc854f44` (matches manifest).
- Scored job `9dabcd9a13554bad`: 2.105628x, 13/13 tests, CI 2.0740-2.1372; the 12 published latencies
  reproduce 2.105599x (`muse_review_checks.py`).

## 2. Correctness item: all-remote replay. DONE, bug CONFIRMED (not fixed in P36)
`experiments/boards/diag/zerodiag.py` on the judge H200 (`28294206`): MAIN build, M=32, mixed eager and
graph replays bit-equal with counters reset; replaying the same captured graph with every route remote
HANGS (2 s watchdog). P36's judge runs are unaffected (natural EP8 routing; all 17+ tokens remote ~1e-8),
but `moe_fused` MAIN must not be reused in a server without a J==0 guard.

## 3. Kernel top-50 gate: NO KERNEL RUN
kernel-mint's public post (guild t/116) only lists board totals (Decode 2.2464, MoE 2.3866, Attention 2.6619,
Dense 1.5590); per-shape timings are offered only as a team-to-team trade, which needs the user. Without a
per-shape comparison there is no evidence for a target-scale (8.4% uniform) improvement.

## 4. Serving track (pinned SGLang a55ab7d5; checkout WSL ~/muse-serving/sglang)

### S1: Marlin small-token fused post-expert norm. Op level on H200: PASS. Server: UNMEASURED
Patch `serving/s1_postnorm.patch.diff` (2 files, 2,466 bytes). H200 op diagnostic `serving/s1s2_opdiag.py`
(`cd2e95e7`), image module identical to the pin (`img_eq=1`, MIN=32), read flush, CUDA graphs:

| T | stock us | fused us | speedup |
|---:|---:|---:|---:|
| 1 | 33.8 | 6.8 | 4.9x |
| 4 | 37.9 | 9.8 | 3.9x |
| 8 | 39.1 | 8.5 | 4.6x |
| 9 | 39.7 | 9.1 | 4.4x |
| 16 | 42.3 | 9.2 | 4.6x |
| 24 | 45.9 | 9.2 | 5.0x |
| 31 | 48.3 | 9.2 | 5.3x |
| 32 / 33 | 9.6 / 8.9 | 9.4 / 8.9 | 1.0x (both fused already) |

Correctness: finite everywhere, all-remote output exactly 0 with NaN/Inf-poisoned remote rows, rel-L2 vs
FP64 identical for both (1.66-1.70e-3), max |stock - fused| <= 0.0078 (BF16 ulp-level). About 25-39 us saved
per MoE layer per forward with < 32 tokens (C=1 every step; C=4 when < 4 requests are batched).
Risk: serving requires answers to match stock; BF16-ulp differences could flip a greedy token. Only an
official `okc serve check` (opens Nov 5 12:00 PT) can confirm, and its C=16 rung barely exercises S1; the
C=1 effect is only visible in a scored serving run.

### S2: rank-local expert-id alignment. REJECT
Patch `serving/s2_local_align.patch.diff`. Same diagnostic: sorting with 32 ids instead of 256 is slower at
small T (align 8.6 vs 5.5 us at T=1, 9.0-11.8 vs 7.4-8.5 at T=8-33; equal at 128/256; Marlin GEMMs unchanged),
and the Marlin output differs from stock in the `hot` pattern (gemm_eq=0, tail_eq=0 at T=16/32/128/256).
Per-expert slot sets and padded totals match, but the change has no upside and a correctness question. Dropped.

## 5. Next actions (need the serving window, Nov 5 12:00 PT onward)
1. `okc serve check` with S1 only (correctness/answer match at C=16; 3 checks/day).
2. If it passes, one scored serving run with S1 (1/day); compare against stock's published serving numbers.
3. Do not combine S2; do not reuse moe_fused in the server.
