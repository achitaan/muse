# Dead ends (Oct 30 evening, branch argon/decode-next-20261030)

Measured on the judge H200. Each item: what was tried, the evidence, and why it is closed.

## Decode level
- **L2 weight prefetch during attention/norm at T = 1 / 8 (L2P)**. Jobs `94c68965`, `bad2bc21`.
  - Bulk prefetch from 8 or 132 blocks: no gain on the o weights, +20-25 us once pooler weights were added.
  - Load warm-up from all SMs: at best -0.8 us on a 52.5 us chain (1.5%).
  - Closed: the projections at 1 / 8 tokens are not DRAM-bound, and the handoff gaps are only about 0.5-1 us.
- **A3 attention tuning in Decode** (`9171012a`): no change on the target shape. Closed.

## MoE kernel
- **Removing the grid barrier / per-token finalize.**
  - Diagnostic `63c17c9c`: the tail after the last phase-B block is only 1.6-2.5 us at 1, 8 and 16 tokens. Phase A already streams at about 4.3 TB/s at M >= 8.
  - Closed (at most about 2 us).

## Attention kernel
- **Fewer, longer KV splits on small sliding shapes** (loop trial 1, `543fe60e`). Reverted. Every changed shape got slower (judge, us):

  | Shape | Before | After |
  |---|---:|---:|
  | decode 1 | 5.18 | 6.28 |
  | verify 1 | 7.66 | 9.43 |
  | decode 32 | 10.14 | 11.51 |
  | verify 16 | 12.02 | 14.41 |

- **Launch-plan sweep** (`a7c4e6e6`, setup-time paired CUDA-graph timing on task-shaped inputs). Every variant was slower than the current plan, so the plan is a local optimum. Closed:
  - clusters cs 4 / 8 / 16 on sliding shapes: +41-46%;
  - other split counts on sliding shapes: +2-39%;
  - full-layer grids of 132 / 396 / 528 CTAs: +3-17%;
  - coop on/off: +3-38%;
  - single-request full grids 132 / 264 / 396: +3-54%.
- **Timeline of decode 1 sliding** (`eaebaa8f`, globaltimer marks). The 4.8 us is a chain of dependent round trips:

  | Step | us |
  |---|---:|
  | entry to segment start (likely cold instruction cache; the speculative seq mode is already used) | 0.93 |
  | page indices to K/V issued | 0.62 |
  | queries | 0.40 |
  | main loop | 0.90 |
  | atomic merge + counter | 1.13 |
  | last-CTA finalize | 0.82 |

  No single dominant step.
- **Spreading the merge's atomics over more workspace copies** (`nc` 2 / 4 / 8 on all six sliding shapes; `0b48e4f5`, paired setup timing). Relative to nc = 1:

  | Shape | nc = 2 / 4 / 8 |
  |---|---|
  | decode 1 | +5.0% / +5.7% / +9.7% |
  | verify 1 | +2.8% / +2.6% / +1.8% |
  | decode 16 | +3.1% / +5.2% |
  | verify 16 | -1.6% (noise) / +2.6% |
  | decode 32 | +2.2% / +4.1% |
  | verify 32 | +2.0% / +9.3% |

  Closed: same-address atomic contention is not what limits the merge.

## Noise
- The same attention source scored 1441.7 us (`8347b4ca`) and 1490.9 us (`804f8d7c`): about 3.4% run-to-run. Single-check keep/revert cannot resolve changes below about 3-4%.
- Only setup-time paired timing can.
