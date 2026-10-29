# GPU checks and submissions, October 29, 2026 (22:41â€“00:00 UTC)

The user authorized these in chat: "use as many checks as make as many updates
as you can till then (8pm Toronto)". Every number below was measured by the
official service on an H200. The raw JSON and profiles are in this directory
and in `experiments/opus_fresh/candidates/profiles/`.

| Job | Mode | Source (SHA256 prefix) | Result |
|---|---|---|---|
| `efbe095b7aae43e3` | check | `moe_finreg.py` `fe72a4b2` | 5/5 passed, **2.3541x** CI [2.3188, 2.3894]. WGS loads 32/128/256: 1.629/2.575/2.862, versus retained check 1.642/2.591/2.875. **No gain; tied within noise.** |
| `bfa220e08a4143ed` | check | `dense_o128.py` `b638ca2a` | 9/9 passed, 1.4941x. The O128 kernel was slower at every shape (see below), so all six shapes stayed on the old kernel and this point says nothing about O128. **Rejected.** |
| `ddda5cdcbfe64203` | check | `decoder_finreg.py` `267fe3e1` | 13/13 passed, **1.9949x** CI [1.9650, 2.0248], versus D9 1.9902. **Tied within the interval.** |
| `1216cb6725c84c7a` | check | `decoder_o128.py` `e55b3821` | 13/13 passed, 1.9961x. **Fallback only:** `O128_SKIP` at 258.9 s of preparation, so no new kernel ran. |
| `b6556f86c6e4429a` | scored | `decoder_finreg.py` | 13/13 passed, **1.9876x** CI [1.9578, 2.0174]. The board keeps the earlier 1.9940 row, so the Decode board is unchanged. |
| `b43cb2fef3a94aa9` | scored | `moe_finreg.py` | 5/5 passed, **2.3507x** CI [2.3154, 2.3860]. The board MoE row updated from 2.3483 to 2.3507 (+0.1%, within noise). The first submit attempt failed in transport (TLS error) and created no job. |

## O128 measured timings

These are paired cold-L2 graph timings measured during `prepare` in job
`bfa220e0`, in Âµs: the best O128 variant against the old kernel.

| Shape | O128 best | Old kernel |
|---|---:|---:|
| M128 qkv_gate | 28.5 | 18.0 |
| M128 pooler | 30.2 | 24.7 |
| M128 unpooler | 30.1 | 22.8 |
| M256 qkv_gate | 34.7 | 22.2 |
| M256 pooler | 49.7 | 27.9 |
| M256 unpooler | 56.4 | 27.6 |

The resource marker was `1440000` (141 registers, 0 local bytes).

The O128 kernel was slower at every shape. This is consistent with the
predicted risk: each 64-row CTA re-reads the full activation and waits
synchronously on each wgmma group. Saving 25% of weight bytes does not offset
that. Close O128 as designed; do not retry variants of it.

## FINREG

It is correct: both tracks pass all tests and benchmarks. It is speed-neutral
within the reported intervals. It was published because it is a correct source
and the Decode publication replaces an older board entry that lacks the
required precision boundaries. **No speedup is claimed.**

## Leaderboard after submissions (22:56 UTC)

- MoE: rank 85 of 1,129, speedup 2.3507.
- Decode: rank 225 of 999, speedup 1.994 (the old row is kept).
- Overall: rank 225 of 1,625.

The overall rank fell from the 18:55 snapshot (207) because other teams improved, not because of these runs.

