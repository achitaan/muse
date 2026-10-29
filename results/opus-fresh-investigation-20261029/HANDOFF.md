# Fresh investigation handoff (October 29, 2026)

Branch `opus/fresh-investigation-20261029`, worktree
`.slim/worktrees/opus-fresh`. It starts from
`6b7209fd89fc859ca658e3148b60b05d1052b44e` on `codex/h200-2-5x-20261028`.

This was offline analysis only. There were **no official checks, scored
submissions, GPU jobs, okc/API calls, network fetches, subagents or merges**.
No `best/` or `candidates/` file changed. All five retained hashes still match
`MODEL_HANDOFF.md`:

| File | SHA256 |
|---|---|
| best/verified.py | edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b |
| best/moe_verified.py | 331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50 |
| best/decoder_verified.py | d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988 |
| best/attention_checked.py | 9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f |
| best/prefill_verified.py | 93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7 |

## Evidence types

- **Archived snapshots:** These are read-only leaderboard responses from
  18:55 UTC, stored in `results/model-handoff-20261029/*.json`. I did not fetch
  new ones.
- **Scored or checked results:** These come from the official run artifacts.
  D9 Decode was checked at 1.9902248796x in job `9560c915e85f4f50` and is
  unscored. The account's Decode board row is 1.9940x, but no source hash is
  attached to that row.
- **Derived arithmetic:** I recomputed values from those artifacts. These are
  calculations, not new measurements.
- **Hypotheses:** These are labelled as hypotheses below. None was implemented
  or measured.

## Finding 1: The overall rank follows the Decode-layer result (derived from the archived snapshot)

`experiments/opus_fresh/overall_model.py` joins the five archived boards with
the overall board by team name. Its output is saved in
`overall_model_output.txt`.

- For overall ranks 1–314, the overall rank equals the `muse_decode_layer`
  rank exactly. The first mismatch is rank 315, a team with no Decode entry.
  Our account is 207 overall and 207 on Decode.
- I tested this model: overall metric = `max(decode_speedup, composite)`.
  Here `composite = 1/(0.25/s_dense + 0.11/s_attn + 0.48/s_moe + 0.16)`. A
  missing track counts as 1.0. The weights come from a grid fit at 0.01 steps.
  - Mean absolute rank error is 1.53 over 1,597 teams, with a median of 1.
  - Error is **0.00 for the top 300**. The maximum error is 120, deep in the
    tail.
  - It predicts our rank as 207, which matches the board.
  - Ranking by Decode alone, without the max, gives a mean error of 15.8.
  - Sum-of-times, geometric-mean and rank-sum models fit badly (adjacent-order
    agreement about 0.58).
- Our composite is 1.75 under this fit. Even if Dense, Attention and MoE were
  all at their board-leading values, it would only reach about 1.89. That is
  below the Decode value already needed for rank about 50.
- **Implication (inference):** Within this model, only `muse_decode_layer`
  can move this account into the top 50. Dense, Attention, MoE and Prefill
  track gains matter for the overall rank only through code that Decode
  executes.
- Decode speedup needed, from snapshot rows (the target moves over time):

  | Overall rank | Decode speedup | Gain over 1.9940 |
  |---|---|---|
  | 150 | 2.091 | +4.9% |
  | 100 | 2.123 | +6.4% |
  | 50 | 2.184 | +9.5% |
  | 25 | 2.235 | +12.1% |

  The model's own predictions are 2.18 → rank 53, 2.19 → 46 and 2.25 → 23.
- **Caveat:** This uses one snapshot, and the API does not publish the
  formula. The model is a fit, not a documented rule.

## Finding 2: How much Decode time must be removed (derived)

`experiments/opus_fresh/decode_sensitivity.py` reconstructs 1.9902 exactly
from the job `9560c915e85f4f50` benchmark medians. Its output is saved in
`decode_sensitivity_output.txt`.

- **Per-load speedups:**

  | Load | Speedup |
  |---|---|
  | decode 1 | 2.008 |
  | decode 16 | 1.942 |
  | **decode 32** | **1.739 (weakest)** |
  | verify 1 | 2.033 |
  | verify 16 | 2.050 |
  | verify 32 | 2.199 |

- **Ratio of our time to the stated floor:**

  | Load | Ratio |
  |---|---|
  | decode 1 | about 2.1 |
  | decode 16 / decode 32 sliding | 1.80 / 1.87 |
  | full-layer shapes | 1.5–1.85 |
  | verify 32 sliding | 2.31 (largest) |

- **Rank about 50 needs one of these:**
  - About 14.3 µs saved on every one of the 12 shapes.
  - About 8.9% saved on every shape.
- **Upper bounds**, assuming the removed kernel time is on the critical path:

  | Change | Decode speedup |
  |---|---|
  | MoE kernel 15% faster | 2.138 |
  | MoE kernel 25% faster | 2.250 |
  | Full-layer attention 20% faster | 2.034 |
  | All norm/boundary/pre-expert kernels removed | 2.100 |
  | MoE −15% plus full attention −20% | 2.188 |

## Finding 3: The MoE kernel uses about half the bandwidth floor (derived from saved results)

- **MoE track** (job `d600231c8a8244cb`): floor divided by our benchmark time
  is 0.46 / 0.54 / 0.57 / 0.59 / 0.56 / 0.51 for decode1 / decode16 / decode32
  / verify1 / verify16 / verify32.
- **Decode layer, inferred.** I subtracted the dense-weight floor (qkv, o,
  pooler, unpooler slice or full, FP32 router at 4.8 TB/s) and the KV floor
  from each stated shape floor. The remainder implies about this many distinct
  local experts:

  | Tokens | Distinct local experts |
  |---|---|
  | T=1 | about 2 |
  | T=8 | about 5.6 |
  | T=16 | about 5.7 |
  | T=32 | about 6.7 |
  | T=128/256 | about 12.6 |

  I used 28.3 MB per expert (w13 and w2 codes plus FP8 scales). Against the
  D9 `moe_fused` profile durations, that gives about 40–57% of 4.8 TB/s.
  - This rests on an assumption: that the judge's floor counts the target
    distinct-expert bytes. The reference's `weights_fields` targets
    `router_load`/`router_distinct` per load.
  - It is consistent with the MoE track's own decode1 floor of 11.4 µs, which
    is about 2 experts.
- The MoE kernel's source is byte-identical in the MoE and Decode retained
  files. The extracted `CUDA_SRC` string has SHA256 `707f9584...` (computed
  from the Python string, not the stored payload encoding).
- **Inference:** MoE is the only single component whose plausible improvement
  alone reaches the rank-50 band. Decode-only components such as norms,
  routing and overlap are secondary levers.

## Finding 4: At T=1 the layer is latency-bound (derived, partly a hypothesis)

At decode1 sliding, the profile kernels sum to 87.4 µs. The raw graph median
is also 87.4 µs (84.6 µs after the 4 µs replay subtraction). The total dense
plus inferred-expert bytes are about 195 MB, which is about 41 µs at
4.8 TB/s or about 49 µs at 4.0 TB/s.

So HBM is idle for roughly 35–45% of that shape. The idle time comes from
latency-bound kernels: attention at 4.9 µs, norm/boundary kernels at 4.6 µs,
2.0 µs and 3.7 µs, the unpooler slice GEMV at 5.1 µs against a 1.7 µs floor,
and launch gaps.

`_fpr_PDL = 0` in D9, so programmatic dependent launch is off for the fused
pooler-router path. I did not measure how much the side-stream router
(`route_fp32<1>`, 11.2 µs) overlaps with the pooler. The profile sum alone
cannot settle it.

## Independently developed possibilities (hypotheses only; none implemented or measured)

These are ordered by the upper-bound arithmetic above. All need GPU evidence
before anyone acts on them.

1. **MoE streaming efficiency.** This is the largest lever on every load. A
   15–25% MoE kernel gain inside Decode maps to 2.14–2.25. Earlier
   repository experiments closed several micro-variants, and the cause of the
   remaining roughly 2× gap to the floor (phase serialization, dequant
   instruction pressure, ring depth or tails) is not established by any saved
   per-phase trace I found. A per-phase timeline (phase A / phase B /
   finalize per CTA, as `%globaltimer` stamps into a debug buffer, in a
   private non-scored harness) would be the first evidence to collect. That is
   a measurement step, not an optimization.
2. **Overlap expert-weight streaming with the pooler tail.** Small-T routing
   is known before the pooler finishes.
   - The idea: fold the pre-expert RMSNorm and local-ID mapping
     (`_pn_inputs_kernel`, about 2 µs plus a launch gap) into the MoE prologue.
     Launch MoE with a programmatic dependency after the pooler. Let its
     prologue build the route table and issue the first weight-ring TMA loads
     before `griddepcontrol.wait`.
   - It moves no extra bytes; it only uses HBM that is otherwise idle.
     Rough upper bound: about 4–8 µs per small-T shape.
   - Numerics stay unchanged if the RMSNorm arithmetic is kept bit-identical,
     including FP32 statistics, the BF16 output boundary and
     `enable_fp_fusion=False` equivalence.
3. **Within-call L2 staging of the next dense weights during latency-bound
   phases.** For example, the O weights (16.8 MB) and the unpooler slice
   (8.4 MB) during attention and the boundary norm.
   - This is legitimate in-call work. It must never touch the next call's
     weights or carry anything across calls.
   - Rough upper bound: a few µs on T≤8 shapes only. The risk is SM/L2
     contention with attention.
4. **Fold the norm/boundary kernels** (about 8–10 µs per shape at T≥16:
   `muse_spark_norm_residual_kernel` ×2, routing) into neighbouring producers.
   - The removal upper bound is 2.10.
   - The repository's earlier cooperative and cluster attempts lost badly,
     so a different design would be required: for example, last-CTA
     finalization that keeps the existing reduction order. Those losses say
     nothing either way about that design.
5. **Full-layer attention at T≥16** runs at about 60–75% of its KV floor. The
   upper bound is modest (+2–3% for a 20–30% faster kernel).

## Changed files on this branch

All files are new. Nothing existing was modified.

- `experiments/opus_fresh/overall_model.py`: overall-board aggregation fit
  (offline).
- `experiments/opus_fresh/decode_sensitivity.py`: Decode per-shape and
  per-load sensitivity (offline).
- `experiments/opus_fresh/list_strings.py`: AST lister and extractor for
  embedded source strings (offline).
- `results/opus-fresh-investigation-20261029/overall_model_output.txt`,
  `decode_sensitivity_output.txt` and this `HANDOFF.md`.

The extracted `.cu` files in `experiments/opus_fresh/scratch/` were left
untracked on purpose. They are derived copies.

## Tests actually performed

- Recomputed the retained hashes (all five match).
- Ran the two offline Python analyses: CPU only, on archived JSON.
- Byte-compared the MoE CUDA string between `best/moe_verified.py` and
  `best/decoder_verified.py` (identical).
- No compilation, GPU execution, correctness test or timing was performed.

## Uncertainties

- The overall aggregation is inferred from one snapshot. Its exact weights are
  a fit and may not be the official formula. Rank thresholds move as others
  submit.
- The expert counts are inferred from floors. The real per-replay routing
  varies.
- Profile kernel durations are not critical-path fractions. Overlap is
  unmeasured.
- Every possibility listed above is an unmeasured hypothesis. The upper bounds
  assume the removed time is on the critical path.
- The evaluation window ends 2026-10-30T00:00Z. Any GPU use after that needs
  new authorization.
