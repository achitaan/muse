# Muse optimization checkpoint

Updated November 4, 2026, Toronto time (November 5 UTC). This documentation refresh uses saved evidence through October 31; it does not reproduce GPU measurements or refresh the live board, queue, or quotas.

## Repository and retained work

Repository: [achitaan/muse](https://github.com/achitaan/muse), branch `argon/decode-moe-20261030`. Starting revision for this refresh: `3d1d67707989dbdf5a5193edf2c5e040b82c4ee3`, which saved remaining results, logs, profiles and offline sources. HEAD matched the local tracking ref before this refresh; no remote fetch was performed. Regenerable CUBIN binaries remain excluded.

The goal remains top 50 overall. Recent experiments have not established an additional speedup. Verified sources remain unchanged.

| Track | Retained source | SHA256 prefix | Retained scored speedup |
|---|---|---|---:|
| Dense | [best/verified.py](../../best/verified.py) | `edc61a1e` | 1.527771327x |
| MoE | [best/moe_verified.py](../../best/moe_verified.py) | `33114871` | 2.348299935x |
| Decode | [decoder_pdli.py](../../experiments/decode_moe/decoder_pdli.py) | `c15df9d3` | 2.075466170x |
| Attention | [best/attention_checked.py](../../best/attention_checked.py) | `9e922c35` | 2.537000514x |
| Prefill | [best/prefill_verified.py](../../best/prefill_verified.py) | `93e57ab4` | 1.663469245x |

These are separate track results, not an overall speedup. Prefill is excluded from the overall board. The overall API [snapshot](scoring-review/overall-20261031.json) reports **277 of 1,730** on October 31 and supplies ranks only. Exact overall aggregation remains unconfirmed. Decode rank 228 and top-50 threshold 2.2252x in older state fields are historical track observations, not a current overall placement or gap.

## Latest completed experiments

| Experiment | Job and source | Result | Decision |
|---|---|---|---|
| Dense one-token router/pooler specialization | `8efb0121db7848c2`; `9bd760ff` | 9 official tests, 30 benchmarks and integrity passed; 20 private byte probes passed. **0/4** comparisons passed the greater-than-3% timing gate. Candidate/parent ratios: 0.984886681 to 1.007623872. | Closed; no score, retry, or Decode transfer. |
| MoE WGS finalizer row-read reuse | `5a9d8a25bda34b31`; `ae387de3` | Public fallback passed 5 tests, 6 benchmarks and integrity. Both private BIG entry markers executed. **0/12** comparisons passed the timing gate; ratios 0.994471703 to 1.005541232. | Closed; no score, retry, or Decode transfer. |

Sources and raw paired samples: [M1 outcome](router-m1/outcome.json), [WGS outcome](wgs-final-reuse/outcome.json). Neither fallback benchmark point is a candidate gain. WGS saved logical ybuf requests without improving measured complete-kernel time; that does not establish saved HBM traffic or an exposed bottleneck.

WGS evidence limitation: the server's 16,000-character log tail retained 9 of 40 individual byte-pass messages and clipped exact GPU resource values. Completion of all probes and resource guards is inferred from the frozen control flow reaching both markers and all timing comparisons. CPU resource measurements and complete raw timing arrays are retained; do not describe clipped GPU attributes as directly observed.

Earlier tiny routing, whole-row norm, grouped/helper norm, router sidecar and Attention specialization experiments remain closed. Their immutable records are preserved. A passed checker or a higher fallback point does not reopen them.

## Planned work and current limits

The next investigation is the retained MoE kernel's phase timing, using its existing disabled timestamp hooks to distinguish routing, gather, phase A/B, grid join and finalization. See [measurement note](wgs-final-reuse/NEXT_MEASUREMENT.md). **No runnable instrumentation payload, CPU qualification, GPU admission, or new measurement exists yet.** A timer escape/compiler-ordering correction, exact source binding, both BIG-entry resource checks, live-buffer correctness, instrumentation overhead calibration and bounded lossless logs are prerequisites. This is a diagnostic proposal, not a promised optimization. Per-CTA duration sums cannot establish critical-path fractions.

The last owned check is terminal. Live queue state has not been refreshed in this update. No GPU job, scored submission or profiling job was launched by this documentation refresh.

| Track | Checks left in October 31 snapshot | Scores left in that snapshot |
|---|---:|---:|
| Dense | 16 | 10 |
| MoE | 18 | 10 |
| Attention | 18 | 10 |
| Decode | 20 | 10 |
| Prefill | 20 | 10 |

Source: [saved quotas](wgs-final-reuse/quotas-after.json). Limits reset daily at 00:00 UTC, so these are **historical**, not today's available attempts. Refresh before any launch.

Human removed the fixed 60-minute interval. Use checks sparingly, preserve a four-check reserve per track, require an idle shared GPU queue and fresh evaluator/source/quota admission, and avoid noise retries. Save Decode checks until a credible component gain justifies whole-layer validation. The [owned policy override](EVALUATION_POLICY_OVERRIDE.json) supersedes older spacing records.

The approved MoE norm/output variance exception remains limited to those boundaries: norm-sum relative L2 <= 1e-5, output relative L2 <= 0.001 and worst-row error <= 0.005, with the recorded denominator floor. Live GEMM/SwiGLU and FP32 ybuf remain byte-exact. Decode and Attention byte gates, precision/materialization rules and the greater-than-3% performance gate remain unchanged. WGS read reuse used stricter byte gates throughout.

The current constraint is missing causal evidence for a worthwhile next change. The results establish a plateau for the tested mechanisms; they do not prove that further optimization is impossible. No fresh rank, quota, overall speedup, or speedup prediction is asserted here.

Machine-readable checkpoint: [CURRENT_STATE.json](CURRENT_STATE.json). Earlier entries in [RESULTS.md](RESULTS.md), [HANDOFF.md](../../HANDOFF.md) and [BLOCKED_HANDOFF.md](BLOCKED_HANDOFF.md) remain historical evidence; this checkpoint takes precedence for current status.
