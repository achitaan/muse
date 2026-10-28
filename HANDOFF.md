# Current handoff - October 28, 2026

Goal: approach top 50 overall, main chat only, no deadline. Conserve checks and
scores; no blind sweeps, repeated baselines, cosmetic resubmissions, subagents,
paid GPUs, external model providers, global changes or credential output.
The requested ordered pass MoE -> Decode -> Attention -> Prefill is complete
with full passing coverage. Top 50 overall is still unmet; Goal remains active.

| Track | Published speedup | Last observed place | Job |
| --- | ---: | ---: | --- |
| Dense | 1.527771327x | 99/1245, 17:04 UTC | 6131bc96427e4486 |
| MoE | 2.348299935x | 53/990, 17:04 UTC | b6c70c9b83724d2e |
| Decode | 1.965324622x | 150/869, 17:04 UTC | 02fd620987114785 |
| Attention | 2.562402309x | 82/951, 17:04 UTC | 20524d06e3894595 |
| Prefill | 1.663469245x | 123/576, 17:04 UTC | 23e7c8b4c3e54869 |

Overall: **150/1432 at 17:04:44 UTC**. MoE's 50th-place score was 2.3539x,
about 0.24% ahead. Decode's 50th-place score was 2.1421x. Placements can
change. Prefill does not count toward overall. No job or controller is pending.

The newer whole-decoder check is **1.994319167x**, job 09b34f4e921e49a6, exact
09badc9b source in best/decoder_verified.py, full 13/12/integrity pass. It adds
checked exact MoE, implicit small-token FFN zero padding, and early FFN input
loads. Its interval overlaps the published check; no scored run was spent.
best/decoder_published.py remains a61735ee. Source distinctions matter.

MoE checked and published sources are both 33114871: paired BF16 epilogues and
exact DQ8 bit placement, with all original norm precision retained. One scored
confirmation followed two independent passing checks of identical timed CUDA.
The rejected selector chose no alternate; folding its metadata branch recovers
the simpler kernel AST. Individual intervals overlap, so report measured points
and ranks without claiming statistical significance. No repeated source checks
or scored intermediates were used. See MOE_NOTES.md and full replication proof.

The phase7 small-projection trial afd35730c7d3476f passed full 13/12/integrity
at 1.997916935x, but the old prepare took 257 s and triggered the 230-second
cutoff. New paths were skipped before packing/compilation; their GPU correctness
and performance remain unverified. The guard discarded the trial. Its timed
compute is another independent observation of baseline 09badc9b, not evidence
of a new optimization. One check and zero scored runs were used. The controller
is archived and no job is pending; evidence is in results/resumed-decoder-phase7-20261028/.

Next: reduce prepare overhead or add the curated instantiations to the existing
extension before another check. Do not simply raise cutoffs toward the 300-second
limit. The prepared builder and CPU audit in experiments/prepared/ recover
1,441,792 BF16 values exactly for the three layouts and retain all baseline
bytes; packaged trial size is 995430 bytes. Retained source remains 988314 bytes.
The public o32h_r100 configuration also targets the profiled 16-token output
projection (~15 us); it is only a follow-up hypothesis, not implemented or tested.

Read STATE.json, per-track NOTES, official TASK.md/AGENTS.md and notices. best/
holds exact sources and metadata; results/ holds frozen sources, raw official
records, profiles and immutable controller history. Decoder phase6 and MoE
phase5 and decoder phase7 are archived. Initialize one scoped phase with cached baseline before
trials, preserve handles across observation failures, never edit journals, and
push complete evidence at boundaries. Dense's last observed quotas were empty;
query other quotas only when needed.

Preserve BF16 GEMM/global-scale and effective-weight boundaries, FP32 norm
intermediates, every route/key, fresh inputs, cache writes and all residual
outputs. NVFP4 weights stay exact, no persistent BF16 expansion or output/data
caching. Shared native scratch requires sequential calls. Keep Apache/MIT
licenses and Vjeux, Nathen Fernandes, Lee Penkman and SGLang attribution.
