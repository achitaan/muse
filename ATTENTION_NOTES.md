> CURRENT CANDIDATE, October29: Attention identity-rescale short circuit
> ed810c7a is frozen and CPU-audited; no GPU result or speedup yet. It changes
> only L1 flat update control, preserving all keys, precision and geometry.
> Packet: results/attention-skipscale-preflight-20261029/. Fixed review/gates:
> experiments/prepared/attention_skip_identity_scale_review.md. Next new check
> no earlier than04:22:28.724212UTC, one job, no fresh baseline or sweep.
> No GPU job/controller is active at this checkpoint. Retain every best;
> the rejected MoE factor trial remains closed. Earlier running headers
> below are historical. Goal remains active; main chat only.

> Latest checkpoint: October28,23:08UTC. Attention QK operand swap469d8b04
> passed16/12/integrity as e663e5cbca7041b1 at2.500663288x; slower than2.537000514.
> Changed GM0.984233346, L8controls0.990164299; normalized0.9940101.
> Controller discarded before guard, no score. Retain9e922c35. All other bests unchanged.
> No pending job/controller. Checks/scores left: Dense0/0,MoE0/7,Decode1/4,
> Attention8/5,Prefill18/9; total27/25, derived from22:53API and one check.
> Dense CPU preflight inverted28672BF16codes and exposed escaped-negative-zero
> sign loss in public codec equality check; no Dense candidate/GPU evaluation.
> Read dense_pinned_source_review.md for next bounded source investigation. Goal active.

> Latest checkpoint: October28,22:53UTC. No pending job/controller. Goal active; top50unmet.
> Two checks this continuation, no scored runs. Prefill28d04382 passed5/8/integrity
> at1.680162079x but fixed normalized-profile guard failed1.004422(<1.02); reverted.
> MoEf5737777 passed5/6/integrity at2.348851666x; native65280gate audit passed,
> but aggregate declined vs2.365437416; controller discarded before guard. No Decode transfer.
> Retain MoE33114871(scored2.348299935x),Prefill93e57ab4(scored1.663469245x),
> Attention9e922c35(scored2.537000514x),DecodeD9checked1.990224880x/unscoredrepair.
> API limits: Dense0/0,MoE0/7,Decode1/4,Attention9/5,Prefill18/9(checks/scores).
> Total28checks/25scores; reset00UTC. Latest overall snapshot remains150/1447at22:34UTC.
> Next source inspection: dense_pinned_source_review.md; Dense cannot evaluate before reset.
> Conserve other tracks: no new candidate justified yet. Main chat only.

> Latest checkpoint: 2026-10-28T22:34:05.967226+00:00. Attention9e922c35 scored2.537000514x,
> full16/12/integrity; retained segment/index-domain repair. DecodeD9FE7558
> checked1.990224880x as9560c915e85f4f50, full13/12/integrity. Unscored repair;
> previous published Decode remains1fb5 at1.986659677x. Keep source-bound scores separate.
> ThreeCTA was discarded (real3/no spills, no normalized gain). No pending job/controller.
> Overall API150/1447; no combined-speedup field. Best rows may still reference old sources.
> Checks/scores left: Dense0/0,MoE1/7,Decode1/4,Attention9/5,Prefill19/9; totals30/25.
> Next: prefill_small_expert_mma_review.md. Zero-row plan deferred before GPU use.
> Main chat only; no score reruns for noise. Top50Goal remains active.

> Checkpoint: Attention coverage repair9e922c35 is fully validated and retained.
> Scored8b767ea7b2b74de6 passed16/12/integrity at2.537000514x; native CUDA parent
> c05090e2 checked2.594354656x. Large-L8 fallback guarded and CPU validated.
> ThreeCTA65527644 was discarded: real3CTA/no spills but no normalized gain.
> Decode repaird9fe7558 is CPU-audited and ready for one full13/12 check.
> It retains original FP16 QK/PV and all dense/router/MoE/residual computation.
> Next optimization packet: prefill_zero_row_unpooler_review.md. Goal active.

> Latest checkpoint: October28,21:47UTC. Attentiona310b988 FP8 PV passed
> full16tests/12benchmarks/integrity; check574bd76165e54556 at2.580662730x,
> scored3e2a317e289a4cc8 at2.548282764x. Effectively tied with parent2.541743138x;
> changed and unchanged controls rose similarly. Do not rescore for noise.
> No pending job/controller. Best/candidate Attention nowa310. Decode remains1fb5.
> Next: audit attention_fp8pv_three_cta_review.md; no new3CTA candidate yet.
> API overall149/1445; Attention91/960 still historical2.5624, not a310's rank.
> Live remaining checks/scores: Dense0/0,MoE1/7,Decode2/4,Attention11/6,Prefill19/9.
> Total33checks/26scores. Preserve Decode checks; PV transfer screen~0.5%overall.

# Attention Decode checkpoint - October 28, 2026

Current corrected check: **2.522139036x**, job `1a16e2d658a8400d`, all16tests,
12benchmarks and integrity passed. Interval: [2.484306950,2.559971122].
Exact source: `best/attention_checked.py`; SHA256
`24782695e507801dcf4e753349a97de723d99e3463b2aac0c67a645ff53a884a`.
Scored confirmation `d0cdadc0960446ca` passed at **2.541743138x**,
interval [2.503616991,2.579869285], full16/12/integrity.
The older2.562402309x publication20524d06 used omitted BF16 boundaries;
withdrawal approval is pending. Read ATTENTION_PRECISION_AUDIT.md.
Do not repeat it for timing noise.

The candidate derives from Vjeux's Apache-2.0 implementation at
https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393.
Full license and local change notices are embedded in the source and preserved
in `THIRD_PARTY_NOTICES.md` and `licenses/vjeux-Apache-2.0.txt`.
The upstream leaderboard result is not a measurement of our candidate.

The native CUDA path combines Q/K preprocessing, cache writes, paged attention
and output normalization/gating. It traverses every visible key, retains causal
and sliding-window masks, and performs stable softmax reduction. The CUDA
prologue/finalizers now restore BF16 norm, RoPE and output materializations.
Dispatch uses public shapes and device data.

Local cache changes retain scratch per device and launch signature and reuse
immutable RoPE constants. The first attempt used stream identity in these
keys. Eager warm-up and graph capture then allocated separate buffers, causing
initialization to become replayed graph operations. Removing that distinction
raised the historical unfixed check from2.016854097x to2.589760797x. Those
numbers predate the BF16 correction and do not measure the current source.
Scratch is reset by
the native kernel. Same-signature calls require sequential replay; this source
does not provide independent scratch for concurrent calls sharing a signature.

Earlier work and negative evidence:

- Original fused preprocessing: check `89f9f7d227424d20`, 1.120441329x;
  scored `f6b899f674914ef2`, 1.116457695x.
- Custom partitioned online softmax: `cdc1670c333d4ceb`, 0.699554216x,
  correct but slower. Serial partition loops were expensive.
- Adaptive split count and shorter key chunks: `449fd95cf481493a`,
  0.825614005x, correct but still slower. Rejected without parameter sweeps.
- First native implementation with per-stream initialization:
  `2885135a42f441ca`, 2.016854097x, correct; the cache correction above retained.

Evidence, exact trial sources, profiles and archived controller history are in
`results/resumed-attention-phase2-20261028/`. The previous phase remains in
`results/resumed-attention-20261028/`. The phase is archived to allow publishing
the checkpoint; the overall native Goal remains active and top 50 is unmet.

Use fewer checks and scored runs, per the user's latest request. Each new run
must resolve a concrete question with a material expected gain. Reuse cached
baselines and job handles. A small timing fluctuation is not a reason to repeat
a check or spend another scored submission. Prefill remains last; its official
TASK states it does not contribute to the overall board.


## BF16 boundary correction: full check passed

Corrected Attention24782695 passed16tests/12benchmarks/integrity as
1a16e2d658a8400d at2.522139036x, CI[2.484306950,2.559971122]. This is now
best/attention_checked.py. Exact-source scored confirmationd0cdadc0960446ca
and Decode transfer check9a176eaead1a44ec (source1fb5d05d) are pending.
See ATTENTION_PRECISION_AUDIT.md and results/attention-bf16-boundaries-20261028/.
Withdrawal of five affected historical publications needs explicit approval;
automatic review rejected it and no withdrawal executed. Do not claim that
historical unfixed scores belong to these corrected source files.


Corrected Attention publication d0cdadc0960446ca passed16tests/12benchmarks/
integrity at2.541743138x, CI[2.503616991,2.579869285]. It measures exactly
24782695, the source of the primary2.522139036x check. Historical20524d06
still belongs to the omitted-boundary source and remains on the board pending
withdrawal approval; do not report its2.562402309x as this corrected source's
score. Source, publication result and profiles are saved under
results/attention-bf16-boundaries-20261028/publication/.


## One-query FP8 QK trial closed at21:25UTC

Job60aa54a9feeb49e9/source8463eaf3 passed16tests/12benchmarks/integrity
at2.542655363x, CI[2.504515532,2.580795193]. The six changed L1 cases
had geometric ratio0.999962266 vs corrected247; unchanged L8 controls
rose2.3628%. Batch-one sliding ratio0.96775 violated the fixed0.98 floor.
The controller discarded it (trial91092e8, revert04741e8). No scored run
or Decode transfer. Retain corrected247, scored2.541743138x. One check
used; expected Attention12checks/7scores left. Full payload, profiles,
source and CPU audits, fixed guard failure and immutable controller history
are in results/resumed-attention-fp8q-20261028/. Do not rerun for noise
or retain a selector based on these small individual fluctuations.

Next inspect L1 PV: its helper loads FP8 V and expands32packed FP16
registers per32-key block, with16FP16 MMA operations. A register-only
4x4 byte transpose could feed8FP8 K32 MMAs without extra shared memory.
Keep the retained FP16 QK path. Before a candidate, audit P scaling,
PTX A/B/C mapping, every BF16 boundary and actual FA3 internal precision.


## Complete Attention segment-domain repair

Retained9e922c35 scored8b767ea7b2b74de6 at2.537000514x,
CI[2.498945507,2.575055522], full16tests/12benchmarks/integrity. Native CUDA
is exactly checkedc05090e2 (job14d661f21adc4403,2.594354656x); final source
adds a locally verified fallback guard for L8B>64. Source9e9 itself has a
full scored validation, not a separate check. Driver reuses that completed
scored handle for a cached baseline, avoiding another evaluation.

The source-derived interval audit found an eligible B512/pairs8 one-long-
request case losing154keys under the old64segment cap. L1 now stores512;
public index-space bounds guard int32 prefix overflow. L8B>64 uses the
unchanged general fallback, whose B65/full+sliding CPU comparison against
FP64 had maxL2 error0.0000541864. That fallback may be slower for large
L8 batches; all scored B<=32 native arithmetic remains unchanged. A future
native expansion could compact unused flat-only shared fields, but is not
needed for the current scoring shapes and has no measured gain.

This is a correctness repair, not a demonstrated speedup. Prior scores
2.5483/2.5417 are effectively tied with2.5370. Prior sources had the general
segment-domain limitation; do not claim their coverage extends to these
skewed large batches. Full evidence: results/attention-segment-repair-20261028/.
Do not repeat scored submissions for timing noise.
