> Current checkpoint: October28,18:57UTC. MoE33114871 retained:2.365437416x
> checked,2.348299935x published. Latest three trials lost/tied and were reverted;
> no score used. No pending job/controller. Read phase9 at the end of this file
> and experiments/prepared/moe_register_split_review.md for next work. Goal active.

# Current handoff - October 28, 2026

Goal: approach top 50 overall, main chat only, no deadline. Preserve the selected
model and conserve official checks/scores. No subagents, paid GPUs, external
model providers, global changes, credential output, blind sweeps, repeated
baselines or cosmetic resubmissions. The ordered MoE -> Decode -> Attention ->
Prefill pass has full passing coverage. Top 50 overall remains unmet; Goal active.

| Track | Published speedup | Last observed place | Job |
| --- | ---: | ---: | --- |
| Dense | 1.527771327x | 105/1252, 18:24 UTC | 6131bc96427e4486 |
| MoE | 2.348299935x | 54/995, 18:24 UTC | b6c70c9b83724d2e |
| Decode | 1.993956388x | 146/871, 18:24 UTC | 441eb580f3074669 |
| Attention | 2.562402309x | 86/954, 18:24 UTC | 20524d06e3894595 |
| Prefill | 1.663469245x | 125/577, 18:24 UTC | 23e7c8b4c3e54869 |

Overall: **146/1438 at 18:24:33 UTC**. The overall API gives ranks only,
not a combined speedup; whole-decoder-layer speedup is 1.993956388x. Cutoffs last
read at 17:58 were Decode 2.1438x and MoE 2.3539x. Prefill does not count toward
overall. No job or controller is pending.

Current daily quotas at 18:24:33 UTC (checks / scored left): Dense 0/0,
MoE 8/7, Decode 5/5, Attention 14/8, Prefill 19/9. Totals: 46 checks and
29 scored runs, 75 evaluations, allocated separately by track and mode. Full
read-only snapshot: results/resumed-moe-phase6-20261028/quota-and-score-snapshot.json.

## Decoder publication and rejected trials

Exact 09badc9b source is both best/decoder_verified.py and
best/decoder_published.py, 988314 bytes. Primary full check 09b34f4e921e49a6:
1.994319167x. Scored confirmation 441eb580f3074669: 1.993956388x, interval
[1.964047042,2.023865734], full 13 tests/12 benchmarks/integrity. It transfers
the corrected 33114871 MoE, adds implicit small-token FFN zero padding and early
FFN loads, preserving all BF16/FP32 norm boundaries and full FP32 small routing.
This replaces a61735ee's 1.965324622x publication; frozen history remains.

Phase7 skipped new projection setup at its budget gate. Phase8 compared three
fixed SK2 small-token projections: all tied or lost. Phase9 compared one output
projection: 4.9% slower. Both explicit choice tuples were empty and the guard
discarded them despite higher noisy aggregate points. No rejected projection
source was scored. Those full trials reused incumbents for timed calls, so do
not claim full official correctness of unselected new kernels. Evidence:
results/resumed-decoder-phase7-20261028/ through phase9, including full sources,
profiles, audits and immutable controller histories. Observed setup varied from
215 to 257 seconds; stay below the official 300-second prepare and 1 MB limits.

Publication used one confirmation of the retained update after its primary
check and three incumbent-only observations. Individual intervals overlap;
report measured points and placements without claiming formal significance.

Profile correction: m16trace_ is the 16-token output projection (~5.6 us).
The ~14.6 us dd_cap2_b_cpssm stage is unpooling. Resolve dispatch and symbols
before attributing truncated profile names. Do not repeat the mistaken O16
bottleneck attribution or the rejected projection comparisons.

## Next: bounded 32-token WGS specialization

MoE 33114871 remains checked and published at 2.348299935x, full 5/6/integrity.
Paired BF16 epilogues and exact DQ8 placement preserve original GEMM rounding
and FP32 norm arithmetic. The prior mixed skinny/fat selector also chose none.

The phase6 scale-six candidate passed all 5/6/integrity with compression enabled
but regressed to 2.198081320x, interval [2.165110100,2.231052539]. All three WGS
loads lost 11..14% relative speedup; unchanged small paths stayed close. It was
discarded; one check and no score used. Full sources, profiles, reversible
source ledger, full-size byte audits and controller history are preserved in
results/resumed-moe-phase6-20261028/. Retain 33114871 and original byte scales.
Do not retry scale encodings: lower traffic did not overcome added decoding.

Read experiments/prepared/moe_wgs32_review.md. One source-backed next hypothesis
is a four-tile main WGS module for 17..32 tokens, while retaining the original
kernel elsewhere. Current WGS_NT is eight even for this band. No new candidate
is implemented or measured yet. Prune wider MMA variants consistently, preserve
all jobs/routes and precision, and use a separate scratch bank because counter
offsets change. The existing allocation is sized for both capacities; prove
launcher offsets and alternating-shape isolation before one focused check.
Reuse original packed weights, at most one extra module, no geometry sweep.

Read STATE.json, per-track notes, official TASK.md/AGENTS.md and notices. All
controllers are archived; initialize one scoped phase with cached baseline
before trials. Preserve handles across observation failures and push complete
evidence at boundaries. Query quotas only when needed. Latest per-track quotas are saved above; query them again only when needed.

Preserve every route/key, fresh inputs, cache writes and residual outputs.
No persistent higher-precision weight expansion, input/output caching or
private harness/seed dependence. Shared native scratch requires sequential
calls. Retain Apache/MIT notices and attribution to Vjeux, Nathen Fernandes,
Lee Penkman and SGLang.


## Phase 7: four-tile WGS rejected (18:41 UTC)

Full check 47f44d0ff8704a44, exact source d0ae4a9a775b28d614fee1768d37fad609024c44ad2189058565263b03c9f9b4,
passed all 5 tests, 6 benchmarks and integrity at 2.353666801x, interval
[2.318361799,2.388971803]. Specialization was enabled. Baseline remains
33114871, checked at 2.365437416x and published at 2.348299935x. No score used.

The 32-token speedup was 1.638191005x versus baseline 1.642387556x.
Actual resource queries: original 96 registers/thread, zero local bytes,
231168 dynamic shared bytes; specialization 95 registers, 8 local bytes,
183552 dynamic shared bytes (both BIG instantiations agree). Shared memory
fell, but register capacity barely moved and a spill appeared. Separate-job
timings do not isolate the spill's cost. Do not retry tile sizes as a sweep.

The discarded wrapper also lacks a final prepare synchronization after its
new scratch allocation. It was never published. Any future reuse must fix
that readiness issue and check the resulting exact source. There is no reason
to spend another check repairing this losing experiment alone.

Complete frozen source, raw results/profiles, CPU/source audits and immutable
controller history: results/resumed-moe-phase7-20261028/. Controller reverted
the candidate to the verified baseline and was archived at 18:43 UTC.
One MoE check consumed after the 18:24 quota snapshot (expected 7 checks/7
scores remaining, not a fresh API quota observation). Goal remains active.


## Phase 8: scheduler estimate rejected (18:52 UTC)

Full check35f15c0fcf9247d6, source df1871fad29b04f70aed36c7e3e1a5085066346c966aee5d07fe96e1a928a843,
passed5tests/6benchmarks/integrity at2.341433747x, CI[2.306312241,2.376555253].
WGS32/128/256 speedups1.624659818/2.557936989/2.856700906 did not improve
on the primary baseline1.642387556/2.590667418/2.874509152. Separate-job
intervals overlap; no isolated code-layout or instruction-cost claim. Controller
reverted to33114871 and was archived. One check, no score. Full source, raw
profiles/results, exact-owner CPU audit and history: results/resumed-moe-phase8-20261028/.
Do not transplant this quotient change to Decode or Attention.

Next reviewed hypothesis: const __grid_constant__ Params in the native MoE
kernel declaration. All device helpers already take const references and the
parameter object has no member mutations; pointed-to scratch/output stays
mutable. This may avoid per-thread parameter copies, but no generated-code
or timing gain is established. See experiments/prepared/moe_gridparams_review.md.
No pending job at this phase boundary. Expected MoE quota6checks/7scores after
two checks since18:24; not a fresh quota API observation. Goal remains active.


## Phase 9: grid-parameter annotation rejected (18:57 UTC)

Full checke5ed457feaff4119, source1613f98e749538dde707677bd268fb6d94fe39b45341147d2eaadae243067c71,
passed5tests/6benchmarks/integrity at2.354389959x, CI[2.319074110,2.389705809].
No aggregate gain over retained33114871; reverted without a score. The source
audit recovers the exact original kernel body and module AST. This experiment
does not establish whether parameter copies were present or eliminated in SASS.
Raw sources/profiles/results, source audit and controller history are saved in
results/resumed-moe-phase9-20261028/. No trial from phases7..9 is published or
transferred to Decode. Baseline remains2.365437416x checked/2.348299935x scored.

Next investigation is design-only: experiments/prepared/moe_register_split_review.md.
A complete producer warpgroup may allow explicit register redistribution, but
compile-time resource admission, all thread roles and barrier counts need proof
before implementation/evaluation. Keep GS2 and every original numerical path.
Do not apply setmaxnreg to a partial producer warpgroup or mix in losing trials.

All controllers are archived. No official job is pending. Three checks and
zero scores used since18:24. Expected MoE quota5checks/7scores remaining, an
accounting estimate rather than a fresh API observation. Overall rank remains
the last observed146/1438 at18:24; no new placement claim. Top50Goal is active.
Public Vjeux and LeePenkman repository HEADs still match their already-pinned
f35a41ee andc565e373 revisions; no newer public patch was found in those repos.


## Phase10: full producer warpgroup, no retained speedup (19:08 UTC)

Checkfc4ff22b0b184caf/source17adb3e6 passed5tests/6benchmarks/integrity at
2.340478636x, CI[2.305371456,2.375585815]. Resource preflight was
(640,96,96,0,0,231168,1,1,65536,132): both initial allocations96, no static
local bytes, one resident block/SM. Split enabledTrue. No changed WGS load
improved over primary33114871. Reverted, archived, no score. All math helpers
and GS2 were unchanged. Static attributes do not directly measure dynamic
ownership or establish a compiler-stall cause. Full evidence and audits:
results/resumed-moe-phase10-20261028/. No official job pending at this boundary.

An inactive Decode transfer draft21bf949f is991255bytes with all non-MoE AST
unchanged, but it was never evaluated or promoted because standalone timing
did not improve. Retain09badc9b in Decode and33114871 in MoE.

One source-backed follow-up: four-step WGMMA groups on this admitted register
split. Original source specifically says GS4 serialized16..32-token variants
at96registers; GS4 needs16more live A-fragment registers, matching the reviewed
96to112compute budget change. Test that paired mechanism once, retaining all
geometry, original layouts and GSFCI3; capture ordinary compiler diagnostics.
Do not retry tile/encoding/grid parameters or score a flat result. Expected
MoE quota4checks/7scores after four checks since18:24 (not a fresh API query).
