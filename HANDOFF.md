> Current checkpoint: October28,21:00UTC. Read ATTENTION_PRECISION_AUDIT.md.
> Corrected Attention24782695 passed full coverage at2.522139036x and scored
> 2.541743138x asd0cdadc0960446ca. Decode1fb5d05d passed full13/12/integrity
> at1.988085030x and scored1.986659677x as75f2c7f464f34c61. Earlier native
> attention sources omitted BF16 boundaries; withdrawal needs user approval.
> No job/controller pending. Redux was discarded by its performance guard.
> MoE phase13 two-group design passed full5/6 but regressed to2.045556897x;
> discarded without a score. Actual occupancy2/SM, no spills. One MoE check left.
> Top50Goal active. API overall147/1442 still includes older publications.

# Current handoff - October 28, 2026

Goal: approach top 50 overall, main chat only, no deadline. Preserve the selected
model and conserve official checks/scores. No subagents, paid GPUs, external
model providers, global changes, credential output, blind sweeps, repeated
baselines or cosmetic resubmissions. The ordered MoE -> Decode -> Attention ->
Prefill pass has full passing coverage. Top 50 overall remains unmet; Goal active.

Live API snapshot at20:33UTC, with source-bound scores separated from API bests:

| Track | Current retained scored source | API best speedup | API place |
| --- | ---: | ---: | ---: |
| Dense | 1.527771327x,6131bc96427e4486 | 1.5278x | 106/1255 |
| MoE | 2.348299935x,b6c70c9b83724d2e | 2.3483x | 54/1001 |
| Decode | corrected1.986659677x,75f2c7f464f34c61 | historical1.9940x | 147/874 |
| Attention | corrected2.541743138x,d0cdadc0960446ca | historical2.5624x | 88/959 |
| Prefill | 1.663469245x,23e7c8b4c3e54869 | 1.6635x | 125/582 |

Overall: **147/1442 at20:33UTC**. API gives ranks only, no combined speedup.
Attention/Decode API places still use earlier omitted-boundary publications
pending withdrawal approval; they are not verified ranks for corrected sources.
Top50cutoffs at this snapshot: Dense1.5523,MoE2.3557,Decode2.1456,
Attention2.6112,Prefill1.7982. Prefill does not count toward overall.
Snapshot: results/decoder-attention-bf16-boundaries-20261028/placement-20261028-203300.json.

Current daily quotas at20:28:30UTC (checks/scored left): Dense0/0,
MoE2/7, Decode2/4, Attention13/7, Prefill19/9. Phase13 then consumed one MoE
check, so expected remaining MoE1/7 and totals35checks/27scores (62evaluations).
Limits are separate by track and mode. Original read-only snapshot:
results/decoder-attention-bf16-boundaries-20261028/quotas-20261028-202830.json.

## Historical decoder publication and rejected trials

Historical09badc9b is archived; both current best decoder files are corrected
1fb5d05d (988827bytes). For the old source, primary check09b34f4e921e49a6 was
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


## Phase11: GS4 serialized, rejected (19:16 UTC)

Checkb47f9fd54cbb4d25/source844c0165 passed5tests/6benchmarks/integrity at
2.282485686x, CI[2.248248401,2.316722971]. Compiler diagnostics explicitly
report C7512 for both BIG instantiations: WGMMA serialized due to insufficient
register resources. Initial96registers, zero reported local bytes, one resident
640-thread block/SM. Full result/audit/source/profiles/controller preserved in
results/resumed-moe-phase11-20261028/. Reverted, no score, no Decode transfer.
Close the GS/allocation tuning path; do not sweep group sizes or register counts.

A two-compute-warpgroup design was assessed, not implemented/evaluated. It
would need128-row WGS units and a repack, double phase-A/B units/staged X reads,
change readiness indexing, and preserve512virtual gather/finalizer threads to
cover4096columns and the old FP32 norm tree. Extra activation traffic makes
this a poor next check without stronger bottleneck evidence.

Next: one isolated WGS-weight cache policy from retained33114871. Fat jobs
split at64tokens, so multiple jobs on a hot expert reread identical weights
within a launch; the existing evict-first comment assumes weights stream once.
Own NumPy public-formula routing draws (seed20261028,64draws/shape) found repeats
in50/64draws atM256, mean logical load/unique-weight ratio1.07969, and17/64
atM128, ratio1.01287. These are synthetic opportunity counts, not official
inputs, cache-hit measurements or a speedup prediction. CPU model is saved
in phase11. Preserve intermediate evict-last stores and all numerical work.

No job/controller pending at this boundary. Expected MoE3checks/7scores left
after five checks since18:24, not a new quota API observation. Goal active.


## Phase12: WGS normal weight priority rejected; move to Decode (19:28 UTC)

Check7b9052f66cab49a2/source273c1f1a passed5tests/6benchmarks/integrity at
2.312809011x, CI[2.278116875,2.347501146]. WGS-normal-enabledTrue. All changed
WGS loads lost about3% relative speedup against primary33114871. Reverted, no
score. Do not sweep eviction constants or transfer this change. Full frozen
source/results/profiles, CPU audits and history: results/resumed-moe-phase12-20261028/.
The GS2cache variant's captured compiler log reports96registers/16barriers,
zero stack/spills and no C7512. This is not SASS or cache-hit attribution.

Inactive Decode cache draft4c910181 is988840bytes and has unchanged non-MoE
AST; it was never evaluated/promoted because standalone timing lost. Both
Decode09badc9b and MoE33114871 stay published/retained.

Six MoE checks and zero scores used since18:24. Expected MoE2checks/7scores
left (accounting estimate, not fresh API observation); reserve those for new
evidence rather than more register/cache tuning. Other track quotas unchanged
by this work. No job/controller pending. Overall remains last observed146
at18:24, no new rank claim. Native top50Goal stays active.

Next in the requested order: Decode. Correct profile attribution at one token:
router_fp32~11.6us, pooler v18s_gemv~13.4us, unpooler smtrace_gemv~4.0us.
At8tokens router~14.9..17.9us. Do not attribute pooler time to unpooling or
spend a check on pruning a tiny zero unpooler path. The full-FP32 router input
currently reloads the same1024-column K-split for all8warps/CTA. Assess one
shared-input cache per CTA:4KB atM1,32KB atM2..8, fresh each launch. Preserve
FP32weights/input, every __fmaf_rn and shuffle/reduction order, sigmoid/top8/ties,
128block counter protocol, output format and side-stream ordering. No BF16
router approximation. No new Decode source or check yet.


## Decode phase10: shared router input rejected after causal review (19:46 UTC)

Checka2960d1516fc4243/source41d475ae passed13tests/12benchmarks/integrity at
2.003057078x, CI[1.973011222,2.033102934]. New xshared kernel appears in all
four small-token profiles. Its affected load speedup ratios against09badc9b
were0.98336/0.99071/0.97434/0.97027: every changed load lost. Router spans
were12.7/11.0/17.0/18.3us versus11.6/11.6/14.9/17.9us. The whole-layer point
gain came from unchanged controls, notably32-token loads. Do not claim a gain.

The controller correctly recorded its numeric keep by the configured whole-layer
metric; its immutable log remains intact. After archiving, review explicitly
restored09badc9b in commit2324c43. No score and no best-source promotion.
Full source/profile/audits/history: results/resumed-decoder-phase10-20261028/.
This demonstrates why future router guards must require affected-load and
component improvement, in addition to full correctness and selected-path proof.

Next: replace top8's repeated float/ID/raw shuffle tree with exact unsigned
ordered-FP32-key max plus integer ID min and one raw-score broadcast. This
changes selection implementation, not score precision. Signed zeros need one
canonical key because float comparisons equate them; NaNs never win the
original local comparisons. A preliminary CPU proof matched old shuffle and
scalar global selection on404vectors/3232picks, including ties/extremes and
NaNs, with raw-score bits unchanged. No CUDA candidate yet. Preserve all
FP32FMA, split reduction, sigmoid, normalization, counters/fences and shape
dispatch. Main chat only; one check after complete review, no repeated baseline.

No job/controller pending at this boundary. Decode quota expected4checks/5scores
after one new check; not a fresh API observation. Published09badc9b remains
1.993956388x. Native top50Goal active; no updated rank claim.


## Decode phase11: exact warp reductions, October28 20:09UTC

Check95d566e83fa74e33 passed13tests/12benchmarks/integrity at2.020471502x,
CI[1.990164430,2.050778575]. All GPU router bitwise probes passed. Controlled
eight-pair cold graph medians for incumbent/new were1.03084/1.06426 at M1
normal/impulse,1.01534/1.01327 at M8. The predeclared guard required every
median>1.03; M8 failed. Controller discarded/reverted automatically. No score
used, no threshold revision or repeat. Prepare255.533s. Complete evidence:
results/resumed-decoder-phase11-20261028/. Original09badc9b remains retained
pending the attention precision correction documented in ATTENTION_PRECISION_AUDIT.md.


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


## Corrected Decode full pass, October28 20:24UTC

Source1fb5d05d passed13tests/12benchmarks/integrity as9a176eaead1a44ec at
1.988085030x, CI[1.958263754,2.017906305]. Full guard passed, including BF16
attention materializations and the original FP32 small router. It is now
best/decoder_verified.py. Exact-source scored confirmation75f2c7f464f34c61
is pending. All other Decode AST nodes are unchanged from09badc9b; no rejected
router experiment is included. Full evidence:
results/decoder-attention-bf16-boundaries-20261028/.
best/decoder_published.py still contains historical09badc9b until confirmation;
its metadata flags the required precision correction. No withdrawals executed.


Corrected Decode publication75f2c7f464f34c61 passed13tests/12benchmarks/
integrity at1.986659677x, CI[1.956859782,2.016459572]. It is exactly1fb5d05d,
now both best/decoder_verified.py and best/decoder_published.py. Primary check
was1.988085030x. Both corrected Attention and Decode have full scored coverage;
no job/controller pending. Earlier published sources remain pending withdrawal
approval and must not supply this source's speedup/rank. All sources/results/
profiles are archived. Top50overall remains unproven and the Goal stays active.


## MoE phase13: two consumer groups rejected, October28 20:59UTC

Check0fb1da41657d4346, source8f6b8c76, passed5tests/6benchmarks/integrity
and analytic all-remote probes. Actual resources match the CPU layout:
320threads, Smem22744bytes, dynamic110848bytes,96registers,0local bytes,
2active CTAs/SM for both variants,132SMs and264CTA capacity. The new path
ran in all M32/128/256 profiles. This proves the intended resource change,
not a speedup: whole point2.045556897x, CI[2.014873544,2.076240250].
Affected relative speedups were0.80046,0.74681,0.72405. Controller discarded
and reverted; guard was not run because whole metric already lost. No score.
Retain33114871 (checked2.365437416x/published2.348299935x).

All changed loads lost, including M32 where smaller job capacity need not
increase jobs. Do not blame the result solely on larger-load repeated weights
or try neighboring occupancy geometry without new evidence. Preserve the last
MoE check for a stronger idea. Full source, result, profiles, edits, CPU audits
and controller history: results/resumed-moe-phase13-20261028/.
No job/controller pending. Next ordered focus is Decode; corrected1fb5d05d
remains checked1.988085030x/published1.986659677x.
