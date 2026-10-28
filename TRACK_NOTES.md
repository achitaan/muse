# Challenge track handoff â€” October 28, 2026

## Current result after the ordered pass

Overall rank snapshot: **221**. Scored: dense 1.527771327x, MoE 2.312613084x,
Decode 1.804641371x, Attention 2.562402309x. Prefill check passed at
1.639290758x; scored job 23e7c8b4c3e54869 is pending. All official test and
benchmark coverage passes for each saved source. Prefill does not contribute
to overall. Top 50 remains unmet. Controllers are archived; native Goal active.
Read HANDOFF.md and the profile-backed next decoder review. Conserve runs.
Earlier status sections below are historical and superseded.


## Current checkpoint (supersedes all earlier historical snapshots)

MoE scored 2.312613084x. Attention scored 2.562402309x. Decoder checked
1.815511785x, passing 13 tests/12 benchmarks; scored job 95bbfd2c81224715 is
pending. Prefill is prepared for one full check, last in the requested order;
read PREFILL_NOTES.md. No Prefill GPU result yet. The overall Goal is active
and top 50 remains unmet. Conserve evaluations; no sweeps or repeat baselines.


## Latest checkpoint (supersedes historical sections below)

User requested fewer checks and runs. Reuse cached evidence and require a
material expected gain before another GPU experiment. Read ATTENTION_NOTES.md:
attention checks at 2.589760797x; scored job 20524d06e3894595 passed at 2.562402309x.
Whole decoder scored at 1.172780190x; MoE at 1.268508633x. Dot's precision review
has arrived and is incorporated. Prefill remains last and does not count toward
overall. Native Goal remains active; the attention controller is archived.


## Current direction

The user removed the original 04:00 Toronto deadline and wants the best possible
progress toward top 50 **overall**, using the main chat only. Preserve evaluation
quota: use checks for focused, justified hypotheses, and submit only validated
candidates. Latest instruction: focus on **muse_moe_w4a16**, with ideas from Dot
expected later. No Dot ideas have arrived yet.

The overall board is separate from individual problem ranks. At the resumption
snapshot, overall rank was 1000 and dense-projection rank was 92. Refresh these
before making a current-rank claim.

## Attention decode: preserved and set aside

- Baseline check `9ef787b1f17340df`: 0.998158016x, all 16 tests and 12 benchmarks passed.
- Fused preprocessing check `89f9f7d227424d20`: **1.120441329x**, all 16 tests,
  12 benchmarks and integrity checks passed.
- Exact source: `best/attention_checked.py` and `candidates/attention_active.py`.
  SHA256 `481e472af7daac9e079547ad3b15614596e6cc650b23c4ac2486013dfc314bc0`.
- Scored job `f6b899f674914ef2` was launched once. Observe it; do not resubmit it.
- Change: one Triton kernel fuses Q/K RMSNorm, interleaved RoPE, FP8 conversion
  and cache writes; FA3 attention and SGLang output norm/gate remain.
- Evidence and controller history: `results/resumed-attention-20261028/`.
- A custom partitioned attention hypothesis is saved in
  `experiments/prepared/attention_split_addon.py`. Its CPU interval coverage and
  softmax merge model passed; GPU compilation, correctness and speed are untested.
  It must not be described as a verified improvement.
- Quota after the scored launch: 18 checks and 9 scored runs left today.

## Whole decoder layer: preserve the baseline, then set aside

- Official starter/specification: `muse_decode_layer/`.
- Candidate baseline: `candidates/decode_active.py`, SHA256
  `2bddf929e2ba027db0f56a506625d038fc99b1c1a9780ebeaff44f9fdc4c3208`.
- Baseline check `65ab5fffad0c4995` completed at **1.002013444x**, with all
  13 tests and 12 benchmarks passing. Do not repeat this baseline.
- Baseline controller is archived under results/resumed-decoder-20261028/controller/.
- Controller verifier: `scripts/measure_decode.py`; immutable raw evidence is
  cached under `.decode-measurements/<source SHA>/`.
- No optimization trial or scored run has been launched on this track.
- Prepared hypothesis: fuse global-to-local expert-ID mapping; pass 32 local
  experts to alignment instead of 256 global experts after IDs are mapped.
  CPU exhaustive mapping equivalence passed all 256 IDs on all 8 EP ranks.
  Addon/audit are `.tools/decode-route-addon.py` and
  `.tools/decode-routing-cpu-audit.json`; no GPU correctness/speed claim yet.
- Dense projection integration is another hypothesis. Module-level intersections
  with the verified dense source are only Config, F, Inputs, TypedDict, Weights,
  _bf16_gemm_dispatch_impl, kernel_fn, prepare and torch; preserve namespaces and
  the upstream MIT notice if integrating. No integration has been made.
- Quota after the baseline launch: 19 checks, 10 scored runs left today.

## MoE: current focus

- Official starter and specification: `muse_moe_w4a16/`; candidate baseline: `candidates/moe_active.py`; verifier: `scripts/measure_moe.py`.
- Full untouched quota at focus change: 20 checks, 10 scored runs.
- Task: NVFP4 expert weights, BF16 activations, FP32 reductions, 32 local experts
  of 256 total, top 8 routes; per-expert post norm before weighted sum.
- Official coverage: 5 correctness tests and 6 scored benchmark cases.
- Baseline: Marlin repacking in prepare, route alignment, two Marlin W4A16 GEMMs,
  SwiGLU, fused post-expert norm and weighted sum.
- First evidence-backed hypothesis to assess: baseline tile selection uses all
  token routes divided by 32 local experts even though most routes are remote;
  this selects M=48/64 tiles for M=128/256, potentially padding heavily. Consider
  a focused 8/16-token block choice and local-expert alignment after profiling.
- Keep all NVFP4 codes, FP8 block scales and FP32 global scales exact. No output
  caches, dropped routes, activation quantization or harness/timing changes.

## Dense projections: previous verified result

Best scored job `6131bc96427e4486`: **1.527771327x**, all 9 tests and 30 benchmark
cases passed; source `best/verified.py`, SHA256
`edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b`.
Both quotas are exhausted today (20 checks and 10 scores). Historical rank 85
was provisional and is no longer a live placement claim. Preserve upstream MIT
attribution in THIRD_PARTY_NOTICES.md and the embedded license.

## Workflow and resource boundaries

Use the existing native Goal and one experiment controller at a time. The decoder baseline phase is archived; the new MoE phase uses its own controller.
Keep frozen source bytes, official full-case records, confidence intervals and
honest keep/revert history. User requested main-chat-only work; do not delegate.
Use the existing authenticated free official H200 service. Never print or commit
credentials. Quotas are per problem and reset at 00:00 UTC. No paid GPU service
or external model provider is authorized.

## Latest MoE update

Read MOE_NOTES.md for measured progress, Dot review, precision correction, and remaining track order. Attention scored result is 1.116457695x; its earlier pending status above is historical.

## Track transition after MoE

MoE retained a fully checked FP32-norm candidate at 1.268945620x. See best/moe_metadata.json and MOE_NOTES.md. Moving to Decode Layer next, then Attention Decode, with Prefill last.

## Decoder checkpoint and next track

Read DECODE_NOTES.md and best/decoder_metadata.json. Whole-layer checked score is 1.168638714x; a single scored publication is pending. Attention Decode is the current focus, with Prefill last. MoE scored publication passed at 1.268508633x.

Latest MoE update: check 6792a5c832ee4aff passed at 2.299797371x with
all 5 tests and 6 benchmarks. Exact source best/moe_verified.py. Scored job
b4a10b76347e4cb0 is pending. Read MOE_NOTES.md; the old 1.2685x result is
historical. Current next phase is one combined native decoder check, preserving
the FP32 router and all residual boundaries. Latest overall snapshot: rank 619
after attention, before the new MoE score. No sweeps; Prefill remains last.

Latest: Prefill scored job 23e7c8b4c3e54869 passed at 1.663469245x.
Current decoder fusion draft has passed bitwise codec/layout and Python
binding audits; GPU correctness and speed remain unmeasured. One check will
test 9..32-token fusion, preserving the complete FP32 small-token router.
Publish only if its checked interval excludes the retained interval.

## Live placement snapshot, 2026-10-28 13:55 UTC

| Track | Rank |
|---|---:|
| Overall | 221 / 1426 |
| Dense | 96 / 1239 |
| MoE | 107 / 988 |
| Decode | 221 / 862 |
| Attention | 79 / 942 |
| Prefill | 118 / 570 |

Read-only board queries; no evaluation quotas were spent.
