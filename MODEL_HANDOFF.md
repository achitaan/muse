# Muse Open Kernel Challenge — factual handoff

Snapshot: **October 29, 2026, approximately 2:56 p.m. America/Toronto**
(18:56 UTC). This document contains state and evidence, with no proposed
optimization roadmap. Earlier plans elsewhere in the repository are history.

## Repository and execution state

- Repository: https://github.com/achitaan/muse — private.
- Working branch: **`codex/h200-2-5x-20261028`**.
- Local workspace: `C:\Users\Achita\Documents\ChatGPT\Muse 2`.
- All five retained sources matched the already-pushed commit
  `62868e253e424efe022edbed090ad1f50ab3f6ee` byte for byte when this handoff began.
- This handoff and the new untested draft are included in a subsequent commit
  on the same branch. The branch's delivery commit is reported in the chat.
- Git's configured upstream is `origin/main`; the working branch has been
  pushed explicitly to its matching remote branch.
- **No live official GPU job** at 18:55:57 UTC. Autoresearch status is
  **`not_initialized`**. No new GPU check was started for the tail prototype.
- The active challenge goal is not achieved. No verified performance gain
  occurred in the audited 11-hour window, and the subsequent completed
  experiments also produced no retained replacement.
- Work remained in the main chat; no subagent or model switch was used here.

## Retained sources and source-bound results

These are the working files. The hashes identify exact source bytes; a checked
file and a published predecessor are different evidence.

| Track | Retained file | SHA256 | Source-bound result |
|---|---|---|---|
| Dense | `best/verified.py` | `edc61a1e1fb26a10bcfacda3ef3fc06833c17ec7a078d5ded8eb8717509b765b` | Scored **1.5277713270942548x**, job `6131bc96427e4486` |
| MoE | `best/moe_verified.py` | `331148716e8d1ba92b45cd29725be997d44b3a5235af74d75770bb9444100d50` | Scored **2.348299935218348x**, job `b6c70c9b83724d2e`; primary check **2.365437416088433x**, `d600231c8a8244cb` |
| Decode | `best/decoder_verified.py` | `d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988` | Checked **1.9902248796281297x**, job `9560c915e85f4f50`; this repair is **unscored**. Corrected published predecessor `1fb5d05d...` scored **1.986659677x**, `75f2c7f464f34c61` |
| Attention | `best/attention_checked.py` | `9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f` | Scored **2.5370005142373167x**, job `8b767ea7b2b74de6` |
| Prefill | `best/prefill_verified.py` | `93e57ab45fb71414c65b71a2b03241a62b3d6097890cb7612616e620c050aae7` | Scored **1.6634692445639083x**, job `23e7c8b4c3e54869` |

The corresponding `candidates/` active files remain the retained versions.
The Decode active path is `candidates/decode_active.py`; Dense is
`candidates/active.py`. Exact hashes and the pushed-commit comparison are in
`results/model-handoff-20261029/retained-sources.json`.

## Fresh account leaderboard and quota snapshot

Read-only official API observations at **18:55–18:56 UTC October 29**:

**Overall: 207 / 1,597.** The API says its overall measure is rank across
problems and supplies ranks only. It does not expose an overall speedup or
an aggregation formula in this response. Prefill is explicitly excluded from
the overall competition.

| Track | Account board rank / ranked entrants | Account board speedup, rounded by API | Checks left today | Scored submissions left today |
|---|---:|---:|---:|---:|
| Dense | 147 / 1,384 | 1.5278x | 18 | 10 |
| Attention | 132 / 1,063 | 2.5624x | 16 | 10 |
| MoE | 79 / 1,109 | 2.3483x | 12 | 10 |
| Decode | 207 / 977 | 1.9940x | 17 | 10 |
| Prefill | 164 / 637 | 1.6635x | 20 | 10 |
| Total remaining | — | — | **83** | **50** |

**The account board table is separate from the retained-source table.** The
leaderboard response does not bind its `you` row to a source hash. In
particular, the account Attention/Decode board values above must not be
assigned to the corrected retained files. Historical publications include
earlier versions with omitted precision boundaries. Previous withdrawal
attempts were rejected by automatic approval review; none were withdrawn.
Owner authorization remains required for withdrawing publications.

The daily limits are separate per problem: 20 checks and 10 scored submissions.
A check uses private evaluation quota and does not publish to the board.
Limits reset at 00:00 UTC. Quotas and ranks can change after this snapshot.
Raw API responses, timestamps and the no-live-job observation are archived
under `results/model-handoff-20261029/`.

## Current evaluation authorization and policy

The human authorized sparing GPU use until **October 29 at 8 p.m. Toronto**
(`2026-10-30T00:00:00Z`). That authorization has not been extended.
`EVALUATION_POLICY.json` records:

- At least 60 minutes globally between newly accepted checks.
- At most one active GPU job.
- Four checks and four scored submissions reserved per track.
- No new check in the last 20 minutes of the window.
- A retry needs a diagnosed bounded defect; quota is not a spending target.
- Publication follows source-bound correctness, actual new-path execution
  and material performance evidence.

Last accepted check: Dense `f026c3b14a864be0`,
**18:03:28.127090 UTC October 29**. The next pacing slot is therefore no
earlier than **19:03:28.127090 UTC / 3:03 p.m. Toronto**. This timestamp is
an eligibility boundary, not a scheduled job. There is no controller waiting
to launch automatically.

## Implementation facts

The target is **H200 / Hopper**. Each source is a self-contained Python
submission with the official `prepare(weights)` and `kernel_fn(data,state)`
interfaces. Preparation is untimed but capped at 300 seconds; runtime is
captured and replayed as a CUDA graph with fresh inputs and cold L2.

### MoE

The retained CUDA payload is
`1e60cf469ebff8d0f1e63a223f41f041374a1f30f90e4201a819ee178353270c`.
It is already one persistent native kernel covering routing, both GEMMs,
SwiGLU, norm statistics and final weighted reduction.

Dispatch: TINY for M<=4, TINY2 for M5–8, MID for M9–16, MAIN/WGS above 16
when eligible, with the original non-PRMT fallback. `_make_scratch`, `_ext`,
`_scales_prmt_exact`, `_pack_codes`, `_pack_scales` and `_wgs_copy` are real
symbols in the standalone implementation. The fixed chunk capacity is 384.

All 256 experts occupy 6 GiB packed codes plus 0.75 GiB FP8 scales, before
global scales. A 32-expert shard occupies 864 MiB plus global scales. Fixed
scratch capacity is approximately 157 MiB; capacity is not per-call traffic.
Persistent expanded BF16 experts are disallowed by the task.

The retained implementation preserves Marlin-equivalent BF16 accumulator,
global-scale and product boundaries; FP32 SiLU-and-multiply followed by BF16;
BF16 effective `(1+w)` postnorm gain; FP32 normalization and route reduction
with epsilon 1e-8; final BF16 output. The standalone checker uses 1.5% global
relative L2 and 7.5% worst-row error with the documented denominator floor.

### Decode

D9's sequence is QKV -> native attention/cache/norm/gate -> O -> attention
residual/norm -> pooler/router work -> join -> pre-expert norm/local-ID mapping
-> persistent MoE -> unpooler -> FFN residual/norm. Router/pooler overlap
already exists for small tokens. Small-token router arithmetic is FP32.

Pre-expert norm and ID mapping are already fused with each other. For T<=8,
the unpooler computes 1024 local columns with an implicit zero tail. The
final norm still spans **8192** elements; above eight tokens the full unpooler
is required. D9 is 989,018 bytes; historical preparation was approximately
215–257 seconds. Source size and preparation headroom are real constraints.

Decode has **six loads**, each weighted as `46*sliding + 16*full`. Its twelve
benchmark rows are not equally weighted independent loads. Dense load totals
combine five projections. MoE has six equal-weight loads. Profile kernel
duration sums do not establish non-overlapped critical-path fractions.

### Attention and shared numerical behavior

Attention retains every valid key and causal suffix, BF16 Q/K norm and
applicable RoPE results, FP8 cache/scales, BF16 attention output before
norm/gate, and final BF16 output. Its last-CTA finalizer already exists.
Decode also checks routing, cache writes, residual update and normalized
output. Task checkers differ; MoE's tolerances do not apply to every track.

Routing and inputs are fresh every replay. No timed host routing reads,
input-dependent allocations, cached data-derived decisions or lazy compile
are permitted. Invalid/nonlocal routes are masked before loads/arithmetic;
NaN times zero is not a valid mask. All-nonlocal MoE outputs must be zero.
The retained scratch contract is sequential same-state replay.

## Measurements since the retained wins

`results/optimization-return-audit-20261029-1510/report.json` documents nine
accepted private checks between 04:10 and 15:10 UTC October 29, **zero retained
performance wins**, and unchanged hashes for all five best sources.

Selected subsequent and recent terminal evidence:

| Experiment | Measured outcome | Evidence directory under `results/` |
|---|---|---|
| Dense fixed QKV cluster3 / pooler cluster2 | M128 QKV 3.7–4.1% slower; pooler 3.2–7.0% slower in paired probes. Empty choices. Full 9/30/integrity fallback point 1.501728512x is not new-kernel evidence. Job `f026c3b14a864be0`; no score | `dense-cluster-resources-check-20261029/` |
| Decode dual-order cluster norm | 36 numerical probes and eight replays passed. All eight paired boundaries lost; combined M1/8/16/32 new/old time ratios 3.066/2.833/1.723/1.866. Job `545f0b524917423b`; no score | `decoder-cluster-norm-dual-order-check-20261029/` |
| Decode first cluster norm | Native resources passed, setup FP32 residual gate failed before full suite. Job `63acbd4d99324fea` | `decoder-cluster-norm-setup-failure-20261029/` |
| Attention two-score pipeline | Full 16/12/integrity; actual new path; point 2.44137533x, normalized affected gain 0.83209 | `attention-two-score-pipeline-check-20261029/` |
| Attention external finalizer | Full suite; point 2.575983343x, normalized affected 1.01555 below its frozen gate; one case also failed its gate | `attention-external-finalize-check-20261029/` |
| Decode cooperative O projection + boundary | Correctness/resource probes passed; paired new/old 1.3874 and 3.0763. Job `7f85bd70a3484dc7`; no new path retained | `decoder-cooperative-boundary-check-20261029/` |
| MoE FP32 finalizer cache | Correctness/resources passed; affected improvements only 0.27%/0.03%/0.12%, below its frozen materiality gates. Job `94ce080b1fb148a3` | `moe-fp32-final-cache-check-20261029/` |
| MoE cluster activation multicast | Correctness/resources passed; M128/M256 1.44%/0.70% slower; nearby M385 2.29% slower. Job `6627d54067774130` | `moe-cluster-x-check-20261029/` |
| MoE paired epilogue stores | M32/M128 slightly slower, M256 0.77% faster; selection false. Job `5262c2c560c94197` | `moe-epair-store-repaired-check-20261029/` |

Other recorded negative results include MoE skinny16 (34.6–45.6% slower),
two consumer warpgroups (two resident blocks/SM and no local spill, but slower),
GS4/register allocation, six-bit scales, cache priority and scheduling variants.
CPU-only direct-stage gather/private-fixup drafts increased spill fields and
were not GPU-launched. `MOE_NOTES.md`, `DECODE_NOTES.md`, `ATTENTION_NOTES.md`,
`PREFILL_NOTES.md` and source-bound result packets contain the detailed history.

An empty-choice check can pass the official suite while timing only the old
fallback. Such points have not been counted as improvements to the new kernel.
The approximately ±1.5% interval floor is not an acceptance threshold or a
joint confidence interval comparing two independent jobs.

## Untested work saved with this handoff

**No GPU run or speedup claim exists for the following prototype.**

- Python source: `1ed5ba7b1c87ae0142bf0d52fba08e7938c1be9dcc5d3200d388fabee15c951f`,
  956,861 bytes, in `results/decoder-moe-tail-preflight-20261029/submission.py`.
- Native CUDA: `95abcfa8857ca584e03c86a5ba1ac583a26b6abde4ff13b5c4fcafe5a2052091`.
- It combines the tiny MoE launch, unpooler and final FFN residual/norm.
  It retains the global BF16 MoE output and uses per-token publication plus a
  second grid join. It uses private sequential scratch and a cooperative launch.
- Both offline tiny variants compile at 150 registers, zero stack/spills.
  All 4,194,304 dense BF16 values reconstruct bitwise; 2,048 finite replay
  schedules, 30 boundary fixtures, eight impulse GEMMs and CPU host tests pass.
  Twenty-five corrupt result fixtures are rejected; the positive fixture is
  explicitly synthetic. All six freshly fetched official files match pinned.
- Native Torch/CUDA ABI, H200 occupancy, GPU correctness/replay, latency,
  preparation time and actual peak memory are **unverified**.
- `launch-admission.json` is **false**. The driver refuses to launch it in
  this state. It has not replaced any `best/` or active candidate file.
- Builder, helper, host, addon, CPU model, review and draft guards are under
  `experiments/prepared/decoder_moe_tail*`, `build_decoder_moe_tail.py`,
  `freeze_decoder_moe_tail.py`, `model_decoder_moe_tail.py`,
  `audit_decoder_moe_tail_admission.py`, and `scripts/*decoder_moe_tail*`.

Also archived: a single-loop MoE six-state ring recurrence, CUDA `d61835dd...`,
at `results/moe-ring-recurrence-preflight-20261029/`. It uses no GPU quota.
It preserves the finite slot/parity model and avoids the earlier C7512 warning,
but generated static instructions increase 25,088 -> 25,256. No Python
candidate or performance result exists. Earlier three-body variants in
`moe-static-ring-preflight-20261029/` newly serialized WGMMA.

## Tooling and evidence layout

- `muse_*/TASK.md`, `reference.py`, `checks.py`, `baseline.py`, `AGENTS.md`:
  pinned official interfaces, semantics and checks. Fresh identity evidence
  for the new draft is included in its result packet.
- `STATE.json` and `HANDOFF.md`: accumulated state/history; some fields and
  headers are historical. Exact source-bound terminal packets take precedence.
- `results/retained-load-opportunities-20261029/loads.json`: saved load weights,
  benchmark latencies and attributed profile summaries.
- `scripts/official.py::cli`: official CLI access wrapper. Its old `main()`
  deadline is October 28 and is obsolete; the newer per-experiment drivers use
  `EVALUATION_POLICY.json`.
- Official CLI on this machine is the ignored workspace `.tools/okc` under
  Ubuntu WSL. It is authenticated as of the read-only observations above.
  Credentials and ignored compiler/SDK caches are **not** included in GitHub.
- `experiments/prepared/local_nvrtc_compile.py`: Windows NVRTC 13.0 CPU-only
  device compilation. Version hashes and flags are saved with its artifacts.
  The compiler wheels are in ignored `.tools`; this is not a GPU measurement.
- Exact source/evidence bytes are protected by `.gitattributes`. Large binary
  caches and per-experiment working measurement directories are ignored.
- The last completed autoresearch trial was reverted and archived. No active
  controller owns the repository at handoff.

This handoff contains neither credentials nor a proposed optimization strategy.
