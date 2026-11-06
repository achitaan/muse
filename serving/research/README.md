# Serving predictor: CPU-only evidence and prediction engine

This is an analytical evidence engine, not a trained serving-score predictor.
No GPU experiment, official check, submission, external agent or paid service is
launched by it. Stock is the validated serving baseline; S1 is still a candidate.

## Current data and reusable work

354 archived rows were imported from the paired S1/native measurements and finalizer/
ordered-buffer numerical probes. Rows include graph-capture blocks and dependent
replays: they are **not 354 independent experiments**. All serving targets are missing.
There is no target distribution, serving correlation, held-out accuracy, calibrated
probability or learned score mapping to report. Kernel-board scores are not labels.
The historical dataset has source-file hashes and line numbers; raw evidence remains
in the original directories. Existing SGLang fixtures, paired timing scripts, layout
oracles and patch builders are reusable. S1 has a prospective frozen record, but its
serving estimates are null until workload/critical-path features exist.

The latest follow-up supersedes the old MoE design: a resident-Marlin implementation
already exists, and the ordered-buffer follow-up already failed T128 equivalence.
Do not restart those as novel hypotheses. Read ../FOLLOWUP_HANDOFF.md, ../../TRIED.md
(repository root), and ../../notes/loop-2026-11-04/LOOP_LOG.md before candidate work.

## Interface and versioned feature contract

Prediction/ledger schema version 2 uses JSON with explicit nulls for unknown values.
Version-1 ledger records remain readable and unchanged. Candidate fields:

| Field | Meaning |
|---|---|
| id / lineage / mechanism | Identity and failure-history grouping |
| patch_sha256 | Exact artifact, null for a research proposal |
| by_concurrency | Separate keys 1,4,16,32; missing is not zero |
| critical_path_fraction | Bounded fraction of measured non-overlapped work |
| stage_new_old_ratio | Bounds from matching workload/operator evidence |
| added_time_fraction | Added traffic/launch/reduction time divided by baseline |
| overlap_basis | Must explicitly identify non-overlapped critical-path evidence |
| additional_workspace_bytes / workspace_cap_bytes | Workspace screening only |
| evaluation_cost | Unknown until a bounded experiment is specified |
| validity / evidence / next_experiment | Gate state, sources, smallest useful test |

The model evaluates all endpoints of `new/old = 1-f+f*r+a`. Returned ranges are
conditional scenario bounds, **not 80% prediction intervals**. No uniform prior is
silently imposed. Throughput stays null unless the caller explicitly assumes fixed
work and unchanged queueing/batching. Interactivity and official score stay null.
`critical_path(nodes, changed)` computes a DAG longest path and handles joins using
max rather than summing concurrent durations. Resource serialization must be encoded
as dependencies; this is not a full scheduler simulator.

Per-concurrency `trace` inputs now call that model directly. Supply `nodes` with
`duration_us`, explicit dependencies, `changed_duration_us` bounds, a `source` and
`resource_edges_complete: true`. Do not also supply serial Amdahl fractions. The
model evaluates the fastest and slowest fixed-DAG schedules, including critical-path
switches. Additional launches must be explicit nodes (zero baseline duration and
positive candidate duration where appropriate); changed topology requires a separately
justified graph, not silently dropping edges. This code does not infer dependencies
from a list of profiler kernel durations.

Observation ingestion requires a previously frozen prediction, exact candidate and
patch bindings, post-freeze measurement start, environment/GPU/workload/session/
lineage identifiers, fidelity, execution evidence, gates and source. Negative results
remain recordable. Hash-named exclusive-create ledger files reject overwrite and
detect changed contents. Historical imports are separate and cannot masquerade as
prospective validation. Only one coordinator writes this ledger.

New usable freezes must include `evaluation_context` with baseline patch SHA,
SGLang revision, environment, GPU and workload identity. Ingestion compares these
exactly, checks metric units, timestamps and a unique `measurement_id`, and rejects
future starts or duplicate results submitted through another prediction. Existing
unbound S1 freezes remain historical records, not valid prospective evaluation
contracts. Freeze again with the actual evaluator context before measuring.

```bash
python serving/research/engine.py predict --input serving/research/s1.json
python serving/research/engine.py queue --input serving/research/candidates.json
python serving/research/engine.py freeze --input serving/research/s1.json
python serving/research/engine.py ingest --input measured-observation.json
python serving/research/engine.py calibration
python serving/research/engine.py trace --input serving/research/examples/trace_demo.json
python serving/research/engine.py split --input grouped-observations.json
python serving/research/engine.py review --input measured-observation.json
python serving/research/operator_features.py --output serving/research/operator_features.json
python -m unittest discover -s serving/research -p 'test_*.py'
```

`bootstrap.py` is a one-time import and intentionally refuses to overwrite its dataset.
The queue currently identifies evidence gaps and rejects failed candidates; it does
not claim to rank expected value without meaningful probabilities and costs.
`promotion_review` never changes an incumbent. Every correctness, rule, memory, graph,
TTFT and reproducible-serving-gain gate must be true before human review is eligible.
The measured direction must also improve; setting a gate to true cannot turn an
observed slowdown into an eligible candidate. Review does not independently verify
user-supplied gate assertions and never modifies a serving patch.

`operator_features.json` contains 36 S1 and 18 native-v3 source-bound feature groups.
These summarize existing measurements, not new independent experiments. Token count,
routing pattern, cache regime and EP offset remain separate; the three capture blocks
retain observed ranges and null controls. Their source/patch/harness hashes, runtime,
image and revision are attached. Missing CUDA metadata in a log stays explicitly
missing. Do not insert these ranges into concurrency features without matching
serving call mix, overlap and critical-path evidence.

## Statistical design once serving labels exist

Target log candidate/stock latency or throughput ratios, separately by concurrency
and metric. Fit residuals from the analytical model with regularized shared effects
plus family/concurrency offsets; session effects capture hardware drift. Keep the
operator-to-server transfer distinct from server-to-official-score calibration.
This regression is **not yet fitted or implemented as a production learner**.

Before fitting, group observations into connected components sharing either candidate
lineage or session. Hold out entire components; preprocessing and noise estimates
must use training components only. Replays are within-session noise observations.
Compare no-change, analytical, nearest-family and residual models on log-ratio MAE,
direction accuracy, interval coverage, rank correlation and Brier score for improvement.
Use group-level resampling for uncertainty. Keep the analytical model unless held-out
evidence supports the added model. Report scenario-bound inclusion separately from
prediction-interval coverage; the current calibration CLI does the former only.
The `split` command now implements those transitive lineage-or-session components.
It refuses to produce folds from fewer than two components. Two components only make
a split possible; they do not establish sufficient data or validated performance.
Incorrect-output or rules-unverified results remain in the ledger but are excluded
from performance calibration, and component measurements cannot become serving labels.

## Three distinct research candidates

1. **S1 finalizer fusion:** implemented and operator-measured. Missing call frequencies,
   critical-path shares and batching distributions prevent a numerical serving forecast.
   First calibration is the authorized official check after opening; C16 alone cannot
   establish or disprove C1/C4 benefit. Output agreement remains unresolved.
2. **Routing workspace reuse:** parked after source audit. Output buffers are already
   passed into GEMMs, cache1/cache3 share storage, and capture/replay is separate.
   Reopen only with exposed uncaptured allocation/initialization and lifetime evidence.
3. **DFlash metadata transfer batching:** investigate whether verification metadata
   copies can share a transfer without changing dependencies or draft semantics.
   Parked: buffers are reused and device/dtype guards protect transfers. Reopen only
   when a selected-path trace shows actual exposed transfers and explains the guards.

These are mechanism proposals, not three validated optimizations. The rejected
ordered-buffer candidate is also queued as blocked to preserve negative history.
The exact pinned-source hashes and parking decisions are in `source_audit.json` and
an append-only `source_audit` ledger event. `record_source_audit.py` refreshes the
candidate views idempotently without overwriting previous ledger records.
Typed evolutionary mutations and automatic cost-aware acquisition are deferred:
without known legal profitable configurations or serving labels they would recreate
the uninformative search this project is meant to avoid.

## Rules, limits and remaining work

The live manifest was re-read during this implementation: Nov10 09:00 Pacific opening,
Nov20 00:00 Pacific closing; 409600-byte patch cap, 400-character added-line cap,
3 C16 checks/day, 1 scored run/day. These quotas do not grant spending authority.
The prior $50 total and $15 additional pre-opening limits remain; this work spent $0.
No eight-GPU rental is authorized. Previously rented SGLang checkouts were deleted
with the pods. Previously recorded macOS can/serving paths are not assumed local.
Located `/home/achita/muse-serving/sglang` in WSL. Its active `my-patch` branch is at
`ecc0ea98fb46d27f4e468035e873be35e570d996`, not the evaluator revision. The pinned
`a55ab7d5` object exists and was read with `git show`; the checkout was not changed.
Pinned DFlash source already allocates reusable metadata buffers around lines
1067–1082. Mere buffer reuse is therefore not a novel candidate. Pinned Marlin code
has temporary allocations around lines 77, 114 and 128; their presence does not prove
exposed serving cost because graph capture may already amortize allocation.

This milestone supplies the schema, historical dataset, source-bound operator features,
callable serial/DAG analytical models, freeze/ingest/calibration loop, leakage-safe split
builder, promotion guards and source-audited queue. Remaining work is measured serving
trace collection and construction of its dependency DAG, probabilistic residual
training/validation when data supports it, acquisition/mutation, and actual prospective
serving calibration. The first real prediction-versus-serving result cannot be
completed from current archived component measurements.

The referenced Auto-research-agent repository was reviewed for context; no external
controller was installed or invoked. Its public README describes an orchestration
harness, not evidence that a calibrated serving predictor exists. The supplied X post
could not be fetched (403), so no claims are attributed to it.
