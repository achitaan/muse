# GOAL MODE — OPEN KERNEL CHALLENGE
## Autonomous Codex-Only GPU Optimization Sprint

### PRIMARY OBJECTIVE

Compete in the Muse Spark Open Kernel Challenge:

https://openkernelchallenge.dev/

Achieve the best technically valid kernel performance possible during this sprint.

**Start immediately. Continue autonomously. Stop at 1:00 PM America/Toronto on October 23, 2026.**

This is an intensive, approximately one-hour engineering sprint. Do not treat it as an ordinary one-shot coding task.

Use my existing global Ralph-style loop engineering system where available.

The continuous workflow is:

**Understand → Research → Plan → Implement → Test → Benchmark → Analyze → Improve → Repeat**

Your purpose is to maximize meaningful engineering progress and competitive performance before the deadline, while making aggressive use of my available Codex subscription usage ahead of its expected reset.

---

## 1. MODEL AND USAGE POLICY

I am using **ONLY Codex**.

Do not invoke:
- OpenCode
- Claude Code
- External model providers
- Paid model APIs
- Third-party coding orchestrators
- Paid GPU/cloud infrastructure

Use models available through my authenticated Codex subscription.

### Model routing

Prefer the strongest available Codex model appropriate for the work:

- GPT-6 Astra / Astra Pro, if supported: difficult mathematical reasoning, GPU optimization strategy and architecture.
- GPT-6.1 Sol: implementation, debugging, experiments and integration.
- Native Codex subagents: independent research, parallel experiments, mathematical verification and adversarial code review.

These are preferences, not mandatory assignments. Inspect actual installed models and supported reasoning effort settings before changing anything.

Use High, XHigh, Max or Ultra reasoning where available and justified.

**AGGRESSIVE USAGE POLICY**

I expect my Codex usage to reset soon. I explicitly prefer spending substantial available subscription usage on productive engineering now rather than preserving it unnecessarily.

Therefore:
- Do not minimize reasoning effort merely to save credits.
- Use stronger models when they may improve solutions.
- Proactively spawn native Codex subagents for independent useful tasks.
- Perform deep research and additional validation where they can reveal better optimizations.
- Do not stop when usage becomes moderately low.
- Do not throttle productive work to preserve subscription allowance.
- Check actual usage/reset information where supported, without inventing numbers or reset guarantees.

However, do not burn credits on meaningless repetition, fake experiments or redundant agent conversations. Spend aggressively on solving the competition.

Do not purchase additional credits or incur pay-as-you-go costs.

---

## 2. THE HARD TIME LIMIT

Deadline: **October 23, 2026, 13:00 America/Toronto (EDT).**

Check the local system clock.

Run the actual engineering loop until the deadline.

At approximately 12:55 PM:
- Begin final integration and verification.
- Preserve the best correct implementation.
- Save experimental results and ongoing work.
- Finish any feasible official evaluation.
- Prepare a concise final report.

At 1:00 PM:
- Stop launching new work.
- Terminate or gracefully stop the local loop and its workers.
- Do not leave Codex consuming usage after the deadline.
- Do not delete useful progress.

If a task finishes early, **start another useful task automatically**.

Do not declare completion simply because the original implementation works.

The objective is continuous improvement until the time limit or an actual blocking resource exhaustion.

---

## 3. INITIAL RECONNAISSANCE

Immediately inspect:

1. The official competition website, current rules, scoring system and kernel specifications.
2. The local workspace and any existing challenge repository.
3. My current Codex configuration and installed version.
4. My existing global Ralph-style loop runner, if present.
5. Available Codex model configurations and native subagent support.
6. Available CUDA/GPU hardware, dependencies and development tools.
7. The official `okc` CLI and authentication status.
8. Available official evaluation/submission quotas.

Avoid unnecessary setup. Do not spend most of the sprint configuring infrastructure.

If the workspace is empty, create a clean, dedicated challenge repository.

The official CLI supports commands such as:

```bash
okc problems
okc init muse_bf16_proj
okc check --help
okc submit --help
okc leaderboard muse_bf16_proj
```

Verify command syntax with the installed version before execution.

On Windows, use a suitable existing WSL2/Linux environment where necessary. Do not undertake a major OS installation during this sprint.

If authentication requires human interaction, request that action once and resume independently afterward.

---

## 4. COMPETITION STRATEGY

The current kernel track contains:

1. `muse_bf16_proj`
2. `muse_attention_decode`
3. `muse_moe_w4a16`
4. `muse_decode_layer`
5. `muse_prefill_layer`

These are evaluated on NVIDIA H200 hardware.

The eventual objective is a strong overall leaderboard position, ideally top 50 or higher, to support entry into the serving track.

**For this one-hour sprint, prioritize actual measurable progress on one kernel rather than superficially modifying all five.**

Quickly inspect the specifications and choose the kernel offering the strongest combination of:

- Achievable speedup.
- Implementation complexity.
- Expected scoring impact.
- Time to a correct working result.
- Opportunities for further optimization.

Tentatively start with `muse_bf16_proj` unless technical evidence favors another.

Make the choice autonomously and begin implementation immediately.

Do not get stuck endlessly comparing problems.

---

## 5. ENGINEERING APPROACH

For the selected kernel, inspect:

- `TASK.md`
- `AGENTS.md`
- `reference.py`
- `baseline.py`
- `submission.py`
- Relevant tests and numerical tolerances.

Understand the exact computation and the baseline's bottlenecks.

Look for high-impact optimizations in:

### GPU computation
- Tensor Core utilization.
- Hopper-specific capabilities.
- Triton and native CUDA.
- CUTLASS, CuTe DSL and available libraries.
- Warp scheduling and specialization.
- Register allocation and occupancy.
- Tile dimensions and thread-block geometry.
- Kernel fusion and launch overhead.

### Memory performance
- Global memory transactions.
- Shared-memory use.
- Coalesced loads and stores.
- Data-layout transformations.
- Redundant memory traffic.
- Caching and register reuse.
- Permitted lossless weight repacking.

### Algorithmic improvements
- Mathematically equivalent transformations.
- Reduced redundant operations.
- Improved parallelism.
- Specialized dispatch based on public shapes.
- Better handling of individual batch sizes.
- Reducing synchronization and intermediate tensors.

Use roofline analysis or profiling when appropriate.

Do not assume an optimization is faster just because it is theoretically sophisticated.

Measure everything.

---

## 6. NATIVE CODEX SUBAGENTS

Use Codex's built-in subagent functionality when available.

Do not create an unnecessary agent framework.

Suggested independent roles:

**Agent A — GPU Optimization Research**

Investigate the selected operation, Hopper hardware, existing optimized implementations and potentially superior algorithms.

Return concrete mathematical or implementation recommendations.

**Agent B — Implementation Experiment**

Develop a separate candidate optimization in an isolated worktree or separate file.

Test and benchmark it wherever possible.

**Agent C — Correctness and Performance Review**

Independently inspect numerical correctness, competition-rule compliance and potential performance bottlenecks.

Challenge the main implementation and propose improvements.

**Main Agent — Integration and Optimization**

Own the objective, manage the experiment queue, integrate successful results and maintain the best correct submission.

You may spawn additional specialists when genuinely useful.

Use available models and reasoning levels intelligently. Subagents do not need to use Astra if Sol or another available Codex model is a better fit.

Avoid simultaneous edits to shared files.

Parallelize independent work without creating recursive agent explosions.

---

## 7. TRUE CONTINUOUS GOAL LOOP

This is the most important section.

**A normal Codex session finishing is not sufficient to end Goal Mode.**

Inspect my existing global autonomous loop setup first.

If a working loop controller already exists, reuse it without changing its global project-agnostic behavior.

If none exists, establish a minimal, temporary, Codex-only local loop controller using the installed supported Codex interfaces.

The controller must:

1. Maintain a persistent objective.
2. Keep a durable record of progress.
3. Launch a Codex work iteration.
4. Wait for that iteration to finish.
5. Inspect exit status and updated state.
6. Select the next useful experiment.
7. Launch another iteration automatically.
8. Recover from ordinary failures.
9. Repeat until the hard deadline.

Use supported `codex exec` and continuation functionality where appropriate, verifying flags against the installed CLI.

Prefer fresh-context iterations reading persistent state over allowing one chat to accumulate unlimited context.

Do not recursively create controllers.

Only one controller owns the deadline and can launch primary iterations.

An iteration must not declare final completion merely because tests pass or a candidate becomes faster.

### Iteration procedure

Each iteration must:

1. Read the current goal and state.
2. Determine the highest-value unresolved task.
3. Generate a concrete technical hypothesis.
4. Implement and test the change.
5. Benchmark appropriately.
6. Compare with the best known correct version.
7. Accept, reject or revise the candidate.
8. Record the results.
9. Update the next-action queue.
10. Continue.

If an experiment fails, investigate or choose another candidate.

If a benchmark shows no improvement, analyze its bottleneck instead of randomly changing parameters.

If no obvious optimizations remain, use additional research and independent reviews to find new directions.

If a session crashes, recover from persistent files and continue when possible.

If persistent looping is technically unavailable, explicitly report that limitation rather than pretending an autonomous loop has started.

---

## 8. EXPERIMENT MANAGEMENT

Maintain lightweight project-local state, such as:

```text
GOAL.md
STATE.json
EXPERIMENTS.md
RESULTS.md
best/
```

Record:

- Current objective and selected kernel.
- Active experiment.
- Baseline performance.
- Best correct candidate and commit/hash.
- Measured speedup.
- Per-shape performance where available.
- Failed experiments and explanations.
- Unresolved issues.
- Next experiment.
- Remaining official check/submission quota.
- Deadline and completion status.

Keep the best-known correct implementation recoverable.

Do not overwrite it with an unverified experiment.

Use Git branches or worktrees for independent implementations.

Do not commit generated caches, credentials or large temporary artifacts.

---

## 9. OFFICIAL VALIDATION AND SUBMISSION

Follow the current official rules.

The documented kernel limits are 20 checks and 10 submissions per problem per day, subject to change. Verify live quotas using `okc problems`.

Use local diagnostics whenever practical, but recognize that local GPU measurements may not reflect H200 performance.

Use official checks strategically:

- Establish correctness.
- Validate meaningful candidate improvements.
- Compare competing optimizations.
- Catch hardware-specific issues.
- Verify the best final candidate.

Do not consume limited checks repeatedly for unchanged code.

Official H200 performance is authoritative.

If a candidate clearly improves performance and passes validation, submit it through the official CLI when appropriate.

Continue optimizing after a successful submission if time remains.

### Strict competition integrity

Comply with the current official rules, including:

- Required numerical precision and tolerances.
- CUDA graph compatibility.
- Required full attention computation.
- Weight preparation restrictions.
- No harness modification.
- No cached or hardcoded outputs.
- No exploiting evaluation implementation details.
- No prohibited computation outside timed regions.

A faster invalid result is not an improvement.

The goal is legitimate performance engineering that could be upstreamed.

---

## 10. MAXIMIZE PRODUCTIVE WORK BEFORE RESET

This sprint is intentionally aggressive.

Prefer substantive parallel research and high-effort optimization over minimal-token operation.

Keep the main agent actively engaged in implementation while independent agents work on useful alternatives.

Prioritize:

1. A correct baseline and functioning evaluation pipeline.
2. A genuine first performance improvement.
3. Additional promising optimizations.
4. Independent correctness validation.
5. The strongest valid final submission.

If official evaluations are queued, continue independent work rather than waiting idly when possible.

Do not allow excess coordination, repeated planning or unnecessary formatting to consume the sprint.

The goal is maximum **useful** Codex compute and engineering output, not maximum conversation length.

---

## 11. STATUS AND REPORTING

Keep reporting concise.

Periodically display:

```text
OPEN KERNEL CHALLENGE — GOAL MODE

Time remaining:
Current kernel:
Best correct speedup:
Latest experiment:
Official evaluations remaining:
Active agents:
Next action:
```

Avoid long explanations between iterations.

When useful, show mathematical derivations, benchmark evidence and implementation reasoning.

Do not ask me to approve ordinary technical decisions already covered by this objective.

Only interrupt for necessary authentication, unavoidable permission barriers, or spending decisions. Do not purchase anything without my approval.

---

## 12. HARD STOP AND FINAL OUTPUT

At the deadline:

1. Stop the loop.
2. Preserve the best correct implementation.
3. Save benchmarks and experiments.
4. Save pending ideas and next actions.
5. Stop all workers.
6. Provide a final concise engineering report.

Include:
- Kernel selected and why.
- What was implemented.
- Best verified speedup, or clearly state if none was verified.
- Official check/submission results.
- Major successful and failed experiments.
- Remaining issues.
- What to try next.
- Location of the best submission file.

Leave the repository in a clean, recoverable state.

Do not permanently alter global Codex orchestration settings for this temporary sprint.

---

## 13. EXECUTION ORDER — START NOW

Your first actions must be:

1. Check the clock and confirm the 1 PM hard deadline.
2. Inspect the current workspace and Codex environment.
3. Check for an existing global loop controller.
4. Read the official challenge instructions.
5. Initialize and select a kernel.
6. Start useful research and implementation, with native subagents where beneficial.
7. Activate or reuse the actual persistent execution loop.
8. Continue autonomous optimization until the deadline.

**DO NOT JUST WRITE A PLAN.**

**DO NOT STOP AFTER A FIRST SUCCESS.**

**DO NOT RETURN A FINAL ANSWER WHILE USEFUL WORK CAN STILL CONTINUE WITHIN THE SPRINT.**

Your mission is to produce the strongest valid kernel improvement possible before 1:00 PM today.

BEGIN EXECUTION.
