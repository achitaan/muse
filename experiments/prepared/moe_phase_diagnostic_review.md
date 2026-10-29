# MoE phase diagnostic: one paced private check

Source `db0f6481654d41790e09b309a9f28fdf95e79d86a4f1b0a00183b18be26cba0b`,
190,539 bytes. Frozen evidence is in
`results/moe-phase-diagnostic-db0f-preflight-20261029/`.
The earlier unlaunched d55d packet remains preserved separately; this source
adds software/device version metadata to that diagnostic.

This is an information-gathering check. It is not an optimization candidate,
and neither a higher aggregate point nor successful probes justify scoring it.
The exact retained33114871 source is its prefix. The timed `kernel_fn`, its
globals and its original CUDA remain unchanged. Only `prepare` is wrapped.

## Question and design

Determine whether exposed time lies in routing/gather, phase A, phase B and its
readiness waits, the post-B grid join, or finalization. Summed kernel profiles
cannot answer these questions inside the existing persistent kernel.

Preparation generates its own synthetic M32/128/256 inputs from fixed seeds
20261029+M and a public-formula-inspired popularity distribution. These are
not official timed inputs or observed challenge routing. Instrumentation
results cannot establish a critical-path fraction for the official loads.

One separate MAIN extension enables the retained DBG timestamp macro and adds
a host resource query. The diagnostic changes no numerical instruction or
ownership rule. It uses separate scratch; the incumbent's weights are shared.
No diagnostic tensor or data-derived decision is returned for the timed path.

Each shape compares ten pairs of original/instrumented graph replays on the
same inputs. Five AB and five BA pairs are shuffled with a recorded seed.
Every arm starts after a134,217,728-byte L2 flush. CUDA event milliseconds are
converted to nanoseconds with1e6. These are within-job repeated measurements;
the three shapes and ten pairs are not independent H200 jobs or a confidence
interval. Instrumentation is interpretable only when median calibration is
within5% in either direction and registers/local memory/residency match the
historical incumbent. Any resource mismatch is retained and flagged.

The retained `%globaltimer` records eight points for every CTA. NVIDIA defines
it as a nanosecond timer but notes target-specific behavior; interpret it only
on the measured H200 target. Source:
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer

Per-CTA timestamp order is entry, route, gather, range-join, A-end, B-end,
global-join exit and finalization-end. A phase-B span includes readiness waits.
The B-end-to-join span includes surrounding completion code, so it is not a
pure spin-wait counter. The final tail is measured from the latest B-end to
latest finalization-end; do not sum overlapping spans into an Amdahl fraction.
Each clock is thread 0's observation at the existing marker. In this WGS
finalizer, a CTA barrier follows every token's output stores before DBG(5).
The trace still omits other helper threads' final returns and kernel-launch
retirement; calibrating CUDA event duration remains necessary.

Route counts and logical split-unit counts describe only the generated
synthetic workloads. Logical requests are not measured HBM traffic. The log
retains raw paired samples and losslessly encoded timestamps. If needed to
fit the13000-character diagnostic limit, it retains the union of the latest16
A/B/final CTAs; full-grid summaries stay labelled as derived summaries.

## Gates and limitations

- Preserve BF16 outputs; instrumentation must stay within0.5% global and2%
  worst-row error relative to the checked incumbent on the synthetic probes.
  This is an instrumentation-equivalence gate, not a new mathematical oracle.
- The fresh official task, reference, checker, baseline, NVFP4 helper and
  AGENTS files match the pinned copies exactly. Require the official full
  five tests, six benchmarks and integrity checks as well.
- CPU fixtures recover the trace exactly, validate analytical phase intervals,
  test local/all-nonlocal route metadata, and reject malformed timing evidence.
  They do not prove native compilation, H200 legality or timing quality.
- Original preparation remains intact. Skip diagnostics if the preparation
  budget is insufficient; an incomplete trace set is not a successful study.
  No retry is allowed just to obtain a more attractive aggregate point.
- Observe EVALUATION_POLICY.json. The next slot is at or after02:21:46UTC,
  October29, one hour after the Dense check. One active GPU job, no score.

CPU reproduction:

```text
python experiments/prepared/build_moe_phase_diagnostic.py
python experiments/prepared/audit_moe_phase_diagnostic_cpu.py
python experiments/prepared/test_moe_phase_diagnostic_guard.py
python scripts/check_moe_phase_diagnostic.py
```

The last command reports readiness or resumes an existing handle. Only
`--launch` permits a new check. `--wait-for-slot` waits locally without API
polling. A persisted launch intent blocks re-upload after an unknown outcome;
reconcile the official job before retrying. Observation failures retain the
same handle. No verified or active source is replaced by this diagnostic.
