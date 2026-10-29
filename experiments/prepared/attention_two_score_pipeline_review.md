# L8 full Attention: two score banks before a GPU experiment

Status on October 29, 2026: a CUDA prototype and offline compiler/source audits
are complete. No GPU job, score, incumbent change, or measured speedup belongs
to this proposal yet. The first budget spilled and was rejected offline. One
diagnosed repair compiled without stack/spill traffic or C751 warnings.

The frozen candidate is `c51f843e34f0780ed77483bcb864574e2dc3d69f0b0deb987eb2b75ffe1f3d61`
under `results/attention-two-score-pipeline-preflight-20261029/admission-v3/`.
Its native source is `375726fddea48bc8641167f3c9b4a458b585125ddb4d3987b4e766e88a036816`.
It uses the one-CTA grid only for L8 full, B2..64, flat/nc1, retained cooperative
mode, without PDL/debug/clusters, and only after exact native resource admission.
All other Python behavior and original CUDA functions remain intact.

Parent Attention source:
`9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f`.
The prior external-finalizer experiment is closed and rejected; this proposal
preserves the parent's global cooperative finalizer.

## Hypothesis

For L8 full-context, multi-request flat/nc1 work, two distinct score register
banks could allow the next QK group to remain pending during scalar softmax
work on the current tile. Both QK and PV already use FP8 in the parent.
The proposed change is dependency scheduling, with the existing tile order,
arithmetic, producer stages, and numerical boundaries preserved.

The cost is 32 additional score registers per consumer and a proposed one-CTA
per-SM schedule. Reduced parallelism and changed split boundaries can outweigh
any overlap. No critical-path saving has been measured. The hypothesis must
earn a complete-suite affected/control improvement in one paced check.

## CPU screening evidence

`model_attention_two_score_pipeline.py` produced
`results/attention-two-score-pipeline-model-20261029/model.json`:

- 445 modeled schedules, including zero work, short tails, patch transitions,
  four-stage reuse, two consumer warpgroups, and an 8192-tile arithmetic stress.
- At most two committed WGMMA groups in the modeled FIFO.
- Seven deliberately unsafe scheduling/resource variants rejected.
- Explicit score-bank, raw-stage, transposed-V, P-register, accumulator, and
  tail-drain lifetime checks.

These checks establish properties of the model. They do not prove CUDA
correctness, compiler scheduling, actual residency, or performance.

The first 152-register consumer budget produced 88 bytes of stack and spill
traffic (126 store/344 load bytes in the compiler log). It is closed.
The one bounded repair uses 384 threads, 32 producer and 232 consumer registers.
Offline compilation reports 168 initial registers: a 64,512-register CTA pool
covering the 63,488-register redistributed need. Native admission requires those
same 168 registers, zero local/static memory, 114,688 dynamic bytes and exactly
one resident block per SM. The grid must fit every KV pair simultaneously.

The repaired compile has distinct score bases R60 and R92, with zero spills,
and six sampled control bodies match the retained assembly including branch
targets and appended helper bodies. Separate CPU models cover 360 unequal-key
partitions and 256 changing-grid epochs. All five freshly downloaded official
task/reference/checker/baseline/instruction files match the pinned copies.
The result guard accepts a synthetic positive and rejects 25 defective fixtures;
these fixtures are not measurements.

## Next gate

One paced H200 check must establish native resources, real new-path execution,
all 16 tests/12 benchmarks/integrity, and the fixed performance gates. Both
changed cases must improve at least 1% after control normalization, affected
geomean and normalized affected geomean at least 2%, full weighted track ratio
at least 1.005, and control geomean within 0.95..1.05. A private check is not a
publication. Reject inconclusive evidence; no noise retry, post-hoc batch
selector, or Decode transfer of a failed design.

PTX documents the CTA-local register pool and the completion guarantees of
WGMMA waits in the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/).
Those semantics motivate the model; they provide no performance forecast.
