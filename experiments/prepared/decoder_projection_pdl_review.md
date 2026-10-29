# Decode Attention-to-output-projection overlap: CPU feasibility screen

Baseline: `best/decoder_verified.py`, SHA256
`d9fe7558624a1f070bbacb493b94302e90ff9bf77cdd8d4239f9d7a83ab39988`.
This is a new hypothesis, with no candidate installed and no GPU job used.
The retained sources and selected arithmetic remain unchanged.

`audit_decoder_projection_pdl.py` binds the findings to the exact baseline and
the saved weighted-load packet. Its report is
`results/decoder-projection-pdl-audit-20261029/audit.json`.

## Existing mechanism and proposed change

The native Attention source already calls `pdl_launch_dependents()` at lines
1544 and 1984 of its decoded CUDA literal. The nearby comment claiming there
is no early trigger is stale. Its Python `pdl` flag controls the incoming
dependency from QKV, so enabling that flag alone does not implement this idea.

The proposed downstream consumer is the output projection. O1 uses the V12
smallm2 route, O8 the V18 v18s route. Both retained instances have `OPT & 2`,
so their initial two weight-prefetch units do not load activations. With
`ST=KU=2`, `WK=8` and K1024, these units cover the full K domain. The logical
packed code size is 12 MiB per projection, excluding escape metadata; this is
not a measurement of HBM traffic or exposed prefetch time.

A future implementation would enable a programmatic dependency on the O
launch and wait after independent prefetch, before every activation read.
The existing Attention trigger would stay unchanged. It does not publish
unfinished outputs. Address calculation alone is independent; input loads are
not. Input loaders currently use `ld.global.nc`, whose cache semantics also
need explicit review before adopting cross-kernel overlap.

M16 actually uses `m16trace` with `next_early`; M32 uses the native BF16 olayer
configuration `(32,128,8,1,1,1)`. They must be audited separately. The old
generic M32 table is not evidence for the selected O32 path. M128/M256 remain
controls unless a separate source/resource argument supports their inclusion.

## Opportunity screen

The report reconstructs all six Decode loads with weights 46 sliding and
16 full, using the saved D9 benchmark latencies. Hypothetical *net* savings of
1 us per case on M1/M8 would improve the Decode track by about 0.323%; including
M16/M32 would yield about 0.514%. At 2 us on all four bands, the hypothetical
track gain is 1.034%. These are dilution calculations, not forecasts, measured
overlap, or bounds derived from summed profiler time. The overall aggregation
is still unknown, so none is an overall-board prediction.

The narrow two-band implementation has a modest opportunity. Do not spend a
GPU check merely to toggle PDL. First establish viable coverage and a
source-bound mechanism; then freeze an honest experiment-specific materiality
criterion. Earlier failed MoE gates remain closed and unchanged.

## Requirements before a GPU ticket

- Prove the wait dominates every dependent read in generated code. Preserve
  every arithmetic operation and required BF16 materialization.
- Validate the host launch attribute, actual CUDA graph dependency, replay
  lifetimes, and every serial/fallback path. Do not patch the harness.
- Prove forward progress without concurrent execution. Early consumer blocks
  must never prevent unfinished producer work from progressing.
- Check actual H200 registers, shared memory and local memory. CPU compilation
  would not establish legal residency or numerical correctness.
- Keep packing, scratch capacity, source size and 300-second preparation within
  limits. The lossless packaging draft remains unmeasured preparation work.
- Charge contention, waits and launch changes. Use paired whole-layer graph
  samples with unchanged controls, not isolated kernel point estimates.
- Require the full Decode numerical/routing/cache/residual/output suite and
  real new-path execution before retention or publication.

No GPU ticket is ready from this source screen alone.

## Primary mechanism sources

NVIDIA's [Programmatic Dependent Launch guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
documents downstream independent work, dependency synchronization, optional
overlap and stream-capture support. NVIDIA's
[CUDA 13.0 PTX ISA](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol)
defines the trigger and wait memory behavior. These establish available
mechanisms, not profitability. Consult its `ld.global.nc` section for the
noncoherent read-only cache distinction. Sources reviewed October 29, 2026.
