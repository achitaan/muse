# Decode VMS32

Check file: `kb/exp/decode_vms32.py` (standalone).

Sources pinned in `build_decode_vms32.py`:
- Decode incumbent: `f50a51d7`, `kb/v1/decode_v1.py`.
- Control-only VMS engine: `8736a4b1`, `candidates/moe_vms32.py`.
- Existing routing adapter/diagnostic helpers: `bf9357dd`.

Only d32S and d32F can select VMS32, independently. The incoming MoE
arguments are already BF16 `[32,4096]` activations, int32 local ids
`[32,8]` (-1 remote), and FP32 normalized route weights. The VMS adapter
uses range `[0,32)` without modifying ids/weights. Stage A is normally
stream ordered after routing; the engine's B/finalizer PDL chain is unchanged.
All other production paths retain the incumbent.

Per shape, three fresh-input passes each measure 16 rounds. Each round
includes experimental ABBA + BAAB and distinct incumbent/incumbent
ABBA + BAAB controls, with alternating control placement. Every replay
restores the same metadata, residual stream and new-KV slots, then READ512
flushes outside the CUDA-event interval. Admission requires three complete
passes; both median paired and median-time-ratio nulls in `[.995,1.005]`
each pass; every experimental paired ratio `<1`; and their three-pass
median `<=.995`.

Numerics do not demand MoE bit identity. First and last replays require
identical expert ids and byte-identical pre-MoE activations/local ids/route
weights versus the incumbent, plus identical new cache bytes. Actual-input
MoE uses an independent NVFP4-dequant/FP32 reference (global .015, row .075).
Final hidden and residual-update outputs use full independent FP32 layer
equations, the same expert ids, and the judge's BF16-vs-E4M3-query-derived
3–15% tolerance with 5x row factor. Each output's global error must be no
worse than 1.10x the incumbent error against that same reference. Checks
run before/after timing and veto sticky device errors. Diagnostic streams
use the public fixed CPU-generated stream basis (seed 5354), not a newly
randomized mean that would break router alignment.

The incumbent's optional MFAB ablation is disabled only during prepare;
production arithmetic/dispatch is unaffected. Every inline build has
explicit host/NVCC C++20 and SM90a gencode. The builder removes only Python
comments/docstrings/blank lines, with an AST-equality assertion proving no
embedded source or other constant changes; original copyright notices remain.

Validation: 12 CPU/structural tests pass (PyTorch 2.14.1 CPU), including
analytical zero/nonzero full-layer solutions, rank-one routed MoE,
NVFP4 code values, lossless repack, signature consistency, adversarial
admission boundaries, exact timing order/count, per-shape static dispatch,
all inline-build flags, size, and forbidden-runtime scan. AST parse passes.
GPU compilation/correctness/timing and prepare <285 seconds are UNVERIFIED
locally (no GPU/nvcc); the external check is required.

Ledger claim/result (caller should publish to the shared main ledger; this
subagent does not write/push main): reopened T32 only, unlike the closed
T16 decode VMS experiment, based on clean-null 32-draw MoE evidence
`e115610794554d17` (bs32 geomean .987). Honest reference-error gate replaces
the previous bit-identity veto. Offline prepared; no GPU check spent here.
