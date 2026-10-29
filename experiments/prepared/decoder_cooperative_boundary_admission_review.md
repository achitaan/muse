# One bounded cooperative Decode check: frozen admission plan

Candidate `ad7b55cfdf01408842510fca48eef235b95e73fcd9d721b73ce418d9f561ea92`
is917645bytes, at `results/decoder-cooperative-boundary-preflight-20261029/v3/`.
Native CUDA is51180052ffd10dfb11ab1bb747c11192922c7effdd2669b4389185184028e4b1.
Baseline D9 remains verified/active until the one-iteration controller installs
the trial. Broad top50 goal remains active; no claim of rank or speedup.

This integrates the cooperative O1/O8 boundary described in
decoder_cooperative_boundary_review.md. It reuses the retained plan's original
lossless packed weights or BF16 fallback; all other original module nodes are
unchanged from the source-preserving307dd791package. The bounded original
kernel_fn edit and appended code have an exact inverse restoring that package.
That package separately binds back to D9 via lossless literal packaging and
the documented unreachable router factories/unread RoPE table preparation.
The new module adds no timed host read or lazy compile. Unsupported token
bands, layouts, devices and backend/norm configurations use original paths.

## Actual admission work in prepare

One extra compact native module compiles four instances, before graph capture.
For each family and weight format, query cooperative capability, real SM count,
actual function local/static/register resources, occupancy at the exact dynamic
shared size, and grid capacity. Require zero local/static memory, O1>=1 and
O8>=4blocks/SM, and all grid blocks fitting the cooperative capacity. No host
resource query runs inside the timed call. Native launch uses the cooperative
attribute; an ordinary unsupported spin-wait grid is never attempted.

Correctness probes compare against the actual retained GEMV plus selected JIT
CUDA boundary: BF16 projection result must be bitwise identical; final BF16
FFN output global/worst-row relative L2<=1e-3; FP32 residual/router<=2e-5.
Each format tests random, zero, outlier and tiny inputs; O8 additionally tests
M2..7; four changing-input graph replays per format detect stale buffers.
The total is80probe cases. Raw errors are losslessly compressed in stdout
markers so the immutable result can retain them within log-tail limits.

The two original-format component comparisons use16paired samples with arm
order alternating, each amortizing16calls through a captured graph. CUDA event
milliseconds are converted to nanoseconds with1e6. These repeated components
are warm-cache diagnostics, not representative cold-L2 whole-layer timings.
For each family require geometric old/new>=1.08 and>=12winning pairs. This
is a conservative filter, not a significance test. Do not publish from it.
Both families must be selected to pass the fixed experiment's later guard.

## Full-suite retention gate fixed before launch

- Exact source, H200, full13tests/12benchmarks/integrity and matching cases.
- All four format/family resource markers and80numerical/replay probes pass.
- Distinct `cb_smallm2_gemv` in profiles0/1 and `cb_v18s_gemv` in6/7; absent
  from the eight unchanged cases. Empty/partial selection is not this win.
- Reconstruct all six load costs using46sliding+16full, not twelve equal rows.
- Affected case speedup ratios GM>=1.005 versus cached D9; controls' four
  weighted-load ratio GM between.95and1.05; complete weighted track ratio
  divided by that control estimate>=1.003. Each affected load divided by the
  same control estimate must be>=.995. The controller additionally requires
  actual aggregate metric improvement. No statistical significance claim.
- Preserve raw paired samples and full evaluator result/profiles. If tied,
  inconclusive, failed, fallback-only, or resource-invalid, restore D9 without
  a score, parameter sweep, or noise rerun. One diagnosed bounded defect may
  be reviewed for a later repair within the pacing/reserve policy.

The current official Decode TASK/reference/checks/baseline/AGENTS/nvfp4 files
match all six pinned copies byte for byte. CPU bridge checks establish matching
20-argument declarations/calls, explicit pybind header, source Python syntax,
and unsupported-layout rejection. They do not compile the actual CUDA host
extension. The retention guard passed its synthetic positive fixture and
rejected15broken fixtures; fixtures are not GPU performance measurements.

Draft v1 lacked an explicit pybind header with no_implicit_headers; it was
rejected locally before any upload. V2 fixes that and labels diagnostic cache
behavior; v3 adds distinct kernel names and compact lossless probe logs. These
are three preparation sources, zero GPU jobs. The earlier CPU bridge checker
miscounted arguments as22; inspection corrected it to20 before admission.

All inherited licenses remain in the integrated source. Peak memory includes
the original harness weights, original packed plans, transient alternate-format
validation plans, probe graphs, module build allocations and new output tensors.
No expanded persistent NVFP4 experts are introduced. The unused512MiB RoPE
table removal provides representation headroom; preparation latency and actual
peak memory remain unmeasured until this complete-source GPU check.

Use one foreground controller, cached exact D9 baseline, candidates/decode_active.py
as sole mutation scope, and the existing deadline/60-minute pacing/four-check
and four-score reserves. A GPU check is off-board; only a material retained
whole-layer result can be considered for a scored publication.
