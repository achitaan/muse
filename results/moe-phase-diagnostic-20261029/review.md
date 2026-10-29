# MoE phase diagnostic complete; timing admission failed

Checke66873fa2adc4030/source db0f6481 passed all5tests,6benchmarks and integrity.
Preparation took219.095s. All three synthetic probes measured zero output
error relative to the incumbent. Median instrumented/original timing ratios
were1.01622,1.00961and1.00818for M32,128and256, inside the predefined5% band.

The instrumented kernels reported96registers,8local bytes,231168dynamic shared
bytes and one resident CTA per SM, for both BIG variants. The historical
incumbent reference reports zero local bytes. The predefined resource gate
therefore failed, and every `timing_interpretation_allowed` flag is false.
The study completed; its quantitative timing admission did not pass.

The current original module was not queried for its resources in this job.
Do not attribute the local-memory difference specifically to instrumentation
or call it measured spill traffic. The runtime metadata was PyTorch2.14.1,
CUDA13.0on NVIDIA H200. No threshold is relaxed after seeing these results.

Raw clocks and derived summaries remain preserved for this synthetic,
instrumented source. They cannot establish incumbent phase fractions, exact
whole-layer critical-path time, judged routing or HBM traffic. Only the last
instrumented replay supplies each shape's raw trace; ten paired replays are
within-job calibration samples, not independent GPU jobs. A zero count of
CTAs completing B before the latest A completion does not prove B never
started earlier.

The official2.336029160x point timed the unchanged incumbent path. It is not
a new-algorithm speedup, and no score or promotion follows. Preserve MoE33114871
and every other verified/active source. Do not retry solely for cleaner traces.
The one-stage ownership rule remains parked after its mixed CPU cost screen.

One MoE check, zero scored submissions. Expected postlaunch quota from the
fresh prelaunch snapshot minus this job: Dense19,MoE17,Decode20,Attention20,
Prefill20checks;10scores each. No later quota query is implied. Next new check
cannot start before03:22:01UTC on October29 and needs a concrete hypothesis.
The authorized GPU window still ends at20:00Toronto October29.

`review.json` binds the exact source, job and raw result hash. `result.json`,
`diagnostic-verdict.json`, launch records and prelaunch quota are immutable.
No GPU job or foreground driver remains active after this terminal result.
