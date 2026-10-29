# Original-FP32 finalizer cache: one fixed candidate

**Terminal, October29 10:08UTC:** accepted94ce080b1fb148a3 compiles and passes
the runtime resource/synthetic correctness gates. Affected32/128/256 paired
improvements are only0.27%/0.03%/0.12%, below frozen gates; selected_main=false.
The official full-suite2.329737405x point runs retained paths. Reverted419962d,
archived20261029-100754; no score/Decode transfer or noise retry. Full evidence:
results/moe-fp32-final-cache-check-20261029/. The initial TLS upload created
no job, proven by unchanged official job IDs/quota before the accepted retry.
All preparation/readiness statements below are the prelaunch record.

Baseline33114871/nativeCUDA1e60cf46 remains verified. Candidatef200e6e0,
335794bytes, embeds nativeCUDA403bd123, deviceprefixb0e49b0f. All retained
Python/CUDA remains an exact prefix; six reversible native edits recover every
parent CUDA byte. No GPU result or speedup exists for this candidate yet.

The retained WGS finalizer reads each valid FP32 expert-output row for its
norm statistics and again for route weighting. The candidate keeps those exact
values in two float4[8] arrays per consumer thread, removing the second read.
It changes no expert-output representation, phase-B store, gain multiplication,
norm arithmetic/reduction order, epsilon, route order or BF16 materialization.
It adds no scratch layout or persistent allocation. This differs from the
rejected factor-y trial: that trial stored pre-gain BF16 bits and reconstructed
FP32 products; this candidate stores the original FP32 product and reuses it.

The global completion join remains exact. All phase-B output producers have
published before either finalizer pass; sequential-state use prevents the next
invocation from overwriting these rows mid-call. Values are loaded only for valid
route slots, and both later uses are under the same validity branch. Uninitialized
nonlocal entries are never loaded or used. Empty job/range guards avoid the known
uninitialized-row read in the original empty WGS range; nonempty scheduling and
all counters/generation/fixup/arrival algebra remain exact.

Holding64FP32 values per thread risks register pressure. Both BIG variants now
compile in the CPU NVRTC13.0 Windows proxy at the same96base registers,16offline
stack bytes and56/124static spill-store/load fields as the exact retained proxy.
The two compiled variants remove32 LDG.E.128 instructions in total,16each.
LDL/STL, HGMMA and warpgroup arrive/wait static counts match, with no C751x
serialization warning. This is instruction evidence, not runtime attributes,
instruction-sequence equivalence, executed bytes or measured speed.

Logical reread reduction is16384bytes per local route. Those original reads may
hit L2, so this is not an HBM saving or a latency bound. There is no admissible
exposed finalizer-time fraction. The increased register lifetime could still lose
on H200 despite unchanged offline resource fields. A single bounded comparison
will decide whether the source-level reuse earns complete-kernel time.

The CPU model passes72 finite FP32 fixtures with identical arithmetic bits,
including zero-to-eight valid routes and NaN-poisoned invalid rows. It rejects
invalid-row reads, premature BF16 caching and wrong-route reuse. This does not
establish GPU numerical equivalence. The separate fixed-mask codec screen checks
4495 shift triples and rules out a four-rotation-only exact bit cover for mask
0x81c081c0; it rules out no other encoding or decoder.

Frozen preflight: results/moe-fp32-final-cache-preflight-20261029/. It includes
exact source, inverse ledger, both compiler inputs/logs/full SASS/resource reports,
CPU/source/host audits, evaluator identity and12 malformed-result rejection
fixtures. Namespace/export/query bindings agree;1608 dispatch cases retain all
M<=16 and ineligible paths. Only PRMT-eligible M>16 can select this extension,
after complete resource, correctness/replay and performance admission. Official
affected indices are2/4/5 (M32/128/256), controls0/1/3 (M1/16/8).

Prepare shares original packed weights and adds a separate157MiB scratch bank
and a temporary128MiB cold-cache buffer, not another expert layout. Count all
original input/packing/compilation/captured-output allocations in peak memory.
Original preparation>185s skips the extra build; compile>270s skips probes;
no new case starts after280s. All reachable extensions/attributes must be ready
before graph capture. No timed host input reads, cached inputs or lazy builds.

Both BIG runtime resource queries must pass1..112registers, zero actual local/
static bytes,231168dynamic bytes,576threads, at least one resident block and
grid1..160. Failure skips every new launch and preserves the incumbent. The
query uses the exact okc_f32cache namespace in all nine references; the host
declaration, exported method and prepare call are audited together.

One future paced check runs17 normal nearby/chunked cases plus skewed128 in ten
cold graph-replay AB/BA pairs, with milliseconds converted to nanoseconds by1e6.
Fresh input/routes are replayed twice at32; four all-nonlocal zero cases and a
duplicate-route case are separate analytic probes. All local routes and required
BF16/FP32 boundaries remain. Runtime selection requires0.5% global/2% worst-row
incumbent comparison error, complete probes, no more than2% paired nearby loss,
and more than3% paired gain on each32/128/256 case. Repeats belong to one job.

Retention/publication also requires full5tests/6benchmarks/integrity against
the official reference/checker, exact evaluator identity, real new namespace
in all three affected profiles and no controls, affected geometric gain>3%,
control-normalized affected gain>3%, no affected normalized ratio below0.99,
control drift within0.95..1.05 and track speedup>2.4009189773297592x. These
gates are materiality filters, not a statistical-significance claim. A failed
or inconclusive result retains33114871; no score, Decode transfer or noise retry.

The earliest next check is09:53:13.031328UTC/05:53Toronto. Existing one-job,
60-minute pacing, four-check reserve and20:00Toronto window apply. This packet
does not itself launch a job. Preserve any later upload intent/handle across
observation timeouts. Broad top50 goal remains active regardless of this trial.
