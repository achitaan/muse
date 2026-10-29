# L8 full Attention: parallel finalization after a normal stream join

Retained parent:9e922c3555484081ca45a99a90284e26eccab181fb677700d8227a4b3402412f,
native CUDA241b67fe0008bbae53435b53d1bb6d647ded0b5718118875c0ec368b718b41db.
No active source changed and no GPU job was used for this study. The top50
objective remains active. This is a new completion mechanism; it does not
repeat a split-count/coop flag, wider-tile or serial-partition comparison.

The CPU draft is4cbe0266,177564B, native7e81b545, under
results/attention-external-finalize-preflight-20261029/v1/. The later resource
admission source is88026272155440d2dce8c565ac2bcbbe28a4fef97c0cf4909a3b5d6a90fdf794,
179965B, nativec02c053a7afcab82fb1d09df7b74361855d6c24c6b664633db3dec675d06eeb7,
under admission-v2/. Neither source is GPU-admitted yet.

## Mechanism and exact scope

The retained flat L8 mainloop publishes fixed-reference FP32 numerator/denominator
partials with global reductions. In cooperative mode it then waits for every
CTA's arrival before distributing norm/gate finalization across the grid.
The host can disable cooperation on a grid that does not fit; source comments
and the Python plan alone do not prove the actual mode. GPU admission must
observe the old instance's resources and enough residency for the fixed grid.

The new MODE16 keeps every mainloop arithmetic, key partition, causal mask,
FP8 probability/cache operation, BF16 boundary and payload reduction. It
removes the completion-counter/finalizer block and the final grid join from
that instance. Critically, the full end-of-segment producer/consumer CTA
barrier remains: producer cp.async draining and shared-memory reuse are not
short-circuited. The unused epoch copy is removed only in MODE16.

A second ordinary launch on the same stream then finalizes one token/KV-pair
per128-thread CTA. Its math is copied exactly from fin_nc1, specialized as
L1 only for indexing: token=b*8+j indexes QKV/output, while
workspace_row=(b*pairs+pair)*8+j indexes the retained[B,pairs,8] workspace.
Each CTA handles sixteen complete64-element heads. One added full-warp sync
orders all eight lanes' denominator reads before part0 clears that location.
There is no new intermediate BF16 round between norm and sigmoid multiplication.

Normal stream ordering provides the join between kernels; warp synchronization
orders the local denominator read/clear. These mechanisms are documented by
[NVIDIA's stream-ordering guide](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)
and [CUDA synchronization semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).
The draft excludes PDL calls; actual graph replay/race and accuracy validation
on H200 remains required.

Host dispatch changes only L8,full,flat,nc1,B>1,no PDL/debug, with the retained
host's cooperation still enabled and new-main occupancy>=2. Other copies,
single-request/sliding/L1 paths and all Python fallbacks remain original.
The native large-L8 fallback at B>64 is unchanged. The CPU indexing study's
larger shapes do not enable those fallback shapes.

## Completed CPU evidence

The inverse edit ledger restores the entire retained CUDA exactly. The draft
Python AST is unchanged. The resource admission revision adds only a native
query/export and _build checks; its numerical device prefix is exact.

Offline NVRTC13.0.88/sm90a/C++20 results:

| Instance | Registers | Stack/spill fields | Dynamic shared |
|---|---:|---:|---:|
| Retained L8 flat |80|0/0/0|114688B|
| Deferred L8 flat MODE16 |80|0/0/0|114688B|
| New128-thread finalizer |31|0/0/0|0B|

No C7514/5 serialization warning appeared. The new main has zero integer
global-arrival atomic instructions versus three in the old generic body;
FP32 payload reductions remain. It has4304instructions including appended
helpers versus6672in the old generic entry. These are static counts, not
executed instruction counts or a latency forecast.

All six instantiated control bodies match in instruction operands, branch/call
targets and appended helper code: L1-flat0,L8-flat0,L1-nonflat9/13 and
L8-nonflat3/15. The initial filename-only comparison reported four differences
because new entries renumbered division helpers and namespace encodings.
The final comparison binds every section-defined symbol to its instruction
offset, preserving targets and complete helper bodies; external symbols and
constants are not normalized. No CUDA source change was made to hide a diff.
This covers these six instances, not every possible native specialization.

The CPU model covers44ownership shapes up to524288head rows, the1024values
owned by one finalizer CTA,35old/external sequential epoch cases and45numerical
fixtures. Maximum relative L2 against a higher-precision row oracle is1.747e-5.
It detects a pair-sensitive workspace permutation and dropped BF16 attention,
premature BF16 norm-before-gate and wrong8192-element norm-domain mutations.
These are CPU mathematical screens, not official tolerance or GPU-race proof.

## Cost model and admission still needed

The draft adds one timed kernel launch and no workspace. It retains the same
FP32 partial payload and finalizer payload:260B read and260B zeroed per head
row, plus128B BF16 gate input and128B output. These are representation bytes;
scalar broadcasts, instruction requests/cache lines and measured HBM traffic
are not inferred. Atomic ordering may change after completion code is removed.
The finalizer clears every payload row before sequential scratch reuse; old
epoch counters remain untouched and valid for a later original cooperative call.
Concurrent same-workspace calls remain outside the retained contract.

The affected benchmark cases are9/11, L8 full B16/B32. Cached source-bound
latencies are101.35/185.20us. Hypothetical *net* savings of1/2/4us after the
extra launch imply weighted track gains1.001958/1.003932/1.007931. They are
opportunity screens, not measured exposed join costs or promised gains.

The admission-v2 build queries actual old/new registers,local/static memory,
occupancy at114688B, finalizer resources and SM count before timing. It requires
old/new<=80regs,zero local/static memory,>=2blocks/SM; finalizer<=64regs,zero
local/static memory and>=1block/SM; >=132SMs,64head dimension,384main/128final
threads andMODE16. All five current official Attention files match the pinned
TASK/reference/checker/baseline/AGENTS byte for byte.

Before launch, finish independent source/precision inverse and result guards
with broken fixtures, cached9e baseline binding, actual profile markers and
full16tests/12benchmarks/integrity. Require MODE16 plus the new finalizer only
in9/11; do not treat an old-path aggregate point as this algorithm's win.
Precommitted retention thresholds in admission-v2 are affected and normalized
affected GM>=1.02, each normalized affected case>=1.01, unchanged controls
within[.95,1.05], and weighted track ratio>=1.005. These engineering gates do
not establish statistical significance or automatically authorize publication.

Reject failures/ties without a noise rerun or neighbor sweep. No check before
13:50:40.319601UTC on October29; preserve four checks/scores per track and
the20Toronto deadline. A future justified Decode transfer must port only this
control/finalization mechanism onto D9's own FP16-PV source. It must not import
the tied standalone FP8-PV mainloop or associate its score with the new source.
