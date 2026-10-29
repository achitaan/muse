# Dense resource binding draft: not admitted for GPU

Prepared source7de2abc3 extends unlaunched13224d30 with native resource queries.
Its original CUDA is an exact prefix; both fixed timed kernels, dispatch,
FP64 numerical probes and nearby paired selection AST remain unchanged.
The complete inverse edit ledger restores13224d30. Attribution and license
remain in the single-file submission. No Dense candidate was installed.

The two existing device bodies compile offline for sm90a: GQX90registers,
GQ120registers, both zero stack/spills. The first proxy failed because NVRTC
does not provide host stdlib.h to cuda.h. The second uses the actual CUDA13
opaque tensor-map layout,16uint64 aligned128bytes, and omits host-only
functions/headers. Both proxy sources/logs are retained. This establishes
device compilation for that proxy, not host ABI, H200 launch legality,
occupancy, correctness or speed.

The proposed untimed host query records registers, local/static shared bytes,
SM count, active cluster capacity, dynamic shared memory, block/cluster/grid
dimensions, configured dynamic limit, active blocks per SM and cluster
support. QKV keeps cluster3/230544bytes/grid108; pooler keeps
cluster2/215184bytes/grid128. Both keep160threads. The guard requires a legal
single wave at native capacity, zero local/static bytes and<=128registers.

Before considering one paced check: bind the new exact source to guards and
driver; fault-test the native resource evidence; preserve the old CPU/math,
timing-unit and nearby evidence through explicit source equivalence; take
current official task identities. Keep original accuracy/performance gates.
The old Dense driver lacks the newer local pacing wait/live-job reconciliation
pattern; update it prospectively before use. Do not run it blindly.

Current GPU authorization ends20:00Toronto; next global check slot is no
earlier than18:03:10.328859UTC,14:03Toronto, subject to admission and one
live-job/reserve policy. No parameter sweep, new geometry or GPU job is
part of this draft. Its conditional whole-track opportunity remains the
earlier2.84% listed-floor bound, not a forecast or rank promise.
