# Next Prefill hypothesis: compact structurally zero MoE rows before unpooling

Inspection only: no candidate, GPU evaluation, speedup or readiness claim.
Parent93e57ab4 passed5/8/integrity; scored23e7c8b4c3e54869 at1.663469245x.
Prefill remains last and does not contribute to overall placement. Finish
the Attention coverage repair and its embedded Decode repair first.

Source best/prefill_verified.py retains SGLang dense/attention/router/norm
modules and replaces experts with native grouped persistent NVFP4. Primary
profiles: experts278.5us/44.4% atT753,811.2us/40.3% atT3012,
3807.6us/37.2% atT16384. Large dense projections form much of the rest.
Vendor kernel names combine multiple projections; resolve the actual
unpooler call/weight dimensions before attributing any whole row to it.
Do not claim that the1400.6us first-turn vendor row is solely unpooling.

Algebraic opportunity: _pg_route_metadata stores local IDs0..31 or-1.
_pg_combine adds all local expert outputs in its existing eight-route loop.
When none is local, its BF16 output is exactly the zero vector. The following
linear unpooler has no bias, so that token's8192-column result is also zero.
The original implementation still runs a dense projection on these rows.
This is an exact structural zero, not a small-value threshold. Preserve
every nonzero/local expert, all routing outputs and all BF16 boundaries.

With uniform top8 routing among384experts and32local experts, a rough
prior for zero rows is(352/384)^8, about50%. Actual routers are correlated;
this is a hypothesis, not a measurement of challenge inputs. Never change
routing or cache activations to increase this fraction.

Proposed bounded feature:

- In the existing combine kernel, track whether any of eight LOCAL IDs is
  nonnegative. Keep its FP32 norm/gain/weighted-sum order and BF16 store.
- Compact active token IDs with one GPU atomic per active token. Each
  token owns one slot; queue order may vary, but no duplicates or omissions.
- Reuse existing counter initialization: allocate33int32counts, pass the
  first32as the exact expert-count view to native MoE, use slot32for the
  active count. Its zero initialization precedes routing and combine. Do
  not race a store-to-zero against atomics within the same kernel.
- The combine kernel can write8192BF16 zeros for inactive unpooler rows
  using two4096-column stores. Active rows are written by the new matmul.
  This avoids an extra full output memset and leaves no uninitialized row.
- A BF16 Triton persistent matmul reads the GPU active count and row-ID
  queue. It gathers only active4096-element MoE rows, multiplies by the
  original BF16 unpooler weight (8192x4096), accumulates FP32, rounds BF16,
  and scatters to original token rows. No CPU count read or host branch.
- A first concrete layout is BM64,BN128,BK64,num_warps8,num_stages3,
  grid2*SMs. Each program loops tile_id+=grid_width until it covers
  ceil(active_count/BM)*(8192/BN) tiles. An inner K loop covers every4096
  element. This is one initial design, not authorization for a blind sweep.
- Keep kernel_fn's outputs/residual modules unchanged. Carry this call's
  active queue/count/output as local return values, never activation data
  cached in prepare/state. Weights and immutable device properties can be
  prepared normally. All work derived from inputs stays in timed kernels.
- Preserve the original small-token path (including its1024-column
  unpooler rule). A hardware-based threshold such as two BM tiles can
  select compaction only for sufficiently large T; scored Prefill T>=200.

Before implementing, read the current Triton documentation through Context7
and resolve actual kernel_fn/prepare/weight keys. The local CPU runtime
cannot compile Triton. Match allowed BF16 input/output and FP32 accumulator
behavior; use BF16 dot operands rather than quantizing activations. The
reference dense projection is F.linear without bias.

CPU/source acceptance: queue uniqueness and completeness under arbitrary
ordering, zero/all-active/mixed routes, tails1/63/64/65/127/129/753,
repeated/alternating-call counter reset, every output row initialized,
complete dot-product K coverage, and compact+scatter against an independent
FP64 dense oracle with BF16 rounding. Prove both expert-count and active-
count pointer bounds. Preserve all other layer AST/math except the explicit
integration. No input-dependent host allocation/branching or host sync.

Performance risk: vendor cuBLAS is fast; half as many rows helps only if
the custom projection's efficiency stays competitive. Full-active inputs
must remain correct and may be slower. Include this in nearby-shape checks
and retain all samples, rather than assuming the50% routing prior. Profile
the actual new kernel in every affected benchmark. Require full5tests/
8benchmarks/integrity and a material whole-layer gain before a single score.
An unchanged fallback that times faster by noise is not a selected new path.

Potential later Decode transfer is a separate decision. At small T the
projection can be bandwidth bound, so FLOP reduction may not help. Use its
actual unpooler fractions and controlled evidence before a scarce check.
