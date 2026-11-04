# RSTREAM v3: quantitative diagnosis and bounded repair

Check file: `experiments/moe_restart/moe_restart.py`, branch `loop/moe-restart`.
Parent: `57e115d`. No new GPU check/submission/push by this lane.

## Evidence, and what it does not establish

Judge `e61f95e5a91944c8` built v2 in 40.9 s, verified all tested reference
draws (~0.0055 relative error), admitted **zero** cases, and finished prepare
in 179.5 s. Log:

`/private/var/folders/pm/r8xm1mr525v913lct5tdg_5w0000gn/T/opencode/subs/profiles/moe_restart2.e61f95e5a91944c8.log`

| M | Incumbent median us | v2 median us | Gate AB ratio |
|---:|---:|---:|---:|
| 1 | 26.152 | 114.976 | 4.653 |
| 8 | 68.872 | 420.008 | 6.011 |
| 16 | 102.944 | 1060.120 | 10.019 |
| 32 | 121.608 | 1230.688 | 10.100 |
| 128 | 174.288 | 2371.072 | 12.914 |
| 256 | 235.072 | 3410.208 | 13.908 |

The gate is a geometric aggregation of per-draw paired ratios, not the ratio
of the two displayed medians. Do not try to make those columns divide exactly.

Two hard findings:

1. **No token-slab weight reload loop exists.** `load_group` executes once
   per K64 group (including unique lookahead), `decode` four times per group,
   and a native-N WGMMA covers the entire padded token axis. The token loop
   is the epilogue, after the entire reduction, and reads only accumulators.
   v3 retains this reuse; it does not hide extra weight passes behind N64 slabs.
2. **The wide kernel poisoned small cases and occupancy.** M16/32 used the
   same R128/CAP256 binary as M256: 168 registers/thread, 1928 local B/thread,
   one resident CTA/SM. Tiny used 87 registers, zero local bytes, and supported
   four resident CTAs/SM, but the launch STILL requested only 132 CTAs.

The old log has no active-expert/padding histograms, phase timings, SASS or
dynamic local-memory counts. Growth with M can reflect touched experts,
padded-token tensor work, local accesses, routing and finish work. **It does
not prove linear per-token re-streaming, nor uniquely identify a 13x term.**
`cudaFuncAttributes.localSizeBytes` includes stack/compiler arrays as well as
spills; it is not ptxas's spill-load/store count.

## Exact work and traffic model

H=I=4096. One touched expert contains:

| Phase | Weight bytes (codes + scales) | R64 tiles | R128 tiles |
|---|---:|---:|---:|
| A: gate/up | 18,874,368 | 128 | 64 |
| B: down | 9,437,184 | 64 | 32 |
| Total | **28,311,552** | 192 | 96 |

Per CTA **per K64**: R64 loads 2304 B (2048 codes + 256 scales); R128 loads
4608 B. Full K: 147456 / 294912 B. None depends on N. The optimistic 4.4 TB/s
unique weight floor is **6.434 us/touched expert**, A 4.290 + B 2.145 us.
The new common K256 stage consumes 9216 / 18432 weight B per R64/R128 CTA,
without changing the full-K total.
For 32 touched experts it is 205.90 us, already close to the 235.07 us old
M256 median. The exact old routing count was not logged.

For each consumer thread/K64: four words x eight weights are decoded. Each
word has about eight integer bit/rotate/multiply operations, four BF16x2
multiplies, two scale-half selects, PRMT and scale-rebase IMAD: approximately
**64 ordinary thread instructions for 32 weights**, plus **two scale
shuffles**. Exact SASS may fuse half selects or expand instructions. Across
one whole expert this is about **3,145,728 ordinary warp instructions**,
**98,304 shuffle warp instructions**, and 50,331,648 decoded scalar weights.
At an *assumed* 1.5 GHz and four issuing schedulers/SM on 132 SMs, decode alone
has a ~3.97 us/expert optimistic issue floor, excluding waits, loads, address
arithmetic and WGMMA control. One/two ready warps per scheduler cannot be
assumed to attain that floor through dependent instructions.

Each WG issues four m64nNk16 operations/K64, 256/full K; one expert has
192 WG-output tiles, hence **49,152 native-N WGMMA instructions**, independent
of N. N increases the work/duration of each instruction, not the number of
weight loads. In N8-slab-equivalent work, a K64 group is `4*(N/8)` m64n8k16
operations, encoded as four native-N instructions, not N/8 weight passes.
Each WG also executes **128 fence/commit/wait<1> triples plus a final drain**
per full K, with at most two committed groups outstanding. That count is
unchanged by batching activation copies. Exposing merely 100 ns per wait
would cost 12.8 us/CTA, versus a 4.42 us R64 weight-stream time at one
SM's peak-bandwidth share; 300 ns would cost 38.4 us. These are latency
equivalents, not measured WGMMA wait latencies. They show why additional
independent ready WGs matter even when there are zero spills.

FLOPs/expert = **100,663,296 x Npad**. At 989 dense BF16 TF/s:

| Npad | Tensor-only peak us/expert | Accumulator registers/thread |
|---:|---:|---:|
| 8 | 0.814 | 4 |
| 32 | 3.257 | 16 |
| 64 | 6.514 | 32 |
| 128 | 13.028 | 64 |
| 256 | 26.056 | 128 |

Activation traffic/CTA/full K is `8192*Npad` B. Shared logical writes + WG
reads per expert are **3,145,728*Npad B for R64** and **2,359,296*Npad B for
R128** (N/9 or N/12 of weight bytes). This cost is genuinely N-dependent.
Eight all-local experts with N256 have 206.16 GFLOPs: ~208.4 us at tensor
peak, NOT 3.4 ms; R64 shared traffic is 6.44 GB, also substantial. This hot
pattern is a correctness stress, not necessarily a timed random draw.

Gather is bounded by padded activation rows, not output tiles: input reads
and writes are O(sum(Npad)*8192). B writes 16384 B/real local route. Finish
reads that row twice plus stores 8192 B/token: at most 67.1 MB of row reads
for 2048 local routes (~15.3 us at HBM peak). Routing redundantly scans
32*8M IDs per CTA, so its instruction cost does scale with M and with resident
CTA count; the phase counters explicitly isolate it.

Scheduling uses one global atomic/task, at most 6144 R64 A+B tasks for all
32 experts. Three grid barriers and CTA synchronizations add latency, but
neither their count nor the number of weight tasks grows with N for fixed
active experts. They cannot explain 13x simply by counting tokens.

### Local-memory equivalent: plausibility, not a fake measurement

If all excess median time were HBM traffic, with the *maximum* 32 active
experts, the equivalent extra bytes/thread/K64 in the R128 kernel would be:

| M | Required extra B/thread/K64 |
|---:|---:|
| 16 | 74.4 |
| 32 | 86.2 |
| 128 | 170.7 |
| 256 | 246.7 |

These are lower bounds under that illustrative model (fewer experts raises
them). 247 B is only 6.4% of one 1928-byte frame read+write per K64, so a hot
fraction of that frame could plausibly account for milliseconds. If accesses
hit L1, the required traffic is larger; static local-byte size alone cannot
assign the measured time. Conversely M1/8 have **zero** local bytes and still
lose 4.6–6x: eliminating spills alone cannot fix the structure.

The incumbent source also documents NVVM outlining/CALL-induced WGMMA
serialization and insufficient register-pipeline fits (`C7510`/`C7512` in
`wgs_phase`/`wgs_seg`). v2's huge local frame may involve that phenomenon,
not just HBM spill bytes. It cannot be resolved from captured attributes.

### The other concrete inefficiency: K64 synchronization granularity

v2 performed 64 activation bulk copies, full waits, and empty releases per
tile. At N8 each request was only 1024 B. Its three requests in flight were
only 3072 B/CTA, with one actual CTA/SM and one/two consumer WGs. Assuming
300 ns request latency, that window is at most ~10.2 GB/s/SM of activation
copies. R64 would need 14.8 GB/s/SM of activation copies to accompany its
33.3 GB/s/SM weight share at peak HBM, capping weight throughput at ~69%
even before decode/dependencies (41% at 500 ns). R128 needs only 7.4 GB/s/SM
for N8, so this copy-window argument ALONE cannot explain its 10x loss.
These are illustrative latency assumptions, not measured TMA latencies.
Combine them with the fixed WGMMA wait/dependency path, one actual CTA/SM,
and the large kernel's local-memory/compiler hazards; do not invent a
single proven 13x TMA or spill factor from aggregate times.

v3 uses K256 for N<=64, K128 for N<=128, K64 otherwise. The common N8 request
is 4096 B; two requests give 8192 B/CTA, with 2/4 actual CTAs/SM. Full waits,
empty releases and TMA setup are reduced 4x in the common band. Weight/decode
work and WGMMA count are unchanged. The 4x stage and 4x occupancy changes
are NOT independent multiplicative speedup promises: both attack latency.

## Implemented repair

| Input M | R / CAP | Threads | Producer warps/CTA | Target resident CTAs/SM |
|---|---|---:|---:|---:|
| <=8 | 64 / 8 | 160 | 1 | 4 |
| <=16 | 128 / 16 | 320 | 2 | 2 |
| <=32 | 128 / 32 | 320 | 2 | 2 |
| <=64 | 128 / 64 | 288 | 1 | 2 |
| <=128 | 64 / 128 | 160 | 1 | 2 |
| <=256 | 64 / 256 | 160 | 1 | 2 |

- Actual grid is SM count * min(actual occupancy, target), not always 132.
  Simultaneous-residency checking remains mandatory before spin barriers.
- Min-block launch bounds match those targets. Lookahead is reduced from
  6/3 groups to 4/2/1; N8/16/32 no longer pays the CAP256 allocation.
- `const __grid_constant__ Params` avoids a per-thread local copy when its
  address is taken. Native CUDA supports it on this judge; the local Clang
  stub does not model the qualifier's code-generation contract.
- **No production case may use nonzero local bytes or miss its residency
  target.** Resource rejection falls back only that case and continues the
  other cases. This is a guard, not a claim local validation proved zero
  spills. Actual NVCC allocation is still unknown.
- Two K-adaptive ring slots use at most **87040 B** including 21504 B metadata,
  allowing two large CTAs/SM. Allocation covers the nonmonotone N96/N192
  stage sizes; no hidden under-allocation from f(CAP).
- Medium producer warps independently issue alternate stages. Tiny/large
  gain producer concurrency from additional CTAs without spending registers
  on an unnecessary extra warp. Two producer warps do NOT double a two-slot
  ring's in-flight capacity; their benefit is experimental, and their
  register cost is subject to the same admission guard. Weights never
  visit shared memory.
- Native full-N accumulators reuse each decoded weight across the token
  axis. Processing N64 passes would require reloading weights or storing
  partial accumulators between K groups; neither is silently introduced.

## Diagnostics, prepare time and remaining limits

Six prepare-only samples plus three first-use warmups: hot/random at
M8/32/256. Warmups exclude module loading and host resource printing from
sample event intervals, followed by a read flush. Separate PROFILE template
instances record route, gather+sync, A, A-sync, B, B-sync, finish clock64 spans;
sampled WG0 full-wait/work and producer-0 empty-wait/bulk-issue cycles isolate
consumer versus producer pressure. `RSTREAM_PROFILE` also logs active experts,
route/padding totals, actual CTA count and event time. Counters perturb the
profiled build/register allocation; its `RSTREAM_RESOURCES profile=1` must be
read alongside them. Counts are concurrent regions, not additive wall time.
Clock64 and trace stores are compile-time discarded from production.

The optional incumbent selector selected nothing in both checks and cost
37.9 s; v3 removes it, preserving exact default fallback builds/arithmetic.
The observed fallback-ready 177.3 s becomes roughly 139.4 s *if other costs
are unchanged*. Build now has six production and three profile families,
so compile time can rise and that estimate is not a guarantee. Admission
was 2.2 s in v2. Existing 275 s soft checks / 300 s reporting remain;
**new total prepare <300 s is not verified**, and compilation is not
cancellable. No extra tuning sweep or external compiler process is added.

Expected remaining gap is **not measured**. M8/16/32 have the best chance:
common K256 staging, 4 consumer WGs/SM and small accumulator budgets. M1
still has only 128 A / 64 B tiles for a single expert: more launch CTAs cannot
create output tiles, and the last phase uses only one CTA/token. It probably
remains an incumbent case without a smaller mma.sync tile/stream-K redesign.

Large CAP256 still fundamentally supports fewer consumer WGs than a narrow
token-slab kernel. Four full-N256 WGs would need 512*128=65536 registers
**for accumulators alone**, leaving none for weights/addresses/producers.
v3's 2 single-WG CTAs therefore give two consumer WGs/SM, not the incumbent
MAIN's four. R64 also has 4/3 the old R128 logical activation shared traffic.
Even if it removes the 10–14x pathology, an order-one residual loss (roughly
1.3–2x is plausible, not predicted/measured) is a serious risk at M128/256.
With 32 experts the bandwidth floor leaves only ~12.4% headroom versus the
235 us incumbent; it is not evidence for an easy 19% aggregate gain.

Thus whole-expert register streaming is not proven fundamentally impossible,
but **no large-case win should be assumed**. If zero-local v3 still loses,
the next decision requires its phase/region counters and generated PTX/SASS,
not another speculative producer-warp increase. The retained numerical and
paired-performance gate is the only authority for production admission.

Reproduce static model: `python3 experiments/moe_restart/performance_model.py`.
Output: `results/moe_restart_performance_model.json`. Numerical/build/frontend
checks and their explicit GPU-validation limits remain in `MOE_RESTART.md`.
Current evidence: `results/moe_restart_v3_validation.log` and
`results/moe_restart_v3_layout.log` (frontend checks, not real NVCC allocation).
