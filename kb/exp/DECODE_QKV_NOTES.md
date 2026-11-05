# Decode QKV16/32: existing SK2 NACC=2 configuration

Candidate: `kb/exp/decode_qkv.py`, based on `origin/argon/keell-base-20261102`.
Only optimization: use the dense board's existing `qkv16t_r200` configuration
at T=16 and T=32, individually admitted for d16S/F and d32S/F.

## Why this configuration

Incumbent qkv16_r176 / qkv32_r208:

| Property | T=16 | T=32 |
|---|---:|---:|
| NT / NWR / WSK / UPW / NACC / CL | 3 / 1 / 4 / 2 / 1 / 1 | same |
| Rows per CTA / CTAs | 24 / 96 | 24 / 96 |
| K units per stage / stages over K | 8 / 16 | 8 / 16 |
| Weight bytes per stage per CTA | 18,432 | 18,432 |
| X bytes per stage per CTA | 16,384 | 32,768 |
| Stage bytes / ring stages | 34,816 / 5 | 51,200 / 4 |
| Ring bytes | 174,080 | 204,800 |
| Threads per CTA | 288 | 288 |
| Useful compressed weight bytes | 28,311,552 | 28,311,552 |
| Measured in-layer projection time | 7.8–8.1 us | 10.6 us |
| Weight-payload effective BW | 3.50–3.63 TB/s | 2.67 TB/s |
| Payload BW / 4.8 TB/s HBM ceiling | 72.8–75.6% | 55.6% |

Escape tables add a small weight-dependent amount to these payload numbers.
The payload-only HBM floor is 5.90 us, not a predicted achievable kernel time.
X is replicated across CTAs: 25.17/50.33 MB read into shared at T=16/32, but
is reusable in L2, so this is **not** additional compulsory HBM traffic.

Both rings exceed half of the SM shared-memory capacity, forcing one resident
CTA/SM. There is a single underfilled wave: 96/132 = 72.7% of SMs receive a
CTA, with 36 SMs idle (not a second-wave tail). Nine warps per occupied SM
give 14.1% thread/warp occupancy on those SMs, about 10.2% device-wide;
only four of those nine warps perform the matrix multiplies. Actual register
usage and stalls require a GPU compiler/profile and are not claimed here.
The larger T=32 X tiles and doubled MT consume more work without more CTAs.

The selected existing configuration changes NACC from 1 to 2, providing two
independent FP32 accumulation chains. Its r200 cap still gives exactly five
ring stages at T=16 and four at T=32. Therefore CTA count, weight stream,
escape order, shared-memory size and single BF16 epilogue store are unchanged.
It attacks dependency latency, **not** the underfilled wave. The alternative
can change FP32 summation order, which is why it never bypasses exact checks.

## Admission and containment

- Production default is the complete incumbent pipeline, with original tuning.
- New SK2 extension compiles on a CPU thread while the incumbent prepares.
- Whole-layer CUDA graphs run against the actual incumbent per shape.
- Two fresh synthetic draws; two passes; each arm has 12 alternating
  ABBA/BAAB rounds (24 observations per arm per pass).
- Every replay follows a 512 MiB **read** reduction outside the timing events.
  The current-token KV locations are restored before the read flush.
- Base-vs-base control in each pass, with opposite control placement in pass 2.
- Every pass requires median, paired, ABBA and BAAB candidate ratios <= .992;
  every null metric must be in [.995, 1.005].
- Repeated base and candidate QKV, hidden output, stream output, expert IDs
  and changed KV bytes must all be bit-identical. This is stricter than the
  allowed incumbent-repeatability envelope; no numerical tolerance is widened.
- Native attention, selected bfp QKV and exactly T=16/32 decode are required.
- Admission is state-local and keyed by batch, query length and attention mode.
  Incomplete/failed/over-budget shapes keep the incumbent.
- Existing optional MFIN diagnostic timings were removed; production MFIN
  kernels and dispatch are unchanged. All 13 inline build sites explicitly
  normalize host/NVCC C++20 and SM90a flags for the CUDA 13 judge.
- Optional gate deadline: 280 s total from prepare entry, leaving 10 s margin
  below the requested 290 s. The incumbent's own prepare has no new hard timeout.

## Validation

`python3 -m unittest kb/exp/test_decode_qkv_cpu.py -v`: seven tests passed.
`python3 -m py_compile kb/exp/decode_qkv.py`: passed.
`git diff --check`: passed.
Forbidden process / graph-version grep: no matches.
Source: 991,507 bytes at validation time.
No local GPU or nvcc: native compile, numerical identity, timing and total
prepare wall time remain for the caller's check. No check submitted here.

Ledger search: `scripts/tried.py search qkv nacc`, `search qkv16t` found no
matching closed idea. `search qkv ring` returned unrelated/previously closed
routes, not this NACC=2 configuration. A local-only TRIED row records the claim;
the caller must publish its shared-ledger status (this lane never writes main).
