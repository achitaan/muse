# Full-context pre-wait staging

Check file: `kb/exp/decode_attnfull.py`; parent: `51fe86a`.

## Main finding: d16F/d32F already do this

`_plan7` selects flat=1 for B>1, window=0. `_decode_attention7`
passes PDL=True at T=16/32; native `launch` uses
`cudaLaunchAttributeProgrammaticStreamSerialization`.

The flat kernel splits consumers (ROLE=2) and producers (ROLE=1).
Only consumers execute the segment's `pdl_wait`. Producers independently
gather cached rows with `pos < min(end, seq)`, publish `S.full[stage]`,
and only block on `S.empty[stage]` starting at tile STAGES=3. Thus
**all three existing ring slots already stage before QKV completes**.
The consumer's unchanged `ws_patch` supplies new-token rows after waiting.
The candidate leaves this producer code byte-identical and does not time
an identical d16/d32 arm. Its DEC_AFULL lines mark no_timing=1, admitted=0.

## Residency: not 264 CTAs overlapping QKV

CPU C++20 layout evaluation of the original q1 `Smem` gives 62,464 dynamic
shared bytes, including the native launcher's 1,024-byte alignment pad;
256 threads/CTA. The native diagnostic queries actual registers, total
shared allocation, and standalone blocks/SM on the check GPU.

For the default bf16 sk2 QKV configurations:

| Shape | Config | Stages | Ring bytes | QKV dynamic smem lower bound | +attention |
|---|---|---:|---:|---:|---:|
| d16 | qkv16_r176 | 5 | 174080 | 181744 | 244208 |
| d32 | qkv32_r208 | 4 | 204800 | 218848 | 281312 |

Lower bounds follow `bfp__sk2_geom` and `launch_sk2` with pmax=0; actual
escape-table storage adds bytes. Both sums exceed H200's 233,472 bytes/SM:
**no attention CTA can co-reside on a SM occupied by these QKV CTAs**,
regardless of register allocation. QKV's grid is 2304/24=96 CTAs, so on
132 SMs there can still be 36 free SMs, with up to 72 early attention CTAs
if attention standalone occupancy is two. Actual launch/dependency timing
may allow fewer; no GPU residency measurement was possible locally.
If prepare selects cuBLAS/Triton or another dense backend, these QKV bounds
do not apply. Native diagnostic explicitly says early residency unmeasured.
Even the all-resident ring capacity is only 264*3*16384=12,976,128 bytes;
the proposed overlap already exists and cannot hide the whole memory stream.

## Distinct candidate retained only at d1F/v1F

B=1 full context uses a non-WS per-request kernel. Move BOTH existing
prologue tiles (STAGES-1=2) before the PDL wait, rather than the parent's
tile-zero-only arm. Page indices are drained/CTA-synchronized first;
SPEC length is checked before KV addresses. Both gathers join the original
post-wait Q/constants cp.async group. All original commit calls and
wait_group counts stay intact; post-wait duplicate gathers are skipped.
New rows are not gathered and remain patched by the original post-wait pass.
ONE builds retain their original straight-line copy. Ring depth is unchanged.

Default is incumbent. Only d1F/v1F can be admitted, separately, after both
12-round whole-layer graph ABBA/BAAB passes with a 512 MiB READ before every
timed replay. Both median and paired ratios must be <=0.992; independent
base/base null median and paired ratios must be in [0.995,1.005]. Outputs
must fit measured incumbent repeatability; new KV bytes and expert ids must
match the first incumbent snapshot exactly; all other KV bytes are checked.
No isolated-attention timings consume prepare budget. Deadline remains 275s.

## Validation and ledger handoff

Five dependency-free CPU tests pass: AST/source rewrite, unchanged WS
producer, exactly-once tile dispatch model, gate boundaries, plans/PDL,
C++20/SM90a flags, and file-size limit. Forbidden process/graph-API grep is
clean. No GPU, nvcc, judge timing, actual register query, or conformance run
was available; none was submitted. Parent must run any desired check.

Ledger searched through `scripts/tried.py search cached KV`. The requested
`.tools/tried.py` path is absent. The actual helper pushes main, so it was
not used to write: the task explicitly forbids subagent pushes/main commits.
Proposed row for the parent's shared-ledger update:

| Idea | Outcome | Number | Job | Where | Date |
|---|---|---|---|---|---|
| Full-context pre-wait KV staging: flat d16F/d32F already fill three-stage ring before consumer wait; distinct B1 full-context arm fills both existing non-WS prologue stages, not only tile zero | diagnostic primary/no-op; B1 candidate unmeasured | loop/dec-attnfull | not submitted | kb/exp/decode_attnfull.py | Nov 04 |
