# Next MoE hypothesis: a bounded WGS specialization for 17..32 tokens

This is implemented and CPU-audited, but GPU correctness and speed remain
unverified. Use exact retained 33114871 as the baseline. Do not inherit the
rejected scale-six layout. It passed full correctness but lost 11..14% speedup
on all WGS loads; original byte scales, PRMT decode and NVFP4 code layout stay.

The default CUDA hardcodes OKC_FT_T=64. WGS_NT=FT_T/8 is therefore eight, with
float d[8][4] accumulator capacity and three 32768-byte x-ring slots, even in
32-token work. The source comments state that register pressure can serialize
wgmma pipelines across the wide N variants. This is a reason to investigate,
not proof of the compiler's actual register allocation or the next speedup.

Candidate boundary: one additional main WGS module with FT_T=32 for 17..32
tokens when PRMT is eligible. All 1..16 and >32 token paths remain original;
ineligible scales retain the original main fallback. No tile sweep or format
selector. The 32-token benchmark is the slowest individual MoE speedup in the
saved primary check (~1.642x), and MoE is a major component of decoder32.

The source route() already chunks fat jobs using ceil(count/FT_T) and
min(FT_T,count-q*FT_T), so FT_T=32 caps every job at 32 rows without dropping
routes. Do not assume distinct IDs or rely on a benchmark shape's routing.
Audit all count distributions and routing entries, including duplicate routes,
empty experts, skewed experts and nearby M17..33. The 33-token case must take
the original module. Preserve the 384-token chunk size and every output.

Implementation constraints to inspect before editing:

- Make OKC_FT_T an overridable compile define while retaining default64.
- For the 32-row build, prune WGSMMA<5..8> and switch cases5..8; do not leave
  PTX operands referring to d[4..7] when WGS_NT is four. Inspect the separate
  older fat-path MMA wrappers too, even though WGS does not execute them.
- Derive accumulator, WGS_XS, WGS_FIXW, fixup and JMAX layouts consistently.
  Do not change K accumulation, rounding, epilogues, norms or routing math.
- Give the new variant a separate scratch bank. WGS_FIXW changes fixup counter
  offsets; sharing banks with the original variant can expose stale counters
  when M alternates. The existing _make_scratch allocation is already padded
  for both capacities: 1024 doneA slots, WGS_FIXW4096 maximum and
  WGS_XROWS=EMAX+8*(32+EMAX//32+8). Prove launcher offsets and bounds explicitly.
- Reuse original packed weights; WGS weight pieces/stage bytes are independent
  of FT_T. Do not recompress scales or rebuild the large static weight copies.
- At most one extra compiled variant and one extra scratch bank. Check the
  300-second prepare budget; last observed full preparations ranged165..208s.

A useful prepare diagnostic could report cudaFuncGetAttributes register counts,
local-memory bytes and shared-memory size for the old/new real kernels. This
is ordinary static resource metadata, not permission to inspect private harness
state or generated outputs. Do not claim register reduction from source arrays
alone; report hardware/compiler attributes or leave the gap explicit.

Before one full official check, CPU-audit layout bounds for both configurations,
scratch separation across alternating M, complete job/route coverage, flag and
factory consistency, host dispatch near all thresholds, full source inversion
and the original BF16/FP32 precision contract. The guard must require full
5/6/integrity plus evidence the specialization actually compiled/was enabled.
Reject a fallback-only trial. Score only a material measured improvement.

## Implemented candidate and pre-check evidence

Source d0ae4a9a775b28d614fee1768d37fad609024c44ad2189058565263b03c9f9b4
is 181513 bytes. The original CUDA source is retained byte for byte with only
an appended host resource-query function. The alternate source is constructed
by ordinary string operations from that source: FT_T32, removal of the wider
WG/WGSMMA definitions, and trap defaults for impossible job tile counts above
four. The original variants retain their full original code and 64-token cap.

The host dispatch uses the new module and a distinct scratch bank only at
17..32 tokens. Packing, weights and all numerical helpers are unchanged. The
wrapper reuses the cached original modules, adds just one module and one bank,
and skips the new build if original preparation exceeds235 seconds. Require
full 5/6/integrity plus explicit enabled/resource diagnostics before retention.

audit_moe_wgs32_cpu.py reversed all enumerated edits to recover the original
module AST. It independently reconstructed and reversed alternate-source edits,
exhausted 6146 single-expert count cases and checked 560 routing distributions,
including duplicates and skew. It checked 292 host-dispatch cases, exact chunk
coverage, fresh outputs, dtype/ID-alignment behavior, both scratch layouts,
budget/domain/hardware gates and the real factory's five-key cache isolation.
These are control-flow, representation and bounds checks, not GPU numerics.

Both allocations are large enough for both declared layouts, but counter offsets
differ (15204352 versus17825792 float-sized positions from ybuf); banks remain
separate across alternating batches. The specialization cuts x-ring payload
from98304 to49152 bytes. Actual register/local-memory/shared-memory values will
come from the real kernels, not the CPU mocks. NVIDIA's field definitions:
https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaFuncAttributes.html

One focused full official check is planned using cached baseline 2.365437416x.
No geometry sweep, synthetic timing selector or scored intermediate is added.
