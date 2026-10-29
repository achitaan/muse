# Attention L8 wait removal: rejected before GPU evaluation

CPU-only assembly screen of retained Attention source 9e922c35, CUDA SHA256
241b67fe0008bbae53435b53d1bb6d647ded0b5718118875c0ec368b718b41db.
The instantiated device kernel is attn_kernel_fp8pv<8,true,0>, matching the
flat L8 host launch. No GPU kernel was loaded or executed. No challenge check,
score, candidate promotion, or Decode integration was performed.

The retained loop drains PV at each tile end. A simple omission could in
principle overlap PV(i) with QK(i+1), given correct lifetimes. This screen tests
only whether two fixed source formulations preserve asynchronous assembly:

| Device prefix | C7514 diagnostic | Static QGMMA | Wait 0 | Wait 1 |
| --- | --- | ---: | ---: | ---: |
| Retained | absent | 12 | 7 | 3 |
| Omit tile-end drain and its empty fences | present | 12 | 12 | 0 |
| Same omission; move output/lacc empty fences after the next full QK wait | present | 12 | 12 | 0 |

Both changes cause ptxas to report serialization due to non-WGMMA reads of
accumulator registers during a pipeline stage. The complete disassembly shows
a wait-zero between each matrix instruction, rather than the retained grouped
issues and partial waits. Moving the two output-accumulator fences alone does
not resolve it. These static counts include emitted loop copies; they are not
dynamic counts or timing estimates. No performance regression was measured.
All three builds report 80 base registers and no stack/spills in this proxy;
that does not establish dynamic consumer register allocation or residency.

Reject these exact two simple formulations from GPU evaluation. Keep the
retained implementation. A later design must address the compiler pipeline
constraint and prove every operand, accumulator, transpose buffer and producer
release lifetime. This screen neither rules out all overlap designs nor proves
correctness of either edited prefix. The separate NVIDIA partial-wait register
copy issue remains a concern to audit, not an established fault in this source.

Compiler: workspace-only NVIDIA NVRTC 13.0.88, Windows, sm_90a. The same pinned
compiler/header wheels as the MoE NVRTC preflight were used. The proxy omits the
Torch host launcher and removes its two Torch includes plus unused driver
cuda.h. It supplies the explicitly recorded integral_constant/stdint virtual
headers and defines INFINITY as FP32 bits 0x7f800000 because host math headers
are absent. It uses default-device execution space. Neither device arithmetic
nor a source initializer was silently changed to fit the proxy. The two setup
failures from unavailable host headers/INFINITY are preserved. No global
package/PATH mutation occurred. The actual judge NVCC build is not identified.

Each variant's JSON preserves exact options, source hashes and adaptations;
full device prefixes, logs and deterministic compressed assembly are included.
summary.json binds their hashes and lists every matrix/wait instruction.
The three included scripts reproduce the builds using the workspace compiler
from experiments/prepared/local_nvrtc_compile.py. Their output paths stay in
.tools. Do not import the compiler or these diagnostics into a submission.

Prepared while the foreground MoE repair experiment waited for its hourly
launch slot. Move this packet into results and commit only after that controller
has retained or reverted its scoped trial. The current MoE source is unchanged.

## Decode binding

A separate CPU build instantiates Decode D9's embedded attn_kernel<8,true,0>
with the same proxy adaptations. It compiles successfully and emits the same
matrix/wait instruction sequence as retained standalone Attention L8. Complete
instruction text is not claimed identical; the raw comparison says false and
preserves both disassemblies. decoder-binding.json binds exact source/SASS
hashes. The source diff shows the standalone version adds L1 FP8-PV helpers and
replaces the L1 PV calculation; that earlier tied result is not a Decode win.
Neither a runtime integration nor another GPU trial follows from this screen.
