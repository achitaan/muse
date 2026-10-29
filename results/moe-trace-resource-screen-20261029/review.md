# Trace resource screen: no GPU retry earned

Three fixed CPU-only compiler formulations were compared with the retained
MoE device prefix. Both BIG variants were instantiated with the retained MAIN
defines and the same documented NVRTC proxy adaptations. No kernel was loaded
or executed, no official check was used, and no runtime candidate was changed.

| Device prefix | Base registers | Offline stack bytes | Ptxas spill store/load bytes |
| --- | ---: | ---: | ---: |
| Retained, tracing disabled | 96 | 16 | 56 / 124 |
| Original 64-bit DBG markers enabled | 96 | 24 | 156 / 300 |
| Low 32-bit timestamps in the same slots | 96 | 24 | 156 / 300 |
| Self-contained pointer reload and 64-bit timestamp asm | 96 | 32 | 204 / 356 |

The low-word timer does not remove the extra stack in this proxy. The proposed
pointer-rematerialization block makes it larger. All dumps report LOCAL0 and
SHARED1024; those offline fields are not substituted for runtime CUDA attributes.
Ptxas's spill byte counts are not measured memory traffic. No timing or occupancy
conclusion follows from this table.

This is additional evidence against spending a GPU check on these formulations.
It does not identify the cause of the historical Linux diagnostic's local8-byte
attribute: that job did not query the contemporary original module, and the
Windows proxy is a different compilation environment. The old trace's timing
admission remains false; no threshold or historical interpretation changes.

The low-word input only changes DBG's store to the lower word of each existing
64-bit trace slot. NVIDIA documents `%globaltimer_lo` as the low 32 bits of the
[target-specific global timer](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer).
A host implementation would need initialized high words and an explicit bounded
wraparound reconstruction. No such host implementation or correctness claim is
made here, because the compiler screen did not support pursuing this approach.

The pointer-rematerialization input replaces DBG with an inline asm block that
reloads a diagnostic-only global pointer at each marker. It adds eight bytes
of device-global storage and leaves the numerical code and marker sites intact.
It has no host pointer setter and is not a runnable challenge submission. No
diagnostic global is introduced into a verified kernel or timed path.

The original numerical source is exactly CUDA1e60cf46 from MoE33114871. Each
transformation is invertible to that source. Raw compiler inputs, full logs,
metadata/options, cubin hashes and resource dumps are saved beside this review.
The retained comparison is source-bound in
`results/moe-epair-nvrtc-preflight-20261029/`. The same pinned NVIDIA compiler
and header wheels were used; no global package or PATH was changed.

Reproduce saved-evidence validation with
`python experiments/prepared/screen_moe_trace_resources.py`. With the pinned
workspace compiler present, `--compile` recompiles the three device prefixes
on the CPU. It does not call the challenge API or a CUDA kernel-launch API.
Compilation timing is not deterministic; use unused paths, for example
`--compile --work-dir .tools/trace-reproduction --output-dir .tools/trace-reproduction-result`.
The script refuses to compile over an existing evidence packet.

Keep the original diagnostic result and all verified sources. This screen
does not qualify another phase-only GPU retry or a scored submission. The
next evaluation still needs a distinct runtime hypothesis with credible
whole-layer value and the established correctness/resource/replay gates.
