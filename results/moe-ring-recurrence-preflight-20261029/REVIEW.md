# MoE six-state ring recurrence: held before GPU

Parent CUDA 1e60cf46; draft d61835dd. Unlike the earlier three-body drafts,
this has one dynamic loop body and a six-state cursor. The two three-slot
rings retain the same slot/parity, first accumulation, releases and exact
stage count in 65,536 finite CPU sequences. Four bad recurrences are detected.

Both NVRTC proxies retain 96 registers and 16/56/124 stack/store/load fields,
matching the control. They avoid the earlier C7512 serialization warning.
However, static instruction count grows 25,088 -> 25,256 and the binary grows
3,860,368 -> 3,874,448 bytes. HGMMA count remains 512. This does not establish
runtime cost or improvement. No credible exposed whole-load saving offsets
the extra control work; hold without a GPU check or compiler/register sweep.
No Python candidate or speedup claim. Both saved best kernels remain original.
