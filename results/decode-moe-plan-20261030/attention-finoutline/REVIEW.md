# L1 flat finalizer outlining: CPU admission rejected

The human asked to continue the other tracks and preserve Decode checks until strong results. This packet uses no GPU evaluations. The next paced GPU candidate remains the separate MoE whole-row norm2f725371; the final Decode check stays reserved.

Retained Attention9e922c35 embeds native CUDA241b67fe. Retained Decode452fbd9d profiles attribute88.1us/171.2us to L1 full attention at B16/B32. These are kernel spans,not measured instruction-cache stalls or critical-path fractions. Saved history already rejects identity-rescale skipping,QK operand swap,L1 wide tiles,L8 two-score banks and external finalization; none is reopened here.

The distinct hypothesis was to outline fin_nc1 only for L1 flat kernels. Its body is force-inlined despite a nearby comment describing an out-of-line cold path. A copied noinline helper preserves the exact arithmetic,loads,clears and stores. Only the L1-flat call boundary changes; launch plans,split counts,cooperation,precision,keys,cache writes,fallbacks and other Python definitions remain original. An exact inverse recovers retained CUDA. Candidate fdf4947ac4d35060600bd89fb2eee3c4a1844750315a7b7dfc7d40bf9ae65488 is175387bytes; CUDA554882be51ec56dea92eecd5781a5b7a97c77ba8f4e6b37b368013308785fef0.

The source already marks its kernel parameter grid-constant. NVIDIA documents that this avoids creating per-thread parameter copies when its address is used: [CUDA13 programming guide](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html). No new grid-constant flag was added.

The predeclared compiler screen required zero added stack/local/spills and at least5% less compiled kernel code before admitting a GPU check. Both parent and candidate use128registers and zero stack/spills. However,cuobjdump counts3560 parent instructions and3608 candidate instructions,ratio1.013483146. The candidate adds a relative leaf call; its subroutine remains embedded in the compiled kernel text. These counts include leaf subroutines,not dynamic executed instructions or measured cache misses.

The footprint gate fails. Park this source without a GPU check,score,selector or Decode transfer. This is a CPU readiness rejection,not a measured H200 slowdown. No numerical GPU correctness or speedup is claimed. Source,compiler logs,cubins and SASS are retained here.
