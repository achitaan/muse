# L8 overlap: dependency screen, not a GPU candidate

Retained9e922c35 source inspection while the independent L1 identity-rescale
trial waits. No current source edits, new GPU evaluation or speedup claim.

The L8 flat WS loop commits its current QK group, waits until at most one group
is pending, transposes V, drains QK, builds P and commits its PV group. It then
drains PV at the loop end. A separate drain remains after the entire loop.
Earlier comments describe overlap across tiles, but the current end-of-tile
drain prevents PV from remaining in flight into the next tile.

The source has two4608-byte V-transpose buffers per KV head, toggled by tile,
and eight P-fragment registers fenced after the next tile's partial wait.
Conceptually, keeping PV(i) pending until QK(i+1) has been committed would let
wait_group1 complete the older PV while the new QK remains pending. The next
transpose targets the opposite V buffer. Output/denominator rescaling must
occur only after the old PV completed, and the final loop drain must remain.
This is an incomplete dependency argument, not a safe patch or timing bound.
Every A register, accumulator, shared operand and producer release needs a
complete lifetime proof, including first/last tiles, masked tiles, offset
updates and segment boundaries.

NVIDIA defines partial-wait group semantics and forbids premature access to
pending WGMMA operands/accumulators:
https://docs.nvidia.com/cuda/pdf/ptx_isa_9.0.pdf

NVIDIA also documents a compiler copy-propagation issue for certain WGMMA
register-copy patterns across wait_group N>0. Its suggested workarounds include
a complete wait before the next ldmatrix or loading individual fragments
directly. See the CUDA compiler known issue, not the unrelated library fixes:
https://docs.nvidia.com/cuda/archive/13.2.2/cuda-toolkit-release-notes/index.html#cuda-compiler-issues
The exact risky pattern is not established for this kernel. Do not generalize
the release note into a claim that the retained source is faulty.

Saved diagnostic software reports torch2.14.1+cu130/CUDA runtime13.0. That does
not identify the exact NVCC/ptxas build. The official evaluator image identity
is preserved in raw results; it is not permission to change its environment.
A prospective overlap candidate needs compiler-build evidence and generated
code inspection or an applicable safe formulation, plus real residency and
fresh-replay validation. Holding more fragments live can also serialize the
pipeline or spill under the flat L8 consumer104-register budget.

Decision: park implementation/evaluation. This is a distinct architectural
option rather than a reason to repeat an old tile/occupancy/commit flag.
The queued L1 identity-rescale source changes none of these WGMMA waits.
