# Skinny16 compiler repair: one diagnosed retry

First check46ded9fefcb844db stopped at compilation. The evaluator's integrity
check passed, but setup failed before any correctness tests or benchmarks.
It supplied no speed result. The controller recorded metric0, discarded and
reverted to checked33114871. Complete evidence and controller history:
`results/resumed-moe-skinny16-20261029/attempt1/`.

The static-ring audit had inspected that function, but missed two FAST1
initialization blocks in moe_fused. The original supported tiny variants use
a helper warp. Pruning had removed their `#if OKC_WS` guards. Our newly enabled
non-helper configuration has no wse/wsrf/wsre fields, and no OKC_WSB define,
so those unguarded statements caused four NVCC errors. This is a missed
preflight requirement for the new variant, not a flaw in the retained SKT8
incumbent under its existing flags.

The unpruned public source contains the exact two conditional guards. The
repair restores them in the alternate only, adding40source bytes. It changes
no arithmetic, tile/launch geometry, data layout, synchronization operation
on the selected non-helper path, selector, numerical probe, timing comparison
or retain threshold. The affected helper initializations simply do not compile
when OKC_WS=0. Ordinary ring-barrier initialization remains unchanged.

Repaired source50dfae559fc8ce4bb6224d0dce6e7fc1ccc7a9097b7b4e19a66ea8a456efd3cd,
329582bytes, is saved under results/moe-skinny16-repair-20261029/.
Its inverse restores the complete failed f20caadb trial byte-for-byte. That
restores the prior full prototype and checked parent prefix as well. A lexical
audit checks all11helper-only references in the complete alternate, verifies
their OKC_WS guards and catches the original four unguarded statements as a
mutant. The selection/routing/dispatch audit reruns48/770cases against this
exact repaired source. Prior numerical/layout models remain applicable because
their code is unchanged. This still does not establish CUDA compilation.

Reuse the fixed comparison and full5/6/integrity/selected-profile guard from
moe_skinny16_selection_review.md. Only source binding and the required compiler
repair proof change. Retain exactly the original numerical and timing gates,
including three pattern medians and the affected-versus-control requirement.
One diagnosed retry is justified; this is not another tuning configuration.
No fresh GPU baseline or scored run. If it loses, preserve the negative result;
do not sweep nearby tile, warp, stage or selection thresholds.

At00UTC all tracks reset to20checks/10scores. Attempt1 consumed one MoE check;
expected remaining MoE19/10 and other tracks20/10, total99checks/50scores before
this retry. These are derived counts from the saved pre-launch API snapshot.
The broader top50Goal remains active, and all verified kernels stay unchanged.
