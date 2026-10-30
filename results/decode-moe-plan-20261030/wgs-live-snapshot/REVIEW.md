# Correct live-buffer evidence for WGS MoE

Retained MoE33114871 writes phase-A SwiGLU directly into swizzled xB on its WGS path. Its finalizer computes norm sums in shared memory. The earlier diagnostic9dc89929 sampled row-major inter and global SS instead. When WGS is selected at M32,those fields are unused; byte stability there does not establish live SwiGLU equality or WGS norm statistics. Its ybuf/output observations remain valid. The tiny/MID inter and atomic-SS observations remain meaningful. Raw results are preserved; DIAGNOSIS.md now starts with this correction.

This correction does not affect prepared tiny-norm2f725371,whose M1..8 paths use live row-major inter and publish current SS. No permission or numerical gate is changed.

New untimed helper wgs_live_snapshot.py reconstructs the retained fat-first/skinny job order from a fresh host copy of probe routes. It reads only valid xB rows,returning expert-major route order. EMAX3072,JMAX88 and WGS_XROWS3776 put xB at BF16 offset(3072+3776)*4096. The inverse handles the stage/token-tile/column-quad/token/column swizzle. It ignores uninitialized padded rows and never reads cached routing decisions during a timed call.

An independent forward oracle packs canonical rows by tensor-axis permutation. All32 cases at M1,8,17,32,65,128,257,384 across none/one/hot/all-local routing pass. Tests poison padding with NaNs,exercise64-row job splits,and detect26 changed-live-value/wrong-buffer mutations. A full4096-column offset case distinguishes live xB from a poisoned unused inter region. These are CPU layout tests,not GPU correctness or performance evidence. No kernel or frozen submission changed.
