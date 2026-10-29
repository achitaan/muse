import triton as _pf_triton
import triton.language as _pf_tl
@_pf_triton.jit
def _pf_l2_kernel(Tab, WARPS: _pf_tl.constexpr):
    pid = _pf_tl.program_id(0).to(_pf_tl.int64)
    nprog = _pf_tl.num_programs(0).to(_pf_tl.int64)
    e = _pf_tl.arange(0, WARPS * 32)
    lane = (e % 32).to(_pf_tl.int64)
    wid = (e // 32).to(_pf_tl.int64)
    b0 = _pf_tl.load(Tab + 0)
    b1 = _pf_tl.load(Tab + 1)
    b2 = _pf_tl.load(Tab + 2)
    b3 = _pf_tl.load(Tab + 3)
    c1 = _pf_tl.load(Tab + 4)
    c2 = _pf_tl.load(Tab + 5)
    c3 = _pf_tl.load(Tab + 6)
    n = _pf_tl.load(Tab + 7)
    step = nprog * WARPS
    iters = (n + step - 1) // step
    # Chunk c goes to warp c % step in iteration c // step, so the whole grid
    # walks the buffers in priority order (output projection first).
    for it in range(0, iters):
        c = _pf_tl.minimum(it * step + pid * WARPS + wid, n - 1)
        a = _pf_tl.where(c < c1, b0 + c * 4096,
                         _pf_tl.where(c < c2, b1 + (c - c1) * 4096,
                                      _pf_tl.where(c < c3, b2 + (c - c2) * 4096, b3 + (c - c3) * 4096)))
        _pf_tl.inline_asm_elementwise(
            "prefetch.global.L2 [$1];\n\tprefetch.global.L2 [$1+1024];\n\t"
            "prefetch.global.L2 [$1+2048];\n\tprefetch.global.L2 [$1+3072];\n\tmov.u32 $0, 0;",
            "=r,l", [a + lane * 32], dtype=_pf_tl.int32, is_pure=False, pack=1)
