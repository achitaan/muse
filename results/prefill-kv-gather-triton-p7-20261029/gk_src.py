import triton as _p7_triton
import triton.language as _p7_tl
@_p7_triton.jit
def _p7_gather_kernel(KS, VS, IDX, KD, VD, n_rows, ROW: _p7_tl.constexpr, BLOCK: _p7_tl.constexpr):
    pid = _p7_tl.program_id(0)
    rows = pid * BLOCK + _p7_tl.arange(0, BLOCK)
    mask = rows < n_rows
    src = _p7_tl.maximum(_p7_tl.load(IDX + rows, mask=mask, other=0), 0).to(_p7_tl.int64)
    cols = _p7_tl.arange(0, ROW)
    s_off = src[:, None] * ROW + cols[None, :]
    d_off = rows.to(_p7_tl.int64)[:, None] * ROW + cols[None, :]
    m2 = mask[:, None]
    _p7_tl.store(KD + d_off, _p7_tl.load(KS + s_off, mask=m2), mask=m2)
    _p7_tl.store(VD + d_off, _p7_tl.load(VS + s_off, mask=m2), mask=m2)


