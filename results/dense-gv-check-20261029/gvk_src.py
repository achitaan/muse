import triton as _gv_triton
import triton.language as _gv_tl
@_gv_triton.jit
def _gvk(X, W, O, N, K: _gv_tl.constexpr, M: _gv_tl.constexpr, ROWS: _gv_tl.constexpr):
    pid = _gv_tl.program_id(0)
    offs = _gv_tl.arange(0, K)
    for r in _gv_tl.static_range(ROWS):
        row = pid * ROWS + r
        w = _gv_tl.load(W + row * K + offs).to(_gv_tl.float32)
        for m in _gv_tl.static_range(M):
            x = _gv_tl.load(X + m * K + offs).to(_gv_tl.float32)
            _gv_tl.store(O + m * N + row, _gv_tl.sum(w * x, axis=0))


