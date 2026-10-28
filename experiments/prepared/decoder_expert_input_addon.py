"""Fuse pre-expert RMSNorm and global-to-local route rotation."""
import triton as _pn_triton
import triton.language as _pn_tl


@_pn_triton.jit
def _pn_inputs_kernel(X, Gain, Ids, Start, Out, LocalIds,
                      XStride: _pn_tl.constexpr, IdStride: _pn_tl.constexpr,
                      H: _pn_tl.constexpr, Eps: _pn_tl.constexpr,
                      Block: _pn_tl.constexpr):
    row = _pn_tl.program_id(0)
    col = _pn_tl.arange(0, Block)
    x = _pn_tl.load(X + row * XStride + col, col < H, 0).to(_pn_tl.float32)
    gain = _pn_tl.load(Gain + col, col < H, 0).to(_pn_tl.float32)
    inv = _pn_tl.rsqrt(_pn_tl.sum(x * x, 0) / H + Eps)
    y = (x * inv) * gain
    _pn_tl.store(Out + row * H + col, y, col < H)
    slot = _pn_tl.arange(0, 8)
    ids = _pn_tl.load(Ids + row * IdStride + slot).to(_pn_tl.int32)
    start = _pn_tl.load(Start).to(_pn_tl.int32)
    # Positive modulus 256 is exactly a low-eight-bit mask, including negatives.
    _pn_tl.store(LocalIds + row * 8 + slot, (ids - start) & 255)


def _pn_expert_inputs(x, ids, local_start, state):
    rows, hidden = x.shape
    gain = state['pre_expert_norm']._get_effective_weight()
    assert x.dtype == gain.dtype == torch.bfloat16 and hidden == 4096
    out = torch.empty((rows, hidden), dtype=x.dtype, device=x.device)
    local_ids = torch.empty(ids.shape, dtype=torch.int32, device=ids.device)
    if rows:
        _pn_inputs_kernel[(rows,)](
            x, gain, ids, local_start, out, local_ids, x.stride(0), ids.stride(0),
            hidden, RMS_EPS, _pn_triton.next_power_of_2(hidden), num_warps=4,
            enable_fp_fusion=False)
    return out, local_ids
