"""Fuse implicit unpooler zero padding into the small-token FFN boundary.

Adapted from SGLang's Apache-2.0 fused_norm_residual at pinned revision
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f; the complete Apache license above applies.
"""
import triton as _sfn_triton
import triton.language as _sfn_tl


@_sfn_triton.jit
def _sfn_kernel(Branch, Stream, PostWeight, PreWeight, Alpha, Beta, Output, NewStream,
                BranchStride: _sfn_tl.constexpr, StreamStride: _sfn_tl.constexpr,
                H: _sfn_tl.constexpr, Slice: _sfn_tl.constexpr,
                PostEps: _sfn_tl.constexpr, PreEps: _sfn_tl.constexpr):
    row = _sfn_tl.program_id(0)
    col = _sfn_tl.arange(0, H)
    branch = _sfn_tl.load(Branch + row * BranchStride + col, col < Slice, 0).to(_sfn_tl.float32)
    post_weight = _sfn_tl.load(PostWeight + col).to(_sfn_tl.float32)
    post_input = (branch * post_weight).to(_sfn_tl.bfloat16).to(_sfn_tl.float32)
    # The implicit zero tail is part of the 8192-column RMSNorm domain.
    post_inv = _sfn_tl.rsqrt(_sfn_tl.sum(post_input * post_input, 0) / H + PostEps)
    norm_branch = (post_input * post_inv).to(_sfn_tl.bfloat16).to(_sfn_tl.float32)
    stream = _sfn_tl.load(Stream + row * StreamStride + col).to(_sfn_tl.float32)
    alpha = _sfn_tl.load(Alpha + col).to(_sfn_tl.float32)
    beta = _sfn_tl.load(Beta + col).to(_sfn_tl.float32)
    updated = alpha * stream + beta * norm_branch
    _sfn_tl.store(NewStream + row * H + col, updated)
    # Default pre_norm_dtype=None: keep the residual input in FP32 here.
    pre_inv = _sfn_tl.rsqrt(_sfn_tl.sum(updated * updated, 0) / H + PreEps)
    pre_weight = _sfn_tl.load(PreWeight + col).to(_sfn_tl.float32)
    _sfn_tl.store(Output + row * H + col, (updated * pre_inv) * pre_weight)


def _sfn_boundary(stream, branch, state):
    cached = state['sliced_ffn_boundary']
    if (not branch.shape[0] or cached is None or not branch.is_contiguous()
            or not stream.is_contiguous()):
        padded = F.pad(branch, (0, HIDDEN - UNPOOL_SLICE))
        output, updated, _ = state['ffn_boundary'](stream, padded, branch_dtype=torch.bfloat16)
        return output, updated
    output = torch.empty(stream.shape, dtype=torch.bfloat16, device=stream.device)
    updated = torch.empty_like(stream)
    _sfn_kernel[(branch.shape[0],)](
        branch, stream, cached['post_weight'], cached['pre_weight'], cached['alpha'], cached['beta'],
        output, updated, branch.stride(0), stream.stride(0), HIDDEN, UNPOOL_SLICE,
        cached['post_eps'], cached['pre_eps'], num_warps=8, enable_fp_fusion=False)
    return output, updated


_sfn_original_prepare = prepare


def prepare(weights):
    state = _sfn_original_prepare(weights)
    boundary = state['ffn_boundary']
    state['sliced_ffn_boundary'] = None
    if (boundary._post_norm_is_input_scale and boundary._use_residual_gate
            and boundary._post_norm.has_weight and boundary._pre_norm.has_weight
            and boundary._residual_add.gate.numel() == HIDDEN):
        boundary._ensure_cached_weights(torch.bfloat16)
        state['sliced_ffn_boundary'] = {
            'post_weight': boundary._cc_post_weight, 'pre_weight': boundary._cc_pre_weight,
            'alpha': boundary._cc_alpha, 'beta': boundary._cc_beta,
            'post_eps': boundary._post_norm.eps, 'pre_eps': boundary._pre_norm.eps}
    return state
