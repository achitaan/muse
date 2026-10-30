"""Private SDK norm with earlier outgoing PDL; no change to norm arithmetic."""

_PDLN_EXT = None
_PDLN_ERROR = None


def _pdln_build():
    global _PDLN_EXT, _PDLN_ERROR
    try:
        from pathlib import Path
        import os
        import tempfile
        from sglang.kernels.jit.utils import KERNEL_PATH, load_jit, make_cpp_args
        source = None
        for name, expected in _PDLN_EXPECTED_HEADERS.items():
            rel = 'csrc/muse_spark_v12/' if name == 'norm_residual.cuh' else 'include/sgl_kernel/'
            text = (KERNEL_PATH / rel / name).read_text()
            actual = hashlib.sha256(text.encode()).hexdigest()
            if actual != expected:
                raise RuntimeError('PDLN source identity mismatch: ' + name)
            if name == 'norm_residual.cuh':
                source = text
        source = _PDLN_NOTICE + pdln_derive(source) + _PDLN_QUERY
        assert hashlib.sha256(source.encode()).hexdigest() == _PDLN_SOURCE_SHA
        fd, name = tempfile.mkstemp(prefix='okc_pdln_', suffix='.cuh')
        os.close(fd)
        Path(name).write_text(source)
        args = make_cpp_args(8192, 256, True, True, 1)
        _PDLN_EXT = load_jit('okc_pdln_' + _PDLN_SOURCE_SHA[:12], *args,
            cuda_files=[name], cuda_wrappers=[
                ('run_std', f'PdlnMuseSparkNormResidual<{args}, false, false>::run'),
                ('query', 'pdln_query')], extra_cuda_cflags=['-lineinfo', '-Xptxas=-v'])
        print('PDLN_BUILD', _PDLN_SOURCE_SHA, 'SDK_headers_exact=True', flush=True)
    except Exception as e:
        _PDLN_ERROR = repr(e)[-1500:]
        print('PDLN_BUILD_ERROR', _PDLN_ERROR, flush=True)


_pdln_thread = _okh_bg(_pdln_build)
_pdln_core_prepare = _achita_layer_prepare


def _achita_layer_prepare(weights):
    state = _pdln_core_prepare(weights)
    # Join both new C++ builds before the parent's graph-selection captures.
    # Compilation overlaps weight preparation; all join time stays in prepare.
    _pdln_thread.join()
    _p16_thread.join()
    return state


_pdln_original_plans = _p16_plans


def _p16_plans(state):
    _pdln_original_plans(state)
    state['pdln'] = {'ready': False}
    if _PDLN_EXT is None:
        return
    norm = state['attn_boundary']
    if (not norm.jit_selected() or norm._hidden_size != 8192 or norm._post_norm_is_input_scale or norm._pre_norm_is_input_scale
            or norm._post_norm.has_weight or not norm._pre_norm.has_weight or not norm._use_residual_gate):
        print('PDLN_PLAN_SKIP unsupported_norm', flush=True)
        return
    norm._ensure_cached_weights(torch.bfloat16)
    pre, alpha, beta = norm._cc_pre_weight, norm._cc_alpha, norm._cc_beta
    if (pre.dtype != torch.bfloat16 or pre.numel() != 8192 or not pre.is_contiguous()
            or alpha.dtype != torch.float32 or beta.dtype != torch.float32
            or alpha.numel() not in (1, 8192) or beta.shape != alpha.shape
            or not alpha.is_contiguous() or not beta.is_contiguous()):
        print('PDLN_PLAN_SKIP unsupported_weights', flush=True)
        return
    resources = [int(_PDLN_EXT.query(i)) for i in range(4)]
    print('PDLN_RESOURCE regs_local_shared_blocks=', resources, 'source=', _PDLN_SOURCE_SHA, flush=True)
    regs, local, shared, blocks = resources
    if regs <= 0 or local != 0 or shared < 0 or blocks <= 0:
        return
    state['pdln'] = dict(ready=True, pre=pre, alpha=alpha, beta=beta,
                        post_eps=float(norm._post_norm.eps), pre_eps=float(norm._pre_norm.eps))


def _pdln_boundary(stream, branch, state):
    p = state['pdln']
    if not p['ready']:
        raise RuntimeError('PDLN called without preparation admission')
    if (stream.ndim != 2 or branch.ndim != 2 or stream.shape != branch.shape
            or branch.shape[1] != 8192 or stream.dtype != torch.float32 or branch.dtype != torch.bfloat16
            or not stream.is_contiguous() or not branch.is_contiguous()
            or stream.data_ptr() % 16 or branch.data_ptr() % 16):
        return state['attn_boundary'](stream, branch, branch_dtype=torch.bfloat16, router_dtype=None)
    out = torch.empty_like(branch)
    updated = torch.empty_like(stream)
    _PDLN_EXT.run_std(branch, stream, out, updated, None, None, p['pre'], p['alpha'], p['beta'],
                      p['post_eps'], p['pre_eps'], False)
    # Original pre_norm_dtype=None uses FP32 pre-norm input. Preserve it.
    return out, updated, None
