"""Early stock norm signal for the exact FP32-router T1/T8 path only."""

_PDLS_EXT = None
_PDLS_ERROR = None


def _pdls_build():
    global _PDLS_EXT, _PDLS_ERROR
    try:
        from pathlib import Path
        import os
        import tempfile
        from sglang.kernels.jit.utils import KERNEL_PATH, load_jit, make_cpp_args
        source = None
        for name, expected in _PDLS_HEADERS.items():
            rel = 'csrc/muse_spark_v12/' if name == 'norm_residual.cuh' else 'include/sgl_kernel/'
            text = (KERNEL_PATH / rel / name).read_text()
            if hashlib.sha256(text.encode()).hexdigest() != expected:
                raise RuntimeError('PDLS header identity: ' + name)
            if name == 'norm_residual.cuh':
                source = text
        source = _PDLS_NOTICE + pdln_derive(source) + _PDLS_QUERY
        assert hashlib.sha256(source.encode()).hexdigest() == _PDLS_SOURCE_SHA
        fd, name = tempfile.mkstemp(prefix='okc_pdls_', suffix='.cuh')
        os.close(fd)
        Path(name).write_text(source)
        args = make_cpp_args(8192, 256, True, True, 1)
        _PDLS_EXT = load_jit('okc_pdls_' + _PDLS_SOURCE_SHA[:12], *args,
            cuda_files=[name], cuda_wrappers=[
                ('run_router', f'PdlnMuseSparkNormResidual<{args}, false, true>::run'),
                ('query', 'pdln_query')], extra_cuda_cflags=['-lineinfo', '-Xptxas=-v'])
        print('PDLS_BUILD', _PDLS_SOURCE_SHA, flush=True)
    except Exception as e:
        _PDLS_ERROR = repr(e)[-1500:]
        print('PDLS_BUILD_ERROR', _PDLS_ERROR, flush=True)


_pdls_thread = _okh_bg(_pdls_build)
_pdls_core_prepare = _achita_layer_prepare


def _achita_layer_prepare(weights):
    state = _pdls_core_prepare(weights)
    # The parent's first graph capture occurs after this core hook returns.
    _pdls_thread.join()
    _pdls_plans(state)
    return state


def _pdls_plans(state):
    state['pdls'] = dict(ready=False, selected={})
    if _PDLS_EXT is None:
        return
    norm = state['attn_boundary']
    if (not norm.jit_selected() or norm._hidden_size != 8192 or norm._post_norm_is_input_scale
            or norm._pre_norm_is_input_scale or norm._post_norm.has_weight
            or not norm._pre_norm.has_weight or not norm._use_residual_gate):
        return
    norm._ensure_cached_weights(torch.bfloat16)
    pre, alpha, beta = norm._cc_pre_weight, norm._cc_alpha, norm._cc_beta
    if (pre.dtype != torch.bfloat16 or pre.numel() != 8192 or not pre.is_contiguous()
            or alpha.dtype != torch.float32 or beta.dtype != torch.float32
            or alpha.numel() not in (1, 8192) or beta.shape != alpha.shape
            or not alpha.is_contiguous() or not beta.is_contiguous()):
        return
    resources = [int(_PDLS_EXT.query(i)) for i in range(4)]
    print('PDLS_RESOURCE regs_local_shared_blocks=', resources, flush=True)
    regs, local, shared, blocks = resources
    if regs <= 0 or local != 0 or shared < 0 or blocks <= 0:
        return
    state['pdls'].update(ready=True, pre=pre, alpha=alpha, beta=beta,
        post_eps=float(norm._post_norm.eps), pre_eps=float(norm._pre_norm.eps))


def _pdls_boundary(stream, branch, state):
    p = state['pdls']
    if not p['ready']:
        raise RuntimeError('PDLS without preparation admission')
    if (stream.ndim != 2 or branch.ndim != 2 or stream.shape != branch.shape
            or branch.shape[0] not in (1, 8) or branch.shape[1] != 8192
            or stream.dtype != torch.float32 or branch.dtype != torch.bfloat16
            or not stream.is_contiguous() or not branch.is_contiguous()
            or stream.data_ptr() % 16 or branch.data_ptr() % 16):
        return state['attn_boundary'](stream, branch, branch_dtype=torch.bfloat16,
                                      router_dtype=torch.float32)
    out, updated, router = torch.empty_like(branch), torch.empty_like(stream), torch.empty_like(stream)
    # Preserve the original FP32 pre-norm input and separate FP32 router output.
    _PDLS_EXT.run_router(branch, stream, out, updated, router, None,
        p['pre'], p['alpha'], p['beta'], p['post_eps'], p['pre_eps'], False)
    return out, updated, router


def _pdls_select(state, start):
    plans = state.get('pdli', {}).get('plans', {})
    dev = state['rf']['weight'].device
    for m in (1, 8):
        if 'r' + str(m) not in plans or not _okh_budget(start, 274.):
            continue
        gen = torch.Generator(device=dev).manual_seed(85300 + m)
        ao = torch.randn((m, 8192), device=dev, generator=gen).to(torch.bfloat16)
        stream = torch.randn((m, 8192), device=dev, generator=gen)
        def unit(on):
            b = (_pdls_boundary(stream, ao, state) if on else
                 state['attn_boundary'](stream, ao, branch_dtype=torch.bfloat16, router_dtype=torch.float32))
            # Same incoming-PDL R2f in both arms isolates the outgoing norm signal.
            return tuple(b) + tuple(_pdli_route(b[0], b[2], state))
        def umutate(seed):
            ao.normal_(generator=gen); stream.normal_(generator=gen)
        if not _pdli_ab('PDLS_norm_r2f_M' + str(m),
                {'parent': lambda: unit(False), 'pdl': lambda: unit(True)}, umutate, dev, start):
            continue
        for sliding in (True, False):
            if not _okh_budget(start, 274.):
                break
            data = _pdli_data(m, sliding, dev, gen)
            # One union per regime: retain the parent's selected paths, add norm/R2f.
            mask = state['pdli']['selected'].get((m, bool(sliding)), 0) | 4
            def mutate(seed):
                data['attn_input'].normal_(generator=gen); data['stream'].normal_(generator=gen)
            tag = 'PDLS_layer_M' + str(m) + ('_sliding' if sliding else '_full')
            if _pdli_ab(tag, {'parent': lambda: _pdls_parent_kernel_fn(data, state),
                             'pdl': lambda: _pdls_layer(data, state, mask)},
                         mutate, dev, start, (data['k_cache'], data['v_cache'])):
                state['pdls']['selected'][m, bool(sliding)] = mask
            print('PDLS_SELECT', tag, state['pdls']['selected'].get((m, bool(sliding)), 0), flush=True)


_pdls_parent_prepare = prepare
_pdls_parent_kernel_fn = kernel_fn


def prepare(weights):
    start = _okh_clock.perf_counter()
    state = _pdls_parent_prepare(weights)
    if state.get('pdls', {}).get('ready') and state.get('rf') is not None and _okh_budget(start, 274.):
        try:
            _pdls_select(state, start)
        except Exception as e:
            state['pdls']['selected'].clear()
            print('PDLS_ERROR', repr(e)[-1500:], flush=True)
    torch.cuda.synchronize()
    print('PDLS_DONE', _okh_clock.perf_counter() - start, state.get('pdls', {}).get('selected'), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    mask = state.get('pdls', {}).get('selected', {}).get((cfg['tokens'], bool(cfg['window'])), 0)
    if cfg['batch'] == 1 and cfg['q_len'] in (1, 8) and cfg['tokens'] == cfg['q_len'] and mask:
        return _pdls_layer(data, state, mask)
    return _pdls_parent_kernel_fn(data, state)
