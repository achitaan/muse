"""Unvalidated O16/O32 integration; every path defaults to the scored parent."""

_P16_EXT = None
_P16_ERROR = None
assert hashlib.sha256(_P16_CUDA.encode()).hexdigest() == _P16_CUDA_SHA


def _p16_build():
    global _P16_EXT, _P16_ERROR
    try:
        _P16_EXT = load_inline('okc_p16_' + _P16_CUDA_SHA[:12], _P16_CPP,
            cuda_sources=_P16_CUDA, functions=_P16_FUNCTIONS,
            extra_cuda_cflags=['-O3', '-lineinfo', '-gencode=arch=compute_90a,code=sm_90a'],
            no_implicit_headers=True, with_pytorch_error_handling=False, verbose=False)
        print('P16_BUILD', _P16_CUDA_SHA, flush=True)
    except Exception as e:
        _P16_ERROR = repr(e)[-1500:]
        print('P16_BUILD_ERROR', _P16_ERROR, flush=True)


_p16_thread = _okh_bg(_p16_build)
_p16_core_prepare = _achita_layer_prepare


def _p16_warm_core(weights):
    state = _p16_core_prepare(weights)
    def warm():
        dev = state['dense']['o'].device
        try:
            with torch.cuda.device(dev), torch.cuda.stream(torch.cuda.Stream(device=dev)):
                for m in (16, 32):
                    s = torch.zeros((m, 8192), dtype=torch.float32, device=dev)
                    a = torch.zeros((m, 8192), dtype=torch.bfloat16, device=dev)
                    state['attn_boundary'](s, a, branch_dtype=torch.bfloat16, router_dtype=None)
                torch.cuda.synchronize(dev)
            print('P16_NORM_WARM', flush=True)
        except Exception as e:
            print('P16_NORM_WARM_ERROR', repr(e)[-1000:], flush=True)
    _PDLI_NORM_THREADS.append(_okh_bg(warm))
    return state


_achita_layer_prepare = _p16_warm_core


def _p16_project(x, state):
    m = x.shape[0]
    if (m not in (16, 32) or x.dtype != torch.bfloat16 or not x.is_contiguous()
            or x.data_ptr() % 16 or x.shape[1] != 1024 or _P16_EXT is None):
        return _achita_dense_project(x, 'o', state)
    pp = state['p16']['plans'].get(m)
    if pp is None:
        return _achita_dense_project(x, 'o', state)
    out = torch.empty((m, 8192), dtype=torch.bfloat16, device=x.device)
    stream = torch.cuda.current_stream().cuda_stream
    if m == 16:
        wr, wk, ku, split = pp['cfg']
        rc = getattr(_P16_EXT, _P16_M16_EXPORT)(0, pp['wp'].data_ptr(), x.data_ptr(), out.data_ptr(),
            pp['esc'].data_ptr(), m, 8192, 1024, wr, wk, ku, split, pp['t01'], pp['u01'], stream, 0)
    else:
        rc = _P16_EXT.ol_launch(0, x.data_ptr(), pp['packed'].data_ptr(), out.data_ptr(), m, stream, 0)
    if rc:
        raise RuntimeError('P16 launch rc=' + str(rc))
    return out


def _p16_plans(state):
    dense = state['achita_dense']
    mt = _r2_find(dense, 'mt') or {}
    pp = mt.get('plans', {}).get('o')
    if (pp is not None and pp.get('fmt') == 1 and tuple(pp.get('cfg', ())) == (2, 4, 4, 1)
            and tuple(pp.get('inst', ())) == _P16_M16_INST):
        info = getattr(_P16_EXT, _P16_M16_EXPORT)
        regs = info(0, 0, 0, 0, 0, 16, 8192, 1024, 2, 4, 4, 1, 0, 0, 0, 2)
        occ = info(0, 0, 0, 0, 0, 16, 8192, 1024, 2, 4, 4, 1, 0, 0, 0, 1)
        print('P16_RESOURCE O16', regs, occ, flush=True)
        if regs >= 0 and regs % 1000 == 0 and occ > 0:
            state['p16']['plans'][16] = pp
    bundle = _r2_find(dense, 'bundle2') or {}
    ol = bundle.get('o', {})
    cfg = (32, 128, 8, 1, 1, 1)
    if ol.get('enabled') and cfg in ol.get('configs', ()) and ol.get('packed') is not None:
        regs = _P16_EXT.ol_launch(0, 0, 0, 0, 32, 0, 1)
        occ = _P16_EXT.pdlo32_info(1) if regs >= 0 else -1
        print('P16_RESOURCE O32', regs, occ, 'dynamic_smem=196736', flush=True)
        if regs >= 0 and regs % 10000 == 0 and occ > 0:
            state['p16']['plans'][32] = ol


def _p16_data(m, sliding, dev, gen, ctx=80000):
    # Decode B=m,L=1, with disjoint request page domains and mixed lengths.
    stride = ctx + 1
    pages = m * stride
    cache_shape = (pages, 1, 2, 64)
    k = torch.randn(cache_shape, device=dev, generator=gen).to(torch.float8_e4m3fn)
    v = torch.randn(cache_shape, device=dev, generator=gen).to(torch.float8_e4m3fn)
    seq = torch.arange(m, dtype=torch.int32, device=dev) * -251 + ctx
    bases = torch.arange(m, dtype=torch.int64, device=dev) * stride
    return dict(attn_input=torch.randn((m, 8192), device=dev, generator=gen).to(torch.bfloat16),
        stream=torch.randn((m, 8192), device=dev, generator=gen), seq_lens=seq,
        cache_loc=bases + seq.to(torch.int64),
        page_table=torch.arange(pages, dtype=torch.int32, device=dev).reshape(m, stride),
        k_cache=k, v_cache=v, k_scale=torch.ones((), device=dev), v_scale=torch.ones((), device=dev),
        local_start=torch.zeros((1,), dtype=torch.int32, device=dev), scheduler_metadata=None,
        config=dict(batch=m, q_len=1, tokens=m, num_q_heads=16, num_kv_heads=2, head_dim=64,
            page_size=1, window=2048 if sliding else 0, rope_theta=ROPE_THETA if sliding else 0.,
            qk_scale_factor=5.4730064863875, sm_scale=.125, kv_scale=1., n_experts=256))


def _p16_select(state, start):
    _p16_plans(state)
    dev = state['dense']['o'].device
    for m in (16, 32):
        if m not in state['p16']['plans'] or not _okh_budget(start, 274.):
            continue
        gen = torch.Generator(device=dev).manual_seed(20261030 + m)
        # Isolated projection arithmetic/eligibility before the whole layer.
        x = torch.randn((m, 1024), device=dev, generator=gen).to(torch.bfloat16)
        probes = {'parent': lambda: (_achita_dense_project(x, 'o', state),),
                  'pdl': lambda: (_p16_project(x, state),)}
        if not _pdli_probe(probes, lambda seed: x.normal_(generator=gen), label='P16_projection_' + str(m)):
            print('P16_REJECT_PROJECTION', m, flush=True)
            continue
        for sliding in (True, False):
            if not _okh_budget(start, 274.):
                break
            data = _p16_data(m, sliding, dev, gen)
            cache = (data['k_cache'], data['v_cache'])
            tag = 'P16_M' + str(m) + ('_sliding' if sliding else '_full')
            def mutate(seed):
                data['attn_input'].normal_(generator=gen)
                data['stream'].normal_(generator=gen)
            if _pdli_ab(tag, {'parent': lambda: _p16_parent_kernel_fn(data, state),
                            'pdl': lambda: _p16_layer(data, state)}, mutate, dev, start, cache):
                state['p16']['selected'].add((m, bool(sliding)))
            print('P16_SELECT', tag, (m, bool(sliding)) in state['p16']['selected'], flush=True)


_p16_parent_prepare = prepare
_p16_parent_kernel_fn = kernel_fn


def prepare(weights):
    start = _okh_clock.perf_counter()
    state = _p16_parent_prepare(weights)
    state['p16'] = {'plans': {}, 'selected': set()}
    # No compiler thread may remain active during graph capture or timed calls.
    _p16_thread.join()
    if _P16_EXT is not None and _okh_budget(start, 274.):
        try:
            _p16_select(state, start)
        except Exception as e:
            state['p16']['selected'].clear()
            print('P16_ERROR', repr(e)[-1500:], flush=True)
    torch.cuda.synchronize()
    print('P16_DONE', _okh_clock.perf_counter() - start, sorted(state['p16']['selected']), flush=True)
    return state


def kernel_fn(data, state):
    cfg = data['config']
    key = cfg['tokens'], bool(cfg['window'])
    if (cfg['q_len'] == 1 and cfg['batch'] in (16, 32) and cfg['tokens'] == cfg['batch']
            and key in state.get('p16', {}).get('selected', ())):
        return _p16_layer(data, state)
    return _p16_parent_kernel_fn(data, state)
