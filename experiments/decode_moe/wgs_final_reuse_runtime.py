"""Prepare-only WGS read-reuse qualification; appended to the pinned parent."""

_wr_parent_prepare = prepare
_wr_parent_kernel = kernel_fn


class _WRAdapter:
    def __init__(self, module):
        self.okc_run = module.wr_run


def _wr_build(state):
    assert state['sfx'] == ('-DOKC_PRMTSC=1',)
    from torch.utils.cpp_extension import load_inline
    flags = list(MAIN_DEFINES) + list(state['sfx'])
    if DQ7:
        flags.append('-DOKC_DQ7=1')
    module = load_inline(name='okc_wgs_readreuse_v1', cpp_sources=WR_CPP,
        cuda_sources=WR_CUDA, functions=['wr_run', 'wr_set_pdl', 'wr_resources'],
        extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo',
                          '-DOKC_NS=' + str(MAIN_NS), '-DOKC_NSF=3', *flags], verbose=False)
    metadata = module.wr_resources()
    print('WR_RESOURCES', metadata, flush=True)
    assert len(metadata) == 12
    for start in (0, 6):
        regs, local, shared, blocks, dynamic, threads = metadata[start:start + 6]
        assert 0 < regs <= 96 and 0 <= local <= 16 and shared == 0 and blocks >= 1
        # Native attribute/occupancy calls must succeed at this exact size.
        assert threads == 576 and dynamic > 0
    module.wr_set_pdl(1 if PDL else 0)
    return _WRAdapter(module)


def _wr_call(data, state, module):
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].to(torch.int32).contiguous()
    if ids.data_ptr() & 15:
        ids = ids.clone()
    tw = data['topk_weights'].to(torch.float32).contiguous()
    assert 17 <= x.shape[0] <= CHUNK_TOKENS
    assert data['config']['local_start'] == state['lo']
    assert data['config']['local_count'] == state['cnt']
    out = torch.empty((x.shape[0], HIDDEN), dtype=torch.bfloat16, device=x.device)
    module.okc_run(state['w'], state['scratch'], x, ids, tw, out,
                   state['lo'], state['cnt'], state['grid'])
    return out


def _wr_data(m, state):
    dev = state['w'][0].device
    return dict(config=dict(local_start=state['lo'], local_count=state['cnt']),
        hidden_states=torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=dev),
        topk_ids=torch.empty((m, 8), dtype=torch.int32, device=dev),
        topk_weights=torch.empty((m, 8), dtype=torch.float32, device=dev))


def _wr_mutate(data, state, pattern, seed):
    import random
    m = data['hidden_states'].shape[0]
    rng = random.Random(20261031 + 1009 * seed + 31 * m)
    local = list(range(state['lo'], state['lo'] + state['cnt']))
    remote = [e for e in range(256) if e not in local]
    hot = rng.sample(local, 8)
    rows = []
    for t in range(m):
        if pattern == 'normal':
            row = rng.sample(range(256), 8)
        else:
            n = {'none': 0, 'one': 1, 'all': 8, 'skew': 1 + t % 3}[pattern]
            row = rng.sample(hot if pattern == 'skew' else local, n) + rng.sample(remote, 8 - n)
        rows.append(row)
    data['topk_ids'].copy_(torch.tensor(rows, dtype=torch.int32, device=data['topk_ids'].device))
    generator = torch.Generator(device=data['hidden_states'].device).manual_seed(151 + seed)
    data['hidden_states'].normal_(generator=generator)
    if pattern == 'none':
        data['hidden_states'].fill_(float('nan'))
    data['topk_weights'].uniform_(.1, 1., generator=generator)
    data['topk_weights'].div_(data['topk_weights'].sum(-1, keepdim=True))


def _wr_snapshot(state, out, host_ids):
    # WGS writes its live SwiGLU to xB, not the legacy inter or ss buffers.
    jobs, n, _ = jobs_from_routes(host_ids, state['lo'], state['cnt'])
    del jobs
    return (snapshot_wgs_inter(state, host_ids, state['lo'], state['cnt']),
            state['scratch'][1][EMAX:EMAX + n * HIDDEN].clone(), out.clone())


def _wr_bytes(a, b):
    return all(x.shape == y.shape and x.dtype == y.dtype and bool(torch.isfinite(x).all())
               and bool(torch.isfinite(y).all()) and bool(torch.equal(
                   x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))) for x, y in zip(a, b))


def _wr_probe(data, parent, candidate, module):
    _wr_mutate(data, parent, 'normal', 0)
    graphs = []
    for fn in (lambda: _wr_parent_kernel(data, parent), lambda: _wr_call(data, candidate, module)):
        fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fn()
        graphs.append((graph, out))
    for pattern in ('none', 'normal', 'one', 'all', 'skew'):
        _wr_mutate(data, parent, pattern, 3)
        host_ids = data['topk_ids'].cpu().tolist()  # untimed diagnostics only
        snapshots = []
        for index, state in ((0, parent), (1, candidate), (0, parent)):
            before = int(state['scratch'][0][3].item())
            graph, out = graphs[index]
            graph.replay()
            torch.cuda.synchronize()
            assert int(state['scratch'][0][3].item()) == before + 1
            snapshots.append(_wr_snapshot(state, out, host_ids))
        context = str(data['hidden_states'].shape[0]) + ':' + pattern
        assert _wr_bytes(snapshots[0], snapshots[2]), 'unchanged parent repeat differs ' + context
        assert _wr_bytes(snapshots[0], snapshots[1]), 'candidate boundary differs ' + context
        if pattern == 'none':
            assert all(bool((snapshot[-1] == 0).all()) for snapshot in snapshots)
        print('WR_BYTES_PASS', data['hidden_states'].shape[0], pattern,
              'live_xB/ybuf/out;parent-repeat', flush=True)


def prepare(weights):
    start = _okh_clock.perf_counter()
    state = _wr_parent_prepare(weights)
    state['wr_accept'] = False
    if not state['sfx'] or not _okh_budget(start, 252.):
        print('WR_SKIP', 'eligibility_or_build_budget', flush=True)
        return state
    try:
        module = _wr_build(state)  # synchronous; no compilation during capture
        candidate = dict(state, scratch=_make_scratch(weights['w13'].device))
        for m in (17, 31, 32, 33, 128, 256, 257, 384):
            assert _okh_budget(start, 268.), 'probe budget'
            _wr_probe(_wr_data(m, state), state, candidate, module)
        assert _okh_budget(start, 274.), 'marker budget'
        for m in (32, 128):
            data = _wr_data(m, state)
            _wr_mutate(data, state, 'normal', 7)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profile:
                _wr_call(data, candidate, module)
                torch.cuda.synchronize()
            markers = [event.name for event in profile.events() if 'moe_readreuse' in event.name]
            print('WR_MARKERS', m, markers, flush=True)
            assert markers, 'candidate execution marker missing'
        passed = []
        for m in (32, 128, 256):
            data = _wr_data(m, state)
            for pattern in ('normal', 'skew'):
                _wr_mutate(data, state, pattern, 7)
                fns = {'parent': lambda: _wr_parent_kernel(data, state),
                       'candidate': lambda: _wr_call(data, candidate, module)}
                for order in (('parent', 'candidate'), ('candidate', 'parent')):
                    assert _okh_budget(start, 278.), 'timing budget'
                    raw = {}
                    med = _okh_time({key: fns[key] for key in order}, weights['w13'].device, reps=25, samples=raw)
                    passed.append(med['candidate'] < .97 * med['parent'])
                    print('WR_TIME', dict(m=m, population=pattern, order=order, median_us=med, raw_us=raw), flush=True)
        if len(passed) == 12 and all(passed):
            for m in (32, 128, 256):
                data = _wr_data(m, state)
                _wr_mutate(data, state, 'normal', 7)
                for order in (('control_a', 'control_b'), ('control_b', 'control_a')):
                    assert _okh_budget(start, 278.), 'control budget'
                    raw = {}
                    med = _okh_time({key: lambda: _wr_parent_kernel(data, state) for key in order},
                                    weights['w13'].device, reps=25, samples=raw)
                    print('WR_CONTROL', dict(m=m, order=order, median_us=med, raw_us=raw), flush=True)
                    assert .98 <= med['control_a'] / med['control_b'] <= 1.02, 'unchanged timing control drift'
            state['wr_accept'] = True
            state['wr_module'] = module
        print('WR_ACCEPT', state['wr_accept'], flush=True)
    except Exception as error:
        state['wr_accept'] = False
        print('WR_REJECT', repr(error)[-1800:], flush=True)
    torch.cuda.synchronize()
    print('WR_DONE', _okh_clock.perf_counter() - start, state['wr_accept'], flush=True)
    return state


def kernel_fn(data, state):
    m = data['hidden_states'].shape[0]
    if state.get('wr_accept', False) and 17 <= m <= CHUNK_TOKENS:
        return _wr_call(data, state, state['wr_module'])
    return _wr_parent_kernel(data, state)
